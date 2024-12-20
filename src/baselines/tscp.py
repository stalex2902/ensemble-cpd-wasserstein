"""Functions & models for TS-CP2 baseline training and testing."""

from typing import List, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.baselines.prediction_utils import *
from torch.utils.data import DataLoader, Dataset

# --------------------------------------------------------------------------------------#
#                                      Loss                                             #
# --------------------------------------------------------------------------------------#


def nce_loss_fn(
    history: torch.Tensor, future: torch.Tensor, similarity, temperature: float = 0.1
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute loss function for TS-CP2 model.

    :param history: tensor with history window slice
    :param future: tensor with future window slice
    :param similarity: distance function
    :param temperature: temperature coeeficient
    :return: tuple of
        - value of loss function
        - mean similarity value between positive samples
        - mean similarity value between negative samples
    """
    try:
        device = history.device
    except AttributeError:
        device = "cpu"

    criterion = torch.nn.BCEWithLogitsLoss()
    N = history.shape[0]
    sim = similarity(history, future)
    pos_sim = torch.exp(torch.diag(sim) / temperature)

    tri_mask = torch.ones((N, N), dtype=bool)
    tri_mask[np.diag_indices(N)] = False

    neg = sim[tri_mask].reshape(N, N - 1)
    all_sim = torch.exp(sim / temperature)

    logits = torch.divide(torch.sum(pos_sim), torch.sum(all_sim, axis=1))

    lbl = torch.ones(history.shape[0]).to(device)
    # categorical cross entropy
    loss = criterion(logits, lbl)

    mean_sim = torch.mean(torch.diag(sim))
    mean_neg = torch.mean(neg)
    return loss, mean_sim, mean_neg


# --------------------------------------------------------------------------------------#
#                                     Models                                            #
# --------------------------------------------------------------------------------------#


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        dilation_rate: int,
        nb_filters: int,
        kernel_size: int,
        dropout_rate: float = 0,
        use_batch_norm: bool = False,
        use_layer_norm: bool = False,
        use_weight_norm: bool = False,
        training: bool = True,
    ) -> None:
        """Defines the residual block for the WaveNet TCN.

        :param in_channels: number of input channels
        :param dilation_rate: the dilation power of 2 we are using for this residual block
        :param nb_filters: the number of convolutional filters to use in this block
        :param kernel_size: the size of the convolutional kernel
        :param dropout_rate: tloat between 0 and 1. Fraction of the input units to drop
        :param use_batch norm: if True, apply batch norm
        :param use_layer_norms: if True, apply normalization along the layer
        :param use_weight_norm: if True, normalize the weights
        :param training: indicates training or testing mode
        """
        self.in_channels = in_channels
        self.dilation_rate = dilation_rate
        self.nb_filters = nb_filters
        self.kernel_size = kernel_size
        self.dropout_rate = dropout_rate

        # for causal padding
        self.padding = (self.kernel_size - 1) * self.dilation_rate

        self.use_batch_norm = use_batch_norm
        self.use_layer_norm = use_layer_norm
        self.use_weight_norm = use_weight_norm

        self.training = training

        super().__init__()

        self.conv_1 = nn.Conv1d(
            self.in_channels,
            self.nb_filters,
            self.kernel_size,
            padding=0,
            dilation=self.dilation_rate,
        )
        # if self.use_weight_norm:
        #    weight_norm(self.conv_1)
        self.bn_1 = nn.BatchNorm1d(self.nb_filters)
        self.ln_1 = nn.LayerNorm(self.nb_filters)
        self.relu_1 = nn.ReLU()

        self.conv_2 = nn.Conv1d(
            self.nb_filters,
            self.nb_filters,
            self.kernel_size,
            padding=0,
            dilation=self.dilation_rate,
        )
        # if self.use_weight_norm:
        #    weight_norm(self.conv_1)
        self.bn_2 = nn.BatchNorm1d(self.nb_filters)
        self.ln_2 = nn.LayerNorm(self.nb_filters)
        self.relu_2 = nn.ReLU()

        self.conv_block = nn.Sequential()
        self.downsample = (
            nn.Conv1d(in_channels, self.nb_filters, kernel_size=1)
            if in_channels != self.nb_filters
            else nn.Identity()
        )

        self.relu = nn.ReLU()

        self.init_weights()

    def init_weights(self) -> None:
        """Initialize weights with Normal distribution."""
        torch.nn.init.normal_(self.conv_1.weight, mean=0, std=0.05)
        torch.nn.init.zeros_(self.conv_1.bias)

        torch.nn.init.normal_(self.conv_2.weight, mean=0, std=0.05)
        torch.nn.init.zeros_(self.conv_2.bias)

        if isinstance(self.downsample, nn.Conv1d):
            torch.nn.init.normal_(self.downsample.weight, mean=0, std=0.05)
            torch.nn.init.zeros_(self.downsample.bias)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        """Do forward pass through the model.

        :param inp: input tensor
        :return: tuple of
            - result (output)
            - skip_out (with skip-connections)
        """
        # do causal padding
        out = F.pad(inp, (self.padding, 0))
        out = self.conv_1(out)

        if self.use_batch_norm:
            out = self.bn_1(out)
        elif self.use_layer_norm:
            out = self.ln_1(out)
        out = self.relu_1(out)

        # spatial dropout
        out = out.permute(0, 2, 1)  # convert to [batch, time, channels]
        out = F.dropout2d(out, self.dropout_rate, training=self.training)
        out = out.permute(0, 2, 1)  # back to [batch, channels, time]

        # do causal padding
        out = F.pad(out, (self.padding, 0))
        out = self.conv_2(out)
        if self.use_batch_norm:
            out = self.bn_2(out)
        elif self.use_layer_norm:
            out = self.ln_2(out)
        out = self.relu_2(out)
        out = self.relu_2(out)

        out = out.permute(0, 2, 1)  # convert to [batch, time, channels]
        out = F.dropout2d(out, self.dropout_rate, training=self.training)
        out = out.permute(0, 2, 1)  # back to [batch, channels, time]

        skip_out = self.downsample(inp)
        res = self.relu(out + skip_out)
        return res, skip_out


class TCN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        nb_filters: int,
        kernel_size: int,
        nb_stacks: int,
        dilations: List[int],
        use_skip_connections: bool,
        dropout_rate: float,
        use_batch_norm: bool,
        use_layer_norm: bool,
        use_weight_norm: bool,
    ) -> None:
        """Initialize TCN model.

        :param in_channels: number of input channels
        :param nb_filters: the number of convolutional filters to use in this block
        :param kernel_size: the size of the convolutional kernel
        :param nb_stacks: number of stacked residual blocks
        :param dilations: dilations (powers of 2) used for this residual block
        :param use_skip_connection: if True, use skip connections
        :param dropout_rate: tloat between 0 and 1. Fraction of the input units to drop
        :param use_batch norm: if True, apply batch norm
        :param use_layer_norms: if True, apply normalization along the layer
        :param use_weight_norm: if True, normalize the weights
        """
        super().__init__()

        self.dropout_rate = dropout_rate
        self.use_skip_connections = use_skip_connections
        self.dilations = dilations
        self.nb_stacks = nb_stacks
        self.kernel_size = kernel_size
        self.nb_filters = nb_filters

        self.use_batch_norm = use_batch_norm
        self.use_layer_norm = use_layer_norm
        self.use_weight_norm = use_weight_norm
        self.in_channels = in_channels

        if self.use_batch_norm + self.use_layer_norm + self.use_weight_norm > 1:
            raise ValueError("Only one normalization can be specified at once.")

        self.residual_blocks = []
        res_block_filters = 0
        for s in range(self.nb_stacks):
            for i, d in enumerate(self.dilations):
                in_channels = self.in_channels if i + s == 0 else res_block_filters
                res_block_filters = (
                    self.nb_filters[i]
                    if isinstance(self.nb_filters, list)
                    else self.nb_filters
                )
                self.residual_blocks.append(
                    ResidualBlock(
                        in_channels=in_channels,
                        dilation_rate=d,
                        nb_filters=res_block_filters,
                        kernel_size=self.kernel_size,
                        dropout_rate=self.dropout_rate,
                        use_batch_norm=self.use_batch_norm,
                        use_layer_norm=self.use_layer_norm,
                        use_weight_norm=self.use_weight_norm,
                    )
                )

        self.residual_blocks = nn.ModuleList(self.residual_blocks)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        """Do forward pass through the model.

        :param inp: input tensor
        :return: output
        """
        out = inp
        for layer in self.residual_blocks:
            out, skip_out = layer(out)
        if self.use_skip_connections:
            out = out + skip_out
        return out


class BaseTSCPEncoder(nn.Module):
    def __init__(self, args: dict) -> None:
        """Initialize TS-CP2 Encoder model for synthtic Normal experiments.

        :param args: dict with all the parameters
        """
        super().__init__()

        self.c_in = args["model"]["c_in"]

        # print("c_in:", self.c_in)

        self.nb_filters = args["model"]["nb_filters"]
        self.kernel_size = args["model"]["kernel_size"]
        self.nb_stacks = args["model"]["nb_stacks"]
        self.dilations = args["model"]["dilations"]
        self.use_skip_connections = args["model"]["use_skip_connections"]
        self.dropout_rate = args["model"]["dropout_rate"]
        self.use_batch_norm = args["model"]["use_batch_norm"]
        self.use_layer_norm = args["model"]["use_layer_norm"]
        self.use_weight_norm = args["model"]["use_weight_norm"]
        self.seq_len = args["model"]["seq_len"]
        self.n_steps = args["model"]["n_steps"]
        self.code_size = args["model"]["code_size"]

        self.tcn_layer = TCN(
            in_channels=self.c_in,
            nb_filters=self.nb_filters,
            kernel_size=self.kernel_size,
            nb_stacks=self.nb_stacks,
            dilations=self.dilations,
            use_skip_connections=self.use_skip_connections,
            dropout_rate=self.dropout_rate,
            use_batch_norm=self.use_batch_norm,
            use_layer_norm=self.use_layer_norm,
            use_weight_norm=self.use_weight_norm,
        )

        self.fc1 = nn.Linear(self.nb_filters * self.seq_len, 2 * self.n_steps)
        self.fc2 = nn.Linear(2 * self.n_steps, self.n_steps)
        self.output_layer = nn.Linear(self.n_steps, self.code_size)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Do forward pass through the model.

        :param x: input tensor
        :return: output
        """
        batch_size, seq_len = x.shape[0], x.shape[1]
        out = x.reshape(
            batch_size, -1, seq_len
        ).float()  # shape is (batch_size, c_in, timesteps)
        out = self.tcn_layer(out)
        out = out.flatten(1, 2)
        out = self.relu(self.fc1(out))
        out = self.relu(self.fc2(out))
        out = self.output_layer(out)
        return out


class TSCP_model(pl.LightningModule):
    def __init__(
        self,
        args: dict,
        model: nn.Module,
        train_dataset: Dataset,
        test_dataset: Dataset,
        dataloader_seed: int = 42,
    ) -> None:
        """Initialize TC-CP2 model (lightning wrapper).

        :param args: dictionary with models' parameters
        :param model: core model (encoder)
        :param train_dataset: train dataset
        :param test_dataset: test dataset
        """
        super().__init__()

        self.model = model
        self.args = args

        # Feature extractor for video datasets
        if args["experiments_name"] in ["explosion", "road_accidents"]:
            print("Loading extractor...")
            self.extractor = torch.hub.load(
                "facebookresearch/pytorchvideo:main", "x3d_m", pretrained=True
            )
            self.extractor = nn.Sequential(*list(self.extractor.blocks[:5]))

            # freeze extractor parameters
            for param in self.extractor.parameters():
                param.requires_grad = False

        else:
            self.extractor = None

        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

        self.batch_size = args["learning"]["batch_size"]
        self.num_workers = args["num_workers"]

        self.temperature = args["loss"]["temperature"]

        self.lr = args["learning"]["lr"]
        self.decay_steps = args["learning"]["decay_steps"]

        self.window = args["model"]["window"]
        self.window_1 = args["model"]["window_1"]
        self.window_2 = args["model"]["window_2"]

        self.experiments_name = args["experiments_name"]

        self.dataloader_seed = dataloader_seed

    def __preprocess(self, input: torch.Tensor) -> torch.Tensor:
        """Preprocess batch before forwarding (i.e. apply extractor for video input).

        :param input: input torch.Tensor
        :return: processed input tensor to be fed into .forward method
        """
        if self.experiments_name in ["explosion", "road_accidents"]:
            input = self.extractor(input.float())  # batch_size, C, seq_len, H, W
            input = input.transpose(1, 2)

        # do nothing for non-video experiments
        return input

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Do forward pass through the model.

        :param inputs: input tensor
        :return: output
        """
        inputs = self.__preprocess(inputs)
        return self.model(inputs)

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """Training step for TS-CP2 model.

        :param batch: input data
        :param batch_idx: index of batch
        :return: loss function value
        """
        history, future = history_future_separation(batch[0], self.window)
        history_emb = self.forward(history.float())
        future_emb = self.forward(future.float())

        history_emb = nn.functional.normalize(history_emb, p=2, dim=1)
        future_emb = nn.functional.normalize(future_emb, p=2, dim=1)

        train_loss, pos_sim, neg_sim = nce_loss_fn(
            history_emb,
            future_emb,
            similarity=_cosine_simililarity_dim2,
            temperature=self.temperature,
        )

        self.log("train_loss", train_loss, prog_bar=True, on_epoch=True)
        self.log("pos_sim", pos_sim, prog_bar=True, on_epoch=True)
        self.log("neg_sim", neg_sim, prog_bar=True, on_epoch=True)

        return train_loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """Validation step for TS-CP2 model.

        :param batch: input data
        :param batch_idx: index of batch
        :return: loss function value
        """
        history, future = history_future_separation(batch[0], self.window)

        history_emb = self.forward(history.float())
        future_emb = self.forward(future.float())

        history_emb = nn.functional.normalize(history_emb, p=2, dim=1)
        future_emb = nn.functional.normalize(future_emb, p=2, dim=1)

        val_loss, _, _ = nce_loss_fn(
            history_emb,
            future_emb,
            similarity=_cosine_simililarity_dim2,
            temperature=self.temperature,
        )

        self.log("val_loss", val_loss, prog_bar=True)
        return val_loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Initialize optimizer.

        :return: optimizer for training TS-CP2 model
        """
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        lr = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=opt, T_max=self.decay_steps
        )
        return {"optimizer": opt, "lr_scheduler": lr}

    def train_dataloader(self) -> DataLoader:
        """Set train dataloader.

        :return: dataloader
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        """Set val dataloader.

        :return: dataloader
        """
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
