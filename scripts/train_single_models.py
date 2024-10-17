from typing import List
import warnings
from datetime import datetime

import numpy as np
import torch
import yaml
from pytorch_lightning.loggers import TensorBoardLogger
from src.datasets.datasets import CPDDatasets
from src.metrics.evaluation_pipelines import evaluation_pipeline
from src.metrics.metrics_utils import write_metrics_to_file
from src.models.model_utils import get_models_list
from src.utils.fix_seeds import fix_seeds

warnings.filterwarnings("ignore")

def train_single_models(
    model_type: str,
    experiments_name: str,
    loss_type: str = None,
    ens_num_list: List[int] = [1, 2, 3],
    num_models_in_ensemble: int = 10,
):

    # read config file
    if experiments_name in ["human_activity", "yahoo"]:
        path_to_config = "configs/" + experiments_name + "_" + model_type + ".yaml"
        device = "cpu"
    elif experiments_name in ["explosion", "road_accidents"]:
        assert model_type == "seq2seq", "Only seq2seq models are used for video data"
        path_to_config = "configs/" + "video" + "_" + model_type + "_" + loss_type + ".yaml"
        device = "cuda:0"
    else:
        raise ValueError(f"Unknown experiments name {experiments_name}")
    
    with open(path_to_config, "r") as f:
        args_config = yaml.safe_load(f.read())

    args_config["experiments_name"] = experiments_name
    args_config["model_type"] = model_type

    args_config["num_workers"] = 2
    args_config["loss_type"] = loss_type

    # prepare datasets
    train_dataset, test_dataset = CPDDatasets(experiments_name).get_dataset_()

    for ens_num in ens_num_list:
        for s in range(num_models_in_ensemble):
            seed = s + 10 * (ens_num - 1)

            fix_seeds(seed)
            model = get_models_list(args_config, train_dataset, test_dataset)[-1]

            if model_type == "seq2seq":
                model_name = (
                    args_config["experiments_name"]
                    + "_"
                    + args_config["loss_type"]
                    + "_model_num_"
                    + str(seed)
                )
            elif model_type in ["tscp", "ts2vec"]:
                model_name = (
                    args_config["experiments_name"]
                    + "_"
                    + args_config["model_type"]
                    + "_model_num_"
                    + str(seed)
                ) 

            logger = TensorBoardLogger(
                save_dir=f'logs/{args_config["experiments_name"]}',
                name=f'{args_config["model_type"]}_{args_config["loss_type"]}',
            )

            trainer = Trainer(
                max_epochs=args_config["learning"]["epochs"],
                accelerator=device,
                benchmark=True,
                check_val_every_n_epoch=1,
                gradient_clip_val=args_config["learning"]["grad_clip"],
                logger=logger,
                callbacks=EarlyStopping(**args_config["early_stopping"]),
            )

            trainer.fit(model)
            
            if model_type == "seq2seq":
                step, alpha = None, None
                path_to_models_folder = f"saved_models/{loss_type}/{experiments_name}/ens_{ens_num}"
                path_to_metrics = f"results/{args_config['loss_type']}/{experiments_name}/single_model_results.txt"
            
            elif model_type in ["tscp", "ts2vec"]:
                step, alpha = args_config["predictions"].values()
                path_to_models_folder = f"saved_models/{model_type}/{experiments_name}/ens_{ens_num}"
                path_to_metrics = f"results/{args_config['model_type']}/{experiments_name}/single_model_results.txt"

            torch.save(
                model.state_dict(), f"{path_to_models_folder}/{model_name}.pth",
            )

            model.load_state_dict(
                torch.load(f"{path_to_models_folder}/{model_name}.pth")
            )

            model.eval()

            threshold_number = 300
            threshold_list = np.linspace(-5, 5, threshold_number)
            threshold_list = 1 / (1 + np.exp(-threshold_list))
            threshold_list = [-0.001] + list(threshold_list) + [1.001]

            all_metrics = evaluation_pipeline(
                model,
                model.val_dataloader(),
                threshold_list,
                device=device,
                model_type=model_type,
                step=step, 
                alpha=alpha,
                verbose=True,
                margin_list=args_config["evaluation"]["margin_list"],
            )
            
            write_metrics_to_file(
                filename=path_to_metrics,
                metrics=all_metrics,
                seed=None,
                timestamp=datetime.now().strftime("%y%m%dT%H%M%S"),
                comment=f"{experiments_name}, {args_config['model_type']}, {args_config['loss_type']}, seed = {seed}",
            )
