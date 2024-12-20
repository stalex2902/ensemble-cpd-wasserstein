"""Module for experiments' dataset methods."""

import collections
import os
import pickle
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import av
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pytorchvideo.transforms import ShortSideScale
from src.utils.fix_seeds import fix_seeds
from torch.utils.data import Dataset, Subset
from torchvision.datasets.video_utils import VideoClips
from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import (
    CenterCropVideo,
    NormalizeVideo,
)
from tqdm import tqdm

collections.Iterable = collections.abc.Iterable


class CPDDatasets:
    """Class for experiments' datasets."""

    def __init__(
        self,
        experiments_name: str,
        random_seed: int = 123,
        train_anomaly_num: int = None,
    ) -> None:
        """Initialize dataset class.

        :param experiments_name: type of experiments
            Available now:
            - "human_activity"
            - "explosion"
            - "road_accidents"
        :param random_seed: seed for reproducibility, default = 123
        """
        super().__init__()
        self.random_seed = random_seed
        self.train_anomaly_num = train_anomaly_num

        if experiments_name in [
            "human_activity",
            "explosion",
            "road_accidents",
            "yahoo",
        ]:
            self.experiments_name = experiments_name
        else:
            raise ValueError("Wrong experiment_name {}.".format(experiments_name))

    def get_dataset_(self) -> Tuple[Dataset, Dataset]:
        """Load experiments' dataset follow the paper's settings.

        See https://dl.acm.org/doi/abs/10.1145/3503161.3548182
        """
        train_dataset = None
        test_dataset = None

        if self.experiments_name == "human_activity":
            path_to_data = "/data/human_activity/"
            train_dataset = HumanActivityDataset(
                path_to_data=path_to_data, seq_len=20, train_flag=True
            )
            test_dataset = HumanActivityDataset(
                path_to_data=path_to_data, seq_len=20, train_flag=False
            )

        elif self.experiments_name in ["explosion", "road_accidents"]:
            # default initialization for explosion
            path_to_data = "/data/explosion/"
            path_to_train_annotation = path_to_data + "UCF_train_time_markup.txt"
            path_to_test_annotation = path_to_data + "UCF_test_time_markup.txt"

            if self.experiments_name == "road_accidents":
                path_to_data = "../../data/road_accidents/"
                path_to_train_annotation = (
                    path_to_data + "UCF_road_train_time_markup.txt"
                )
                path_to_test_annotation = path_to_data + "UCF_road_test_time_markup.txt"

            # apply transformation before 3D CNN feature extractor
            # https://pytorch.org/hub/facebookresearch_pytorchvideo_resnet/
            mean = [0.45, 0.45, 0.45]
            std = [0.225, 0.225, 0.225]
            side_size = 256
            crop_size = 256

            transform = Compose(
                [
                    Lambda(lambda x: x / 255.0),
                    NormalizeVideo(mean, std),
                    ShortSideScale(size=side_size),
                    CenterCropVideo(crop_size=(crop_size, crop_size)),
                ]
            )

            train_dataset = UCFVideoDataset(
                clip_length_in_frames=16,
                step_between_clips=5,
                path_to_data=path_to_data,
                path_to_annotation=path_to_train_annotation,
                video_transform=transform,
                num_workers=0,
                fps=30,
                sampler="equal",
            )

            test_dataset = UCFVideoDataset(
                clip_length_in_frames=16,
                step_between_clips=16,
                path_to_data=path_to_data,
                path_to_annotation=path_to_test_annotation,
                video_transform=transform,
                num_workers=0,
                fps=30,
                sampler="downsample_norm",
            )

        elif self.experiments_name == "yahoo":
            path_to_data = "../../data/yahoo/raw_yahoo"
            train_dataset = YAHOODataset(
                path_to_data=path_to_data,
                wnd_size=150,
                step=5,
                dataset_size=None,
                seed=self.random_seed,
                is_train=True,
            )
            test_dataset = YAHOODataset(
                path_to_data=path_to_data,
                wnd_size=1000,
                step=50,
                dataset_size=None,
                seed=self.random_seed,
                is_train=False,
            )

        return train_dataset, test_dataset

    def train_test_split_(
        self, dataset: Dataset, test_size: float = 0.3, shuffle: bool = True
    ) -> Tuple[Dataset, Dataset]:
        """Split dataset on train and test.

        :param dataset: dataset for splitting
        :param test_size: size of test data, default 0.3
        :param shuffle: if True - shuffle data, default True
        :return: tuple of
            - train dataset
            - test dataset
        """
        # fix seeds
        fix_seeds(self.random_seed)

        len_dataset = len(dataset)
        idx = np.arange(len_dataset)

        # train-test split
        if shuffle:
            train_idx = random.sample(list(idx), int((1 - test_size) * len_dataset))
        else:
            train_idx = idx[: -int(test_size * len_dataset)]
        test_idx = np.setdiff1d(idx, train_idx)

        train_set = Subset(dataset, train_idx)
        test_set = Subset(dataset, test_idx)
        return train_set, test_set


class HumanActivityDataset(Dataset):
    """Class for Dataset of HAR time series."""

    def __init__(
        self, path_to_data: str, seq_len: int = 20, train_flag: bool = True
    ) -> None:
        """Initialize HAR dataset.

        :param path_to_data: path to data with HAR datasets (train and test)
        :param seq_len: length of considered time series
        :param train_flag: if True - load train dataset (files are separated), default True
        """
        super().__init__()

        self.path_to_data = path_to_data
        self.seq_len = seq_len
        self.train_flag = train_flag

        self.data = self._load_data()
        normal_data, normal_labels = self._generate_normal_data()
        anomaly_data, anomaly_labels = self._generate_anomaly_data()

        self.features = normal_data + anomaly_data
        self.labels = normal_labels + anomaly_labels

    def _load_data(self) -> pd.DataFrame:
        """Load HAR dataset.

        :return: dataframe with HAR data
        """
        if self.train_flag:
            type_ = "train"
        else:
            type_ = "test"

        total_path = self.path_to_data + "/T" + type_[1:]
        data_path = total_path + "/X_" + type_ + ".txt"
        labels_path = total_path + "/y_" + type_ + ".txt"
        subjects_path = total_path + "/subject_id_" + type_ + ".txt"

        data = pd.read_csv(data_path, sep=" ", header=None)
        names = pd.read_csv(self.path_to_data + "/features.txt", header=None)
        data.columns = [x.replace(" ", "") for x in names[0].values]

        labels = pd.read_csv(labels_path, sep=" ", header=None)
        subjects = pd.read_csv(subjects_path, sep=" ", header=None)

        data["subject"] = subjects
        data["labels"] = labels

        return data

    def _generate_normal_data(self) -> Tuple[List[float], List[int]]:
        """Get normal sequences from data.

        :return: tuple of
            - sequences with time series
            - sequences with labels
        """
        slices = []
        labels = []

        for sub in self.data["subject"].unique():
            tmp = self.data[self.data.subject == sub]
            # labels 7 - 12 characterize the change points
            tmp = tmp[~tmp["labels"].isin([7, 8, 9, 10, 11, 12])]
            normal_ends_idxs = np.where(np.diff(tmp["labels"].values) != 0)[0]

            start_idx = 0

            for i in range(0, len(normal_ends_idxs) - 1):
                # get data before change
                end_idx = normal_ends_idxs[i] + 1
                slice_data = tmp.iloc[start_idx:end_idx]

                # if slice len is enough, generate data
                if len(slice_data) > self.seq_len:
                    for j in range(0, len(slice_data) - self.seq_len):
                        slices.append(slice_data[j : j + self.seq_len])
                        labels.append(np.zeros(self.seq_len))
                start_idx = end_idx
        return slices, labels

    def _generate_anomaly_data(self):
        slices = []
        labels = []

        for sub in self.data["subject"].unique():
            tmp = self.data[self.data.subject == sub]
            # labels 7 - 12 characterize the change points
            tmp_change_only = tmp[tmp.labels.isin([7, 8, 9, 10, 11, 12])]
            # find change indexes
            change_idxs = np.where(np.diff(tmp_change_only["labels"].values) != 0)[0]
            change_idxs = [tmp_change_only.index[0]] + list(
                tmp_change_only.index[change_idxs + 1]
            )
            change_idxs = [-1] + change_idxs

            for i in range(1, len(change_idxs) - 1):
                curr_change = change_idxs[i]

                # find subsequences with change with maximum length
                start_idx = max(change_idxs[i - 1] + 1, curr_change - self.seq_len)
                end_idx = min(change_idxs[i + 1] - 1, curr_change + self.seq_len)
                slice_data = tmp.loc[start_idx:end_idx]

                curr_change = list(slice_data.index).index(curr_change)

                # set labels: 0 - before change point, 1 - after
                slice_labels = np.zeros(len(slice_data))
                slice_labels[curr_change:] = np.ones(len(slice_data) - curr_change)

                if len(slice_data) > self.seq_len:
                    for j in range(0, len(slice_data) - self.seq_len):
                        slices.append(slice_data.iloc[j : j + self.seq_len])
                        labels.append(slice_labels[j : j + self.seq_len])
        return slices, labels

    def __len__(self) -> int:
        """Get datasets' length.

        :return: length of dataset
        """
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[np.array, np.array]:
        """Get one time series and corresponded labels.

        :param idx: index of element in dataset
        :return: tuple of
             - time series
             - sequence of labels
        """
        # TODO: fix hardcode
        sel_features = [
            "tBodyAcc-Mean-1",
            "tBodyAcc-Mean-2",
            "tBodyAcc-Mean-3",
            "tGravityAcc-Mean-1",
            "tGravityAcc-Mean-2",
            "tGravityAcc-Mean-3",
            "tBodyAccJerk-Mean-1",
            "tBodyAccJerk-Mean-2",
            "tBodyAccJerk-Mean-3",
            "tBodyGyro-Mean-1",
            "tBodyGyro-Mean-2",
            "tBodyGyro-Mean-3",
            "tBodyGyroJerk-Mean-1",
            "tBodyGyroJerk-Mean-2",
            "tBodyGyroJerk-Mean-3",
            "tBodyAccMag-Mean-1",
            "tGravityAccMag-Mean-1",
            "tBodyAccJerkMag-Mean-1",
            "tBodyGyroMag-Mean-1",
            "tBodyGyroJerkMag-Mean-1",
            "fBodyAcc-Mean-1",
            "fBodyAcc-Mean-2",
            "fBodyAcc-Mean-3",
            "fBodyGyro-Mean-1",
            "fBodyGyro-Mean-2",
            "fBodyGyro-Mean-3",
            "fBodyAccMag-Mean-1",
            "fBodyAccJerkMag-Mean-1",
            "fBodyGyroMag-Mean-1",
            "fBodyGyroJerkMag-Mean-1",
        ]
        return self.features[idx][sel_features].iloc[:, :-2].values, self.labels[idx]


class UCFVideoDataset(Dataset):
    """Class for UCF video dataset."""

    def __init__(
        self,
        clip_length_in_frames: int,
        step_between_clips: int,
        path_to_data: str,
        path_to_annotation: str,
        video_transform: Optional[nn.Module] = None,
        num_workers: int = 0,
        fps: int = 30,
        sampler: str = "all",
        train_anomaly_num: int = None,
    ) -> None:
        """Initialize UCF video dataset.

        :param clip_length_in_frames: length of clip
        :param step_between_clips: step between clips
        :param path_to_data: path to video data
        :param path_to_annotation: path to annotation labelling
        :param video_transform: necessary transformation
        :param num_workers: num of CPUs
        :param fps: frames per second
        :param sampler: type of sampling:
            - equal - number of clips with the anomaly is almost equal to the number of normal clips
            - downsample_norm - downsample number of normal clips
        """
        super().__init__()

        self.clip_length_in_frames = clip_length_in_frames
        self.step_between_clips = step_between_clips
        self.num_workers = num_workers
        self.fps = fps

        # IO
        self.path_to_data = path_to_data

        self.video_list = UCFVideoDataset._get_video_list(
            dataset_path=self.path_to_data
        )

        # annotation loading
        dict_metadata, _ = UCFVideoDataset._parse_annotation(path_to_annotation)

        path_to_clips = "{}_clips_len_{}_step_{}_fps_{}.pth".format(
            self.path_to_data.split("/")[1],  #!!!!!!!!!!!!!!
            self.clip_length_in_frames,
            self.step_between_clips,
            self.fps,
        )
        if "train" in path_to_annotation:
            path_to_clips = "saves/train_" + path_to_clips
        else:
            path_to_clips = "saves/test_" + path_to_clips

        if os.path.exists(path_to_clips):
            with open(path_to_clips, "rb") as clips:
                self.video_clips = pickle.load(clips)
        else:
            # data loading
            self.video_clips = VideoClips(
                video_paths=self.video_list,
                clip_length_in_frames=self.clip_length_in_frames,
                frames_between_clips=self.step_between_clips,
                frame_rate=self.fps,
                num_workers=self.num_workers,
            )

            # labelling
            self.video_clips.compute_clips(
                self.clip_length_in_frames, self.step_between_clips, self.fps
            )
            self._set_labels_to_clip(dict_metadata)

        # transforms and samplers
        self.video_transform = video_transform
        self.sampler = sampler

        # save dataset
        if not os.path.exists(path_to_clips):
            with open(path_to_clips, "wb") as clips:
                pickle.dump(self.video_clips, clips, protocol=pickle.HIGHEST_PROTOCOL)

        if sampler == "equal":
            self.video_clips.valid_idxs = self._equal_sampling()

        elif sampler == "downsample_anomaly" and train_anomaly_num is not None:
            self.video_clips.valid_idxs = self._equal_sampling(
                downsample_anomaly=train_anomaly_num
            )

        elif sampler == "downsample_norm":
            self.video_clips.valid_idxs = self._equal_sampling(downsample_normal=300)

        elif sampler == "all":
            pass

        else:
            raise ValueError("Wrong type of sampling")

    def __len__(self) -> int:
        """Get length of dataset."""
        return len(self.video_clips.valid_idxs)

    def __getitem__(self, idx: int) -> Union[np.array, np.array]:
        """Get clip and proper labelling with for a given index.

        :param idx: index
        :return: video and labels
        """
        idx = self.video_clips.valid_idxs[idx]
        video, _, _, _ = self.video_clips.get_clip(idx)
        video_idx, clip_idx = self.video_clips.get_clip_location(idx)
        label = np.array(self.video_clips.labels[video_idx][clip_idx], dtype=int)
        # should be channel, seq_len, height, width
        video = video.permute(3, 0, 1, 2)
        if self.video_transform is not None:
            video = self.video_transform(video)
        return video, label

    def _set_labels_to_clip(self, dict_metadata) -> None:
        self.video_clips.labels = []
        self.video_clips.valid_idxs = list(range(0, len(self.video_clips)))
        self.video_clips.normal_idxs = defaultdict(list)
        self.video_clips.cp_idxs = defaultdict(list)

        global_clip_idx = -1
        for video_idx, vid_clips in tqdm(
            enumerate(self.video_clips.clips), total=len(self.video_clips.clips)
        ):
            video_labels = []

            video_path = self.video_clips.video_paths[video_idx]
            video_name = os.path.basename(video_path)

            # get time unit to map frame with its time appearance
            time_unit = (
                av.open(video_path, metadata_errors="ignore").streams[0].time_base
            )

            # get start, change point, end from annotation
            annotated_time = dict_metadata[video_name]

            for clip in vid_clips:
                clip_labels = []
                global_clip_idx += 1

                clip_start = float(time_unit * clip[0].item())
                clip_end = float(time_unit * clip[-1].item())

                change_point = None
                not_suit_flag = True
                for start_time, change_point, end_time in annotated_time:
                    if end_time != -1.0:
                        if (
                            (clip_end > end_time)
                            or (clip_start > end_time)
                            or (clip_start < start_time)
                        ):
                            continue
                    else:
                        if clip_start < start_time:
                            continue
                    if (clip_start > change_point) and (change_point != -1.0):
                        # "abnormal" clip appears after change point
                        continue
                    else:
                        not_suit_flag = False
                        # proper clip
                        break

                # change_point is a last values which lead to the break
                if not_suit_flag:
                    # drop clip idx from dataset
                    video_labels.append([])
                    self.video_clips.valid_idxs.remove(global_clip_idx)
                else:
                    if "Normal" in video_path:
                        clip_labels = list(np.zeros(len(clip)))
                        self.video_clips.normal_idxs[video_path].append(global_clip_idx)
                    else:
                        for frame in clip:
                            frame_time = float(time_unit * frame.item())
                            # due to different rounding while moving from frame to time
                            # the true change point is delayed by ~1 frame
                            # so, we've added the little margin
                            if (frame_time >= change_point - 1e-6) and (
                                change_point != -1.0
                            ):
                                clip_labels.append(1)
                            else:
                                clip_labels.append(0)
                        if sum(clip_labels) > 0:
                            self.video_clips.cp_idxs[video_path].append(global_clip_idx)
                        else:
                            self.video_clips.normal_idxs[video_path].append(
                                global_clip_idx
                            )
                    video_labels.append(clip_labels)
            self.video_clips.labels.append(video_labels)

    @staticmethod
    def _get_video_list(dataset_path: str) -> List[str]:
        """Get list of all videos in data folder."""
        assert os.path.exists(
            dataset_path
        ), "VideoIter:: failed to locate: `{}'".format(dataset_path)
        vid_list = []
        for path, _, files in os.walk(dataset_path):
            for name in files:
                if "mp4" not in name:
                    continue
                vid_list.append(os.path.join(path, name))
        return vid_list

    @staticmethod
    def _parse_annotation(path_to_annotation: str) -> Tuple[dict, dict]:
        """Parse file with labelling."""
        dict_metadata = defaultdict(list)
        dict_types = defaultdict(list)

        with open(path_to_annotation) as file:
            metadata = file.readlines()
            for annotation in metadata:
                # parse annotation
                annotation = annotation.replace("\n", "")
                annotation = annotation.split("  ")
                video_name = annotation[0]
                video_type = annotation[1]
                change_time = float(annotation[3])
                video_borders = (float(annotation[2]), float(annotation[4]))
                dict_metadata[video_name].append(
                    (video_borders[0], change_time, video_borders[1])
                )
                dict_types[video_name].append(video_type)
        return dict_metadata, dict_types

    def _equal_sampling(
        self, downsample_normal=None, downsample_anomaly=None
    ) -> List[int]:
        """Balance sets with and without changes.

        The equal number of clips are sampled from each video (if possible).
        """

        def _get_indexes(dict_normal: Dict[str, int], dict_cp: Dict[str, int]):
            """Get indexes of clips from totally normal video and from video with anomaly."""
            # TODO: Fix unnecessary use of a comprehension
            _cp_idxs = list(np.concatenate([idx for idx in dict_cp.values()]))

            _normal_idxs = []
            _normal_cp_idxs = []

            for key, value in zip(dict_normal.keys(), dict_normal.values()):
                if key in dict_cp.keys():
                    _normal_cp_idxs.extend(value)
                else:
                    _normal_idxs.extend(value)
            return _cp_idxs, _normal_idxs, _normal_cp_idxs

        def _uniform_sampling(
            paths: str, dict_for_sampling: Dict[str, List[int]], max_size: int
        ):
            """Uniform sampling  of clips (with determinate step size)."""
            _sample_idxs = []
            for path in paths:
                idxs = dict_for_sampling[path]
                if len(idxs) > max_size:
                    step = len(idxs) // max_size
                    _sample_idxs.extend(idxs[::step][:max_size])
                else:
                    _sample_idxs.extend(idxs[:max_size])
            return _sample_idxs

        def _random_sampling(
            idxs_for_sampling: np.array, max_size: int, random_seed: int = 123
        ):
            """Random sampling  of clips."""
            assert len(idxs_for_sampling) >= max_size, "Trying to sample too many clips"

            fix_seeds(random_seed)
            _sample_idxs = random.choices(idxs_for_sampling, k=max_size)
            return _sample_idxs

        sample_idxs = []

        cp_paths = self.video_clips.cp_idxs.keys()
        normal_paths = [
            x for x in self.video_clips.normal_idxs.keys() if x not in cp_paths
        ]
        normal_cp_paths = [
            x for x in self.video_clips.normal_idxs.keys() if x in cp_paths
        ]

        cp_idxs, normal_idxs, normal_cp_idxs = _get_indexes(
            self.video_clips.normal_idxs, self.video_clips.cp_idxs
        )

        cp_number = len(cp_idxs)

        if downsample_normal is not None:
            max_samples = downsample_normal
        else:
            max_samples = cp_number

        # sample ~50% of normal clips from video with change point
        if len(cp_idxs) > len(normal_cp_paths):
            normal_from_cp = _uniform_sampling(
                paths=normal_cp_paths,
                dict_for_sampling=self.video_clips.normal_idxs,
                max_size=max_samples // (len(normal_cp_paths) * 2),
            )
        else:
            # print("Equal sampling is impossible, do random sampling.")
            normal_from_cp = _random_sampling(
                idxs_for_sampling=normal_cp_idxs, max_size=max_samples // 2
            )

        # sample ~50% of normal clips from normal video
        max_rest = max_samples - len(normal_from_cp)

        if max_rest > len(self.video_clips.normal_idxs.keys()):
            normal = _uniform_sampling(
                paths=normal_paths,
                dict_for_sampling=self.video_clips.normal_idxs,
                max_size=max_rest // len(normal_paths),
            )

        else:
            # print("Equal sampling is impossible, do random sampling.")
            normal = _random_sampling(idxs_for_sampling=normal_idxs, max_size=max_rest)

        # sometimes it's still not enough because of different video length
        if len(normal_from_cp) + len(normal) < max_samples:
            extra = _random_sampling(
                idxs_for_sampling=normal_idxs + normal_cp_idxs,
                max_size=max_samples - len(normal_from_cp) - len(normal),
            )
        else:
            extra = []

        # controlling number of anomaly clips
        if downsample_anomaly is not None:
            cp_idxs = _random_sampling(cp_idxs, max_size=downsample_anomaly)

        sample_idxs = cp_idxs + normal_from_cp + normal + extra
        return sample_idxs


class YAHOODataset(Dataset):
    """Class for YAHOO! dataset."""

    def __init__(
        self,
        path_to_data: str,
        wnd_size: int,
        step: int,
        dataset_size: int,
        seed: int,
        is_train: bool,
    ) -> None:
        """Initialize YAHOODataset. Extract subsequences with/without CPs from the long source time series.

        :param path_to_data - path to the folder with raw YAHOO data
        :param wnd_size - size of the sequnces to be extracted
        :param step - margin between extracted sequences
        :param dataset_size - desired dataset size (total: train + test)
        :param seed - seed to be fixed for reproducibility
        """
        super().__init__()

        fix_seeds(seed)

        data, cp_idxs = self.extract_windows(
            Path(path_to_data) / "A4Benchmark", wnd_size, step, is_train
        )

        if (dataset_size is None) or (
            dataset_size is not None and dataset_size > len(data)
        ):
            self.data, self.cp_idxs = data, cp_idxs
        else:
            self.data, self.cp_idxs = data[:dataset_size,], cp_idxs[:dataset_size,]

        # permute train dataset
        if is_train:
            perm = np.random.permutation(len(self.data))
            self.data, self.cp_idxs = self.data[perm], self.cp_idxs[perm]

    def extract_windows(
        self, path: Path, window_size: int, step: int, is_train: bool
    ) -> Tuple[np.array, np.array]:
        """Extract sequences from the raw YAHOO time series.

        :param path - path to the folder with raw YAHOO data
        :param window_size - size of the sequnces to be extracted
        :param step - margin between extracted sequences
        :return a tuple of:
            * sequences with and without CPs
            * corresponding CP indices (0 in case of no CP in a sequence)
        """

        def _remove_anomalies(ts: np.array, anomalies: np.array) -> np.array:
            """Remove anomalies from raw YAHOO time series."""
            index = np.where(anomalies == 1)[0]
            for i in index:
                if i > 0 and i + 1 < ts.shape[0]:
                    ts[i] = (ts[i - 1] + ts[i + 1]) / 2
                elif i > 0:
                    ts[i] = ts[i - 1]
                if i == 0:
                    ts[i] = ts[i + 1]
            return ts

        if os.path.isdir(path):
            files = os.listdir(path)
        else:
            files = [path]
        files.sort()
        files.remove("A4Benchmark_all.csv")

        train_size = int(0.7 * len(files))
        if is_train:
            files = files[:train_size]
        else:
            files = files[train_size:]

        windows, labels = [], []

        for f in files:
            data = pd.read_csv(path / Path(f))
            cp = data["changepoint"]
            # remove anomalies
            ts = _remove_anomalies(data["value"].values, data["anomaly"])
            # normalize
            ts = (ts - np.min(ts)) / (np.max(ts) - np.min(ts))
            for i in range(0, ts.shape[0] - window_size, step):
                windows.append(np.array(ts[i : i + window_size]))
                is_cp = np.where(cp[i : i + window_size] == 1)[0]
                if is_cp.size == 0:
                    is_cp = [0]
                labels.append(is_cp[0])

        return np.array(windows), np.array(labels)

    def __len__(self) -> int:
        """Get datasets' length.

        :return: length of dataset
        """
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[np.array, np.array]:
        """Get sequence and the corresponding labelling for a given index.

        :param idx: index
        :return a tuple of:
            * sequence
            * CP label mask
        """
        sequence = self.data[idx]
        labels = np.zeros_like(sequence)
        sequence = np.expand_dims(sequence, axis=1)
        cp_idx = self.cp_idxs[idx]
        if cp_idx > 0:
            labels[cp_idx:] = 1

        return sequence, labels


# ------------------------------------------------------------------------------------------------------------#
#                                    Datasets to store model outputs                                          #
# ------------------------------------------------------------------------------------------------------------#


class OutputDataset(Dataset):
    """Fake dataset to store pre-computed all models' outputs for MMD model evaluation."""

    def __init__(self, test_out_bank, test_uncertainties_bank, test_labels_bank):
        super().__init__()

        # every prediction is (batch_size, seq_len)
        self.test_out = list(torch.vstack(test_out_bank))

        self.test_labels = list(torch.vstack(test_labels_bank))

        self.test_uncertainties = list(torch.vstack(test_uncertainties_bank))

    def __len__(self):
        return len(self.test_labels)

    def __getitem__(self, idx):
        return (self.test_out[idx], self.test_uncertainties[idx]), self.test_labels[idx]


class AllModelsOutputDataset(Dataset):
    """Fake dataset to store pre-computed all models' outputs for MMD model evaluation."""

    def __init__(self, test_out_bank, test_labels_bank):
        super().__init__()

        # every prediction is (n_models, batch_size, seq_len)
        self.test_out = list(torch.hstack(test_out_bank).transpose(0, 1))

        self.test_labels = list(torch.vstack(test_labels_bank))

    def __len__(self):
        return len(self.test_labels)

    def __getitem__(self, idx):
        return self.test_out[idx], self.test_labels[idx]
