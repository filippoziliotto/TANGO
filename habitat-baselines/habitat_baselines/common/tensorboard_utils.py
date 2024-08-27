#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import Any, List, Optional

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter

try:
    import wandb
except ImportError:
    wandb = None


def get_writer(config, **kwargs):
    if config.habitat_baselines.writer_type == "tb":
        return TensorboardWriter(
            config.habitat_baselines.tensorboard_dir, **kwargs
        )
    elif config.habitat_baselines.writer_type == "wb":
        return WeightsAndBiasesWriter(config)
    else:
        raise ValueError("Unrecongized writer")


class TensorboardWriter:
    def __init__(
        self,
        log_dir: str,
        *args: Any,
        resume_run_id: Optional[str] = None,
        **kwargs: Any,
    ):
        r"""A Wrapper for tensorboard SummaryWriter. It creates a dummy writer
        when log_dir is empty string or None. It also has functionality that
        generates tb video directly from numpy images.

        Args:
            log_dir: Save directory location. Will not write to disk if
            log_dir is an empty string.
            *args: Additional positional args for SummaryWriter
            **kwargs: Additional keyword args for SummaryWriter
        """
        self.writer = None
        if log_dir is not None and len(log_dir) > 0:
            self.writer = SummaryWriter(log_dir, *args, **kwargs)

    def get_run_id(self) -> Optional[str]:
        return None

    def __getattr__(self, item):
        if self.writer:
            return self.writer.__getattribute__(item)
        else:
            return lambda *args, **kwargs: None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.writer:
            self.writer.close()

    def add_video_from_np_images(
        self,
        video_name: str,
        step_idx: int,
        images: List[np.ndarray],
        fps: int = 10,
    ) -> None:
        r"""Write video into tensorboard from images frames.

        Args:
            video_name: name of video string.
            step_idx: int of checkpoint index to be displayed.
            images: list of n frames. Each frame is a np.ndarray of shape.
            fps: frame per second for output video.

        Returns:
            None.
        """
        if not self.writer:
            return
        # initial shape of np.ndarray list: N * (H, W, 3)
        frame_tensors = [
            torch.from_numpy(np_arr).unsqueeze(0) for np_arr in images
        ]
        video_tensor = torch.cat(tuple(frame_tensors))
        video_tensor = video_tensor.permute(0, 3, 1, 2).unsqueeze(0)
        # final shape of video tensor: (1, n, 3, H, W)
        self.writer.add_video(
            video_name, video_tensor, fps=fps, global_step=step_idx
        )


class WeightsAndBiasesWriter:
    def __init__(
        self,
        config,
        *args: Any,
        resume_run_id: Optional[str] = None,
        **kwargs: Any,
    ):
        r"""
        Integrates with https://wandb.ai logging service.
        """
        wb_kwargs = {}
        if config.habitat_baselines.wb.project_name != "":
            wb_kwargs["project"] = config.habitat_baselines.wb.project_name
        if config.habitat_baselines.wb.run_name != "":
            wb_kwargs["name"] = config.habitat_baselines.wb.run_name
        if config.habitat_baselines.wb.entity != "":
            wb_kwargs["entity"] = config.habitat_baselines.wb.entity
        if config.habitat_baselines.wb.group != "":
            wb_kwargs["group"] = config.habitat_baselines.wb.group
        slurm_info_dict = {
            k[len("SLURM_") :]: v
            for k, v in os.environ.items()
            if k.startswith("SLURM_")
        }
        if wandb is None:
            raise ValueError(
                "Requested to log with wandb, but wandb is not installed."
            )
        if resume_run_id is not None:
            wb_kwargs["id"] = resume_run_id
            wb_kwargs["resume"] = "must"

        self.run = wandb.init(  # type: ignore[attr-defined]
            config={
                "slurm": slurm_info_dict,
                **OmegaConf.to_container(config),  # type: ignore[arg-type]
            },
            **wb_kwargs,
        )

    def __getattr__(self, item):
        if self.writer:
            return self.writer.__getattribute__(item)
        else:
            return lambda *args, **kwargs: None

    def add_scalars(self, log_group, data_dict, step_id):
        log_data_dict = {
            f"{log_group}/{k.replace(' ', '')}": v
            for k, v in data_dict.items()
        }
        wandb.log(log_data_dict, step=int(step_id))  # type: ignore[attr-defined]

    def add_scalar(self, key, value, step_id):
        wandb.log({key: value}, step=int(step_id))  # type: ignore[attr-defined]

    def __enter__(self):
        return self

    def get_run_id(self) -> Optional[str]:
        return self.run.id

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.run:
            self.run.finish()

    def add_video_from_np_images(
        self, video_name: str, step_idx: int, images: np.ndarray, fps: int = 10
    ) -> None:
        raise NotImplementedError("Not supported")

    def log_results(self, aggregated_stats):
        wandb.log(aggregated_stats)  # type: ignore[attr-defined]


# def log_eqa_results_by_distance(self):
#         #TODO: Distinguish results w.r.t. n° of minimum actions 10/30/50
#         # Support for EQA task infinite values distance_to_goal
#         # also support division in 10/30/50 actions required for shortest path
#         # TODO: log eqa results 10/30/50 inot a table???
#         raise NotImplementedError("Not supported")

#         elif self.task_name in ['eqa-TODO']:
#             eqa_actions_dict = {'10': [], '30': [], '50': []}
#             for _, stats in self.stats_episodes.items():
#                 min_actions = stats['minimum_number_of_actions']
#                 if str(int(min_actions)) in eqa_actions_dict:
#                     eqa_actions_dict[str(int(min_actions))].append(stats)

#             mean_values = {}

#             for key, dict_list in eqa_actions_dict.items():
#                 valid_dicts = [d for d in dict_list if d['distance_to_goal'] != float('inf') and d['smallest_distance_to_target'] != float('inf')]
                
#                 if not valid_dicts:
#                     mean_values[key] = {}
#                     continue
                
#                 sum_dict = {k: sum(d[k] for d in valid_dicts) for k in valid_dicts[0].keys()}
#                 mean_dict = {k: sum_dict[k] / len(valid_dicts) for k in sum_dict.keys()}
                
#                 mean_values[key] = mean_dict
#             self.aggregated_stats = mean_values

#             # Print final results
#             print('-----------------------')
#             print('| EVALUATION FINISHED |')
#             print('-----------------------')

#             for k, v in self.aggregated_stats.items():
#                 print('Number of actions:', k)
#                 for i, j in v.items():
#                     print(f"Average episode {i}: {j:.4f}")
#                 print('-----------------------')

