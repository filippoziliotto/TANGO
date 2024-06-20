#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import gzip
import json
import os
from typing import TYPE_CHECKING, List, Optional

from habitat.core.dataset import Dataset
from habitat.core.registry import registry
from habitat.tasks.open_eqa.open_eqa import OPENEQAEpisode, QuestionData
from habitat.tasks.nav.nav import ShortestPathPoint
from habitat.tasks.nav.nav import NavigationGoal

if TYPE_CHECKING:
    from habitat.config import DictConfig

EQA_HM3D_V1_VAL_EPISODE_COUNT = 557
DEFAULT_SCENE_PATH_PREFIX = "data/scene_datasets/"


@registry.register_dataset(name="OPENEQA-v1")
class OpenEQADatasetV1(Dataset):
    r"""Class inherited from Dataset that loads OPENEQA
    Embodied Question Answering dataset.
    """

    episodes: List[OPENEQAEpisode]

    @staticmethod
    def check_config_paths_exist(config: "DictConfig") -> bool:
        return os.path.exists(config.data_path.format(split=config.split))

    def __init__(self, config: "DictConfig" = None) -> None:
        self.episodes = []

        if config is None:
            return

        with gzip.open(config.data_path.format(split=config.split), "rt") as f:
            self.from_json(f.read(), scenes_dir=config.scenes_dir)

        # with gzip.open('data/datasets/open_eqa/val/val.json.gz', "rt") as f:
        #     self.from_json(f.read(), scenes_dir="data/scene_datasets/")

        self.episodes = list(
            filter(self.build_content_scenes_filter(config), self.episodes)
        )

    def from_json(
        self, json_str: str, scenes_dir: Optional[str] = None
    ) -> None:
        deserialized = json.loads(json_str)
        self.__dict__.update(
            deserialized
        )

        for ep_index, episode in enumerate(deserialized["episodes"]):
            episode = OPENEQAEpisode(**episode)

            if scenes_dir is not None:
                if episode.scene_id.startswith(DEFAULT_SCENE_PATH_PREFIX):
                    episode.scene_id = episode.scene_id[
                        len(DEFAULT_SCENE_PATH_PREFIX) :
                    ]
                episode.scene_id = os.path.join(scenes_dir, episode.scene_id)
            episode.question = QuestionData(**episode.question)

            episode.question.answer_id = ep_index
            episode.question.question_id = ep_index

            for g_index, goal in enumerate(episode.goals):
                episode.goals[g_index] = NavigationGoal(**goal)

            if episode.shortest_paths is not None:
                for path in [episode.shortest_paths]: # List of lists needed for shortest_paths
                    for p_index, point in enumerate(path):
                        path[p_index] = ShortestPathPoint(**point)        
            self.episodes[ep_index] = episode
