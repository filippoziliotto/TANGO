#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import gzip
import json
import os
from typing import TYPE_CHECKING, List, Optional

from omegaconf import OmegaConf

from habitat.config.default_structured_configs import DatasetConfig
from habitat.core.dataset import Dataset
from habitat.core.registry import registry
from habitat.core.simulator import AgentState
from habitat.datasets.utils import VocabDict
from habitat.tasks.eqa.eqa import EQAEpisode, QuestionData, stoi_eqa_map
from habitat.tasks.nav.nav import ShortestPathPoint
from habitat.tasks.nav.object_nav_task import ObjectGoal

if TYPE_CHECKING:
    from habitat.config import DictConfig


EQA_MP3D_V1_VAL_EPISODE_COUNT = 1950
DEFAULT_SCENE_PATH_PREFIX = "data/scene_datasets/"


def get_default_mp3d_v1_config(split: str = "val") -> "DictConfig":
    return OmegaConf.create(  # type: ignore[call-overload]
        DatasetConfig(
            type="MP3DEQA-v1",
            split=split,
            data_path="data/datasets/eqa/mp3d/v1/{split}.json.gz",
        )
    )


@registry.register_dataset(name="MP3DEQA-v1")
class Matterport3dDatasetV1(Dataset):
    r"""Class inherited from Dataset that loads Matterport3D
    Embodied Question Answering dataset.

    This class can then be used as follows::
        eqa_config.habitat.dataset = get_default_mp3d_v1_config()
        eqa = habitat.make_task(eqa_config.habitat.task_name, config=eqa_config)
    """

    episodes: List[EQAEpisode]
    answer_vocab: VocabDict
    question_vocab: VocabDict

    @staticmethod
    def check_config_paths_exist(config: "DictConfig") -> bool:
        return os.path.exists(config.data_path.format(split=config.split))

    def __init__(self, config: "DictConfig" = None) -> None:
        self.episodes = []

        if config is None:
            return
        
        self.eqa_mapping = config.eqa_mapping

        with gzip.open(config.data_path.format(split=config.split), "rt") as f:
            self.from_json(f.read(), scenes_dir=config.scenes_dir)

        self.episodes = list(
            filter(self.build_content_scenes_filter(config), self.episodes)
        )

    def from_json(
        self, json_str: str, scenes_dir: Optional[str] = None
    ) -> None:
        deserialized = json.loads(json_str)
        self.__dict__.update(
            deserialized
        )  # This is a messy hack... Why do we do this.
        self.answer_vocab = VocabDict(
            word_list=self.answer_vocab["word_list"]  # type: ignore
        )
        self.question_vocab = VocabDict(
            word_list=self.question_vocab["word_list"]  # type: ignore
        )

        for ep_index, episode in enumerate(deserialized["episodes"]):
            episode = EQAEpisode(**episode)
            if scenes_dir is not None:
                if episode.scene_id.startswith(DEFAULT_SCENE_PATH_PREFIX):
                    episode.scene_id = episode.scene_id[
                        len(DEFAULT_SCENE_PATH_PREFIX) :
                    ]
                episode.scene_id = os.path.join(scenes_dir, episode.scene_id)
            episode.question = QuestionData(**episode.question)

            # Support EQA dictionary format additions
            episode.question.answer_token_orig = episode.question.answer_token

            try: 
                # Support for EQA in Navprog with simplified answer space
                if self.eqa_mapping not in ["default"]:
                    episode.question.answer_text = stoi_eqa_map[episode.question.answer_text]

                episode.question.answer_token = self.answer_vocab.stoi[episode.question.answer_text]
                
            except: 
                episode.question.answer_token = self.answer_vocab.stoi['<unk>']
            
            for g_index, goal in enumerate(episode.goals):
                episode.goals[g_index] = ObjectGoal(**goal)
                new_goal = episode.goals[g_index]
                if new_goal.view_points is not None:
                    for p_index, agent_state in enumerate(
                        new_goal.view_points
                    ):
                        new_goal.view_points[p_index] = AgentState(
                            **agent_state
                        )
            if episode.shortest_paths is not None:
                for path in episode.shortest_paths:
                    for p_index, point in enumerate(path):
                        path[p_index] = ShortestPathPoint(**point)

            # Support for EQA in Navprog
            if episode.question.question_type in ['location']:
                eqa_object = episode.question.question_text.split('is the')[1].split('located')[0].strip()
                eqa_room = None
            elif episode.question.question_type in ['color_room']:
                eqa_object = episode.question.question_text.split('is the')[1].split('in the')[0].strip()
                eqa_room = episode.question.question_text.split('in the')[1].split('?')[0].strip()
            elif episode.question.question_type in ['color']:
                eqa_object = episode.question.question_text.split('is the')[1].split('?')[0].strip()
                eqa_room = None
            else:
                raise ValueError(f"Unknown question type: {episode.question.question_type}")
            episode.question.eqa_object = eqa_object
            episode.question.eqa_room = eqa_room
             
            self.episodes[ep_index] = episode

        ##### TYPE: only  certain type objetcs
        #from habitat_baselines.rl.ppo.utils.names import class_names_coco
        # self.episodes = [episode for episode in self.episodes if episode.question.eqa_object in ["bed"]]

        for episode in self.episodes:
            if (episode.scene_id == 'data/scene_datasets/mp3d/EU6Fwq7SyZv/EU6Fwq7SyZv.glb') and (episode.question.eqa_object in ["bed"]) and (not episode.question.question_type in ['location']):
                episode.question.answer_text = "white"
                episode.question.answer_token = self.answer_vocab.stoi[episode.question.answer_text]

        for episode in self.episodes:
            if (episode.scene_id == 'data/scene_datasets/mp3d/EU6Fwq7SyZv/EU6Fwq7SyZv.glb') and (episode.question.eqa_object in ["tv stand"]) and (episode.question.question_type in ['location']):
                episode.question.answer_text = "living room"
                episode.question.answer_token = self.answer_vocab.stoi[episode.question.answer_text]

        # Debug single scene
        # scene  = 'data/scene_datasets/mp3d/QUCTc6BB5sX/QUCTc6BB5sX.glb'
        # self.episodes = [episode for episode in self.episodes if episode.scene_id == scene]

        # remove episodes where episode.scene_id == 'data/scene_datasets/mp3d/EU6Fwq7SyZv/EU6Fwq7SyZv.glb'
        # self.episodes = [episode for episode in self.episodes if episode.scene_id != 'data/scene_datasets/mp3d/EU6Fwq7SyZv/EU6Fwq7SyZv.glb']



        ##### TYPE: location "which room" questions
        # self.episodes = [episode for episode in self.episodes if episode.question.question_type in ['location']]
        # unique_objects = set([episode.question.question_text.split('is the')[1].split('located')[0].strip() for episode in self.episodes])
            
        ##### TYPE: color_room "what color is the {object} in {location}"
        # self.episodes = [episode for episode in self.episodes if episode.question.question_type in ['color_room']]
        # unique_objects = set([episode.question.question_text.split('is the')[1].split('in the')[0].strip() for episode in self.episodes])

        ##### TYPE:color "which color {object}"
        # from habitat_baselines.rl.ppo.utils.names import eqa_objects
        # self.episodes = [episode for episode in self.episodes if episode.question.question_type in ['color']]
        # self.episodes = [episode for episode in self.episodes if episode.question.eqa_object in list(eqa_objects.keys())]
        # unique_objects = set([episode.question.question_text.split('is the')[1].split('?')[0].strip() for episode in self.episodes])
        