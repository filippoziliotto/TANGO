#!/usr/bin/env python3

import json
import os
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Union
import random

# Set seed
random.seed(100)

import attr
from habitat.core.registry import registry
from habitat.core.simulator import AgentState
from habitat.core.utils import DatasetFloatJSONEncoder
from habitat.datasets.pointnav.pointnav_dataset import (
    CONTENT_SCENES_PATH_FIELD,
    DEFAULT_SCENE_PATH_PREFIX,
    PointNavDatasetV1,
)
from habitat.tasks.nav.instance_image_nav_task import (  # InstanceImageGoalNavEpisode,
    InstanceImageGoal,
    InstanceImageParameters,
)
from habitat.tasks.nav.object_nav_task import ObjectGoal, ObjectViewLocation

from habitat.datasets.goat_bench.languagenav_dataset import (
    LanguageNavEpisode,
    OVONObjectViewLocation,
)
from habitat.tasks.goat_bench.goat_task import GoatEpisodeSingle, GoatEpisode

if TYPE_CHECKING:
    from omegaconf import DictConfig

@registry.register_dataset(name="Goat-v1-single")
class GoatDatasetV1Single(PointNavDatasetV1):
    r"""
    Class inherited from PointNavDataset that loads GOAT dataset.
    """
    episodes: List[GoatEpisodeSingle] = []
    content_scenes_path: str = "{data_path}/content/{scene}.json.gz"
    goals: Dict[str, Sequence[InstanceImageGoal]]
    current_scene_episodes: List[GoatEpisode] = []

    @staticmethod
    def dedup_goals(dataset: Dict[str, Any]) -> Dict[str, Any]:
        if len(dataset["episodes"]) == 0:
            return dataset

        goals = {}
        for i, ep in enumerate(dataset["episodes"]):
            ep = LanguageNavEpisode(**ep)

            goals_key = ep.goals_key
            if goals_key not in goals:
                goals[goals_key] = ep.goals

            dataset["episodes"][i]["goals"] = []

        dataset["goals"] = goals

        return dataset

    def to_json(self) -> str:
        for i in range(len(self.episodes)):
            self.episodes[i].goals = []

        result = DatasetFloatJSONEncoder().encode(self)

        for i in range(len(self.episodes)):
            goals = self.goals[self.episodes[i].goals_key]
            if not isinstance(goals, list):
                goals = list(goals)
            self.episodes[i].goals = goals

        return result

    def __init__(self, config: Optional["DictConfig"] = None, **kwargs) -> None:
        self.goals = {}
        super().__init__(config)

        self.episodes = list(self.episodes)

    @staticmethod
    def __deserialize_objectnav_goal(
        serialized_goal: Dict[str, Any]
    ) -> ObjectGoal:
        g = ObjectGoal(**serialized_goal)

        for vidx, view in enumerate(g.view_points):
            view_location = ObjectViewLocation(**view)  # type: ignore
            view_location.agent_state = AgentState(**view_location.agent_state)  # type: ignore
            g.view_points[vidx] = view_location

        return g

    @staticmethod
    def __deserialize_languagenav_goal(
        serialized_goal: Dict[str, Any]
    ) -> ObjectGoal:
        if serialized_goal.get("children_object_categories") is not None:
            del serialized_goal["children_object_categories"]

        g = ObjectGoal(**serialized_goal)

        for vidx, view in enumerate(g.view_points):
            view_location = OVONObjectViewLocation(**view)  # type: ignore
            view_location.agent_state = AgentState(**view_location.agent_state)  # type: ignore
            g.view_points[vidx] = view_location

        return g

    @staticmethod
    def __deserialize_imagenav_goal(
        serialized_goal: Dict[str, Any]
    ) -> InstanceImageGoal:
        tmp = serialized_goal.copy()
        g = InstanceImageGoal(**tmp)

        for vidx, view in enumerate(g.view_points):
            view_location = ObjectViewLocation(**view)  # type: ignore[arg-type]
            view_location.agent_state = AgentState(**view_location.agent_state)  # type: ignore[arg-type]
            g.view_points[vidx] = view_location

        for iidx, params in enumerate(g.image_goals):
            g.image_goals[iidx] = InstanceImageParameters(**params)  # type: ignore[arg-type]

        return g

    def from_json(
        self, json_str: str, scenes_dir: Optional[str] = None
    ) -> None:
        deserialized = json.loads(json_str)
        if CONTENT_SCENES_PATH_FIELD in deserialized:
            self.content_scenes_path = deserialized[CONTENT_SCENES_PATH_FIELD]

        if len(deserialized["episodes"]) == 0:
            return

        if "goals" not in deserialized:
            deserialized = self.dedup_goals(deserialized)

        self.current_scene_episodes = []
        self.goals = deserialized["goals"]
        num_filtered_eps = 0

        for i, composite_episode in enumerate(deserialized["episodes"]):
            composite_episode["goals"] = []
            composite_episode = GoatEpisode(**composite_episode)

            composite_episode.episode_id = str(i)

            if scenes_dir is not None:
                if composite_episode.scene_id.startswith(
                    DEFAULT_SCENE_PATH_PREFIX
                ):
                    composite_episode.scene_id = composite_episode.scene_id[
                        len(DEFAULT_SCENE_PATH_PREFIX) :
                    ]

                composite_episode.scene_id = os.path.join(
                    scenes_dir, "", composite_episode.scene_id
                )

            composite_episode.goals = []

            filtered_tasks = []
            for goal in composite_episode.tasks:
                goal_type = goal[1]
                goal_category = goal[0]
                goal_inst_id = goal[2]

                dset_same_cat_goals = [
                    x
                    for x in self.goals.values()
                    if x[0]["object_category"] == goal_category
                ]

                if goal_type == "description":
                    goal_inst = [
                        x
                        for x in dset_same_cat_goals[0]
                        if x["object_id"] == goal_inst_id
                    ]
                    if len(goal_inst[0]["lang_desc"].split(" ")) <= 55:
                        filtered_tasks.append(goal)
                    else:
                        num_filtered_eps += 1
                else:
                    filtered_tasks.append(goal)

            for goal in filtered_tasks:
                goal_type = goal[1]
                goal_category = goal[0]
                goal_inst_id = goal[2]

                dset_same_cat_goals = [
                    x
                    for x in self.goals.values()
                    if x[0]["object_category"] == goal_category
                ]
                children_categories = dset_same_cat_goals[0][0][
                    "children_object_categories"
                ]
                for child_category in children_categories:
                    goal_key = "{}_{}".format(
                        composite_episode.scene_id.split("/")[-1],
                        child_category,
                    )
                    if goal_key not in self.goals:
                        continue
                    dset_same_cat_goals[0].extend(self.goals[goal_key])

                assert (
                    len(dset_same_cat_goals) == 1
                ), f"more than 1 goal categories for {goal_category}"

                if goal_type == "object":
                    composite_episode.goals.append(dset_same_cat_goals[0])
                else:
                    goal_inst = [
                        x
                        for x in dset_same_cat_goals[0]
                        if x["object_id"] == goal_inst_id
                    ]
                    composite_episode.goals.append(goal_inst)

            self.current_scene_episodes.append(composite_episode)  # type: ignore [attr-defined]

        episode_list = []
        k = 0 + len(self.episodes) + 1
        for i, goat_ep in enumerate(self.current_scene_episodes):
            episode_list_single = []
            for j, subtask in enumerate(goat_ep.tasks):
                single_episode = {}
                single_episode['episode_id'] = k
                k += 1
                single_episode['scene_id'] = goat_ep.scene_id

                if j == 0:
                    single_episode['is_first_task'] = True
                else:
                    single_episode['is_first_task'] = False

                tmp = goat_ep.goals[j].copy()
                single_episode['goals'] = tmp
                single_episode['object_category'] = subtask[0]
                single_episode['goat_task'] = subtask[1]

                single_episode['start_position'] = goat_ep.start_position
                single_episode['start_rotation'] = goat_ep.start_rotation

                # check rotation is horizontal, check if useful
                single_episode['start_rotation'][0] = 0
                single_episode['start_rotation'][2] = 0

                for w, img_goals in enumerate(tmp):
                    tmp2 = img_goals.copy()
                    try:
                        single_episode['goals'][w] = self.__deserialize_imagenav_goal(tmp2)
                    except:
                        single_episode['goals'][w] = InstanceImageGoal(**tmp2)

                if subtask[1] in ['object', 'description']:
                    single_episode['is_image_goal'] = False
                else:
                    single_episode['is_image_goal'] = True

                episode_list_single.append(GoatEpisodeSingle(**single_episode))
                
            DEBUG, RANDOM, CUT = False, True, True

            if RANDOM:
                episode_list_single = self.randomize(episode_list_single)

            if CUT:
                # Len(episode_list_single) >= 5 & <= 10 randomly, keep first task
                episode_list_single = episode_list_single[:random.randint(5, 10)]

            if DEBUG: 
                # use only is_image_goal or first_task = true
                episode_list_single = [ep for ep in episode_list_single if ep.is_image_goal == True or ep.is_first_task == True]

            episode_list.extend(episode_list_single)

        self.episodes.extend(episode_list)
    
    def randomize(self, episode_list):
        # Randomize the order of episode, the first episode is always the first task the other are in random order
        # Separate the first episode
        first_episode = next(ep for ep in episode_list if ep.is_first_task)
        remaining_episodes = [ep for ep in episode_list if not ep.is_first_task]

        # shuffle the remaining episodes
        random.shuffle(remaining_episodes)
        episode_list = [first_episode] + remaining_episodes
        return episode_list
    
