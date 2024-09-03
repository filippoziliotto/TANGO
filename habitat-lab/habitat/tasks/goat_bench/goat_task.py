import os
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import attr
import habitat_sim
import numpy as np
from habitat.core.embodied_task import EmbodiedTask, SimulatorTaskAction
from habitat.core.logging import logger
from habitat.core.registry import registry
from habitat.tasks.nav.nav import NavigationEpisode, NavigationTask
from habitat.utils.geometry_utils import quaternion_from_coeff
from habitat_sim import bindings as hsim
from habitat_sim.agent.agent import AgentState, SixDOFPose

if TYPE_CHECKING:
    from omegaconf import DictConfig


@attr.s(auto_attribs=True, kw_only=True)
class GoatEpisode(NavigationEpisode):
    r"""Goat Episode

    :param object_category: Category of the obect
    """
    object_category: Optional[str] = None
    tasks: List[NavigationEpisode] = []

    @property
    def goals_keys(self) -> Dict:
        r"""Dictionary of goals types and corresonding keys"""
        goals_keys = {ep["task_type"]: [] for ep in self.tasks}

        for ep in self.tasks:
            if ep["task_type"] == "objectnav":
                goal_key = (
                    f"{os.path.basename(self.scene_id)}_{ep['object_category']}"
                )

            elif ep["task_type"] in ["imagenav", "languagenav"]:
                sid = os.path.basename(self.scene_id)
                for x in [".glb", ".basis"]:
                    sid = sid[: -len(x)] if sid.endswith(x) else sid
                goal_key = f"{sid}_{ep['goal_object_id']}"

            goals_keys[ep["task_type"]].append(goal_key)

        return goals_keys

    def goals_keys_with_sequence(self) -> str:
        r"""The key to retrieve the goals"""
        goals_keys = []

        for ep in self.tasks:
            if ep["task_type"] == "objectnav":
                goal_key = (
                    f"{os.path.basename(self.scene_id)}_{ep['object_category']}"
                )

            elif ep["task_type"] in ["imagenav", "languagenav"]:
                sid = os.path.basename(self.scene_id)
                for x in [".glb", ".basis"]:
                    sid = sid[: -len(x)] if sid.endswith(x) else sid
                goal_key = f"{sid}_{ep['goal_object_id']}"

            # elif ep["task_type"] == "languagenav":
            #     goal_key = f"{os.path.basename(self.scene_id)}_{ep['object_instance_id']}"

            goals_keys.append(goal_key)

        return goals_keys


@registry.register_task(name="Goat-v1")
class GoatTask(NavigationTask):  # TODO
    r"""A GOAT Task class for a task specific methods.
    Used to explicitly state a type of the task in config.
    """
    is_sub_task_stop_called: bool = False
    active_subtask_idx: int = 0
    last_action: Optional[SimulatorTaskAction] = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_sub_task_stop_called = False
        self.active_subtask_idx = 0
        self.last_action = None

    def reset(self, *args: Any, **kwargs: Any) -> None:
        self.is_sub_task_stop_called = False
        self.active_subtask_idx = 0
        self.last_action = None
        return super().reset(*args, **kwargs)

    def _subtask_stop_called(self, *args: Any, **kwargs: Any) -> bool:
        return isinstance(self.last_action, SubtaskStopAction)

    def _check_episode_is_active(
        self, episode, *args: Any, **kwargs: Any
    ) -> bool:
        return not getattr(
            self, "is_stop_called", False
        ) and self.active_subtask_idx < len(episode.goals)

    def step(self, action: Dict[str, Any], episode: GoatEpisode):
        action_name = action["action"]
        if "action_args" not in action or action["action_args"] is None:
            action["action_args"] = {}
        if isinstance(action_name, (int, np.integer)):
            action_name = self.get_action_name(action_name)
        task_action = self.actions[action_name]
        observations = super().step(action, episode)
        self.last_action = task_action
        return observations


@registry.register_task_action
class SubtaskStopAction(SimulatorTaskAction):
    name: str = "subtask_stop"

    def reset(self, task: GoatTask, *args: Any, **kwargs: Any):
        task.is_sub_task_stop_called = False  # type: ignore
        task.active_subtask_idx = 0

    def step(self, task: GoatTask, *args: Any, **kwargs: Any):
        r"""Update ``_metric``, this method is called from ``Env`` on each
        ``step``.
        """
        task.is_sub_task_stop_called = True  # type: ignore
        task.active_subtask_idx += 1
        return self._sim.get_observations_at()  # type: ignore
    

import hashlib
import os
import random
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
from gym import spaces
from habitat.core.embodied_task import EmbodiedTask
from habitat.core.registry import registry
from habitat.core.simulator import RGBSensor, Sensor, SensorTypes, Simulator
from habitat.core.utils import try_cv2_import
from habitat.tasks.nav.nav import NavigationEpisode

from habitat.tasks.goat_bench.goat_task import GoatEpisode

cv2 = try_cv2_import()


from habitat.tasks.goat_bench.utils import load_pickle

if TYPE_CHECKING:
    from omegaconf import DictConfig


@registry.register_sensor
class ClipObjectGoalSensor(Sensor):
    r"""A sensor for Object Goal specification as observations which is used in
    ObjectGoal Navigation. The goal is expected to be specified by object_id or
    semantic category id, and we will generate the prompt corresponding to it
    so that it's usable by CLIP's text encoder.
    Args:
        sim: a reference to the simulator for calculating task observations.
        config: a config for the ObjectGoalPromptSensor sensor. Can contain field
            GOAL_SPEC that specifies which id use for goal specification,
            GOAL_SPEC_MAX_VAL the maximum object_id possible used for
            observation space definition.
        dataset: a Object Goal navigation dataset that contains dictionaries
        of categories id to text mapping.
    """

    cls_uuid: str = "clip_objectgoal"

    def __init__(
        self,
        *args: Any,
        config: "DictConfig",
        **kwargs: Any,
    ):
        self.cache = load_pickle(config.cache)
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.SEMANTIC

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=-np.inf, high=np.inf, shape=(1024,), dtype=np.float32
        )

    def get_observation(
        self,
        observations,
        *args: Any,
        episode: Any,
        task: EmbodiedTask,
        **kwargs: Any,
    ) -> Optional[int]:
        dummy_embedding = np.zeros((1024,), dtype=np.float32)
        try:
            if isinstance(episode, GoatEpisode):
                # print(
                #     "GoatEpisode: {} - {}".format(
                #         episode.tasks[task.active_subtask_idx],
                #         isinstance(episode, GoatEpisode),
                #     )
                # )
                if task.active_subtask_idx < len(episode.tasks):
                    if episode.tasks[task.active_subtask_idx][1] == "object":
                        category = episode.tasks[task.active_subtask_idx][0]
                    else:
                        return dummy_embedding
                else:
                    return dummy_embedding
            else:
                category = (
                    episode.object_category
                    if hasattr(episode, "object_category")
                    else ""
                )
            if category not in self.cache:
                print("ObjectGoal Missing category: {}".format(category))
            # print("ObjectGoal Found category: {}".format(category))
        except Exception as e:
            print("Object goal exception ", e)
        return self.cache[category]


@registry.register_sensor
class ClipImageGoalSensor(Sensor):
    cls_uuid: str = "clip_imagegoal"

    def __init__(
        self,
        sim: "HabitatSim",
        config: "DictConfig",
        *args: Any,
        **kwargs: Any,
    ):
        self._sim = sim
        sensors = self._sim.sensor_suite.sensors
        rgb_sensor_uuids = [
            uuid
            for uuid, sensor in sensors.items()
            if isinstance(sensor, RGBSensor)
        ]
        if len(rgb_sensor_uuids) != 1:
            raise ValueError(
                "ImageGoalNav requires one RGB sensor,"
                f" {len(rgb_sensor_uuids)} detected"
            )
        (self._rgb_sensor_uuid,) = rgb_sensor_uuids
        super().__init__(config=config)
        self._curr_ep_id = None
        self.image_goal = None

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.COLOR

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return self._sim.sensor_suite.observation_spaces.spaces[
            self._rgb_sensor_uuid
        ]

    def _reset(self, episode):
        self._curr_ep_id = episode.episode_id
        sampled_goal = random.choice(episode.goals)
        sampled_viewpoint = random.choice(sampled_goal.view_points)
        observations = self._sim.get_observations_at(
            position=sampled_viewpoint.agent_state.position,
            rotation=sampled_viewpoint.agent_state.rotation,
            keep_agent_at_new_pose=False,
        )
        assert observations is not None
        self.image_goal = observations["rgb"]
        # Mutate the episode
        episode.goals = [sampled_goal]

    def get_observation(
        self,
        observations,
        episode: Any,
        *args: Any,
        **kwargs: Any,
    ) -> np.ndarray:
        if self.image_goal is None or self._curr_ep_id != episode.episode_id:
            self._reset(episode)
        assert self.image_goal is not None
        return self.image_goal


@registry.register_sensor
class ClipGoalSelectorSensor(Sensor):
    cls_uuid: str = "clip_goal_selector"

    def __init__(
        self,
        config: "DictConfig",
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(config=config)
        self._image_sampling_prob = config.image_sampling_probability
        self._curr_ep_id = None
        self._use_image_goal = True

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.TENSOR

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=0,
            high=1,
            shape=(1,),
            dtype=np.bool,
        )

    def _reset(self, episode):
        self._curr_ep_id = episode.episode_id
        self._use_image_goal = random.random() < self._image_sampling_prob

    def get_observation(
        self,
        observations,
        episode: Any,
        *args: Any,
        **kwargs: Any,
    ) -> np.ndarray:
        if self._curr_ep_id != episode.episode_id:
            self._reset(episode)
        return np.array([self._use_image_goal], dtype=np.bool)


@registry.register_sensor
class ImageGoalRotationSensor(Sensor):
    r"""Sensor for ImageGoal observations which are used in ImageGoal Navigation.
    RGBSensor needs to be one of the Simulator sensors.
    This sensor return the rgb image taken from the goal position to reach with
    random rotation.
    Args:
        sim: reference to the simulator for calculating task observations.
        config: config for the ImageGoal sensor.
    """

    cls_uuid: str = "image_goal_rotation"

    def __init__(
        self, *args: Any, sim: Simulator, config: "DictConfig", **kwargs: Any
    ):
        self._sim = sim
        sensors = self._sim.sensor_suite.sensors
        rgb_sensor_uuids = [
            uuid
            for uuid, sensor in sensors.items()
            if isinstance(sensor, RGBSensor)
        ]
        if len(rgb_sensor_uuids) != 1:
            raise ValueError(
                "ImageGoalNav requires one RGB sensor,"
                f" {len(rgb_sensor_uuids)} detected"
            )

        (self._rgb_sensor_uuid,) = rgb_sensor_uuids
        self._current_episode_id: Optional[str] = None
        self._current_image_goal = None
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.PATH

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return self._sim.sensor_suite.observation_spaces.spaces[
            self._rgb_sensor_uuid
        ]

    def _get_pointnav_episode_image_goal(self, episode: NavigationEpisode):
        goal_position = np.array(episode.goals[0].position, dtype=np.float32)

        # Add rotation to episode
        if self.config.sample_angle:
            angle = np.random.uniform(0, 2 * np.pi)
        else:
            # to be sure that the rotation is the same for the same episode_id
            # since the task is currently using pointnav Dataset.
            seed = abs(hash(episode.episode_id)) % (2**32)
            rng = np.random.RandomState(seed)
            angle = rng.uniform(0, 2 * np.pi)
        source_rotation = [0, np.sin(angle / 2), 0, np.cos(angle / 2)]
        episode.goals[0].rotation = source_rotation

        goal_observation = self._sim.get_observations_at(
            position=goal_position.tolist(), rotation=source_rotation
        )
        return goal_observation[self._rgb_sensor_uuid]

    def get_observation(
        self,
        *args: Any,
        observations,
        episode: NavigationEpisode,
        **kwargs: Any,
    ):
        episode_uniq_id = f"{episode.scene_id} {episode.episode_id}"
        if episode_uniq_id == self._current_episode_id:
            return self._current_image_goal

        self._current_image_goal = self._get_pointnav_episode_image_goal(
            episode
        )
        self._current_episode_id = episode_uniq_id

        return self._current_image_goal


@registry.register_sensor
class CurrentEpisodeUUIDSensor(Sensor):
    r"""Sensor for current episode uuid observations.
    Args:
        sim: reference to the simulator for calculating task observations.
        config: config for the ImageGoal sensor.
    """

    cls_uuid: str = "current_episode_uuid"

    def __init__(
        self, *args: Any, sim: Simulator, config: "DictConfig", **kwargs: Any
    ):
        self._sim = sim
        self._current_episode_id: Optional[str] = None

        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.TENSOR

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=np.iinfo(np.int64).min,
            high=np.iinfo(np.int64).max,
            shape=(1,),
            dtype=np.int64,
        )

    def get_observation(
        self,
        *args: Any,
        observations,
        episode: NavigationEpisode,
        **kwargs: Any,
    ):
        episode_uniq_id = f"{episode.scene_id} {episode.episode_id}"
        episode_uuid = (
            int(hashlib.sha1(episode_uniq_id.encode("utf-8")).hexdigest(), 16)
            % 10**8
        )
        return episode_uuid


@registry.register_sensor
class LanguageGoalSensor(Sensor):
    r"""A sensor for language goal specification as observations which is used in
    Language Navigation.
    Args:
        sim: a reference to the simulator for calculating task observations.
        config: a config for the ObjectGoalPromptSensor sensor. Can contain field
            GOAL_SPEC that specifies which id use for goal specification,
            GOAL_SPEC_MAX_VAL the maximum object_id possible used for
            observation space definition.
        dataset: a Object Goal navigation dataset that contains dictionaries
        of categories id to text mapping.
    """

    cls_uuid: str = "language_goal"

    def __init__(
        self,
        *args: Any,
        config: "DictConfig",
        **kwargs: Any,
    ):
        self.cache = load_pickle(config.cache)
        self.embedding_dim = config.embedding_dim
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.SEMANTIC

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.embedding_dim,),
            dtype=np.float32,
        )

    def get_observation(
        self,
        observations,
        *args: Any,
        episode: Any,
        task: EmbodiedTask,
        **kwargs: Any,
    ) -> Optional[int]:
        uuid = ""

        try:
            dummy_embedding = np.zeros((self.embedding_dim,), dtype=np.float32)
            if isinstance(episode, GoatEpisode):
                # print(
                #     "Lang GoatEpisode: {} - {}".format(
                #         episode.tasks[task.active_subtask_idx],
                #         isinstance(episode, GoatEpisode),
                #     )
                # )
                if task.active_subtask_idx < len(episode.tasks):
                    if (
                        episode.tasks[task.active_subtask_idx][1]
                        == "description"
                    ):
                        # print("not retur lang")
                        instance_id = episode.tasks[task.active_subtask_idx][2]
                        # print("instance id", instance_id)
                        # print(
                        #     "episode goals",
                        #     [
                        #         list(g.keys())
                        #         for g in episode.goals[task.active_subtask_idx]
                        #     ],
                        # )
                        goal = [
                            g
                            for g in episode.goals[task.active_subtask_idx]
                            if g["object_id"] == instance_id
                        ]
                        uuid = goal[0]["lang_desc"].lower()
                    else:
                        return dummy_embedding
                else:
                    return dummy_embedding
            else:
                uuid = episode.instructions[0].lower()
                first_3_words = [
                    "prefix: instruction: go",
                    "instruction: find the",
                    "instruction: go to",
                    "api_failure",
                    "instruction: locate the",
                ]
                for prefix in first_3_words:
                    uuid = uuid.replace(prefix, "")
                    uuid = uuid.replace("\n", " ")
                uuid = uuid.strip()

            if self.cache.get(uuid) is None:
                print("Lang Missing category: {}".format(uuid))
        except Exception as e:
            print("Language goal exception ", e)
        return self.cache[uuid]


@registry.register_sensor
class CacheImageGoalSensor(Sensor):
    r"""A sensor for Image goal specification as observations which is used in IIN.
    Args:
        sim: a reference to the simulator for calculating task observations.
        config: a config for the ObjectGoalPromptSensor sensor. Can contain field
            GOAL_SPEC that specifies which id use for goal specification,
            GOAL_SPEC_MAX_VAL the maximum object_id possible used for
            observation space definition.
        dataset: a Object Goal navigation dataset that contains dictionaries
        of categories id to text mapping.
    """

    cls_uuid: str = "cache_instance_imagegoal"

    def __init__(
        self,
        *args: Any,
        config: "DictConfig",
        **kwargs: Any,
    ):
        self.cache_base_dir = config.cache
        self.image_encoder = config.image_cache_encoder
        self.cache = None
        self._current_scene_id = ""
        self._current_episode_id = ""
        self._current_episode_image_goal = np.zeros((1024,), dtype=np.float32)
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.SEMANTIC

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=-np.inf, high=np.inf, shape=(1024,), dtype=np.float32
        )

    def get_observation(
        self,
        observations,
        *args: Any,
        episode: Any,
        task: EmbodiedTask,
        **kwargs: Any,
    ) -> Optional[int]:
        episode_id = f"{episode.scene_id}_{episode.episode_id}"
        if self._current_scene_id != episode.scene_id:
            self._current_scene_id = episode.scene_id
            scene_id = episode.scene_id.split("/")[-1].split(".")[0]

            suffix = "embedding.pkl"
            if self.image_encoder != "":
                suffix = "{}_iin_{}".format(self.image_encoder, suffix)
            if isinstance(episode, GoatEpisode):
                suffix = suffix.replace("iin", "goat")

            print(
                "Cache dir: {}".format(
                    os.path.join(self.cache_base_dir, f"{scene_id}_{suffix}")
                )
            )
            self.cache = load_pickle(
                os.path.join(self.cache_base_dir, f"{scene_id}_{suffix}")
            )

        try:
            if self._current_episode_id != episode_id:
                self._current_episode_id = episode_id

                dummy_embedding = np.zeros((1024,), dtype=np.float32)
                if isinstance(episode, GoatEpisode):
                    if task.active_subtask_idx < len(episode.tasks):
                        if episode.tasks[task.active_subtask_idx][1] == "image":
                            instance_id = episode.tasks[
                                task.active_subtask_idx
                            ][2]
                            curent_task = episode.tasks[task.active_subtask_idx]
                            scene_id = episode.scene_id.split("/")[-1].split(
                                "."
                            )[0]

                            uuid = "{}_{}".format(scene_id, instance_id)

                            self._current_episode_image_goal = self.cache[
                                "{}_{}".format(scene_id, instance_id)
                            ][curent_task[-1]]["embedding"]
                        else:
                            self._current_episode_image_goal = dummy_embedding
                    else:
                        self._current_episode_image_goal = dummy_embedding
                else:
                    self._current_episode_image_goal = self.cache[
                        episode.goal_key
                    ][episode.goal_image_id]["embedding"]
        except Exception as e:
            print("Image goal exception ", e)
            raise e

        return self._current_episode_image_goal


@registry.register_sensor
class GoatCurrentSubtaskSensor(Sensor):
    r"""A sensor for Image goal specification as observations which is used in IIN.
    Args:
        sim: a reference to the simulator for calculating task observations.
        config: a config for the ObjectGoalPromptSensor sensor. Can contain field
            GOAL_SPEC that specifies which id use for goal specification,
            GOAL_SPEC_MAX_VAL the maximum object_id possible used for
            observation space definition.
        dataset: a Object Goal navigation dataset that contains dictionaries
        of categories id to text mapping.
    """

    cls_uuid: str = "current_subtask"

    def __init__(
        self,
        *args: Any,
        config: "DictConfig",
        **kwargs: Any,
    ):
        self.sub_task_type = config.sub_task_type
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.SEMANTIC

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=0, high=len(self.sub_task_type) + 1, shape=(1,), dtype=np.int32
        )

    def get_observation(
        self,
        observations,
        *args: Any,
        episode: Any,
        task: EmbodiedTask,
        **kwargs: Any,
    ) -> Optional[int]:
        current_subtask = task.active_subtask_idx
        current_subtask_id = len(self.sub_task_type)
        if current_subtask < len(episode.tasks):
            current_subtask_id = self.sub_task_type.index(
                episode.tasks[current_subtask][1]
            )

        return current_subtask_id


@registry.register_sensor
class GoatGoalSensor(Sensor):
    r"""A sensor for Goat goals"""

    cls_uuid: str = "goat_subtask_goal"

    def __init__(
        self,
        *args: Any,
        config: "DictConfig",
        **kwargs: Any,
    ):
        self.image_cache_base_dir = config.image_cache
        self.image_encoder = config.image_cache_encoder
        self.image_cache = None
        self.language_cache = load_pickle(config.language_cache)
        self.object_cache = load_pickle(config.object_cache)
        self._current_scene_id = ""
        self._current_episode_id = ""
        self._current_episode_image_goal = None
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.SEMANTIC

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=-np.inf, high=np.inf, shape=(1024,), dtype=np.float32
        )

    def get_observation(
        self,
        observations,
        *args: Any,
        episode: Any,
        task: Any,
        **kwargs: Any,
    ) -> np.ndarray:
        episode_id = f"{episode.scene_id}_{episode.episode_id}"

        if self._current_scene_id != episode.scene_id:
            self._current_scene_id = episode.scene_id
            scene_id = episode.scene_id.split("/")[-1].split(".")[0]
            self.image_cache = load_pickle(
                os.path.join(
                    self.image_cache_base_dir,
                    f"{scene_id}_{self.image_encoder}_embedding.pkl",
                )
            )

        output_embedding = np.zeros((1024,), dtype=np.float32)

        task_type = "none"
        if task.active_subtask_idx < len(episode.tasks):
            if episode.tasks[task.active_subtask_idx][1] == "object":
                category = episode.tasks[task.active_subtask_idx][0]
                output_embedding = self.object_cache[category]
                task_type = "object"
            elif episode.tasks[task.active_subtask_idx][1] == "description":
                instance_id = episode.tasks[task.active_subtask_idx][2]
                goal = [
                    g
                    for g in episode.goals[task.active_subtask_idx]
                    if g["object_id"] == instance_id
                ]
                uuid = goal[0]["lang_desc"].lower()
                output_embedding = self.language_cache[uuid]
                task_type = "lang"
            elif episode.tasks[task.active_subtask_idx][1] == "image":
                instance_id = episode.tasks[task.active_subtask_idx][2]
                curent_task = episode.tasks[task.active_subtask_idx]
                scene_id = episode.scene_id.split("/")[-1].split(".")[0]

                uuid = "{}_{}".format(scene_id, instance_id)

                output_embedding = self.image_cache[
                    "{}_{}".format(scene_id, instance_id)
                ][curent_task[-1]]["embedding"]
                task_type = "image"
            else:
                raise NotImplementedError
        return output_embedding


@registry.register_sensor
class GoatMultiGoalSensor(Sensor):
    r"""A sensor for Goat goals"""

    cls_uuid: str = "goat_subtask_multi_goal"

    def __init__(
        self,
        *args: Any,
        config: "DictConfig",
        **kwargs: Any,
    ):
        self.image_cache_base_dir = config.image_cache
        self.image_encoder = config.image_cache_encoder
        self.image_cache = None
        self.language_cache = load_pickle(config.language_cache)
        self.object_cache = load_pickle(config.object_cache)
        self._current_scene_id = ""
        self._current_episode_id = ""
        self._current_episode_image_goal = None
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.SEMANTIC

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=-np.inf, high=np.inf, shape=(1024 * 3,), dtype=np.float32
        )

    def get_observation(
        self,
        observations,
        *args: Any,
        episode: Any,
        task: Any,
        **kwargs: Any,
    ) -> np.ndarray:
        episode_id = f"{episode.scene_id}_{episode.episode_id}"

        if self._current_scene_id != episode.scene_id:
            self._current_scene_id = episode.scene_id
            scene_id = episode.scene_id.split("/")[-1].split(".")[0]
            self.image_cache = load_pickle(
                os.path.join(
                    self.image_cache_base_dir,
                    f"{scene_id}_{self.image_encoder}_embedding.pkl",
                )
            )

        output_embedding = np.zeros((1024 * 3,), dtype=np.float32)
        scene_id = episode.scene_id.split("/")[-1].split(".")[0]

        task_type = "none"
        if task.active_subtask_idx < len(episode.tasks):
            if episode.tasks[task.active_subtask_idx][1] == "object":
                category = episode.tasks[task.active_subtask_idx][0]
                obj_embedding = self.object_cache[category]
                output_embedding = np.concatenate(
                    (obj_embedding, obj_embedding, obj_embedding)
                )
                task_type = "object"
            elif episode.tasks[task.active_subtask_idx][1] == "description":
                instance_id = episode.tasks[task.active_subtask_idx][2]
                goal = [
                    g
                    for g in episode.goals[task.active_subtask_idx]
                    if g["object_id"] == instance_id
                ]
                uuid = goal[0]["lang_desc"].lower()
                lang_embedding = self.language_cache[uuid]

                uuid = "{}_{}".format(scene_id, instance_id)
                random_idx = random.choice(
                    range(
                        len(
                            self.image_cache[
                                "{}_{}".format(scene_id, instance_id)
                            ]
                        )
                    ),
                )

                img_embedding = self.image_cache[
                    "{}_{}".format(scene_id, instance_id)
                ][random_idx]["embedding"]

                category = episode.tasks[task.active_subtask_idx][0]
                cat_embedding = self.object_cache[category]

                output_embedding = np.concatenate(
                    (lang_embedding, img_embedding, cat_embedding)
                )
                task_type = "lang"
            elif episode.tasks[task.active_subtask_idx][1] == "image":
                instance_id = episode.tasks[task.active_subtask_idx][2]
                curent_task = episode.tasks[task.active_subtask_idx]
                scene_id = episode.scene_id.split("/")[-1].split(".")[0]

                uuid = "{}_{}".format(scene_id, instance_id)

                img_embedding = self.image_cache[
                    "{}_{}".format(scene_id, instance_id)
                ][curent_task[-1]]["embedding"]

                category = episode.tasks[task.active_subtask_idx][0]
                cat_embedding = self.object_cache[category]

                goal = [
                    g
                    for g in episode.goals[task.active_subtask_idx]
                    if g["object_id"] == instance_id
                ]
                uuid = goal[0]["lang_desc"]
                if uuid is not None:
                    uuid = uuid.lower()
                    lang_embedding = self.language_cache[uuid]
                else:
                    lang_embedding = cat_embedding

                output_embedding = np.concatenate(
                    (lang_embedding, img_embedding, cat_embedding)
                )

                task_type = "image"
            else:
                raise NotImplementedError
        return output_embedding


# Goat main metrics

from collections import defaultdict
from typing import TYPE_CHECKING, Any, List, Optional, Tuple, Union

import numpy as np
from habitat.core.embodied_task import Measure
from habitat.core.logging import logger
from habitat.core.registry import registry
from habitat.core.simulator import Simulator
from habitat.tasks.nav.nav import NavigationEpisode, NavigationTask, TopDownMap
from habitat.utils.visualizations import fog_of_war, maps

from habitat.tasks.goat_bench.utils import load_pickle

if TYPE_CHECKING:
    from omegaconf import DictConfig


@registry.register_measure
class OVONDistanceToGoal(Measure):
    """The measure calculates a distance towards the goal."""

    cls_uuid: str = "ovon_distance_to_goal"

    def __init__(
        self, sim: Simulator, config: "DictConfig", *args: Any, **kwargs: Any
    ):
        self._previous_position: Optional[Tuple[float, float, float]] = None
        self._sim = sim
        self._config = config
        self._episode_view_points: Optional[
            List[Tuple[float, float, float]]
        ] = None

        super().__init__(**kwargs)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def reset_metric(self, episode, task, *args: Any, **kwargs: Any):
        self._previous_position = None
        self._metric = None
        if self._config.distance_to == "VIEW_POINTS":
            goals = task._dataset.goals_by_category[episode.goals_key]
            self._episode_view_points = [
                view_point.agent_state.position
                for goal in goals
                for view_point in goal.view_points
            ]

            if episode.children_object_categories is not None:
                for children_category in episode.children_object_categories:
                    scene_id = episode.scene_id.split("/")[-1]
                    goal_key = f"{scene_id}_{children_category}"

                    # Ignore if there are no valid viewpoints for goal
                    if goal_key not in task._dataset.goals_by_category:
                        continue
                    self._episode_view_points.extend(
                        [
                            vp.agent_state.position
                            for goal in task._dataset.goals_by_category[
                                goal_key
                            ]
                            for vp in goal.view_points
                        ]
                    )

        self.update_metric(episode=episode, task=task, *args, **kwargs)

    def update_metric(
        self,
        episode: NavigationEpisode,
        task: NavigationTask,
        *args: Any,
        **kwargs: Any,
    ):
        current_position = self._sim.get_agent_state().position

        if self._previous_position is None or not np.allclose(
            self._previous_position, current_position, atol=1e-4
        ):
            if self._config.distance_to == "POINT":
                goals = task._dataset.goals_by_category[episode.goals_key]
                distance_to_target = self._sim.geodesic_distance(
                    current_position,
                    [goal.position for goal in goals],
                    episode,
                )
            elif self._config.distance_to == "VIEW_POINTS":
                distance_to_target = self._sim.geodesic_distance(
                    current_position, self._episode_view_points, episode
                )
            else:
                logger.error(
                    "Non valid distance_to parameter was provided"
                    f"{self._config.distance_to}"
                )

            self._previous_position = (
                current_position[0],
                current_position[1],
                current_position[2],
            )
            self._metric = distance_to_target


@registry.register_measure
class OVONObjectGoalID(Measure):
    cls_uuid: str = "ovon_object_goal_id"

    def __init__(self, config: "DictConfig", *args: Any, **kwargs: Any):
        cache = load_pickle(config.cache)
        self.vocab = sorted(list(cache.keys()))
        super().__init__(**kwargs)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def reset_metric(self, episode, task, *args: Any, **kwargs: Any):
        self._metric = self.vocab.index(episode.object_category)

    def update_metric(
        self,
        episode: NavigationEpisode,
        task: NavigationTask,
        *args: Any,
        **kwargs: Any,
    ):
        pass


@registry.register_measure
class GoatDistanceToGoal(Measure):
    """The measure calculates a distance towards the goal."""

    cls_uuid: str = "goat_distance_to_goal"

    def __init__(
        self, sim: Simulator, config: "DictConfig", *args: Any, **kwargs: Any
    ):
        self._previous_position: Optional[Tuple[float, float, float]] = None
        self._sim = sim
        self._config = config
        self._episode_view_points: Optional[
            List[Tuple[float, float, float]]
        ] = None
        self._current_subtask_idx = 0
        self.prev_distance_to_target = 0

        super().__init__(**kwargs)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def reset_metric(self, episode, task, *args: Any, **kwargs: Any):
        self._previous_position = None
        self._metric = {"distance_to_target": 0, "prev_distance_to_target": 0}
        self._current_subtask_idx = 0
        self.prev_distance_to_target = 0
        self.update_metric(episode=episode, task=task, *args, **kwargs)

    def update_metric(
        self,
        episode: NavigationEpisode,
        task: NavigationTask,
        *args: Any,
        **kwargs: Any,
    ):
        current_position = self._sim.get_agent_state().position

        self.prev_distance_to_target = self._metric["distance_to_target"]
        subtask_switched = False
        if self._current_subtask_idx != task.active_subtask_idx:
            self._previous_position = None
            self._current_subtask_idx = task.active_subtask_idx
            episode._shortest_path_cache = None
            subtask_switched = True

        if self._current_subtask_idx == len(episode.tasks):
            self._metric = {
                "distance_to_target": self._metric["distance_to_target"],
                "prev_distance_to_target": self.prev_distance_to_target,
                "episode_ended": True,
            }
            return

        if (
            self._previous_position is None
            or not np.allclose(
                self._previous_position, current_position, atol=1e-4
            )
            or True
        ):
            if self._config.distance_to == "VIEW_POINTS":
                viewpoints = [
                    view_point["agent_state"]["position"]
                    for goal in episode.goals[task.active_subtask_idx]
                    for view_point in goal["view_points"]
                ]
                distance_to_target = self._sim.geodesic_distance(
                    current_position, viewpoints, episode
                )
            else:
                logger.error(
                    "Non valid distance_to parameter was provided"
                    f"{self._config.distance_to}"
                )
                raise NotImplementedError

            self._previous_position = (
                current_position[0],
                current_position[1],
                current_position[2],
            )
            self._metric = {
                "distance_to_target": distance_to_target,
                "prev_distance_to_target": self.prev_distance_to_target,
                "episode_ended": False,
            }

        if not np.isfinite(
            self._metric["distance_to_target"]
        ) or not np.isfinite(self._metric["prev_distance_to_target"]):
            print(
                current_position,
                self._previous_position,
                self._current_subtask_idx,
                task.last_action,
                episode.tasks[task.active_subtask_idx],
                episode.scene_id,
                episode.episode_id,
                self._metric,
            )


@registry.register_measure
class GoatSuccess(Measure):
    """The measure calculates a distance towards the goal."""

    cls_uuid: str = "goat_success"

    def __init__(
        self, sim: Simulator, config: "DictConfig", *args: Any, **kwargs: Any
    ):
        self._previous_position: Optional[Tuple[float, float, float]] = None
        self._sim = sim
        self._config = config
        self._current_subtask_idx = 0
        self._success_by_subtasks = defaultdict(int)
        self._subtask_counts = defaultdict(int)

        super().__init__(**kwargs)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def reset_metric(self, episode, task, *args: Any, **kwargs: Any):
        task.measurements.check_measure_dependencies(
            self.uuid, [GoatDistanceToGoal.cls_uuid]
        )
        self._previous_position = None
        self._metric = None
        self._current_subtask_idx = 0
        self._success_by_subtasks = defaultdict(int)
        self._subtask_counts = defaultdict(int)

        self._subtask_success = [0.0] * len(episode.tasks)

        for t in episode.tasks:
            self._subtask_counts[t[1]] += 1

        self.update_metric(episode=episode, task=task, *args, **kwargs)

    def update_metric(
        self,
        episode: NavigationEpisode,
        task: NavigationTask,
        *args: Any,
        **kwargs: Any,
    ):
        if self._current_subtask_idx == len(episode.tasks):
            return

        distance_to_target = task.measurements.measures[
            GoatDistanceToGoal.cls_uuid
        ].get_metric()

        if (
            task._subtask_stop_called()
            and distance_to_target["prev_distance_to_target"]
            < self._config.success_distance
        ):
            self._success_by_subtasks[
                episode.tasks[self._current_subtask_idx][1]
            ] += 1
            self._subtask_success[self._current_subtask_idx] = 1.0

        success_by_subtask = {}
        for k in ["object", "image", "description"]:
            if self._success_by_subtasks[k] == 0:
                success_by_subtask["{}_success".format(k)] = 0.0
            else:
                success_by_subtask["{}_success".format(k)] = (
                    self._success_by_subtasks[k] / self._subtask_counts[k]
                )

        num_subtask_success = sum(self._subtask_success) == len(episode.tasks)
        self._metric = {
            "task_success": num_subtask_success
            and getattr(task, "is_stop_called", False),
            "composite_success": num_subtask_success,
            "partial_success": sum(self._success_by_subtasks.values())
            / sum(self._subtask_counts.values()),
            "subtask_success": self._subtask_success,
            **success_by_subtask,
        }

        if self._current_subtask_idx != task.active_subtask_idx:
            self._current_subtask_idx = task.active_subtask_idx


@registry.register_measure
class GoatSPL(Measure):
    """The measure calculates a distance towards the goal."""

    cls_uuid: str = "goat_spl"

    def __init__(
        self, sim: Simulator, config: "DictConfig", *args: Any, **kwargs: Any
    ):
        self._previous_position: Union[None, np.ndarray, List[float]] = None
        self._start_end_episode_distance: Optional[float] = None
        self._agent_episode_distance: Optional[float] = None
        self._current_subtask_idx = 0
        self._sim = sim
        self._config = config

        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return "spl"

    def reset_metric(self, episode, task, *args: Any, **kwargs: Any):
        task.measurements.check_measure_dependencies(
            self.uuid, [GoatDistanceToGoal.cls_uuid, GoatSuccess.cls_uuid]
        )

        self._previous_position = self._sim.get_agent_state().position
        self._agent_episode_distance = 0.0
        self._start_end_episode_distance = task.measurements.measures[
            GoatDistanceToGoal.cls_uuid
        ].get_metric()["distance_to_target"]
        self._current_subtask_idx = 0

        self._spl_by_subtasks = defaultdict(int)
        self._subtask_counts = defaultdict(int)
        self._subtask_spl = [0] * len(episode.tasks)

        for t in episode.tasks:
            self._subtask_counts[t[1]] += 1
            self._spl_by_subtasks["{}_spl".format(t[1])] = 0

        self.update_metric(  # type:ignore
            episode=episode, task=task, *args, **kwargs
        )

    def _euclidean_distance(self, position_a, position_b):
        return np.linalg.norm(position_b - position_a, ord=2)

    def update_metric(
        self, episode, task: NavigationTask, *args: Any, **kwargs: Any
    ):
        ep_success = task.measurements.measures[
            GoatSuccess.cls_uuid
        ].get_metric()["subtask_success"]

        current_position = self._sim.get_agent_state().position
        self._agent_episode_distance += self._euclidean_distance(
            current_position, self._previous_position
        )

        self._previous_position = current_position

        if task._subtask_stop_called():
            spl = ep_success[self._current_subtask_idx] * (
                self._start_end_episode_distance
                / max(
                    self._start_end_episode_distance,
                    self._agent_episode_distance,
                )
            )
            self._spl_by_subtasks[
                episode.tasks[self._current_subtask_idx][1]
            ] += spl
            self._subtask_spl[self._current_subtask_idx] = spl

        spl_by_subtask = {}
        for k, v in self._subtask_counts.items():
            spl_by_subtask["{}_spl".format(k)] = (
                self._spl_by_subtasks[k] / self._subtask_counts[k]
            )

        self._metric = {
            "composite_spl": sum(self._spl_by_subtasks.values())
            / sum(self._subtask_counts.values()),
            # **spl_by_subtask,
            "spl_by_subtask": self._subtask_spl,
        }

        if self._current_subtask_idx != task.active_subtask_idx:
            self._current_subtask_idx = task.active_subtask_idx
            self._start_end_episode_distance = task.measurements.measures[
                GoatDistanceToGoal.cls_uuid
            ].get_metric()["distance_to_target"]
            self._agent_episode_distance = 0


@registry.register_measure
class GoatSoftSPL(Measure):
    """The measure calculates a SoftSPL."""

    cls_uuid: str = "goat_soft_spl"

    def __init__(
        self, sim: Simulator, config: "DictConfig", *args: Any, **kwargs: Any
    ):
        self._previous_position: Union[None, np.ndarray, List[float]] = None
        self._start_end_episode_distance: Optional[float] = None
        self._agent_episode_distance: Optional[float] = None
        self._current_subtask_idx = 0
        self._sim = sim
        self._config = config

        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return "soft_spl"

    def reset_metric(self, episode, task, *args: Any, **kwargs: Any):
        task.measurements.check_measure_dependencies(
            self.uuid, [GoatDistanceToGoal.cls_uuid, GoatSuccess.cls_uuid]
        )

        self._previous_position = self._sim.get_agent_state().position
        self._agent_episode_distance = 0.0
        self._start_end_episode_distance = task.measurements.measures[
            GoatDistanceToGoal.cls_uuid
        ].get_metric()["distance_to_target"]
        self._current_subtask_idx = 0

        self._softspl_by_subtasks = defaultdict(int)
        self._subtask_counts = defaultdict(int)

        for t in episode.tasks:
            self._subtask_counts[t[1]] += 1
            self._softspl_by_subtasks["{}_softspl".format(t[1])] = 0

        self.update_metric(  # type:ignore
            episode=episode, task=task, *args, **kwargs
        )

    def _euclidean_distance(self, position_a, position_b):
        return np.linalg.norm(position_b - position_a, ord=2)

    def update_metric(
        self, episode, task: NavigationTask, *args: Any, **kwargs: Any
    ):
        distance_to_target = task.measurements.measures[
            GoatDistanceToGoal.cls_uuid
        ].get_metric()["prev_distance_to_target"]

        current_position = self._sim.get_agent_state().position
        self._agent_episode_distance += self._euclidean_distance(
            current_position, self._previous_position
        )

        self._previous_position = current_position

        if task._subtask_stop_called():
            ep_soft_success = max(
                0, (1 - distance_to_target / self._start_end_episode_distance)
            )
            soft_spl = ep_soft_success * (
                self._start_end_episode_distance
                / max(
                    self._start_end_episode_distance,
                    self._agent_episode_distance,
                )
            )
            self._softspl_by_subtasks[
                episode.tasks[self._current_subtask_idx][1]
            ] += soft_spl

        softspl_by_subtask = {}
        for k, v in self._subtask_counts.items():
            softspl_by_subtask["{}_softspl".format(k)] = (
                self._softspl_by_subtasks[k] / self._subtask_counts[k]
            )

        self._metric = {
            "composite_softspl": sum(self._softspl_by_subtasks.values())
            / sum(self._subtask_counts.values()),
            # **softspl_by_subtask,
        }

        if self._current_subtask_idx != task.active_subtask_idx:
            self._current_subtask_idx = task.active_subtask_idx
            self._start_end_episode_distance = task.measurements.measures[
                GoatDistanceToGoal.cls_uuid
            ].get_metric()["distance_to_target"]
            self._agent_episode_distance = 0


@registry.register_measure
class GoatDistanceToGoalReward(Measure):
    """
    The measure calculates a reward based on the distance towards the goal.
    The reward is `- (new_distance - previous_distance)` i.e. the
    decrease of distance to the goal.
    """

    cls_uuid: str = "goat_distance_to_goal_reward"

    def __init__(
        self, sim: Simulator, config: "DictConfig", *args: Any, **kwargs: Any
    ):
        self._sim = sim
        self._config = config
        self._previous_distance: Optional[float] = None
        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def reset_metric(self, episode, task, *args: Any, **kwargs: Any):
        task.measurements.check_measure_dependencies(
            self.uuid, [GoatDistanceToGoal.cls_uuid]
        )
        self._previous_distance = task.measurements.measures[
            GoatDistanceToGoal.cls_uuid
        ].get_metric()["distance_to_target"]
        self.update_metric(episode=episode, task=task, *args, **kwargs)  # type: ignore

    def update_metric(
        self, episode, task: NavigationTask, *args: Any, **kwargs: Any
    ):
        distance_to_target = task.measurements.measures[
            GoatDistanceToGoal.cls_uuid
        ].get_metric()

        subtask_success_reward = 0
        # Handle case when subtask stop is called
        if task._subtask_stop_called():
            self._previous_distance = distance_to_target["distance_to_target"]

            if (
                distance_to_target["prev_distance_to_target"]
                < self._config.success_distance
            ):
                subtask_success_reward = 5.0

        self._metric = (
            -(
                distance_to_target["distance_to_target"]
                - self._previous_distance
            )
            + subtask_success_reward
        )
        self._previous_distance = distance_to_target["distance_to_target"]


@registry.register_measure
class GoatTopDownMap(TopDownMap):
    r"""Top Down Map measure for GOAT task."""

    def __init__(
        self,
        sim: Simulator,
        config: "DictConfig",
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(sim, config)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return "goat_top_down_map"

    def _draw_goals_view_points(self, episode):
        if self._config.draw_view_points:
            for idx, super_goals in enumerate(episode.goals):
                if type(super_goals[0]) != dict:
                    super_goals = super_goals[0]
                for goal in super_goals:
                    task_type = episode.tasks[idx][1]
                    if self._is_on_same_floor(goal["position"][1]):
                        try:
                            if task_type == "object":
                                view_point_indicator = (
                                    maps.OBJECTNAV_VIEW_POINT_INDICATOR
                                )
                            elif task_type == "description":
                                view_point_indicator = (
                                    maps.LANGNAV_VIEW_POINT_INDICATOR
                                )
                            elif task_type == "image":
                                view_point_indicator = (
                                    maps.IMGNAV_VIEW_POINT_INDICATOR
                                )

                            if idx == 0:
                                view_point_indicator = (
                                    maps.FIRST_TARGET_INDICATOR
                                )

                            if goal["view_points"] is not None:
                                for view_point in goal["view_points"]:
                                    self._draw_point(
                                        view_point["agent_state"]["position"],
                                        view_point_indicator,
                                    )
                        except AttributeError:
                            pass
                break

    def _draw_goals_positions(self, episode):
        if self._config.draw_goal_positions:
            for idx, super_goals in enumerate(episode.goals):
                if type(super_goals[0]) != dict:
                    super_goals = super_goals[0]
                for goal in super_goals:
                    task_type = episode.tasks[idx][1]
                    if self._is_on_same_floor(goal["position"][1]):
                        try:
                            if task_type == "object":
                                point_indicator = (
                                    maps.OBJECTNAV_TARGET_INDICATOR
                                )
                            elif task_type == "description":
                                point_indicator = maps.LANGNAV_TARGET_INDICATOR
                            elif task_type == "image":
                                point_indicator = maps.IMGNAV_TARGET_INDICATOR

                            if idx == 0:
                                point_indicator = maps.FIRST_TARGET_INDICATOR

                            self._draw_point(
                                goal["position"],
                                point_indicator,
                            )
                        except AttributeError:
                            pass
                break