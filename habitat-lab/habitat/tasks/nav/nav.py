#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# TODO, lots of typing errors in here

from typing import TYPE_CHECKING, Any, List, Optional, Sequence, Tuple, Union

import attr
import numpy as np
import quaternion
from gym import spaces, Space

from habitat.config import read_write
from habitat.config.default import get_agent_config
from habitat.core.dataset import Dataset, Episode
from habitat.core.embodied_task import (
    EmbodiedTask,
    Measure,
    SimulatorTaskAction,
)
from habitat.core.logging import logger
from habitat.core.registry import registry
from habitat.core.simulator import (
    AgentState,
    RGBSensor,
    Sensor,
    SensorTypes,
    ShortestPathPoint,
    Simulator,
)
from habitat.core.spaces import ActionSpace
from habitat.core.utils import not_none_validator, try_cv2_import
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat.tasks.utils import cartesian_to_polar
from habitat.utils.geometry_utils import (
    quaternion_from_coeff,
    quaternion_rotate_vector,
)
from habitat.utils.visualizations import fog_of_war, maps

try:
    from habitat.sims.habitat_simulator.habitat_simulator import HabitatSim
    from habitat_sim import RigidState
    from habitat_sim.physics import VelocityControl
except ImportError:
    pass

try:
    import magnum as mn
except ImportError:
    pass

if TYPE_CHECKING:
    from omegaconf import DictConfig

# from habitat_baselines.rl.ppo.utils.map.frontier_exploration.base_explorer import BaseExplorer
# from habitat_baselines.rl.ppo.utils.map.frontier_exploration.utils.path_utils import path_time_cost


cv2 = try_cv2_import()


MAP_THICKNESS_SCALAR: int = 128

@attr.s(auto_attribs=True, kw_only=True)
class NavigationGoal:
    r"""Base class for a goal specification hierarchy."""

    position: List[float] = attr.ib(default=None, validator=not_none_validator)
    radius: Optional[float] = None


@attr.s(auto_attribs=True, kw_only=True)
class RoomGoal(NavigationGoal):
    r"""Room goal that can be specified by room_id or position with radius."""

    room_id: str = attr.ib(default=None, validator=not_none_validator)
    room_name: Optional[str] = None


@attr.s(auto_attribs=True, kw_only=True)
class NavigationEpisode(Episode):
    r"""Class for episode specification that includes initial position and
    rotation of agent, scene name, goal and optional shortest paths. An
    episode is a description of one task instance for the agent.

    Args:
        episode_id: id of episode in the dataset, usually episode number
        scene_id: id of scene in scene dataset
        start_position: numpy ndarray containing 3 entries for (x, y, z)
        start_rotation: numpy ndarray with 4 entries for (x, y, z, w)
            elements of unit quaternion (versor) representing agent 3D
            orientation. ref: https://en.wikipedia.org/wiki/Versor
        goals: list of goals specifications
        start_room: room id
        shortest_paths: list containing shortest paths to goals
    """

    goals: List[NavigationGoal] = attr.ib(
        default=None,
        validator=not_none_validator,
        on_setattr=Episode._reset_shortest_path_cache_hook,
    )
    start_room: Optional[str] = None
    shortest_paths: Optional[List[List[ShortestPathPoint]]] = None


@registry.register_sensor
class PointGoalSensor(Sensor):
    r"""Sensor for PointGoal observations which are used in PointGoal Navigation.

    For the agent in simulator the forward direction is along negative-z.
    In polar coordinate format the angle returned is azimuth to the goal.

    Args:
        sim: reference to the simulator for calculating task observations.
        config: config for the PointGoal sensor. Can contain field for
            `goal_format` which can be used to specify the format in which
            the pointgoal is specified. Current options for goal format are
            cartesian and polar.

            Also contains a `dimensionality` field which specifes the number
            of dimensions ued to specify the goal, must be in [2, 3]

    Attributes:
        _goal_format: format for specifying the goal which can be done
            in cartesian or polar coordinates.
        _dimensionality: number of dimensions used to specify the goal
    """
    cls_uuid: str = "pointgoal"

    def __init__(
        self, sim: Simulator, config: "DictConfig", *args: Any, **kwargs: Any
    ):
        self._sim = sim

        self._goal_format = getattr(config, "goal_format", "CARTESIAN")
        assert self._goal_format in ["CARTESIAN", "POLAR", "GOAT"]

        self._dimensionality = getattr(config, "dimensionality", 2)
        assert self._dimensionality in [2, 3]

        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.PATH

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        sensor_shape = (self._dimensionality,)

        return spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=sensor_shape,
            dtype=np.float32,
        )

    def _compute_pointgoal(
        self, source_position, source_rotation, goal_position
    ):
        direction_vector = goal_position - source_position
        direction_vector_agent = quaternion_rotate_vector(
            source_rotation.inverse(), direction_vector
        )

        if self._goal_format == "POLAR":
            if self._dimensionality == 2:
                rho, phi = cartesian_to_polar(
                    -direction_vector_agent[2], direction_vector_agent[0]
                )
                return np.array([rho, -phi], dtype=np.float32)
            else:
                _, phi = cartesian_to_polar(
                    -direction_vector_agent[2], direction_vector_agent[0]
                )
                theta = np.arccos(
                    direction_vector_agent[1]
                    / np.linalg.norm(direction_vector_agent)
                )
                rho = np.linalg.norm(direction_vector_agent)

                return np.array([rho, -phi, theta], dtype=np.float32)
        else:
            if self._dimensionality == 2:
                return np.array(
                    [-direction_vector_agent[2], direction_vector_agent[0]],
                    dtype=np.float32,
                )
            else:
                return direction_vector_agent

    def get_observation(
        self,
        observations,
        episode: NavigationEpisode,
        *args: Any,
        **kwargs: Any,
    ):
        source_position = np.array(episode.start_position, dtype=np.float32)
        rotation_world_start = quaternion_from_coeff(episode.start_rotation)
        goal_position = np.array(episode.goals[0].position, dtype=np.float32)

        return self._compute_pointgoal(
            source_position, rotation_world_start, goal_position
        )


@registry.register_sensor
class ImageGoalSensor(Sensor):
    r"""Sensor for ImageGoal observations which are used in ImageGoal Navigation.

    RGBSensor needs to be one of the Simulator sensors.
    This sensor return the rgb image taken from the goal position to reach with
    random rotation.

    Args:
        sim: reference to the simulator for calculating task observations.
        config: config for the ImageGoal sensor.
    """
    cls_uuid: str = "imagegoal"

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
                f"ImageGoalNav requires one RGB sensor, {len(rgb_sensor_uuids)} detected"
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
        # to be sure that the rotation is the same for the same episode_id
        # since the task is currently using pointnav Dataset.
        seed = abs(hash(episode.episode_id)) % (2**32)
        rng = np.random.RandomState(seed)
        angle = rng.uniform(0, 2 * np.pi)
        source_rotation = [0, np.sin(angle / 2), 0, np.cos(angle / 2)]
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


@registry.register_sensor(name="PointGoalWithGPSCompassSensor")
class IntegratedPointGoalGPSAndCompassSensor(PointGoalSensor):
    r"""Sensor that integrates PointGoals observations (which are used PointGoal Navigation) and GPS+Compass.

    For the agent in simulator the forward direction is along negative-z.
    In polar coordinate format the angle returned is azimuth to the goal.

    Args:
        sim: reference to the simulator for calculating task observations.
        config: config for the PointGoal sensor. Can contain field for
            `goal_format` which can be used to specify the format in which
            the pointgoal is specified. Current options for goal format are
            cartesian and polar.

            Also contains a `dimensionality` field which specifes the number
            of dimensions ued to specify the goal, must be in [2, 3]

    Attributes:
        _goal_format: format for specifying the goal which can be done
            in cartesian or polar coordinates.
        _dimensionality: number of dimensions used to specify the goal
    """
    cls_uuid: str = "pointgoal_with_gps_compass"

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def get_observation(
        self, observations, episode, *args: Any, **kwargs: Any
    ):
        agent_state = self._sim.get_agent_state()
        agent_position = agent_state.position
        rotation_world_agent = agent_state.rotation
        goal_position = np.array(episode.goals[0].position, dtype=np.float32)

        return self._compute_pointgoal(
            agent_position, rotation_world_agent, goal_position
        )


@registry.register_sensor
class HeadingSensor(Sensor):
    r"""Sensor for observing the agent's heading in the global coordinate
    frame.

    Args:
        sim: reference to the simulator for calculating task observations.
        config: config for the sensor.
    """
    cls_uuid: str = "heading"

    def __init__(
        self, sim: Simulator, config: "DictConfig", *args: Any, **kwargs: Any
    ):
        self._sim = sim
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.HEADING

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(low=-np.pi, high=np.pi, shape=(1,), dtype=np.float32)

    def _quat_to_xy_heading(self, quat):
        direction_vector = np.array([0, 0, -1])

        heading_vector = quaternion_rotate_vector(quat, direction_vector)

        phi = cartesian_to_polar(-heading_vector[2], heading_vector[0])[1]
        return np.array([phi], dtype=np.float32)

    def get_observation(
        self, observations, episode, *args: Any, **kwargs: Any
    ):
        agent_state = self._sim.get_agent_state()
        rotation_world_agent = agent_state.rotation

        if isinstance(rotation_world_agent, quaternion.quaternion):
            return self._quat_to_xy_heading(rotation_world_agent.inverse())
        else:
            raise ValueError("Agent's rotation was not a quaternion")


@registry.register_sensor(name="CompassSensor")
class EpisodicCompassSensor(HeadingSensor):
    r"""The agents heading in the coordinate frame defined by the episode,
    theta=0 is defined by the agents state at t=0
    """
    cls_uuid: str = "compass"

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def get_observation(
        self, observations, episode, *args: Any, **kwargs: Any
    ):
        agent_state = self._sim.get_agent_state()
        rotation_world_agent = agent_state.rotation
        rotation_world_start = quaternion_from_coeff(episode.start_rotation)

        if isinstance(rotation_world_agent, quaternion.quaternion):
            return self._quat_to_xy_heading(
                rotation_world_agent.inverse() * rotation_world_start
            )
        else:
            raise ValueError("Agent's rotation was not a quaternion")


@registry.register_sensor(name="GPSSensor")
class EpisodicGPSSensor(Sensor):
    r"""The agents current location in the coordinate frame defined by the episode,
    i.e. the axis it faces along and the origin is defined by its state at t=0

    Args:
        sim: reference to the simulator for calculating task observations.
        config: Contains the `dimensionality` field for the number of dimensions to express the agents position
    Attributes:
        _dimensionality: number of dimensions used to specify the agents position
    """
    cls_uuid: str = "gps"

    def __init__(
        self, sim: Simulator, config: "DictConfig", *args: Any, **kwargs: Any
    ):
        self._sim = sim

        self._dimensionality = getattr(config, "dimensionality", 2)
        assert self._dimensionality in [2, 3]
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.POSITION

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        sensor_shape = (self._dimensionality,)
        return spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=sensor_shape,
            dtype=np.float32,
        )

    def get_observation(
        self, observations, episode, *args: Any, **kwargs: Any
    ):
        agent_state = self._sim.get_agent_state()

        origin = np.array(episode.start_position, dtype=np.float32)
        rotation_world_start = quaternion_from_coeff(episode.start_rotation)

        agent_position = agent_state.position

        agent_position = quaternion_rotate_vector(
            rotation_world_start.inverse(), agent_position - origin
        )
        if self._dimensionality == 2:
            return np.array(
                [-agent_position[2], agent_position[0]], dtype=np.float32
            )
        else:
            return agent_position.astype(np.float32)


@registry.register_sensor
class ProximitySensor(Sensor):
    r"""Sensor for observing the distance to the closest obstacle

    Args:
        sim: reference to the simulator for calculating task observations.
        config: config for the sensor.
    """
    cls_uuid: str = "proximity"

    def __init__(self, sim, config, *args: Any, **kwargs: Any):
        self._sim = sim
        self._max_detection_radius = getattr(
            config, "max_detection_radius", 2.0
        )
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.TACTILE

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=0.0,
            high=self._max_detection_radius,
            shape=(1,),
            dtype=np.float32,
        )

    def get_observation(
        self, observations, *args: Any, episode, **kwargs: Any
    ):
        current_position = self._sim.get_agent_state().position

        return np.array(
            [
                self._sim.distance_to_closest_obstacle(
                    current_position, self._max_detection_radius
                )
            ],
            dtype=np.float32,
        )

def global_to_episodic_xy(episodic_start, episodic_yaw, pt):
    """
    All args are in Habitat format.
    """
    # Habitat to xy
    pt = np.array([pt[2], pt[0]])
    episodic_start = np.array([episodic_start[2], episodic_start[0]])

    rotation_matrix = np.array(
        [
            [np.cos(-episodic_yaw), -np.sin(-episodic_yaw)],
            [np.sin(-episodic_yaw), np.cos(-episodic_yaw)],
        ]
    )
    episodic_xy = -np.matmul(rotation_matrix, pt - episodic_start)

    return episodic_xy
@registry.register_sensor
class FrontierSensor(Sensor):
    cls_uuid: str = "frontier_sensor"

    def __init__(
        self, sim: HabitatSim, config: "DictConfig", *args: Any, **kwargs: Any
    ) -> None:
        super().__init__(sim, config, *args, **kwargs)
        self._curr_ep_id = None
        self.episodic_yaw = 0.0

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any) -> SensorTypes:
        return SensorTypes.TENSOR

    def _get_observation_space(self, *args: Any, **kwargs: Any) -> Space:
        return spaces.Box(
            low=-float("inf"), high=float("inf"), shape=(1, 2), dtype=np.float32
        )

    def get_observation(
        self, task: EmbodiedTask, episode, *args: Any, **kwargs: Any
    ) -> np.ndarray:
        """Return the 3D coordinates of each frontier."""
        if self._curr_ep_id != episode.episode_id:
            self._curr_ep_id = episode.episode_id
            heading_sensor: HeadingSensor = task.sensor_suite.sensors[  # type: ignore
                "heading"
            ]
            self.episodic_yaw = heading_sensor.get_observation(None, episode)[0]

        explorer_key = [
            k for k in task.sensor_suite.sensors.keys() if k.endswith("_explorer")
        ][0]

        explorer: BaseExplorer = task.sensor_suite.sensors[explorer_key]  # type: ignore

        if len(explorer.frontier_waypoints) == 0:
            return np.zeros((1, 2), dtype=np.float32)

        global_frontiers = explorer._pixel_to_map_coors(explorer.frontier_waypoints)

        # Sort the frontiers by completion time cost
        completion_times = []

        for frontier in global_frontiers:
            completion_times.append(
                path_time_cost(
                    frontier,
                    explorer.agent_position,
                    explorer.agent_heading,
                    explorer._lin_vel,
                    explorer._ang_vel,
                    explorer._sim,
                )
            )
            # completion_times.append(
            #     path_dist_cost(frontier, explorer.agent_position, explorer._sim)
            # )
        global_frontiers = global_frontiers[np.argsort(completion_times)]

        episode_origin = np.array(episode.start_position)

        episodic_frontiers = []
        for g_frontier in global_frontiers:
            pt = global_to_episodic_xy(episode_origin, self.episodic_yaw, g_frontier)
            episodic_frontiers.append(pt)
        episodic_frontiers = np.array(episodic_frontiers)

        return episodic_frontiers


# @registry.register_measure
# class Success(Measure):
#     r"""Whether or not the agent succeeded at its task

#     This measure depends on DistanceToGoal measure.
#     """

#     cls_uuid: str = "success"

#     def __init__(
#         self, sim: Simulator, config: "DictConfig", *args: Any, **kwargs: Any
#     ):
#         self._sim = sim
#         self._config = config
#         self._success_distance = self._config.success_distance

#         super().__init__()

#     def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
#         return self.cls_uuid

#     def reset_metric(self, episode, task, *args: Any, **kwargs: Any):
#         task.measurements.check_measure_dependencies(
#             self.uuid, [DistanceToGoal.cls_uuid]
#         )
#         self.update_metric(episode=episode, task=task, *args, **kwargs)  # type: ignore

#     def update_metric(
#         self, episode, task: EmbodiedTask, *args: Any, **kwargs: Any
#     ):
#         distance_to_target = task.measurements.measures[
#             DistanceToGoal.cls_uuid
#         ].get_metric()

#         if (
#             hasattr(task, "is_stop_called")
#             and task.is_stop_called  # type: ignore
#             and distance_to_target < self._success_distance
#         ):
#             self._metric = 1.0
#         else:
#             self._metric = 0.0

@registry.register_measure
class Success(Measure):
    r"""Whether or not the agent succeeded at its task
    This measure depends on DistanceToGoal measure.
    """

    cls_uuid: str = "success"

    def __init__(
        self, *args: Any, sim: Simulator, config: "DictConfig", **kwargs: Any
    ):
        self._sim = sim
        self._config = config

        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return self.cls_uuid

    def reset_metric(self, *args: Any, episode, task, **kwargs: Any):
        task.measurements.check_measure_dependencies(
            self.uuid, [SubSuccess.cls_uuid]
        )
        self.update_metric(*args, episode=episode, task=task, **kwargs)

    def update_metric(
        self, *args: Any, episode, task: EmbodiedTask, **kwargs: Any
    ):
        subsuccess = task.measurements.measures[
            SubSuccess.cls_uuid
        ].get_metric()

        if subsuccess ==1 and task.currGoalIndex >= len(episode.goals):
            self._metric = 1
        else:
            self._metric = 0




@registry.register_measure
class SPL(Measure):
    r"""SPL (Success weighted by Path Length)

    ref: On Evaluation of Embodied Agents - Anderson et. al
    https://arxiv.org/pdf/1807.06757.pdf
    The measure depends on Distance to Goal measure and Success measure
    to improve computational
    performance for sophisticated goal areas.
    """

    def __init__(
        self, sim: Simulator, config: "DictConfig", *args: Any, **kwargs: Any
    ):
        self._previous_position: Union[None, np.ndarray, List[float]] = None
        self._start_end_episode_distance: Optional[float] = None
        self._agent_episode_distance: Optional[float] = None
        self._episode_view_points: Optional[
            List[Tuple[float, float, float]]
        ] = None
        self._sim = sim
        self._config = config

        self.current_step = 0

        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return "spl"

    def reset_metric(self, episode, task, *args: Any, **kwargs: Any):
        task.measurements.check_measure_dependencies(
            self.uuid, [DistanceToGoal.cls_uuid, Success.cls_uuid]
        )

        self._previous_position = self._sim.get_agent_state().position
        self._agent_episode_distance = 0.0

        self._start_end_episode_distance = task.measurements.measures[
            DistanceToGoal.cls_uuid
        ].get_metric()

        self.update_metric(  # type:ignore
            episode=episode, task=task, *args, **kwargs
        )

        self.current_step = 0

    def _euclidean_distance(self, position_a, position_b):
        return np.linalg.norm(position_b - position_a, ord=2)

    def update_metric(
        self, episode, task: EmbodiedTask, *args: Any, **kwargs: Any
    ):
        ep_success = task.measurements.measures[Success.cls_uuid].get_metric()

        current_position = self._sim.get_agent_state().position

        # The distance is the current position to the goal
        if (self.current_step == 0) and (episode.is_first_task == False):
            self._agent_episode_distance = 0.
            self._previous_position = current_position

        else:
            self._agent_episode_distance += self._euclidean_distance(
                current_position, self._previous_position
            )

        self._previous_position = current_position

        # If subtask we take the distance to goal from current position
        if (self.current_step == 1) and (episode.is_first_task == False):
            self._start_end_episode_distance = task.measurements.measures[
                DistanceToGoal.cls_uuid
            ].get_metric()


        try:
            self._metric = ep_success * (
                self._start_end_episode_distance
                / max(
                    self._start_end_episode_distance, self._agent_episode_distance
                )
            )
            
        # If the denominator is zero then the agent starts lready close to the target. We set a minimum 0.1 meters
        except ZeroDivisionError:
            self._start_end_episode_distance = 0.1
            self._metric = ep_success * (
                self._start_end_episode_distance
                / max(
                    self._start_end_episode_distance, self._agent_episode_distance
                )
            )
        self.current_step += 1



@registry.register_measure
class SoftSPL(SPL):
    r"""Soft SPL

    Similar to spl with a relaxed soft-success criteria. Instead of a boolean
    success is now calculated as 1 - (ratio of distance covered to target).
    """

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return "soft_spl"

    def reset_metric(self, episode, task, *args: Any, **kwargs: Any):
        task.measurements.check_measure_dependencies(
            self.uuid, [DistanceToGoal.cls_uuid]
        )

        self._previous_position = self._sim.get_agent_state().position
        self._agent_episode_distance = 0.0
        self._start_end_episode_distance = task.measurements.measures[
            DistanceToGoal.cls_uuid
        ].get_metric()
        self.update_metric(episode=episode, task=task, *args, **kwargs)  # type: ignore

    def update_metric(self, episode, task, *args: Any, **kwargs: Any):
        current_position = self._sim.get_agent_state().position
        distance_to_target = task.measurements.measures[
            DistanceToGoal.cls_uuid
        ].get_metric()

        ep_soft_success = max(
            0, (1 - distance_to_target / self._start_end_episode_distance)
        )

        self._agent_episode_distance += self._euclidean_distance(
            current_position, self._previous_position
        )

        self._previous_position = current_position

        self._metric = ep_soft_success * (
            self._start_end_episode_distance
            / max(
                self._start_end_episode_distance, self._agent_episode_distance
            )
        )


@registry.register_measure
class Collisions(Measure):
    def __init__(self, sim, config, *args: Any, **kwargs: Any):
        self._sim = sim
        self._config = config
        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return "collisions"

    def reset_metric(self, episode, *args: Any, **kwargs: Any):
        self._metric = {"count": 0, "is_collision": False}

    def update_metric(self, episode, action, *args: Any, **kwargs: Any):
        self._metric["is_collision"] = False
        if self._sim.previous_step_collided:
            self._metric["count"] += 1
            self._metric["is_collision"] = True


@registry.register_measure
class TopDownMap(Measure):
    r"""Top Down Map measure"""

    def __init__(
        self,
        sim: "HabitatSim",
        config: "DictConfig",
        *args: Any,
        **kwargs: Any,
    ):
        self._sim = sim
        self._config = config
        self._grid_delta = config.map_padding
        self._step_count: Optional[int] = None
        self._map_resolution = config.map_resolution
        self._ind_x_min: Optional[int] = None
        self._ind_x_max: Optional[int] = None
        self._ind_y_min: Optional[int] = None
        self._ind_y_max: Optional[int] = None
        self._previous_xy_location: List[Optional[Tuple[int, int]]] = None
        self._top_down_map: Optional[np.ndarray] = None
        self._shortest_path_points: Optional[List[Tuple[int, int]]] = None
        self.line_thickness = int(
            np.round(self._map_resolution * 2 / MAP_THICKNESS_SCALAR)
        )
        self.point_padding = 2 * int(
            np.ceil(self._map_resolution / MAP_THICKNESS_SCALAR)
        )
        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return "top_down_map"

    def get_original_map(self):
        top_down_map = maps.get_topdown_map_from_sim(
            self._sim,
            map_resolution=self._map_resolution,
            draw_border=self._config.draw_border,
        )

        if self._config.fog_of_war.draw:
            self._fog_of_war_mask = np.zeros_like(top_down_map)
        else:
            self._fog_of_war_mask = None

        return top_down_map

    def _draw_point(self, position, point_type):
        t_x, t_y = maps.to_grid(
            position[2],
            position[0],
            (self._top_down_map.shape[0], self._top_down_map.shape[1]),
            sim=self._sim,
        )
        self._top_down_map[
            t_x - self.point_padding : t_x + self.point_padding + 1,
            t_y - self.point_padding : t_y + self.point_padding + 1,
        ] = point_type

    def _draw_goals_view_points(self, episode):
        if self._config.draw_view_points:
            for goal in episode.goals:
                if self._is_on_same_floor(goal.position[1]):
                    try:
                        if goal.view_points is not None:
                            for view_point in goal.view_points:
                                self._draw_point(
                                    view_point.agent_state.position,
                                    maps.MAP_VIEW_POINT_INDICATOR,
                                )
                    except AttributeError:
                        pass

    def _draw_goals_positions(self, episode):
        if self._config.draw_goal_positions:
            for goal in episode.goals:
                if self._is_on_same_floor(goal.position[1]):
                    try:
                        self._draw_point(
                            goal.position, maps.MAP_TARGET_POINT_INDICATOR
                        )
                    except AttributeError:
                        pass

    def _draw_goals_aabb(self, episode):
        if self._config.draw_goal_aabbs:
            for goal in episode.goals:
                try:
                    sem_scene = self._sim.semantic_annotations()
                    object_id = goal.object_id
                    assert int(
                        sem_scene.objects[object_id].id.split("_")[-1]
                    ) == int(
                        goal.object_id
                    ), f"Object_id doesn't correspond to id in semantic scene objects dictionary for episode: {episode}"

                    center = sem_scene.objects[object_id].aabb.center
                    x_len, _, z_len = (
                        sem_scene.objects[object_id].aabb.sizes / 2.0
                    )
                    # Nodes to draw rectangle
                    corners = [
                        center + np.array([x, 0, z])
                        for x, z in [
                            (-x_len, -z_len),
                            (-x_len, z_len),
                            (x_len, z_len),
                            (x_len, -z_len),
                            (-x_len, -z_len),
                        ]
                        if self._is_on_same_floor(center[1])
                    ]

                    map_corners = [
                        maps.to_grid(
                            p[2],
                            p[0],
                            (
                                self._top_down_map.shape[0],
                                self._top_down_map.shape[1],
                            ),
                            sim=self._sim,
                        )
                        for p in corners
                    ]

                    maps.draw_path(
                        self._top_down_map,
                        map_corners,
                        maps.MAP_TARGET_BOUNDING_BOX,
                        self.line_thickness,
                    )
                except AttributeError:
                    pass

    def _draw_shortest_path(
        self, episode: NavigationEpisode, agent_position: AgentState
    ):
        if self._config.draw_shortest_path:
            _shortest_path_points = (
                self._sim.get_straight_shortest_path_points(
                    agent_position, episode.goals[0].position
                )
            )
            self._shortest_path_points = [
                maps.to_grid(
                    p[2],
                    p[0],
                    (self._top_down_map.shape[0], self._top_down_map.shape[1]),
                    sim=self._sim,
                )
                for p in _shortest_path_points
            ]
            maps.draw_path(
                self._top_down_map,
                self._shortest_path_points,
                maps.MAP_SHORTEST_PATH_COLOR,
                self.line_thickness,
            )

    def _is_on_same_floor(
        self, height, ref_floor_height=None, ceiling_height=2.0
    ):
        if ref_floor_height is None:
            ref_floor_height = self._sim.get_agent(0).state.position[1]
        return ref_floor_height <= height < ref_floor_height + ceiling_height

    def reset_metric(self, episode, *args: Any, **kwargs: Any):
        self._top_down_map = self.get_original_map()
        self._step_count = 0
        agent_position = self._sim.get_agent_state().position
        self._previous_xy_location = [
            None for _ in range(len(self._sim.habitat_config.agents))
        ]

        if hasattr(episode, "goals"):
            # draw source and target parts last to avoid overlap
            self._draw_goals_view_points(episode)
            self._draw_goals_aabb(episode)
            self._draw_goals_positions(episode)
            self._draw_shortest_path(episode, agent_position)

        if self._config.draw_source:
            self._draw_point(
                episode.start_position, maps.MAP_SOURCE_POINT_INDICATOR
            )

        self.update_metric(episode, None)
        self._step_count = 0

    def update_metric(self, episode, action, *args: Any, **kwargs: Any):
        self._step_count += 1
        map_positions: List[Tuple[float]] = []
        map_angles = []
        for agent_index in range(len(self._sim.habitat_config.agents)):
            agent_state = self._sim.get_agent_state(agent_index)
            map_positions.append(self.update_map(agent_state, agent_index))
            map_angles.append(TopDownMap.get_polar_angle(agent_state))
        self._metric = {
            "map": self._top_down_map,
            "fog_of_war_mask": self._fog_of_war_mask,
            "agent_map_coord": map_positions,
            "agent_angle": map_angles,
            "current_view_mask": self._current_view_mask,
        }

    @staticmethod
    def get_polar_angle(agent_state):
        # quaternion is in x, y, z, w format
        ref_rotation = agent_state.rotation
        heading_vector = quaternion_rotate_vector(
            ref_rotation.inverse(), np.array([0, 0, -1])
        )
        phi = cartesian_to_polar(heading_vector[2], -heading_vector[0])[1]
        return np.array(phi)

    def update_map(self, agent_state: AgentState, agent_index: int):
        agent_position = agent_state.position
        a_x, a_y = maps.to_grid(
            agent_position[2],
            agent_position[0],
            (self._top_down_map.shape[0], self._top_down_map.shape[1]),
            sim=self._sim,
        )
        # Don't draw over the source point
        if self._top_down_map[a_x, a_y] != maps.MAP_SOURCE_POINT_INDICATOR:
            color = 10 + min(
                self._step_count * 245 // self._config.max_episode_steps, 245
            )

            thickness = self.line_thickness
            if self._previous_xy_location[agent_index] is not None:
                cv2.line(
                    self._top_down_map,
                    self._previous_xy_location[agent_index],
                    (a_y, a_x),
                    color,
                    thickness=thickness,
                )
        angle = TopDownMap.get_polar_angle(agent_state)
        self.update_fog_of_war_mask(np.array([a_x, a_y]), angle)

        self._previous_xy_location[agent_index] = (a_y, a_x)
        return a_x, a_y

    def update_fog_of_war_mask(self, agent_position, angle):
        if self._config.fog_of_war.draw:
            self._fog_of_war_mask, self._current_view_mask = fog_of_war.reveal_fog_of_war(
                self._top_down_map,
                self._fog_of_war_mask,
                agent_position,
                angle,
                fov=self._config.fog_of_war.fov,
                max_line_len=self._config.fog_of_war.visibility_dist
                / maps.calculate_meters_per_pixel(
                    self._map_resolution, sim=self._sim
                ),
            )



@registry.register_measure
class DistanceToGoal(Measure):
    """The measure calculates a distance towards the goal."""

    cls_uuid: str = "distance_to_goal"

    def __init__(
        self, sim: Simulator, config: "DictConfig", *args: Any, **kwargs: Any
    ):
        self._previous_position: Optional[Tuple[float, float, float]] = None
        self._sim = sim
        self._config = config
        self._episode_view_points: Optional[
            List[Tuple[float, float, float]]
        ] = None
        self._distance_to = self._config.distance_to

        super().__init__(**kwargs)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def reset_metric(self, episode, *args: Any, **kwargs: Any):
        self._previous_position = None
        if self._distance_to == "VIEW_POINTS":
            self._episode_view_points = [
                view_point.agent_state.position
                for goal in episode.goals
                for view_point in goal.view_points
            ]
        # Support EQA task
        elif self._distance_to == "EQA_POINTS":
            self._episode_view_points = [
                view_point.position
                for view_point in episode.goals[0].view_points
            ]
        self.update_metric(episode=episode, *args, **kwargs)  # type: ignore

    def update_metric(
        self, episode: NavigationEpisode, *args: Any, **kwargs: Any
    ):
        current_position = self._sim.get_agent_state().position

        if self._previous_position is None or not np.allclose(
            self._previous_position, current_position, atol=1e-4
        ):
            if self._distance_to == "POINT":
                distance_to_target = self._sim.geodesic_distance(
                    current_position,
                    [goal.position for goal in episode.goals],
                    episode,
                )
            elif self._distance_to == "VIEW_POINTS":
                distance_to_target = self._sim.geodesic_distance(
                    current_position, self._episode_view_points, episode
                )
            # Support EQA task
            # EQA-MP3D task is a mess on Habitat we need to handle
            # non-navigable points to avoid np.inf distance to goal
            # still this does not solve completely the problem 
            elif self._distance_to == "EQA_POINTS":
                distance_to_target = self._sim.geodesic_distance(
                    current_position, self._episode_view_points, episode
                )           
            else:
                logger.error(
                    f"Non valid distance_to parameter was provided: {self._distance_to }"
                )

            self._previous_position = (
                current_position[0],
                current_position[1],
                current_position[2],
            )
            self._metric = distance_to_target


@registry.register_measure
class DistanceToGoalReward(Measure):
    """
    The measure calculates a reward based on the distance towards the goal.
    The reward is `- (new_distance - previous_distance)` i.e. the
    decrease of distance to the goal.
    """

    cls_uuid: str = "distance_to_goal_reward"

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
            self.uuid, [DistanceToGoal.cls_uuid]
        )
        self._previous_distance = task.measurements.measures[
            DistanceToGoal.cls_uuid
        ].get_metric()
        self.update_metric(episode=episode, task=task, *args, **kwargs)  # type: ignore

    def update_metric(
        self, episode, task: EmbodiedTask, *args: Any, **kwargs: Any
    ):
        distance_to_target = task.measurements.measures[
            DistanceToGoal.cls_uuid
        ].get_metric()
        self._metric = -(distance_to_target - self._previous_distance)
        self._previous_distance = distance_to_target


class NavigationMovementAgentAction(SimulatorTaskAction):
    def __init__(self, *args, config, sim, **kwargs):
        super().__init__(*args, config=config, sim=sim, **kwargs)
        self._sim = sim
        self._tilt_angle = config.tilt_angle

    def _move_camera_vertical(self, amount: float):
        assert (
            len(self._sim.agents) == 1  # type: ignore
        ), "For navigation tasks, there can be only one agent in the scene"
        sensor_names = list(self._sim.agents[0]._sensors.keys())  # type: ignore
        for sensor_name in sensor_names:
            sensor = self._sim.agents[0]._sensors[sensor_name].node  # type: ignore
            sensor.rotation = sensor.rotation * mn.Quaternion.rotation(
                mn.Deg(amount), mn.Vector3.x_axis()
            )


@registry.register_task_action
class MoveForwardAction(SimulatorTaskAction):
    name: str = "move_forward"

    def step(self, task: EmbodiedTask, *args: Any, **kwargs: Any):
        r"""Update ``_metric``, this method is called from ``Env`` on each
        ``step``.
        """
        if hasattr(
            task, "is_found_called"
        ):
            task.is_found_called = False

        return self._sim.step(HabitatSimActions.move_forward)


@registry.register_task_action
class TurnLeftAction(SimulatorTaskAction):
    def step(self, task: EmbodiedTask, *args: Any, **kwargs: Any):
        r"""Update ``_metric``, this method is called from ``Env`` on each
        ``step``.
        """
        if hasattr(
            task, "is_found_called"
        ):
            task.is_found_called = False
        return self._sim.step(HabitatSimActions.turn_left)


@registry.register_task_action
class TurnRightAction(SimulatorTaskAction):
    def step(self, task: EmbodiedTask, *args: Any, **kwargs: Any):
        r"""Update ``_metric``, this method is called from ``Env`` on each
        ``step``.
        """
        if hasattr(
            task, "is_found_called"
        ):
            task.is_found_called = False

        return self._sim.step(HabitatSimActions.turn_right)


@registry.register_task_action
class StopAction(SimulatorTaskAction):
    name: str = "stop"

    def reset(self, task: EmbodiedTask, *args: Any, **kwargs: Any):
        task.is_stop_called = False  # type: ignore
        if hasattr(
            task, "is_found_called"
        ):
            task.is_found_called = False

    def step(self, task: EmbodiedTask, *args: Any, **kwargs: Any):
        r"""Update ``_metric``, this method is called from ``Env`` on each
        ``step``.
        """
        if hasattr(
            task, "is_found_called"
        ):
            task.is_found_called = True

        task.is_stop_called = True  # type: ignore


@registry.register_task_action
class LookUpAction(NavigationMovementAgentAction):
    def step(self, *args: Any, **kwargs: Any):
        r"""Update ``_metric``, this method is called from ``Env`` on each
        ``step``.
        """
        self._move_camera_vertical(self._tilt_angle)


@registry.register_task_action
class LookDownAction(NavigationMovementAgentAction):
    def step(self, *args: Any, **kwargs: Any):
        r"""Update ``_metric``, this method is called from ``Env`` on each
        ``step``.
        """
        self._move_camera_vertical(-self._tilt_angle)


@registry.register_task_action
class TeleportAction(SimulatorTaskAction):
    # TODO @maksymets: Propagate through Simulator class
    COORDINATE_EPSILON = 1e-6
    COORDINATE_MIN = -62.3241 - COORDINATE_EPSILON
    COORDINATE_MAX = 90.0399 + COORDINATE_EPSILON

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return "teleport"

    def step(
        self,
        *args: Any,
        position: List[float],
        rotation: Sequence[float],
        **kwargs: Any,
    ):
        r"""Update ``_metric``, this method is called from ``Env`` on each
        ``step``.
        """

        if not isinstance(rotation, list):
            rotation = list(rotation)

        if not self._sim.is_navigable(position):
            return

        self._sim.set_agent_state(  # type:ignore
            position, rotation, reset_sensors=False
        )

    @property
    def action_space(self) -> spaces.Dict:
        return spaces.Dict(
            {
                "position": spaces.Box(
                    low=np.array([self.COORDINATE_MIN] * 3),
                    high=np.array([self.COORDINATE_MAX] * 3),
                    dtype=np.float32,
                ),
                "rotation": spaces.Box(
                    low=np.array([-1.0, -1.0, -1.0, -1.0]),
                    high=np.array([1.0, 1.0, 1.0, 1.0]),
                    dtype=np.float32,
                ),
            }
        )


@registry.register_task_action
class VelocityAction(SimulatorTaskAction):
    name: str = "velocity_control"

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.vel_control = VelocityControl()
        self.vel_control.controlling_lin_vel = True
        self.vel_control.controlling_ang_vel = True
        self.vel_control.lin_vel_is_local = True
        self.vel_control.ang_vel_is_local = True

        config = kwargs["config"]
        self.min_lin_vel, self.max_lin_vel = config.lin_vel_range
        self.min_ang_vel, self.max_ang_vel = config.ang_vel_range
        self.min_abs_lin_speed = config.min_abs_lin_speed
        self.min_abs_ang_speed = config.min_abs_ang_speed
        self.time_step = config.time_step
        self._allow_sliding = self._sim.config.sim_cfg.allow_sliding  # type: ignore

    @property
    def action_space(self):
        return ActionSpace(
            {
                "linear_velocity": spaces.Box(
                    low=np.array([self.min_lin_vel]),
                    high=np.array([self.max_lin_vel]),
                    dtype=np.float32,
                ),
                "angular_velocity": spaces.Box(
                    low=np.array([self.min_ang_vel]),
                    high=np.array([self.max_ang_vel]),
                    dtype=np.float32,
                ),
            }
        )

    def reset(self, task: EmbodiedTask, *args: Any, **kwargs: Any):
        task.is_stop_called = False  # type: ignore

    def step(
        self,
        *args: Any,
        task: EmbodiedTask,
        linear_velocity: float,
        angular_velocity: float,
        time_step: Optional[float] = None,
        allow_sliding: Optional[bool] = None,
        **kwargs: Any,
    ):
        r"""Moves the agent with a provided linear and angular velocity for the
        provided amount of time

        Args:
            linear_velocity: between [-1,1], scaled according to
                             config.lin_vel_range
            angular_velocity: between [-1,1], scaled according to
                             config.ang_vel_range
            time_step: amount of time to move the agent for
            allow_sliding: whether the agent will slide on collision
        """
        if allow_sliding is None:
            allow_sliding = self._allow_sliding
        if time_step is None:
            time_step = self.time_step

        # Convert from [-1, 1] to [0, 1] range
        linear_velocity = (linear_velocity + 1.0) / 2.0
        angular_velocity = (angular_velocity + 1.0) / 2.0

        # Scale actions
        linear_velocity = self.min_lin_vel + linear_velocity * (
            self.max_lin_vel - self.min_lin_vel
        )
        angular_velocity = self.min_ang_vel + angular_velocity * (
            self.max_ang_vel - self.min_ang_vel
        )

        # Stop is called if both linear/angular speed are below their threshold
        if (
            abs(linear_velocity) < self.min_abs_lin_speed
            and abs(angular_velocity) < self.min_abs_ang_speed
        ):
            task.is_stop_called = True  # type: ignore
            return

        angular_velocity = np.deg2rad(angular_velocity)
        self.vel_control.linear_velocity = np.array(
            [0.0, 0.0, -linear_velocity]
        )
        self.vel_control.angular_velocity = np.array(
            [0.0, angular_velocity, 0.0]
        )
        agent_state = self._sim.get_agent_state()

        # Convert from np.quaternion (quaternion.quaternion) to mn.Quaternion
        normalized_quaternion = agent_state.rotation
        agent_mn_quat = mn.Quaternion(
            normalized_quaternion.imag, normalized_quaternion.real
        )
        current_rigid_state = RigidState(
            agent_mn_quat,
            agent_state.position,
        )

        # manually integrate the rigid state
        goal_rigid_state = self.vel_control.integrate_transform(
            time_step, current_rigid_state
        )

        # snap rigid state to navmesh and set state to object/agent
        if allow_sliding:
            step_fn = self._sim.pathfinder.try_step  # type: ignore
        else:
            step_fn = self._sim.pathfinder.try_step_no_sliding  # type: ignore

        final_position = step_fn(
            agent_state.position, goal_rigid_state.translation
        )
        final_rotation = [
            *goal_rigid_state.rotation.vector,
            goal_rigid_state.rotation.scalar,
        ]

        # Check if a collision occurred
        dist_moved_before_filter = (
            goal_rigid_state.translation - agent_state.position
        ).dot()
        dist_moved_after_filter = (final_position - agent_state.position).dot()

        # NB: There are some cases where ||filter_end - end_pos|| > 0 when a
        # collision _didn't_ happen. One such case is going up stairs.  Instead,
        # we check to see if the the amount moved after the application of the
        # filter is _less_ than the amount moved before the application of the
        # filter.
        EPS = 1e-5
        collided = (dist_moved_after_filter + EPS) < dist_moved_before_filter

        self._sim.set_agent_state(  # type:ignore
            final_position, final_rotation, reset_sensors=False
        )
        # TODO: Make a better way to flag collisions
        self._sim._prev_sim_obs["collided"] = collided  # type: ignore


@registry.register_task(name="Nav-v0")
class NavigationTask(EmbodiedTask):
    def __init__(
        self,
        config: "DictConfig",
        sim: Simulator,
        dataset: Optional[Dataset] = None,
    ) -> None:
        super().__init__(config=config, sim=sim, dataset=dataset)

    def overwrite_sim_config(self, config: Any, episode: Episode) -> Any:
        with read_write(config):
            config.simulator.scene = episode.scene_id
            if (
                episode.start_position is not None
                and episode.start_rotation is not None
            ):
                agent_config = get_agent_config(config.simulator)
                agent_config.start_position = episode.start_position
                agent_config.start_rotation = [
                    float(k) for k in episode.start_rotation
                ]
                agent_config.is_set_start_state = True
        return config

    def _check_episode_is_active(self, *args: Any, **kwargs: Any) -> bool:
        return not getattr(self, "is_stop_called", False)
    

@registry.register_measure
class SubSuccess(Measure):
    r"""Whether or not the agent succeeded in finding it's
    current goal. This measure depends on DistanceToGoal measure.
    """

    cls_uuid: str = "sub_success"

    def __init__(
        self, *args: Any, sim: Simulator, config: "DictConfig", **kwargs: Any
    ):
        self._sim = sim
        self._config = config

        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return self.cls_uuid

    def reset_metric(self, *args: Any, episode, task, **kwargs: Any): ##Called only when episode begins
        task.measurements.check_measure_dependencies(
            self.uuid, [DistanceToCurrGoal.cls_uuid]
        )
        task.currGoalIndex=0  
        self.update_metric(*args, episode=episode, task=task, **kwargs)

    def update_metric(
        self, *args: Any, episode, task: EmbodiedTask, **kwargs: Any
    ):
        distance_to_subgoal = task.measurements.measures[
            DistanceToCurrGoal.cls_uuid
        ].get_metric()

        if (
            hasattr(task, "is_found_called")
            and task.is_found_called
            and distance_to_subgoal <= self._config.success_distance
        ):
            self._metric = 1
            task.currGoalIndex+=1
            task.foundDistance = distance_to_subgoal

        else:
            self._metric = 0

@registry.register_measure
class PercentageSuccess(Measure):
    r"""Variant of SubSuccess. It tells how much of the episode 
        is successful
    """

    cls_uuid: str = "percentage_success"

    def __init__(
        self, *args: Any, sim: Simulator, config: "DictConfig", **kwargs: Any
    ):
        self._sim = sim
        self._config = config

        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return self.cls_uuid

    def reset_metric(self, *args: Any, episode, task, **kwargs: Any): ##Called only when episode begins
        task.measurements.check_measure_dependencies(
            self.uuid, [DistanceToCurrGoal.cls_uuid]
        )
        self._metric=0
        self.update_metric(*args, episode=episode, task=task, **kwargs)

    def update_metric(
        self, *args: Any, episode, task: EmbodiedTask, **kwargs: Any
    ):
        distance_to_subgoal = task.measurements.measures[
            DistanceToCurrGoal.cls_uuid
        ].get_metric()

        if (
            hasattr(task, "is_found_called")
            and task.is_found_called
            and distance_to_subgoal < self._config.success_distance
        ):
            self._metric += 1/len(episode.goals)

@registry.register_measure
class MSPL(Measure):
    """SPL, but in multigoal case
    """
    cls_uuid: str = "mspl"

    def __init__(
        self, sim: Simulator, config: "DictConfig", *args: Any, **kwargs: Any
    ):
        self._previous_position = None
        self._start_end_episode_distance = None
        self._agent_episode_distance = None
        self._sim = sim
        self._config = config
        self._episode_view_points = None
        super().__init__(**kwargs)


    def _get_uuid(self, *args: Any, **kwargs: Any):
        return self.cls_uuid

    def reset_metric(self, *args: Any, episode, task, **kwargs: Any):
        self._previous_position = self._sim.get_agent_state().position.tolist()

        self._start_end_episode_distance = 0
        for goal_number in range(len(episode.goals) ):  # Find distances between successive goals and keep adding them
            if goal_number == 0:
                self._start_end_episode_distance_tmp = self._sim.geodesic_distance(
                    episode.start_position, episode.goals[0].position
                )
                if self._start_end_episode_distance_tmp == np.inf:
                    self._start_end_episode_distance += self._euclidean_distance(
                        episode.start_position, episode.goals[0].position
                    )
                else:
                    self._start_end_episode_distance += self._start_end_episode_distance_tmp
            else:
                self._start_end_episode_distance_tmp = self._sim.geodesic_distance(
                    episode.goals[goal_number - 1].position, episode.goals[goal_number].position
                )
                if self._start_end_episode_distance_tmp == np.inf:
                    self._start_end_episode_distance += self._euclidean_distance(
                        episode.goals[goal_number - 1].position, episode.goals[goal_number].position
                    )
                else:
                    self._start_end_episode_distance += self._start_end_episode_distance_tmp


        self._agent_episode_distance = 0.0
        self._metric = None
        task.measurements.check_measure_dependencies(
            self.uuid, [Success.cls_uuid]
        )
        self.update_metric(*args, episode=episode, task=task, **kwargs)
        ##

    def _euclidean_distance(self, position_a, position_b):
        return np.linalg.norm(
            np.array(position_b) - np.array(position_a), ord=2
        )

    def update_metric(self, *args: Any, episode, task: EmbodiedTask, **kwargs: Any):
        current_position = self._sim.get_agent_state().position.tolist()
        ep_success = task.measurements.measures[Success.cls_uuid].get_metric()

        self._agent_episode_distance += self._euclidean_distance(
            current_position, self._previous_position
        )
        self._previous_position = current_position

        self._metric = ep_success * (
            self._start_end_episode_distance
            / max(
                self._start_end_episode_distance, self._agent_episode_distance
            )
        )
@registry.register_measure
class PSPL(Measure):
    """SPL, but in multigoal case
    """
    cls_uuid: str = "pspl"

    def __init__(
        self, sim: Simulator, config: "DictConfig", *args: Any, **kwargs: Any
    ):
        self._previous_position = None
        self._start_end_episode_distance = None
        self._agent_episode_distance = None
        self._sim = sim
        self._config = config
        self._episode_view_points = None
        super().__init__(**kwargs)


    def _get_uuid(self, *args: Any, **kwargs: Any):
        return self.cls_uuid

    def reset_metric(self, *args: Any, episode, task, **kwargs: Any):
        self._previous_position = self._sim.get_agent_state().position.tolist()

        self._start_end_episode_distance = 0
        self._start_subgoal_episode_distance = []
        self._start_subgoal_agent_distance = []
        for goal_number in range(len(episode.goals) ):  # Find distances between successive goals and keep adding them
            if goal_number == 0:
                self._start_end_episode_distance += self._sim.geodesic_distance(
                    episode.start_position, episode.goals[0].position
                )
                self._start_subgoal_episode_distance.append(self._start_end_episode_distance)
            else:
                self._start_end_episode_distance += self._sim.geodesic_distance(
                    episode.goals[goal_number - 1].position, episode.goals[goal_number].position
                )
                self._start_subgoal_episode_distance.append(self._start_end_episode_distance)
        self._agent_episode_distance = 0.0
        self._metric = None
        task.measurements.check_measure_dependencies(
            self.uuid, [SubSuccess.cls_uuid, PercentageSuccess.cls_uuid]
        )
        self.update_metric(*args, episode=episode, task=task, **kwargs)
        ##

    def _euclidean_distance(self, position_a, position_b):
        return np.linalg.norm(
            np.array(position_b) - np.array(position_a), ord=2
        )

    def update_metric(self, *args: Any, episode, task: EmbodiedTask, **kwargs: Any):
        current_position = self._sim.get_agent_state().position.tolist()
        ep_percentage_success = task.measurements.measures[PercentageSuccess.cls_uuid].get_metric()
        ep_sub_success = task.measurements.measures[SubSuccess.cls_uuid].get_metric()

        self._agent_episode_distance += self._euclidean_distance(
            current_position, self._previous_position
        )
        self._previous_position = current_position

        if ep_sub_success:
            self._start_subgoal_agent_distance.append(self._agent_episode_distance)

        if ep_percentage_success > 0:
            self._metric = ep_percentage_success * (
                self._start_subgoal_episode_distance[task.currGoalIndex - 1]
                / max(
                    self._start_subgoal_agent_distance[-1], self._start_subgoal_episode_distance[task.currGoalIndex - 1]
                )
            )
        else:
            self._metric = 0
@registry.register_measure
class RawMetrics(Measure):
    """All the raw metrics we might need
    """
    cls_uuid: str = "raw_metrics"

    def __init__(
        self, sim: Simulator, config: "DictConfig", *args: Any, **kwargs: Any
    ):
        self._previous_position = None
        self._start_end_episode_distance = None
        self._agent_episode_distance = None
        self._sim = sim
        self._config = config
        self._episode_view_points = None
        super().__init__(**kwargs)


    def _get_uuid(self, *args: Any, **kwargs: Any):
        return self.cls_uuid

    def reset_metric(self, *args: Any, episode, task, **kwargs: Any):
        self._previous_position = self._sim.get_agent_state().position.tolist()

        self._start_end_episode_distance = 0
        self._start_subgoal_episode_distance = []
        self._start_subgoal_agent_distance = []
        for goal_number in range(len(episode.goals) ):  # Find distances between successive goals and keep adding them
            if goal_number == 0:
                self._start_end_episode_distance += self._sim.geodesic_distance(
                    episode.start_position, episode.goals[0].position
                )
                self._start_subgoal_episode_distance.append(self._start_end_episode_distance)
            else:
                self._start_end_episode_distance += self._sim.geodesic_distance(
                    episode.goals[goal_number - 1].position, episode.goals[goal_number].position
                )
                self._start_subgoal_episode_distance.append(self._start_end_episode_distance)

        self._agent_episode_distance = 0.0
        self._metric = None
        task.measurements.check_measure_dependencies(
            self.uuid, [EpisodeLength.cls_uuid, MSPL.cls_uuid, PSPL.cls_uuid, DistanceToMultiGoal.cls_uuid, DistanceToCurrGoal.cls_uuid, SubSuccess.cls_uuid, Success.cls_uuid, PercentageSuccess.cls_uuid]
        )
        self.update_metric(*args, episode=episode, task=task, **kwargs)
        ##

    def _euclidean_distance(self, position_a, position_b):
        return np.linalg.norm(
            np.array(position_b) - np.array(position_a), ord=2
        )

    def update_metric(self, *args: Any, episode, task: EmbodiedTask, **kwargs: Any):
        current_position = self._sim.get_agent_state().position.tolist()
        ep_success = task.measurements.measures[Success.cls_uuid].get_metric()
        p_success = task.measurements.measures[PercentageSuccess.cls_uuid].get_metric()
        distance_to_curr_subgoal = task.measurements.measures[DistanceToCurrGoal.cls_uuid].get_metric()
        ep_sub_success = task.measurements.measures[SubSuccess.cls_uuid].get_metric()

        self._agent_episode_distance += self._euclidean_distance(
            current_position, self._previous_position
        )
        self._previous_position = current_position
        if ep_sub_success:
            self._start_subgoal_agent_distance.append(self._agent_episode_distance)

        self._metric = {
            'success': ep_success,
            'percentage_success': p_success,
            'geodesic_distances': self._start_subgoal_episode_distance,
            'agent_path_length': self._agent_episode_distance,
            'subgoals_found': task.currGoalIndex,
            'distance_to_curr_subgoal': distance_to_curr_subgoal,
            'agent_distances': self._start_subgoal_agent_distance,
            'distance_to_multi_goal': task.measurements.measures[DistanceToMultiGoal.cls_uuid].get_metric(),
            'MSPL': task.measurements.measures[MSPL.cls_uuid].get_metric(),
            'PSPL': task.measurements.measures[PSPL.cls_uuid].get_metric(),
            'episode_lenth': task.measurements.measures[EpisodeLength.cls_uuid].get_metric()
        }

        if ep_sub_success:
            print("--------------")
            print(f"{task.currGoalIndex} SubGoal Found")
            # print them in line with only two decimal and divide by | metric |
            print("Success: ", round(ep_success, 2), " | Percentage Success: ", round(p_success, 2), " | Distance to Current Subgoal: ", round(distance_to_curr_subgoal, 2), " | Distance to Multi Goal: ", round(self._metric['distance_to_multi_goal'], 2), "\n| Subgoals Found: ", self._metric['subgoals_found'], " | MSPL: ", round(self._metric['MSPL'], 2), " | PSPL: ", round(self._metric['PSPL'], 2), " | Episode Length: ", self._metric['episode_lenth'])

@registry.register_measure
class WPL(Measure):
    """
    MSPL but without the multiplicative factor of Success
    """

    cls_uuid: str = "wpl"

    def __init__(
        self, sim: Simulator, config: "DictConfig", *args: Any, **kwargs: Any
    ):
        self._previous_position = None
        self._start_end_episode_distance = None
        self._agent_episode_distance = None
        self._sim = sim
        self._config = config
        self._episode_view_points = None
        super().__init__(**kwargs)


    def _get_uuid(self, *args: Any, **kwargs: Any):
        return self.cls_uuid

    def reset_metric(self, *args: Any, episode, task, **kwargs: Any):
        self._previous_position = self._sim.get_agent_state().position.tolist()
        self._start_end_episode_distance = 0
        for goal_number in range(len(episode.goals) ):  # Find distances between successive goals and keep adding them
            if goal_number == 0:
                self._start_end_episode_distance += self._sim.geodesic_distance(
                    self._previous_position, episode.goals[0][0].position
                )
            else:
                self._start_end_episode_distance += self._sim.geodesic_distance(
                    episode.goals[goal_number - 1][0].position, episode.goals[goal_number][0].position
                )
        self._agent_episode_distance = 0.0
        self._metric = None
        task.measurements.check_measure_dependencies(
            self.uuid, [Success.cls_uuid]
        )
        self.update_metric(*args, episode=episode, **kwargs)
        ##

    def _euclidean_distance(self, position_a, position_b):
        return np.linalg.norm(
            np.array(position_b) - np.array(position_a), ord=2
        )

    def update_metric(self, *args: Any, episode, task: EmbodiedTask, **kwargs: Any):
        current_position = self._sim.get_agent_state().position.tolist()
        self._agent_episode_distance += self._euclidean_distance(
            current_position, self._previous_position
        )
        self._previous_position = current_position

        self._metric = 1 * (
            self._start_end_episode_distance
            / max(
                self._start_end_episode_distance, self._agent_episode_distance
            )
        )
@registry.register_measure
class DistanceToCurrGoal(Measure):
    """The measure calculates a distance towards the goal.
    """

    cls_uuid: str = "distance_to_curr_goal"

    def __init__(
        self, sim: Simulator, config: "DictConfig", *args: Any, **kwargs: Any
    ):
        self._previous_position = None
        self._start_end_episode_distance = None
        self._agent_episode_distance = None
        self._sim = sim
        self._config = config
        self._episode_view_points = None
        super().__init__(**kwargs)

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return self.cls_uuid

    def reset_metric(self, episode, task, *args: Any, **kwargs: Any):
        task.currGoalIndex = 0
        self._previous_position = self._sim.get_agent_state().position.tolist()
        self._start_end_subgoal_distance = self._sim.geodesic_distance(
            self._previous_position, episode.goals[task.currGoalIndex].position
        )
        self._agent_subgoal_distance = 0.0
        self._metric = None
        if self._config.distance_to == "VIEW_POINTS":
            self._subgoal_view_points = [
                view_point.agent_state.position
                for goal in episode.goals[task.currGoalIndex]
                for view_point in goal.view_points
            ]
        self.update_metric(*args, episode=episode, task=task, **kwargs)

    def _euclidean_distance(self, position_a, position_b):
        return np.linalg.norm(
            np.array(position_b) - np.array(position_a), ord=2
        )

    def update_metric(self, episode, task, *args: Any, **kwargs: Any):
        current_position = self._sim.get_agent_state().position.tolist()
        if self._config.distance_to == "POINT":
            distance_to_subgoal = self._sim.geodesic_distance(
                current_position, episode.goals[task.currGoalIndex].position
            )
            if np.isinf(distance_to_subgoal):
                distance_to_subgoal = self._euclidean_distance(
                    current_position, episode.goals[task.currGoalIndex].position
                )
                # new_position = np.array(current_position, dtype='f')
                # new_position = self._sim.pathfinder.get_random_navigable_point_near(
                #     circle_center=np.array(episode.goals[task.currGoalIndex].position),
                #     radius= 4.0
                # )
                # distance_to_subgoal = self._sim.geodesic_distance(
                #     current_position, new_position
                # )
            # print("Distance to Subgoal: ", distance_to_subgoal)
        elif self._config.distance_to == "VIEW_POINTS":
            distance_to_subgoal = self._sim.geodesic_distance(
                current_position, self._subgoal_view_points
            )

        else:
            logger.error(
                f"Non valid DISTANCE_TO parameter was provided: {self._config.DISTANCE_TO}"
            )

        self._agent_subgoal_distance += self._euclidean_distance(
            current_position, self._previous_position
        )

        self._previous_position = current_position

        self._metric = distance_to_subgoal

    def _euclidean_distance(self, position_a, position_b):
        return np.linalg.norm(
            np.array(position_b) - np.array(position_a), ord=2
        )
    
@registry.register_measure
class DistanceToMultiGoal(Measure):
    """The measure calculates a distance towards the goal.
    """

    cls_uuid: str = "distance_to_multi_goal"

    def __init__(
        self, sim: Simulator, config: "DictConfig", *args: Any, **kwargs: Any
    ):
        self._sim = sim
        self._config = config

        super().__init__(**kwargs)

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return self.cls_uuid

    def reset_metric(self, episode, task, *args: Any, **kwargs: Any):
        self._metric = None
        """if self._config.DISTANCE_TO == "VIEW_POINTS":
            self._episode_view_points = [
                view_point.agent_state.position
                # for goal in episode.goals     # Considering only one goal for now
                for view_point in episode.goals[episode.object_index][0].view_points
            ]"""
        self.update_metric(*args, episode=episode, task=task, **kwargs)

    def update_metric(self, episode, task, *args: Any, **kwargs: Any):
        current_position = self._sim.get_agent_state().position.tolist()

        if self._config.distance_to == "POINT":
            distance_to_target = self._sim.geodesic_distance(
                current_position, episode.goals[task.currGoalIndex].position
            )
            if np.isinf(distance_to_target):
                distance_to_target = self._euclidean_distance(
                    current_position, episode.goals[task.currGoalIndex].position
                )
            for goal_number in range(task.currGoalIndex, len(episode.goals)-1):
                distance_to_target_tmp = self._sim.geodesic_distance(
                    episode.goals[goal_number].position, episode.goals[goal_number+1].position
                )
                if distance_to_target_tmp == np.inf:
                    distance_to_target += self._euclidean_distance(
                        episode.goals[goal_number].position, episode.goals[goal_number+1].position
                    )
                else:
                    distance_to_target += distance_to_target_tmp
        else:
            logger.error(
                f"Non valid DISTANCE_TO parameter was provided: {self._config.DISTANCE_TO}"
            )

        self._metric = distance_to_target

    def _euclidean_distance(self, position_a, position_b):
        return np.linalg.norm(
            np.array(position_b) - np.array(position_a), ord=2
        )
@registry.register_measure
class Ratio(Measure):
    """The measure calculates a distance towards the goal.
    """

    cls_uuid: str = "ratio"

    def __init__(
        self, sim: Simulator, config: "DictConfig", *args: Any, **kwargs: Any
    ):
        self._sim = sim
        self._config = config

        super().__init__(**kwargs)

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return self.cls_uuid

    def reset_metric(self, episode, task, *args: Any, **kwargs: Any):
        self._metric = None
        current_position = self._sim.get_agent_state().position.tolist()
        if self._config.distance_to == "POINT":
            initial_geodesic_distance_to_target = self._sim.geodesic_distance(
                current_position, episode.goals[0].position
            )
            if initial_geodesic_distance_to_target == np.inf:
                initial_geodesic_distance_to_target = self._euclidean_distance(
                    current_position, episode.goals[0].position
                )

            for goal_number in range(0, len(episode.goals)-1):
                
                initial_geodesic_distance_to_target_tmp = self._sim.geodesic_distance(
                    episode.goals[goal_number].position, episode.goals[goal_number+1].position
                )
                if initial_geodesic_distance_to_target_tmp == np.inf:
                    initial_geodesic_distance_to_target = self._euclidean_distance(
                        episode.goals[goal_number].position, episode.goals[goal_number+1].position
                    )
                else:
                    initial_geodesic_distance_to_target += initial_geodesic_distance_to_target_tmp

            initial_euclidean_distance_to_target = self._euclidean_distance(
                current_position, episode.goals[0].position
            )
            for goal_number in range(0, len(episode.goals)-1):
                initial_euclidean_distance_to_target += self._euclidean_distance(
                    episode.goals[goal_number].position, episode.goals[goal_number+1].position
                )
        else:
            logger.error(
                f"Non valid DISTANCE_TO parameter was provided: {self._config.DISTANCE_TO}"
            )
        self._metric = initial_geodesic_distance_to_target / initial_euclidean_distance_to_target

    def _euclidean_distance(self, position_a, position_b):
        return np.linalg.norm(
            np.array(position_b) - np.array(position_a), ord=2
        )

    def update_metric(self, episode, task, *args: Any, **kwargs: Any):

        if hasattr(task, "is_found_called") and task.is_found_called:
            task.CurrGoalIndex=0
        pass
@registry.register_measure
class EpisodeLength(Measure):
    r"""Calculates the episode length
    """
    cls_uuid: str = "episode_length"

    def __init__(
        self, *args: Any, sim: Simulator, config: "DictConfig", **kwargs: Any
    ):
        self._episode_length = None
        self._sim = sim
        self._config = config

        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return self.cls_uuid

    def reset_metric(self, *args: Any, episode, task, **kwargs: Any):
        self._episode_length = 0
        self._metric = self._episode_length

    def update_metric(
        self, *args: Any, episode, task: EmbodiedTask, **kwargs: Any
    ):
        self._episode_length += 1
        self._metric = self._episode_length
@registry.register_task_action
class FoundObjectAction(SimulatorTaskAction):
    name: str = "found"
    def reset(self, *args: Any, task: EmbodiedTask, **kwargs: Any):
        task.is_stop_called = False
        task.is_found_called = False ##C

    def step(self, *args: Any, task: EmbodiedTask, **kwargs: Any):
        r"""Update ``_metric``, this method is called from ``Env`` on each
        ``step``.
        """
        task.is_found_called = True
        return self._sim.get_observations_at()