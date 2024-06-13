#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, Optional
import attr
from gym import Space, spaces

from habitat.core.embodied_task import Action, Measure
from habitat.core.registry import registry
from habitat.core.spaces import ListSpace
from habitat.core.simulator import Observations, Sensor, SensorTypes
from habitat.core.utils import not_none_validator
from habitat.tasks.nav.nav import NavigationEpisode, NavigationTask

@attr.s(auto_attribs=True)
class QuestionData:
    question_text: str
    answer_text: str
    question_id: Optional[int] = None
    answer_id: Optional[int] = None

@attr.s(auto_attribs=True, kw_only=True)
class OPENEQAEpisode(NavigationEpisode):
    r"""Specification of episode that includes initial position and rotation of
    agent, goal, question specifications and optional shortest paths.

    Args:
        scene_id: id of scene inside the simulator.
        start_position: numpy ndarray containing 3 entries for (x, y, z).
        start_rotation: numpy ndarray with 4 entries for (x, y, z, w)
            elements of unit quaternion (versor) representing agent 3D
            orientation.
        goals: relevant goal object/room.
        question: question related to goal object.
    """

    question: QuestionData = attr.ib(
        default=None, validator=not_none_validator
    )

@registry.register_sensor
class QuestionSensor(Sensor):
    def __init__(self, dataset, *args: Any, **kwargs: Any):
        self._dataset = dataset
        super().__init__(*args, **kwargs)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return "question"

    def _get_sensor_type(self, *args: Any, **kwargs: Any) -> SensorTypes:
        return SensorTypes.TOKEN_IDS

    def get_observation(
        self,
        observations: Dict[str, Observations],
        episode: OPENEQAEpisode,
        *args: Any,
        **kwargs: Any
    ):
        return episode.question.question_id
    
    def _get_observation_space(self, *args: Any, **kwargs: Any) -> Space:
        return ListSpace(
            spaces.Discrete(557)
        )

@registry.register_measure
class CorrectAnswer(Measure):
    """CorrectAnswer"""

    def __init__(self, dataset, *args: Any, **kwargs: Any):
        self._dataset = dataset
        super().__init__(**kwargs)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return "correct_answer"

    def reset_metric(self, episode, *args: Any, **kwargs: Any):
        self._metric = episode.question.answer_text

    def update_metric(self, *args: Any, **kwargs: Any):
        pass

@registry.register_measure
class AnswerAccuracy(Measure):
    """AnswerAccuracy"""

    def __init__(self, dataset, *args: Any, **kwargs: Any):
        self._dataset = dataset
        super().__init__(**kwargs)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return "answer_accuracy"

    def reset_metric(self, *args: Any, **kwargs: Any):
        self._metric = 0

    def update_metric(
        self, action=None, episode=None, *args: Any, **kwargs: Any
    ):
        if episode is None:
            return

        if action["action"] == AnswerAction.name or action["action"] == 0:
            self._metric = (
                1
                if episode.question.answer_text
                == action["action_args"]["answer_text"]
                else 0
            )

@registry.register_measure
class StopBeforeEpisodeEnd(Measure):
    """Stops before episode end, 1. == yes, 0. == no"""

    def __init__(self, config, sim, *args: Any, **kwargs: Any):
        self._sim = sim
        self._config = config
        self._max_steps = self._config.max_steps
        super().__init__(**kwargs)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return "stop_before_episode_end"

    def reset_metric(self, episode, task,  *args: Any, **kwargs: Any):
        task.measurements.check_measure_dependencies(
            self.uuid, ['num_steps']
        )
        self._metric = 0.

    def update_metric(
        self, task, episode, action=None,  *args: Any, **kwargs: Any
    ):
        if episode is None:
            return
        
        current_step = task.measurements.measures[
            'num_steps'
        ].get_metric()

        self._metric = (
                1.
                if (action["action"] == AnswerAction.name or action["action"] == 0) \
                and (current_step < self._max_steps)
                else 0.
            )

@registry.register_measure
class SmallestDistanceToTarget(Measure):
    """Smallest distance to target during episode"""

    def __init__(self, sim, config, *args: Any, **kwargs: Any):
        self._sim = sim

        super().__init__(**kwargs)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return "smallest_distance_to_target"

    def reset_metric(self, task, episode, *args: Any, **kwargs: Any):
        task.measurements.check_measure_dependencies(
            self.uuid, ['distance_to_goal']
        )
        self._metric = None

    def update_metric(self, task, episode, *args: Any, **kwargs: Any):
        current_distante_to_target = task.measurements.measures[
            'distance_to_goal'
        ].get_metric()

        if self._metric is None or current_distante_to_target < self._metric:
            self._metric = current_distante_to_target

@registry.register_task(name="OPENEQA-v0")
class OPENEQATask(NavigationTask):
    """
    Embodied Question Answering Task
    """
    is_valid: bool = False
    answer: Optional[int] = None
    invalid_reason: Optional[str] = None

    def _check_episode_is_active(
        self, *args, action, episode, action_args=None, **kwargs
    ) -> bool:
        return self.is_valid and self.answer is None


@registry.register_task_action
class AnswerAction(Action):
    _answer: Optional[str]
    name: str = "answer"

    def __init__(self, *args: Any, sim, dataset, **kwargs: Any) -> None:
        self._sim = sim
        self._dataset = dataset

    def reset(self, task: OPENEQATask, *args: Any, **kwargs: Any) -> None:
        task.answer = None
        task.is_valid = True
        return

    def step(
        self, *args: Any, answer_text: str, task: OPENEQATask, **kwargs: Any
    ) -> Dict[str, Observations]:
        if task.answer is not None:
            task.is_valid = False
            task.invalid_reason = "Agent answered question twice."

        task.answer = answer_text
        return self._sim.get_observations_at()

    @property
    def action_space(self) -> spaces.Dict:
        """Answer expected to be single token."""
        return spaces.Dict(
            {
                "answer_text": spaces.Discrete(
                    557
                )
            }
        )
