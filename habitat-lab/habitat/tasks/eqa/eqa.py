#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import TYPE_CHECKING,Any, Dict, List, Optional

import attr
from gym import Space, spaces
import spacy

from habitat.core.embodied_task import Action, Measure
from habitat.core.registry import registry
from habitat.core.simulator import Observations, Sensor, SensorTypes
from habitat.core.spaces import ListSpace
from habitat.core.utils import not_none_validator
from habitat.tasks.nav.nav import NavigationEpisode, NavigationTask
from habitat.datasets.utils import get_action_shortest_path

@attr.s(auto_attribs=True)
class QuestionData:
    question_text: str
    answer_text: str
    question_tokens: Optional[List[str]] = None
    answer_token: Optional[List[str]] = None
    question_type: Optional[str] = None
    # Added for EQA support
    answer_token_orig: Optional[str] = None 
    eqa_room: Optional[str] = None
    eqa_object: Optional[str] = None

@attr.s(auto_attribs=True, kw_only=True)
class EQAEpisode(NavigationEpisode):
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
        episode: EQAEpisode,
        *args: Any,
        **kwargs: Any
    ):
        return episode.question.question_tokens

    def _get_observation_space(self, *args: Any, **kwargs: Any) -> Space:
        return ListSpace(
            spaces.Discrete(self._dataset.question_vocab.get_size())
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
        self._metric = episode.question.answer_token

    def update_metric(self, *args: Any, **kwargs: Any):
        pass


@registry.register_measure
class EpisodeInfo(Measure):
    """Episode Info"""

    def __init__(self, sim, config, *args: Any, **kwargs: Any):
        self._sim = sim
        self._config = config

        super().__init__(**kwargs)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return "episode_info"

    def reset_metric(self, episode, *args: Any, **kwargs: Any):
        self._metric = vars(episode).copy()

    def update_metric(self, episode, action, *args: Any, **kwargs: Any):
        pass


@registry.register_measure
class AnswerAccuracy(Measure):
    """AnswerAccuracy"""

    def __init__(self, dataset, *args: Any, **kwargs: Any):
        self._dataset = dataset
        super().__init__(**kwargs)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return "answer_accuracy"

    def reset_metric(self, episode, *args: Any, **kwargs: Any):
        self._metric = 0

    def update_metric(
        self, action=None, episode=None, *args: Any, **kwargs: Any
    ):
        if episode is None:
            return

        # Modified to support EQA when AnswerAction is np.int(0)
        if action["action"] == AnswerAction.name or action["action"] == 0:
            self._metric = (
                1
                if episode.question.answer_token
                == action["action_args"]["answer_id"]
                else 0
            )

@registry.register_measure
class AnswerSimilarity(Measure):
    """AnswerSimilarity"""

    def __init__(self, dataset, *args: Any, **kwargs: Any):
        self._dataset = dataset
        self.nlp = spacy.load('en_core_web_md') 
        super().__init__(**kwargs)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return "answer_similarity"

    def reset_metric(self, episode, *args: Any, **kwargs: Any):
        self._metric = 0

    def update_metric(
        self, action=None, episode=None, *args: Any, **kwargs: Any
    ):
        if episode is None:
            return

        if action["action"] == AnswerAction.name or action["action"] == 0:
            ans_token = self.nlp(action['action_args']['answer_text']) 
            gt_token = self.nlp(episode.question.answer_text)

            self._metric = ans_token.similarity(gt_token)

@registry.register_measure
class MinimumNumberOfActions(Measure):
    """
    Minimum number of actions required to reach target
    This should be divided in 10, 30, 50.
    """
    def __init__(self, sim, *args: Any, **kwargs: Any):
        self._sim = sim
        self._cls_distance = { # Check arXiv:2405.16559 for more details on the numbers
            '10': 3.45,
            '30': 5.71,
            '50': 8.21
        }
        super().__init__(**kwargs)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return "minimum_number_of_actions"

    def reset_metric(self, episode, *args: Any, **kwargs: Any):

        self._episode_view_points = [
            view_point.position
            for view_point in episode.goals[0].view_points
        ]

        distance_to_target = self._sim.geodesic_distance(
            episode.start_position, 
            self._episode_view_points,
            episode
        )    

        if distance_to_target <= self._cls_distance['10']:
            self._metric = 10
        elif distance_to_target > self._cls_distance['10'] and distance_to_target <= self._cls_distance['30']:
            self._metric = 30
        elif distance_to_target > self._cls_distance['30'] or distance_to_target is float('inf'):
            self._metric = 50

    def update_metric(self, episode=None, *args: Any, **kwargs: Any):
        pass

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
        # Since we have non-navigable points problems
        # let's initialize with infinity
        self._metric = float('inf')

    def update_metric(self, task, episode, *args: Any, **kwargs: Any):
        current_distante_to_target = task.measurements.measures[
            'distance_to_goal'
        ].get_metric()

        if current_distante_to_target < self._metric:
            self._metric = current_distante_to_target

@registry.register_measure
class StopInCorrectRoom(Measure):
    """TODO:"""
    pass

@registry.register_task(name="EQA-v0")
class EQATask(NavigationTask):
    """
    Embodied Question Answering Task
    Usage example:
        env = habitat.Env(config=eqa_config)

        env.reset()

        for i in range(10):
            action = sample_non_stop_action(env.action_space)
            if action["action"] != AnswerAction.name:
                env.step(action)
            metrics = env.get_metrics() # to check distance to target

        correct_answer_id = env.current_episode.question.answer_token
        env.step(
            {
                "action": AnswerAction.name,
                "action_args": {"answer_id": correct_answer_id},
            }
        )

        metrics = env.get_metrics()
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

    def reset(self, task: EQATask, *args: Any, **kwargs: Any) -> None:
        task.answer = None
        task.is_valid = True
        return

    def step(
        self, *args: Any, answer_id: int, task: EQATask, **kwargs: Any
    ) -> Dict[str, Observations]:
        if task.answer is not None:
            task.is_valid = False
            task.invalid_reason = "Agent answered question twice."

        task.answer = answer_id
        return self._sim.get_observations_at()

    @property
    def action_space(self) -> spaces.Dict:
        """Answer expected to be single token."""
        return spaces.Dict(
            {
                "answer_id": spaces.Discrete(
                    self._dataset.answer_vocab.get_size()
                )
            }
        )