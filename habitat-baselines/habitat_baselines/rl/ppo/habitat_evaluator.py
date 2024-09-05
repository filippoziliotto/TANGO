import os
from collections import defaultdict
from typing import Any, Dict, List

import numpy as np
import torch
import tqdm
import wandb
import math

from habitat import logger
from habitat.tasks.rearrange.rearrange_sensors import GfxReplayMeasure
from habitat.tasks.rearrange.utils import write_gfx_replay
from habitat.utils.visualizations.utils import (
    observations_to_image,
    overlay_frame,
)
from habitat_baselines.common.obs_transformers import (
    apply_obs_transforms_batch,
)
from habitat_baselines.rl.ppo.evaluator import Evaluator, pause_envs
from habitat_baselines.utils.common import (
    batch_obs,
    generate_video,
    get_action_space_info,
    inference_mode,
    is_continuous_action_space,
)
from habitat_baselines.utils.info_dict import extract_scalars_from_info

from habitat_baselines.rl.ppo.code_interpreter.interpreter import PseudoCodeExecuter
from habitat_baselines.rl.ppo.code_interpreter.code_generator import CodeGenerator
from habitat_baselines.rl.ppo.utils.utils import (
    from_xyz_to_polar, from_polar_to_xyz
)
from habitat_baselines.rl.ppo.utils.utils import match_images, log_episode_stats, log_final_results
from habitat_baselines.rl.ppo.code_interpreter.prompts.eqa import eqa_text_to_token
from habitat_baselines.rl.ppo.utils.names import stoi_eqa
from habitat.sims.habitat_simulator.debug_visualizer import DebugObservation

DEBUG = True

class HabitatEvaluator(Evaluator):
    def __init__(self):
        self.current_step = 0
        self.eqa_vars = None
        self.debugger = DebugObservation()
        self.last_agent_pos = None
        pass

    """
    Getting Habitat variables methods
    """
    def _init_variables(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        
    def get_env_variables(self, **kwargs):
        self._init_variables(**kwargs)

        # Task
        self.task_name = self.config.habitat_baselines.task_name
        self.sampling_strategy = self.config.habitat_baselines.sampling_strategy
        print('Task:', self.task_name)
        # Detection
        self.save_obs = self.config.habitat_baselines.save_obs.save_to_disk
        self.object_detector = self.config.habitat_baselines.object_detector
        self.agent_radius = self.config.habitat.simulator.agents.main_agent.radius
        self.object_distance_threshold = self.config.habitat_baselines.object_distance_threshold
        # Feature Matching
        self.matcher = self.config.habitat_baselines.feature_matcher
        # VQA
        self.vqa = self.config.habitat_baselines.vqa
        # Captioner
        self.captioner = self.config.habitat_baselines.captioner
        # Segmentation
        self.segmenter = self.config.habitat_baselines.segmenter
        # LLM
        self.LLM = self.config.habitat_baselines.LLM
        # Room classifier
        self.room_classifier = self.config.habitat_baselines.room_classifier
        # CLIP value mapper
        self.value_mapper = self.config.habitat_baselines.value_mapper

    def init_env(self):
        """
        Methods for initializing/executing/evauluating
        the agent in Habitat environment
        """
        self.observations = self.envs.reset()
        self.observations = self.envs.post_step(self.observations)
        self.batch = batch_obs(self.observations, device=self.device)
        self.batch = apply_obs_transforms_batch(self.batch, self.obs_transforms)  # type: ignore

        self.action_shape, self.discrete_actions = get_action_space_info(
            self.agent.actor_critic.policy_action_space
        )

        self.current_episode_reward = torch.zeros(self.envs.num_envs, 1, device="cpu")

        self.test_recurrent_hidden_states = torch.zeros(
            (
                self.config.habitat_baselines.num_environments,
                *self.agent.actor_critic.hidden_state_shape,
            ),
            device=self.device,
        )

        self.hidden_state_lens = self.agent.actor_critic.hidden_state_shape_lens
        self.action_space_lens = self.agent.actor_critic.policy_action_space_shape_lens

        self.prev_actions = torch.zeros(
            self.config.habitat_baselines.num_environments,
            *self.action_shape,
            device=self.device,
            dtype=torch.long if self.discrete_actions else torch.float,
        )
        self.not_done_masks = torch.zeros(
            self.config.habitat_baselines.num_environments,
            *self.agent.masks_shape,
            device=self.device,
            dtype=torch.bool,
        )
        self.stats_episodes: Dict[
            Any, Any
        ] = {}  # dict of dicts that stores stats per episode
        self.ep_eval_count: Dict[Any, int] = defaultdict(lambda: 0)

        if len(self.config.habitat_baselines.eval.video_option) > 0:
            # Add the first frame of the episode to the video.
            self.rgb_frames: List[List[np.ndarray]] = [
                [
                    observations_to_image(
                        {k: v[env_idx] for k, v in self.batch.items()}, {}
                    )
                ]
                for env_idx in range(self.config.habitat_baselines.num_environments)
            ]
        else:
            self.rgb_frames = None

        if len(self.config.habitat_baselines.eval.video_option) > 0:
            os.makedirs(self.config.habitat_baselines.video_dir, exist_ok=True)

        self.number_of_eval_episodes = self.config.habitat_baselines.test_episode_count
        self.evals_per_ep = self.config.habitat_baselines.eval.evals_per_ep
        if self.number_of_eval_episodes == -1:
            self.number_of_eval_episodes = sum(self.envs.number_of_episodes)
        else:
            self.total_num_eps = sum(self.envs.number_of_episodes)
            # if total_num_eps is negative, it means the number of evaluation episodes is unknown
            if self.total_num_eps < self.number_of_eval_episodes and self.total_num_eps > 1:
                logger.warn(
                    f"Config specified {self.number_of_eval_episodes} eval episodes"
                    ", dataset only has {self.total_num_eps}."
                )
                logger.warn(f"Evaluating with {self.total_num_eps} instead.")
                self.number_of_eval_episodes = self.total_num_eps
            else:
                assert self.evals_per_ep == 1
        assert (
            self.number_of_eval_episodes > 0
        ), "You must specify a number of evaluation episodes with test_episode_count"

        self.pbar = tqdm.tqdm(total=self.number_of_eval_episodes * self.evals_per_ep)
        self.agent.eval()

        # If num_envs is > 1, then we need to call the environment for each env
        self.env_call_list = ['habitat_env' for i in range(self.envs.num_envs)]
        self.current_scene = self.get_current_episode_info().scene_id

    def predict_action(self, coords):
        """
        Given the target pointgoal distance and angle,
        predict the next action to take
        """

        self.space_lengths = {}
        self.batch['pointgoal_with_gps_compass'] = coords.to(self.device)

        # Needed for frontier exploration policy. Is it really???
        if "heading" in list(self.batch.keys()):
            self.batch.pop("heading")

        with inference_mode():
            self.action_data = self.agent.actor_critic.act(
                self.batch,
                self.test_recurrent_hidden_states,
                self.prev_actions,
                self.not_done_masks.to("cuda:0"),
                deterministic=False,
                **self.space_lengths,
            )

            # Other stuff
            if self.action_data.should_inserts is None:
                self.test_recurrent_hidden_states = (
                    self.action_data.rnn_hidden_states
                )
                self.prev_actions.copy_(self.action_data.actions)  # type: ignore
            else:
                self.agent.actor_critic.update_hidden_state(
                self.test_recurrent_hidden_states, self.prev_actions, self.action_data
                )   

            if is_continuous_action_space(self.env_spec.action_space):
                # Clipping actions to the specified limits
                self.step_data = [
                    np.clip(
                        a.numpy(),
                        self.env_spec.action_space.low,
                        self.env_spec.action_space.high,
                    )
                    for a in self.action_data.env_actions.cpu()
                ]
            else:
                self.step_data = [a.item() for a in self.action_data.env_actions.cpu()]

    def execute_step(self, action=None):
        """
        Execute the predicted action in the environment
        and update all the necessary variables
        """

        self.action_to_take = action
        if self.action_to_take in ['stop']:
            self.step_data = [0] 

            # Added for EQA support
            if self.task_name in ['eqa', 'open_eqa']:

                # Case in which the agent does not reach the goal
                if self.max_steps_reached() and 'pred_answer' not in list(self.eqa_vars.keys()):
                    self.eqa_vars['pred_answer'] = '_'

                answer_token, answer_text = eqa_text_to_token(stoi_eqa, self.eqa_vars['pred_answer'])
                self.step_data = [
                    {
                        "action": 0,
                        "action_args": {"answer_id": answer_token,
                                        "answer_text": answer_text},
                    }
                ]

        elif self.action_to_take in ['turn_right']:
            self.step_data = [3]
        elif self.action_to_take in ['turn_left']:
            self.step_data = [2]
        elif self.action_to_take in ['look_up']:
            self.step_data = [4]
        elif self.action_to_take in ['look_down']:
            self.step_data = [5]
        elif (self.action_to_take is None) and self.action_data.actions.item() == 0:
            self.step_data = [a.item() for a in self.action_data.env_actions.cpu()]
            self.prev_actions.copy_(self.action_data.actions) # type: ignore
            self.step_data = [torch.randint(1, 4, (1,), device=self.device).item()]

        else:
            pass
            
        self.outputs = self.envs.step(self.step_data)
        self.observations, self.rewards_l, self.dones, self.infos = [
            list(x) for x in zip(*self.outputs)
        ]
        # Note that `policy_infos` represents the information about the
        # action BEFORE `observations` (the action used to transition to
        # `observations`).

        if not isinstance(self.action_to_take, str):
            self.policy_infos = self.agent.actor_critic.get_extra(
                self.action_data, self.infos, self.dones
            )
            for i in range(len(self.policy_infos)):
                self.infos[i].update(self.policy_infos[i])

        self.observations = self.envs.post_step(self.observations)
        self.batch = batch_obs(  # type: ignore
            self.observations,
            device=self.device,
        )
        self.batch = apply_obs_transforms_batch(self.batch, self.obs_transforms)  # type: ignore

        self.not_done_masks = torch.tensor(
            [[not done] for done in self.dones],
            dtype=torch.bool,
            device="cpu",
        ).repeat(1, *self.agent.masks_shape)

        self.rewards = torch.tensor(
            self.rewards_l, dtype=torch.float, device="cpu"
        ).unsqueeze(1)
        self.current_episode_reward += self.rewards

        self.current_step += 1

    def update_episode_stats(self, display=True):
        """
        After each step, check if the episode is over
        if yes update the stats and generate video
        """
        
        self.next_episodes_info = self.envs.current_episodes()
        self.envs_to_pause = []
        self.n_envs = self.envs.num_envs
        for i in range(self.n_envs):
            if (
                self.ep_eval_count[
                    (
                        self.next_episodes_info[i].scene_id,
                        self.next_episodes_info[i].episode_id,
                    )
                ]
                == self.evals_per_ep
            ):
                self.envs_to_pause.append(i)

            # Exclude the keys from `_rank0_keys` from displaying in the video
            self.disp_info = {
                k: v for k, v in self.infos[i].items() if k not in self.rank0_keys
            }

            if len(self.config.habitat_baselines.eval.video_option) > 0:
                # TODO move normalization / channel changing out of the policy and undo it here
                frame = observations_to_image(
                    {k: v[i] for k, v in self.batch.items()}, self.disp_info
                )
                if not self.not_done_masks[i].any().item():
                    # The last frame corresponds to the first frame of the next episode
                    # but the info is correct. So we use a black frame
                    final_frame = observations_to_image(
                        {k: v[i] * 0.0 for k, v in self.batch.items()},
                        self.disp_info,
                    )
                    final_frame = overlay_frame(final_frame, self.disp_info)
                    self.rgb_frames[i].append(final_frame)
                    # The starting frame of the next episode will be the final element..
                    self.rgb_frames[i].append(frame)
                else:
                    frame = overlay_frame(frame, self.disp_info)
                    self.rgb_frames[i].append(frame)
            

            # episode ended
            if not self.not_done_masks[i].any().item() or (self.action_to_take in ['stop']):
                self.pbar.update()
                self.episode_stats = {
                    "reward": self.current_episode_reward[i].item()
                }

                # Support for EQA task if episode_infos in keys
                if self.task_name in ['eqa']:
                    self.disp_info = {key: value for key, value in self.disp_info.items() if "episode_info" not in key}
                tmp_episode_info = {key: value for key, value in self.infos[i].items() if "episode_info" not in key}

                self.episode_stats.update(extract_scalars_from_info(tmp_episode_info))
                self.current_episode_reward[i] = 0
                k = (
                    self.current_episodes_info[i].scene_id,
                    self.current_episodes_info[i].episode_id,
                )
                self.ep_eval_count[k] += 1
                # use scene_id + episode_id as unique id for storing stats
                self.stats_episodes[(k, self.ep_eval_count[k])] = self.episode_stats

                if len(self.config.habitat_baselines.eval.video_option) > 0:
                    generate_video(
                        video_option=self.config.habitat_baselines.eval.video_option,
                        video_dir=self.config.habitat_baselines.video_dir,
                        # Since the final frame is the start frame of the next episode.
                        images=self.rgb_frames[i][:-1],
                        episode_id=f"{self.current_episodes_info[i].episode_id}_{self.ep_eval_count[k]}",
                        checkpoint_idx=self.checkpoint_index,
                        metrics=extract_scalars_from_info(self.disp_info),
                        fps=self.config.habitat_baselines.video_fps,
                        tb_writer=self.writer,
                        keys_to_include_in_name=self.config.habitat_baselines.eval_keys_to_include_in_name,
                    )

                    # Since the starting frame of the next episode is the final frame.
                    self.rgb_frames[i] = self.rgb_frames[i][-1:]

                gfx_str = self.infos[i].get(GfxReplayMeasure.cls_uuid, "")
                if gfx_str != "":
                    write_gfx_replay(
                        gfx_str,
                        self.config.habitat.task,
                        self.current_episodes_info[i].episode_id,
                    )
                
                if display:
                    self.display_results(per_episode=True)
                self.current_step = 0

            self.not_done_masks = self.not_done_masks.to(device=self.device)
            (
                self.envs,
                self.test_recurrent_hidden_states,
                self.not_done_masks,
                self.current_episode_reward,
                self.prev_actions,
                self.batch,
                self.rgb_frames,
            ) = pause_envs(
                self.envs_to_pause,
                self.envs,
                self.test_recurrent_hidden_states,
                self.not_done_masks,
                self.current_episode_reward,
                self.prev_actions,
                self.batch,
                self.rgb_frames,
            )
            # We pause the statefull parameters in the policy.
            # We only do this if there are envs to pause to reduce the overhead.
            # In addition, HRL policy requires the solution_actions to be non-empty, and
            # empty list of envs_to_pause will raise an error.
            if any(self.envs_to_pause):
                self.agent.actor_critic.on_envs_pause(self.envs_to_pause)
    
    def display_results(self, per_episode=False):

        # This uses Wandb logger but just prints the results on screen
        # Check tensorboard_utils.py for more details
        if per_episode:
            self.stat_episodes = log_episode_stats(
                self.task_name,
                self.stats_episodes,
                self.eqa_vars,
                self.config,
                logger,
            )
            return
            
        self.pbar.close()
        assert (
            len(self.ep_eval_count) >= self.number_of_eval_episodes
        ), f"Expected {self.number_of_eval_episodes} episodes, got {len(self.ep_eval_count)}."

        self.aggregated_stats = {}

        self.all_ks = set()
        for ep in self.stats_episodes.values():
            self.all_ks.update(ep.keys())

        # Log results to Wandb, check tensorboard_utils.py for more details
        self.aggregated_stats, self.metrics = log_final_results(
            self.task_name, 
            self.stats_episodes, 
            self.aggregated_stats, 
            self.all_ks, 
            self.step_id, 
            self.writer, 
            self.config,
            logger
        )

    def execute_action(self, coords=None, action=None):
        # TODO: instead of variables make name of action
        if coords is not None:
            self.predict_action(coords)
            self.execute_step(action=action)
        else:
            self.execute_step(action=action)

    def episode_iterator(self):
        if (len(self.stats_episodes) < (self.number_of_eval_episodes * self.evals_per_ep)
            and self.envs.num_envs > 0
            ):
            return True

        else:
            return False

    def get_habitat_sim(self, env=0):
        """
        Call habitat simulator for the current environment
        """
        return self.envs.call(self.env_call_list)[env].sim

    def get_current_episode_info(self, env=0):
        """
        Get the current episode information
        """
        return self.envs.call(self.env_call_list)[env].current_episode

    def get_current_position(self):
        """
        Get current agent state position & rotation
        """
        return self.get_habitat_sim().get_agent_state()

    def get_current_observation(self, type='rgb'):
        """
        Get the current observation from the environment
        """
        if type in ['rgb']:
            return self.batch['rgb'].squeeze(0).detach().cpu().numpy()
        elif type in ['depth']:
            return self.batch['depth'].squeeze(0).detach().cpu().numpy()
        elif type in ['semantic']:
            return self.batch['semantic'].squeeze(0).detach().cpu().numpy()
        elif type in ['instance_imagegoal']:
            return self.batch['instance_imagegoal'].squeeze(0).detach().cpu().numpy()
        elif type in ['gps']:
            return self.batch['gps'].squeeze(0).detach().cpu().numpy()
        elif type in ['compass']:
            return self.batch['compass'].squeeze(0).detach().cpu().item()
        elif type in ['heading']:
            return self.batch['heading'].squeeze(0).detach().cpu().item()
        else:
            raise ValueError(f"Invalid observation type: {type}")

    def get_current_step(self):
        """
        Get the current step, max is queal to max number of steps
        """
        return self.current_step

    def sample_distant_points(self):
        """
        Sample distant unreachable points and try to navigate to them.
        We try different methods to achieve this.
        Possibility to extend to new and better sampling strategies.
        """
        def is_valid_goal(distance):
            return distance >= min_distance
        
        def adjust_goal_point(goal_point):
            return [p1 + p2 for p1, p2 in zip(goal_point, further_point)]

        # Initial setup
        hab_simulator = self.get_habitat_sim()
        min_distance, max_tries = 10.0, 1000
        current_pos = self.get_current_position()
        agent_pos, agent_ang = current_pos.position, current_pos.rotation
        further_point = [min_distance, 0, min_distance]

        # Raise error early for invalid strategy
        assert self.sampling_strategy in ['unreachable', 'navigable'], f"Invalid sampling strategy: {self.sampling_strategy}"

        # Try to find a valid goal point
        for _ in range(max_tries):
            goal_point = hab_simulator.sample_navigable_point()
            
            if self.sampling_strategy == 'unreachable':
                    goal_point = adjust_goal_point(goal_point)
                    return from_xyz_to_polar(agent_pos, agent_ang, goal_point)

            elif self.sampling_strategy == 'navigable':
                distance = hab_simulator.geodesic_distance(agent_pos, goal_point)            
                if is_valid_goal(distance) and not math.isinf(distance):
                    return from_xyz_to_polar(agent_pos, agent_ang, goal_point)
            
        # If no navigable point is found, return the last navigable point
        print('No Navigable point found, returning last navigable point.')
        return from_xyz_to_polar(agent_pos, agent_ang, goal_point)
            
    def max_steps_reached(self):
        """
        Check if the max steps are reached
        """
        return self.current_step >= self.config.habitat.environment.max_episode_steps - 1

    def get_stereo_view(self, degrees=180):
        """
        Get a view of the current observation based on the specified degrees.
        It captures the observation by turning left and right equally.
        It also saves the jpeg image for debugging purposes.
        """
        if degrees % 2 != 0:
            # approximate to the closest even number
            degrees = degrees - 1

        half_degrees = degrees // 2
        turn_angle = self.config.habitat.simulator.turn_angle
        num_left_turns = half_degrees // turn_angle
        num_right_turns = half_degrees // turn_angle

        views_depth, views_rgb, states = [], [], []

        # Turn left and capture the observation
        for _ in range(num_left_turns):
            self.execute_action(action='turn_left')
            views_depth.append(self.get_current_observation(type='depth'))
            views_rgb.append(self.get_current_observation(type='rgb'))
            states.append(self.get_current_position())

        # Reverse the states list for the left turn observations
        states.reverse()

        # Turn right to go back to the original position
        for _ in range(num_left_turns):
            self.execute_action(action='turn_right')

        # Turn right and capture the observation
        for _ in range(num_right_turns):
            self.execute_action(action='turn_right')
            views_depth.append(self.get_current_observation(type='depth'))
            views_rgb.append(self.get_current_observation(type='rgb'))
            states.append(self.get_current_position())

        # Turn left to go back to the original position
        for _ in range(num_right_turns):
            self.execute_action(action='turn_left')
        self.update_episode_stats()

        stacked_views_rgb = np.hstack(views_rgb)
        stacked_views_depth = np.hstack(views_depth)
        try: stacked_views = match_images(stacked_views_rgb)
        except: stacked_views = stacked_views_rgb[0]

        if self.save_obs:
            self.debugger.save_obs(stacked_views_rgb, prefix=f'stereo_single')
            self.debugger.save_obs(stacked_views, prefix=f'stereo_match')

        views = {
            'rgb': np.array(stacked_views_rgb),
            'depth': np.array(stacked_views_depth),
            'stacked': stacked_views,
            'states': np.array(states)
        }
        return views

    def check_scene_change(self):
        """
        Check if the scene has changed
        """
        current_scene = self.get_current_episode_info().scene_id
        if self.current_scene != current_scene:
            self.current_scene = current_scene
            return True
        return False

    def evaluate_agent(
        self,
        **kwargs,
    ):
        """
        Evalauate the agent in the habitat environment given a certain predefined task
        we iterate through each episode and execute the pseudo code
        """
        self.get_env_variables(**kwargs)
        self.init_env()

        code_generator = CodeGenerator(self, debug=True)
        self.code_interpreter = PseudoCodeExecuter(self)

        while self.episode_iterator():
            self.current_episodes_info = self.envs.current_episodes()

            # Generate the PseudoCode
            self.pseudo_code = code_generator.generate()
                
            # Reset init variables
            self.code_interpreter.parse(self.pseudo_code)

            # Run the code
            if DEBUG:
                self.code_interpreter.run()
            else:
                try:
                    self.code_interpreter.run()
                except:
                    # Handle issues and keep the valuation running
                    self.code_interpreter.handle_errors()

        self.display_results()

    