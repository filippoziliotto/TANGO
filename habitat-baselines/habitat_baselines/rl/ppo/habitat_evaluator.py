import os
from collections import defaultdict
from typing import Any, Dict, List

import numpy as np
import torch
import tqdm
import wandb

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

class HabitatEvaluator(Evaluator):
    def __init__(self):
        self.current_step = 0
        pass

    def _init_variables(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        
        # Task
        self.task_name = self.config.habitat_baselines.task_name
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

    def init_env(self):
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

    def predict_action(self, coords):
        """
        Given the target pointgoal distance and angle,
        predict the next action to take
        """

        self.space_lengths = {}
        self.batch['pointgoal_with_gps_compass'] = coords.to(self.device)

        with inference_mode():
            self.action_data = self.agent.actor_critic.act(
                self.batch,
                self.test_recurrent_hidden_states,
                self.prev_actions,
                self.not_done_masks,
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

    def execute_step(self, force_stop=False, look_around=False):
        """
        Execute the predicted action in the environment
        and update all the necessary variables
        """

        if force_stop:
            self.step_data = [0] 
        elif not force_stop and self.action_data.actions.item() == 0:
            self.step_data = [a.item() for a in self.action_data.env_actions.cpu()]
            self.prev_actions.copy_(self.action_data.actions) # type: ignore
            self.step_data = [torch.randint(1, 4, (1,), device=self.device).item()]
        else:
            pass

        if look_around:
            self.step_data = [2] 
            
        self.outputs = self.envs.step(self.step_data)
        self.observations, self.rewards_l, self.dones, self.infos = [
            list(x) for x in zip(*self.outputs)
        ]
        # Note that `policy_infos` represents the information about the
        # action BEFORE `observations` (the action used to transition to
        # `observations`).
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

    def update_episode_stats(self, force_stop=False, display=False):
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
            
            self.current_step += 1

            # episode ended
            if not self.not_done_masks[i].any().item() or force_stop:
                self.pbar.update()
                self.episode_stats = {
                    "reward": self.current_episode_reward[i].item()
                }
                self.episode_stats.update(extract_scalars_from_info(self.infos[i]))
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
        if per_episode:
            last_key = list(self.stats_episodes.keys())[-1]
            v = self.stats_episodes[last_key]
            episode_info = f"Episode {len(self.stats_episodes)}, {last_key}:"
            formatted_results = (
                f"num_steps: {v['num_steps']} | "
                f"distante_to_goal: {v['distance_to_goal']:.2f} | "
                f"success: {v['success']:.2f} | "
                f"spl: {v['spl']:.2f} | "
                f"soft_spl: {v['soft_spl']:.2f}"
            )
            print(f"{episode_info}\n{formatted_results}")
            print('-----------------------')
            return
            
        self.pbar.close()
        assert (
            len(self.ep_eval_count) >= self.number_of_eval_episodes
        ), f"Expected {self.number_of_eval_episodes} episodes, got {len(self.ep_eval_count)}."

        self.aggregated_stats = {}

        self.all_ks = set()
        for ep in self.stats_episodes.values():
            self.all_ks.update(ep.keys())

        for stat_key in self.all_ks:
            self.aggregated_stats[stat_key] = np.mean(
                [v[stat_key] for v in self.stats_episodes.values() if stat_key in v]
            )

        self.metrics = {k: v for k, v in self.aggregated_stats.items() if k != "reward"}
        for k, v in self.metrics.items():
            self.writer.add_scalar(f"eval_metrics/{k}", v, self.step_id)        
            
        self.writer.add_scalar(
                "eval_reward/average_reward", self.aggregated_stats["reward"], self.step_id
            )

        # logging to wandb
        if self.config.habitat_baselines.writer_type in ['wb']:
            wandb.log(self.aggregated_stats)

        # Print final results
        print('-----------------------')
        print('| EVALUATION FINISHED |')
        print('-----------------------')

        for k, v in self.aggregated_stats.items():
            print(f"Average episode {k}: {v:.4f}")
        print('-----------------------')       

    def execute_action(self, coords=None, force_stop=False, look_around=False):
        if coords is not None:
            self.predict_action(coords)
            self.execute_step(force_stop)
        elif look_around:
            self.execute_step(look_around)
        else:
            self.execute_step(force_stop)

    def episode_iterator(self):
        if (len(self.stats_episodes) < (self.number_of_eval_episodes * self.evals_per_ep)
            and self.envs.num_envs > 0
            ):
            return True

        else:
            return False

    def call_habitat_sim(self):
        """
        Call habitat simulator for the current environment
        """
        return self.envs.call(['habitat_env'])[0].sim

    def get_current_position(self):
        """
        Get current agent state position & rotation
        """
        return self.call_habitat_sim().get_agent_state()

    def get_current_observation(self, type='rgb'):
        """
        Get the current observation from the environment
        """
        assert type in ['rgb', 'depth', 'semantic','instance_imagegoal']
        if type in ['rgb']:
            return self.batch['rgb'].squeeze(0).detach().cpu().numpy()
        elif type in ['depth']:
            return self.batch['depth'].squeeze(0).detach().cpu().numpy()
        elif type in ['semantic']:
            return self.batch['semantic'].squeeze(0).detach().cpu().numpy()
        elif type in ['instance_imagegoal']:
            return self.batch['instance_imagegoal'].squeeze(0).detach().cpu().numpy()

    def get_current_step(self):
        """
        Get the current step, max is queal to max number of steps
        """
        return self.current_step

    def sample_distant_points(self, strategy='navigable'):
        """
        sample distant unreachable points and try to navigate to them
        we try different method to achieve this
        possibility to extend to new and better sampling strategies
        """
        hab_simulator = self.call_habitat_sim()
        min_distance = 20.
        max_tries = 1000
        current_pos = self.get_current_position()
        agent_pos = current_pos.position
        agent_ang = current_pos.rotation
        further_point = [min_distance, 0, min_distance]

        for _ in range(max_tries):
            goal_point = hab_simulator.sample_navigable_point()
            distance = hab_simulator.geodesic_distance(agent_pos, goal_point)

            if strategy in ['unreachable']:
                if distance >= min_distance:
                    goal_point = [p1 + p2 for p1, p2 in zip(goal_point, further_point)]
                    break

            elif strategy in ['navigable']:
                if distance >= min_distance:
                    break
            
            else:
                raise ValueError(f"Invalid sampling strategy")
            
        return from_xyz_to_polar(agent_pos, agent_ang, goal_point)
            
    def max_steps_reached(self):
        """
        Check if the max steps are reached
        """
        return self.current_step >= self.config.habitat.environment.max_episode_steps - 1

    def evaluate_agent(
        self,
        **kwargs,
    ):
        """
        Evalauate the agent in the habitat environment given a certain predefined task
        we iterate through each episode and execute the pseudo code
        """
        self._init_variables(**kwargs)
        self.init_env()

        code_generator = CodeGenerator(self, task=self.task_name, debug=True)
        self.code_interpreter = PseudoCodeExecuter(self)

        while self.episode_iterator():
            self.current_episodes_info = self.envs.current_episodes()

            # Generate the PseudoCode
            self.pseudo_code = code_generator.generate()

            # Reset init variables
            self.code_interpreter.parse(self.pseudo_code)

            # Run the code
            self.code_interpreter.run()

        self.display_results()

    