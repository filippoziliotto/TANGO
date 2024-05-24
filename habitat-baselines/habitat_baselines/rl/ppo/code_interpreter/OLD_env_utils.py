# Base imports
import torch
import numpy as np
import random

# Step interpreters imports
from habitat_baselines.utils.common import (
    inference_mode,
    batch_obs,
    generate_video,
)
from habitat_baselines.common.obs_transformers import (
    apply_obs_transforms_batch,
)
from habitat_baselines.rl.ppo.evaluator import pause_envs
from habitat_baselines.utils.info_dict import extract_scalars_from_info
from habitat.tasks.rearrange.rearrange_sensors import GfxReplayMeasure
from habitat.tasks.rearrange.utils import write_gfx_replay
from habitat.utils.visualizations.utils import (
    observations_to_image,
    overlay_frame,
)

# Utils for Exploration and Detection
from habitat.tasks.utils import cartesian_to_polar
from habitat.utils.geometry_utils import (
    quaternion_rotate_vector,
)

translate = { 0: 'stop',
              1: 'move forward',
              2: 'turn left',
              3: 'turn right'
            }

stoi_eqa = {
'<unk>':0,
'black':11142,
'brown':11496,
'off-white':11367,
'white':11495,
'blue':11494,
'tan':11413,
'grey':10542,
'slate grey': 9584,
'silver':11046,
'green':11352,
'yellow green':3479,
'red brown':7712,
'yellow pink':1374,
'orange yellow':1234,
'bathroom':11462,
'kitchen':11493,
'lounge':11421,
'spa':2047,
'bedroom':11097,
'living room':10734,
'family room':10826,
'light blue':5947,
'tv room':3004,
'closet':11342,
'laundry room':11199,
'olive green':11115,
'foyer':4739,
'hallway':10388,
'dining room':10519,
'purple pink':7510,
'red':10915,
'purple':10636,
'yellow':10223,
'office':11243,
}

def get_current_position(env_vars):
    sim_env = env_vars['envs'].call(['habitat_env'])[0]
    current_pos = sim_env.sim.get_agent_state()
    return current_pos

def from_xyz_to_pointgoal(source_position, source_rotation, goal_position):
    """
    from xyz to agent pointgoal coordinates
    """
    direction_vector = goal_position - source_position
    direction_vector_agent = quaternion_rotate_vector(
        source_rotation.inverse(), direction_vector
    )
    
    rho, phi = cartesian_to_polar(
        -direction_vector_agent[2], direction_vector_agent[0]
    )
    return np.array([rho, -phi], dtype=np.float32)

def from_pointgoal_to_xyz(source_position, source_rotation, rho, phi):
    """
    from agent pointgoal coordinates to xyz
    """
    z = -rho * np.cos(phi)
    x = rho * np.sin(phi)

    direction_vector_agent = np.array([x, source_position[1], z], dtype=np.float32)
    direction_vector = quaternion_rotate_vector(source_rotation, direction_vector_agent)
    goal_position = source_position + direction_vector

    return torch.from_numpy(goal_position)


def sample_pointgoal(env_vars):
    sim_env = env_vars['envs'].call(['habitat_env'])[0]
    # sample navigable point and navigate to it
    while True:
        goal_point = sim_env.sim.sample_navigable_point()
        current_pos = sim_env.sim.get_agent_state()
        explore_dist, explore_angle = from_xyz_to_pointgoal(current_pos.position,\
                                current_pos.rotation, goal_point)
        if explore_dist >= 2.:
            break    
    return goal_point, explore_dist, explore_angle

def compass_from_position(env_vars):
    # Get current agent state
    current_pos = get_current_position(env_vars)

    # Sample random distant points
    x = random.choice([-15.,15.,10.,-10.])
    z = random.choice([-15.,15.,10.,-10.])
    # x = np.random.randint(-20.,20.)
    # z = np.random.randint(-20.,20.)
    goal_point = [current_pos.position[0]+x,
                  current_pos.position[1],
                  current_pos.position[1]+z] 

    if env_vars['step_count'] % 100 == 0:     
        env_vars['explore']['goalpoint'] = goal_point       
    else:
        goal_point = env_vars['explore']['goalpoint']

    # Calculate exploration distance and angle from current position to goal point
    explore_dist, explore_angle = from_xyz_to_pointgoal(
                                    current_pos.position,
                                    current_pos.rotation,
                                    goal_point)    
    # Update exploration point goal in environment variables
    env_vars['explore']['pointgoal'] = [explore_dist, explore_angle] 
    return explore_dist, explore_angle

def update_target_position(env_vars):
    # Get agent position/rotation
    agent_pos = get_current_position(env_vars)
    agent_ang = agent_pos.rotation
    agent_pos = agent_pos.position

    # Get target position/rotation
    target_xyz = env_vars['target']['objs']['xyz']
    polar_coord = from_xyz_to_pointgoal(agent_pos, agent_ang, target_xyz)

    env_vars['target']['objs']['dist'] = polar_coord[0]
    env_vars['target']['objs']['ang'] = polar_coord[1]
    return env_vars

def eqa_text_to_token(env_vars, stoi):
    # Convert text to tokens
    answer_label = stoi.get(env_vars['eqa']['answer'],0)
    env_vars['eqa']['answer_token'] = answer_label
    return answer_label

def predict_action(env_vars, explore_dist, explore_angle):

    env_vars['current_episodes_info'] = env_vars['envs'].current_episodes()
    env_vars['space_lengths'] = {}
        
    # Check correct dimension for forward pass resnet policy
    if isinstance(explore_dist,list) and isinstance(explore_angle,list):
        explore_dist = explore_dist[0]
        explore_angle = explore_angle[0]

    env_vars['batch']['pointgoal_with_gps_compass'] = torch.tensor([[explore_dist, explore_angle]], dtype=torch.float).cuda()
    with inference_mode():
        try: 
            env_vars['action_data'] =  env_vars['agent'].actor_critic.act(
            env_vars['batch'],
            env_vars['test_recurrent_hidden_states'],
            env_vars['prev_actions'],
            env_vars['not_done_masks'],
            deterministic=True,
            **env_vars['space_lengths'])

            if env_vars['action_data'].should_inserts is None:
                env_vars['test_recurrent_hidden_states'] = (
                        env_vars['action_data'].rnn_hidden_states
                        )
                env_vars['prev_actions'].copy_(env_vars['action_data'].actions) # type: ignore
            else:
                env_vars['agent'].actor_critic.update_hidden_state(
                    env_vars['test_recurrent_hidden_states'],\
                    env_vars['prev_actions'],\
                    env_vars['action_data'])
        except:
            pass
                
    return env_vars

def execute_step(env_vars, force_stop=False):

    if force_stop:
        env_vars['step_data'] = [0] 
        # Needed for EQA task
        if env_vars['task_name'] in ['eqa']:
            # env_vars['eqa']['answer'] = 'family room'
            answer = eqa_text_to_token(env_vars, stoi_eqa)
            env_vars['step_data'] = [
                {
                    "action": 0,
                    "action_args": {"answer_id": answer,
                                    "answer_text": env_vars['eqa']['answer'],},
                }
            ]

    elif not force_stop and env_vars['action_data'].actions.item() == 0:
        # If stop action is called before the episode ends by detection, triggers an error so
        # random sample of action from 1 to 3
        env_vars['action_data'].actions = torch.tensor([torch.randint(1, 4, (1,), device=env_vars['device']).item()])
        env_vars['prev_actions'].copy_(env_vars['action_data'].actions) # type: ignore
        env_vars['step_data'] = [torch.randint(1, 4, (1,), device=env_vars['device']).item()]
    else:
        env_vars['step_data'] = [a.item() for a in env_vars['action_data'].env_actions.cpu()]

    env_vars['outputs'] = env_vars['envs'].step(env_vars['step_data'])
    env_vars['observations'], env_vars['rewards_l'], env_vars['dones'], env_vars['infos'] = [
        list(x) for x in zip(*env_vars['outputs'])
    ]
    # Note that `policy_infos` represents the information about the
    # action BEFORE `observations` (the action used to transition to
    # `observations`).
    env_vars['policy_infos'] = env_vars['agent'].actor_critic.get_extra(
            env_vars['action_data'], env_vars['infos'],
            env_vars['dones']
    )
        
    for i in range(len(env_vars['policy_infos'])):
        env_vars['infos'][i].update(env_vars['policy_infos'][i])

    env_vars['observations'] = env_vars['envs'].post_step(env_vars['observations'])
    env_vars['batch'] = batch_obs(  # type: ignore
        env_vars['observations'],
        device=env_vars['device'],
    )
    env_vars['batch'] = apply_obs_transforms_batch(
        env_vars['batch'], env_vars['obs_transforms']
    )  # type: ignore

    env_vars['not_done_masks'] = torch.tensor(
        [[not done] for done in env_vars['dones']],
        dtype=torch.bool,
        device="cpu",
    ).repeat(1, *env_vars['agent'].masks_shape)

    env_vars['rewards'] = torch.tensor(
        env_vars['rewards_l'], dtype=torch.float, device="cpu"
    ).unsqueeze(1)
    env_vars['current_episode_reward'] += env_vars['rewards']

    env_vars['step_count'] += 1
    return env_vars

def is_episode_over(env_vars, force_stop=False):

    env_vars['next_episodes_info'] = env_vars['envs'].current_episodes()
    env_vars['envs_to_pause'] = []
    env_vars['n_envs'] = env_vars['envs'].num_envs
    for i in range(env_vars['n_envs']):
        if (
            env_vars['ep_eval_count'][
                (
                    env_vars['next_episodes_info'][i].scene_id,
                    env_vars['next_episodes_info'][i].episode_id,
                )
            ]
            == env_vars['evals_per_ep']
        ):
            env_vars['envs_to_pause'].append(i)

        # Exclude the keys from `_rank0_keys` from displaying in the video
        env_vars['disp_info'] = {
            k: v for k, v in env_vars['infos'][i].items() if k not in env_vars['rank0_keys']
        }
        # Adding the other variables to the display info
        # env_vars['disp_info']['LLM Suggestion'] = env_vars['LLM']['suggestion']
        env_vars['disp_info']['tau'] = env_vars['superglue']['conf']

        if len(env_vars['config'].habitat_baselines.eval.video_option) > 0:                
            # TODO move normalization / channel changing out of the policy and undo it here
            frame = observations_to_image(
                {k: v[i] for k, v in env_vars['batch'].items()}, env_vars['disp_info']
            )
            if not env_vars['not_done_masks'][i].any().item():
                # The last frame corresponds to the first frame of the next episode
                # but the info is correct. So we use a black frame
                final_frame = observations_to_image(
                    {k: v[i] * 0.0 for k, v in env_vars['batch'].items()},
                    env_vars['disp_info'],
                )
                final_frame= overlay_frame(final_frame, env_vars['disp_info'])
                env_vars['rgb_frames'][i].append(final_frame)
                # The starting frame of the next episode will be the final element..
                env_vars['rgb_frames'][i].append(frame)
            else:
                frame = overlay_frame(frame, env_vars['disp_info'])
                env_vars['rgb_frames'][i].append(frame)

        # episode ended
        if not env_vars['not_done_masks'][i].any().item() or force_stop:
            env_vars['pbar'].update()
                
            env_vars['episode_stats'] = {
                "reward": env_vars['current_episode_reward'][i].item()
            }

            # Need for EQA task if episode_infos in keys
            tmp_episode_info= {key: value for key, value in env_vars['infos'][i].items() if "episode_info" not in key}
            # TODO: drop episode_info key in dict for EQA
            env_vars['episode_stats'].update(extract_scalars_from_info(tmp_episode_info))
            env_vars['current_episode_reward'][i] = 0
            env_vars['k'] = (
                env_vars['current_episodes_info'][i].scene_id,
                env_vars['current_episodes_info'][i].episode_id,
            )
            env_vars['ep_eval_count'][env_vars['k']] += 1
            # use scene_id + episode_id as unique id for storing stats
            env_vars['stats_episodes'][(env_vars['k'], env_vars['ep_eval_count'][env_vars['k']])] = env_vars['episode_stats']

            if len(env_vars['config'].habitat_baselines.eval.video_option) > 0:
                generate_video(
                    video_option=env_vars['config'].habitat_baselines.eval.video_option,
                    video_dir=env_vars['config'].habitat_baselines.video_dir,
                    # Since the final frame is the start frame of the next episode.
                    images=env_vars['rgb_frames'][i][:-1],
                    episode_id=f"{env_vars['current_episodes_info'][i].episode_id}_{env_vars['ep_eval_count'][env_vars['k']]}",
                    checkpoint_idx=env_vars['checkpoint_index'],
                    metrics=extract_scalars_from_info(tmp_episode_info),
                    # metrics=extract_scalars_from_info(env_vars['disp_info']),
                    fps=env_vars['config'].habitat_baselines.video_fps,
                    tb_writer=env_vars['writer'],
                    keys_to_include_in_name=env_vars['config'].habitat_baselines.eval_keys_to_include_in_name,
                )

                # Since the starting frame of the next episode is the final frame.
                env_vars['rgb_frames'][i] = env_vars['rgb_frames'][i][-1:]

            env_vars['gfx_str'] = env_vars['infos'][i].get(GfxReplayMeasure.cls_uuid, "")
            if env_vars['gfx_str'] != "":
                write_gfx_replay(
                    env_vars['gfx_str'],
                    env_vars['config'].habitat.task,
                    env_vars['current_episodes_info'][i].episode_id,
                )

        env_vars['not_done_masks'] = env_vars['not_done_masks'].to(device=env_vars['device'])
        (
            env_vars['envs'],
            env_vars['test_recurrent_hidden_states'],
            env_vars['not_done_masks'],
            env_vars['current_episode_reward'],
            env_vars['prev_actions'],
            env_vars['batch'],
            env_vars['rgb_frames'],
        ) = pause_envs(
            env_vars['envs_to_pause'],
            env_vars['envs'],
            env_vars['test_recurrent_hidden_states'],
            env_vars['not_done_masks'],
            env_vars['current_episode_reward'],
            env_vars['prev_actions'],
            env_vars['batch'],
            env_vars['rgb_frames'],
        )

    return env_vars