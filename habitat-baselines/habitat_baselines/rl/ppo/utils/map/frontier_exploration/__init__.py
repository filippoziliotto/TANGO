"""The following imports are necessary for updating the registry"""
try:
    import habitat_baselines.rl.ppo.utils.map.frontier_exploration.base_explorer
    import habitat_baselines.rl.ppo.utils.map.frontier_exploration.frontier_detection
    import habitat_baselines.rl.ppo.utils.map.frontier_exploration.frontier_sensor
    import habitat_baselines.rl.ppo.utils.map.frontier_exploration.measurements
    import habitat_baselines.rl.ppo.utils.map.frontier_exploration.objnav_explorer
    import habitat_baselines.rl.ppo.utils.map.frontier_exploration.policy
    import habitat_baselines.rl.ppo.utils.map.frontier_exploration.trainer
    import habitat_baselines.rl.ppo.utils.map.frontier_exploration.utils.inflection_sensor
    import habitat_baselines.rl.ppo.utils.map.frontier_exploration.utils.multistory_episode_finder
except ModuleNotFoundError as e:
    # If the error was due to the habitat package not being installed, then pass, but
    # print a warning. Do not pass if it was due to another package being missing.
    if "habitat" not in e.name:
        raise e
    else:
        print(
            "Warning: importing habitat failed. Cannot register habitat_baselines "
            "components."
        )
