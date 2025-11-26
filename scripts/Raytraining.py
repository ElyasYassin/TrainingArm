from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPOConfig
import ray


def make_env(env_config=None):
    # 1) Create the MyoSuite env via Gymnasium
    from shimmy import GymV21CompatibilityV0
    import myosuite
    import gym
    env = gym.make("CenterReachOut-v0")
    # 2) Wrap it to add the new reset/step signature
    return  GymV21CompatibilityV0(env)

if __name__ == "__main__":
    register_env("center_reach", make_env)
    ray.init(ignore_reinit_error=True)

    config = (
        PPOConfig()
        .environment(env="center_reach", disable_env_checking=True)
        .framework("torch")
        .rollouts(num_rollout_workers=4)
        .resources(num_gpus=1)
        .training(model={"fcnet_hiddens": [512, 512]})
    )
    algo = config.build()
    results = algo.train()
    print(f"Reward after first iteration: {results['episode_reward_mean']}")