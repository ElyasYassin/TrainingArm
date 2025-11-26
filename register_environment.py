import gym
from gym.envs.registration import register
from CenterReachOut_v0 import ReachEnvV0

# Register the environment
register(
    id='CenterReachOut-v0',
    entry_point='CenterReachOut_v0:ReachEnvV0',
    max_episode_steps=1000,
)

print("Environment registered successfully!")
print("You can now use gym.make('CenterReachOut-v0')") 