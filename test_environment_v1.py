import gym
import numpy as np
import myosuite
from CenterReachOut_v1 import ReachEnvV1
from gym.envs.registration import register

# Register the new environment
register(
    id='CenterReachOut-v1',
    entry_point='CenterReachOut_v1:ReachEnvV1',
    max_episode_steps=1000,
)

def test_environment():
    """Test the new environment to ensure fixes work correctly"""
    
    # Create environment
    env = gym.make('CenterReachOut-v1')
    
    print("Testing new environment (v1)...")
    print(f"Episode time limit: {env.max_episode_time} seconds")
    print(f"Number of targets: {len(env.round_robin_targets)}")
    
    # Test reset and observation
    obs = env.reset()
    print(f"Observation shape: {obs.shape}")
    print(f"Initial palm position: {env.obs_dict['palm_pos']}")
    print(f"Initial object position: {env.obs_dict['obj_pos']}")
    print(f"Initial reach error: {env.obs_dict['reach_err']}")
    
    # Test a few random actions
    total_reward = 0
    for step in range(50):  # Test for 50 steps (longer due to increased episode time)
        action = np.random.randn(env.action_space.shape[0]) * 0.1  # Small random actions
        obs, reward, done, info = env.step(action)
        total_reward += reward
        
        if step % 10 == 0:  # Print every 10 steps to avoid too much output
            print(f"Step {step}: Reward = {reward:.3f}, Done = {done}")
            print(f"  Palm pos: {env.obs_dict['palm_pos']}")
            print(f"  Object pos: {env.obs_dict['obj_pos']}")
            print(f"  Reach error: {np.linalg.norm(env.obs_dict['reach_err']):.3f}")
        
        if done:
            print(f"Episode ended after {step+1} steps")
            break
    
    print(f"Total reward: {total_reward:.3f}")
    print("New environment test completed successfully!")

if __name__ == "__main__":
    test_environment() 