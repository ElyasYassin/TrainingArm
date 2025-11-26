import gym
import numpy as np
import myosuite
from CenterReachOut_v0 import ReachEnvV0

def test_environment():
    """Test the environment to ensure it's working correctly after fixes"""
    
    # Create environment
    env = gym.make('CenterReachOut-v0')
    
    print("Testing environment...")
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
    for step in range(20):  # Test for 20 steps
        action = np.random.randn(env.action_space.shape[0]) * 0.1  # Small random actions
        obs, reward, done, info = env.step(action)
        total_reward += reward
        
        print(f"Step {step}: Reward = {reward:.3f}, Done = {done}")
        print(f"  Palm pos: {env.obs_dict['palm_pos']}")
        print(f"  Object pos: {env.obs_dict['obj_pos']}")
        print(f"  Reach error: {np.linalg.norm(env.obs_dict['reach_err']):.3f}")
        
        if done:
            print(f"Episode ended after {step+1} steps")
            break
    
    print(f"Total reward: {total_reward:.3f}")
    print("Environment test completed successfully!")

if __name__ == "__main__":
    test_environment() 