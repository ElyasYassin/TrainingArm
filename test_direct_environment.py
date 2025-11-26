import gym
import numpy as np
import myosuite
from CenterReachOut_v1 import ReachEnvV1
import os

def test_direct_environment():
    """Test the environment by directly instantiating it"""
    
    # Find the model path from myosuite
    import myosuite
    model_path = os.path.join(myosuite.__path__[0], 'envs', 'myo', 'assets', 'hand', 'myohand_pose.xml')
    
    print(f"Using model path: {model_path}")
    
    # Create environment directly
    env = ReachEnvV1(model_path=model_path)
    
    print("Testing environment directly...")
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
    for step in range(50):  # Test for 50 steps
        action = np.random.randn(env.action_space.shape[0]) * 0.1  # Small random actions
        obs, reward, done, info = env.step(action)
        total_reward += reward
        
        if step % 10 == 0:  # Print every 10 steps
            print(f"Step {step}: Reward = {reward:.3f}, Done = {done}")
            print(f"  Palm pos: {env.obs_dict['palm_pos']}")
            print(f"  Object pos: {env.obs_dict['obj_pos']}")
            print(f"  Reach error: {np.linalg.norm(env.obs_dict['reach_err']):.3f}")
        
        if done:
            print(f"Episode ended after {step+1} steps")
            break
    
    print(f"Total reward: {total_reward:.3f}")
    print("Direct environment test completed successfully!")

if __name__ == "__main__":
    test_direct_environment() 