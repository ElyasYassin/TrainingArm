import numpy as np
import myosuite
import gym

def test_no_initial_movement():
    """Test that the environment starts without any movement"""
    
    # Create environment
    env = gym.make('CenterReachOut-v0')
    
    # Get the unwrapped environment if it's wrapped
    unwrapped_env = env.unwrapped if hasattr(env, 'unwrapped') else env
    
    print("Testing for initial movement...")
    
    for episode in range(5):
        print(f"\nEpisode {episode + 1}:")
        
        # Reset environment
        obs = env.reset()
        
        # Get initial positions and velocities
        initial_palm_pos = env.obs_dict['palm_pos'].copy()
        initial_hand_vel = env.obs_dict['hand_qvel'].copy()
        initial_time = float(env.obs_dict['time'])
        
        print(f"  Initial time: {initial_time:.3f}")
        print(f"  Initial palm position: {initial_palm_pos}")
        print(f"  Initial hand velocity magnitude: {np.linalg.norm(initial_hand_vel):.6f}")
        
        # Take a few steps with zero action
        zero_action = np.zeros(env.action_space.shape)
        
        # Wait for preparation phase to complete
        while hasattr(unwrapped_env, 'in_prep_phase') and unwrapped_env.in_prep_phase:
            obs, reward, done, info = env.step(zero_action)
            print(f"    Prep phase: time={float(env.obs_dict['time']):.3f}")
        
        print(f"  Preparation phase completed, starting main episode")
        
        # Get the settled start position
        if hasattr(unwrapped_env, 'settled_start_pos'):
            settled_pos = unwrapped_env.settled_start_pos
            print(f"  Settled start position: {settled_pos}")
            print(f"  Original start position: {unwrapped_env.start_pos}")
            print(f"  Position change during settling: {np.linalg.norm(settled_pos - unwrapped_env.start_pos):.6f}")
        
        for step in range(10):
            obs, reward, done, info = env.step(zero_action)
            
            current_palm_pos = env.obs_dict['palm_pos']
            current_hand_vel = env.obs_dict['hand_qvel']
            current_time = float(env.obs_dict['time'])
            
            # Check for movement
            pos_change = np.linalg.norm(current_palm_pos - initial_palm_pos)
            vel_magnitude = np.linalg.norm(current_hand_vel)
            
            print(f"    Step {step}: time={current_time:.3f}, pos_change={pos_change:.6f}, vel={vel_magnitude:.6f}")
            
            if pos_change > 1e-6 or vel_magnitude > 1e-6:
                print(f"    WARNING: Movement detected at step {step}!")
                print(f"      Position change: {pos_change:.8f}")
                print(f"      Velocity magnitude: {vel_magnitude:.8f}")
            
            if done:
                break
        
        print(f"  Episode completed: {done}")

if __name__ == "__main__":
    test_no_initial_movement() 