#!/usr/bin/env python3
"""
Example: Hand Trajectory Analysis with Standard Deviation

This is a simplified example that can be run in a Jupyter notebook or Python environment.
It demonstrates how to generate a figure showing the average hand path with standard deviation.
"""

import myosuite
import gym
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

def quick_trajectory_analysis(num_episodes=20, steps_per_episode=20):
    """
    Quick trajectory analysis function for demonstration
    
    Parameters:
    - num_episodes: Number of episodes to analyze (default: 20 for quick demo)
    - steps_per_episode: Maximum steps per episode (default: 20)
    """
    
    # Setup environment
    print("Setting up environment...")
    env = gym.make('CenterReachOut-v0')
    
    # Load policy (update path as needed)
    policy_path = "./policies/seed_42/best_policy.pickle"
    try:
        pi = pickle.load(open(policy_path, 'rb'))
        print(f"Loaded policy from {policy_path}")
    except FileNotFoundError:
        print(f"Policy file not found at {policy_path}")
        print("Please update the policy_path variable to point to your policy file")
        return None, None
    
    # Collect trajectory data
    print(f"Collecting data for {num_episodes} episodes...")
    all_trajectories = []
    all_velocities = []
    target_position = None
    condition_y = None
    
    # Get a consistent condition
    while True:
        env.reset()
        y_value = env.obs_dict['obj_pos'][1]
        if condition_y is None:
            condition_y = y_value
            target_position = np.copy(env.obs_dict['obj_pos'])
            break
    
    print(f"Target condition Y: {condition_y:.3f}")
    
    for ep in range(num_episodes):
        obs = env.reset()
        # Keep resetting until we get the same condition
        while not np.isclose(env.obs_dict['obj_pos'][1], condition_y, atol=1e-3):
            obs = env.reset()
        
        palm_traj = []
        velocities = []
        
        for step in range(steps_per_episode):
            # Get hand velocity
            hand_qvel = np.copy(env.obs_dict['hand_qvel'])
            vel_mag = np.linalg.norm(hand_qvel)
            velocities.append(vel_mag)
            
            # Get observation and action
            o = env.get_obs()
            action, _ = pi.get_action(o)
            
            # Store palm position
            palm_pos_3d = np.copy(env.obs_dict['palm_pos'])
            palm_traj.append(palm_pos_3d)
            
            # Step environment
            obs, reward, done, info = env.step(action)
            
            if done:
                break
        
        # Pad trajectory if episode ended early
        while len(palm_traj) < steps_per_episode:
            palm_traj.append(palm_traj[-1])
            velocities.append(0.0)
        
        all_trajectories.append(np.array(palm_traj))
        all_velocities.append(np.array(velocities))
        
        if (ep + 1) % 5 == 0:
            print(f"Completed {ep + 1}/{num_episodes} episodes")
    
    # Calculate statistics
    print("Calculating statistics...")
    trajectories_array = np.array(all_trajectories)
    velocities_array = np.array(all_velocities)
    
    avg_trajectory = np.mean(trajectories_array, axis=0)
    std_trajectory = np.std(trajectories_array, axis=0)
    avg_velocities = np.mean(velocities_array, axis=0)
    std_velocities = np.std(velocities_array, axis=0)
    
    # Create the main trajectory plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot individual trajectories (faint lines)
    for i, traj in enumerate(all_trajectories):
        if i < 5:  # Only plot first 5 for clarity
            ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], 
                   color='lightblue', alpha=0.3, linewidth=0.5)
    
    # Plot average trajectory
    ax.plot(avg_trajectory[:, 0], avg_trajectory[:, 1], avg_trajectory[:, 2],
            color='blue', linewidth=3, label=f'Average Trajectory (Y={condition_y:.3f})', marker='o', markersize=4)
    
    # Plot standard deviation as error bars at key points
    steps = avg_trajectory.shape[0]
    key_points = np.linspace(0, steps-1, 6, dtype=int)
    for point_idx in key_points:
        if point_idx < len(avg_trajectory):
            pos = avg_trajectory[point_idx]
            std = std_trajectory[point_idx]
            
            # Draw error bars for each dimension
            ax.plot([pos[0]-std[0], pos[0]+std[0]], [pos[1], pos[1]], [pos[2], pos[2]], 
                   color='red', alpha=0.5, linewidth=1)
            ax.plot([pos[0], pos[0]], [pos[1]-std[1], pos[1]+std[1]], [pos[2], pos[2]], 
                   color='red', alpha=0.5, linewidth=1)
            ax.plot([pos[0], pos[0]], [pos[1], pos[1]], [pos[2]-std[2], pos[2]+std[2]], 
                   color='red', alpha=0.5, linewidth=1)
    
    # Target position
    ax.scatter(target_position[0], target_position[1], target_position[2],
               color='red', marker='X', s=200, edgecolor='k', linewidth=2, label='Target', zorder=10)
    
    # Start position
    start_pos = avg_trajectory[0]
    ax.scatter(start_pos[0], start_pos[1], start_pos[2], 
               color='green', s=150, marker='o', edgecolor='k', linewidth=2, label='Start Position', zorder=10)
    
    # Formatting
    ax.set_xlabel("X Position (m)", fontsize=12)
    ax.set_ylabel("Y Position (m)", fontsize=12)
    ax.set_zlabel("Z Position (m)", fontsize=12)
    ax.set_title(f"Average Hand Path with Standard Deviation\n({num_episodes} episodes)", fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Print summary statistics
    print("\n=== Summary Statistics ===")
    print(f"Number of episodes: {len(all_trajectories)}")
    print(f"Target condition Y: {condition_y:.3f}")
    print(f"Target position: [{target_position[0]:.3f}, {target_position[1]:.3f}, {target_position[2]:.3f}]")
    print(f"Average final distance to target: {np.linalg.norm(avg_trajectory[-1] - target_position):.3f} m")
    print(f"Average peak velocity: {np.max(avg_velocities):.3f} m/s")
    print(f"Trajectory duration: {len(avg_trajectory)} steps")
    
    # Cleanup
    env.close()
    
    return fig, (avg_trajectory, std_trajectory, avg_velocities, std_velocities)

# Example usage
if __name__ == "__main__":
    # Run quick analysis with 20 episodes
    fig, stats = quick_trajectory_analysis(num_episodes=20, steps_per_episode=20)
    
    if fig is not None:
        # Save the figure
        os.makedirs('../figures', exist_ok=True)
        fig.savefig('../figures/example_trajectory_analysis.png', dpi=300, bbox_inches='tight')
        print("\nFigure saved to ../figures/example_trajectory_analysis.png")
        plt.show()
    else:
        print("Analysis failed. Please check your policy file path.") 