#!/usr/bin/env python3
"""
Hand Trajectory Analysis with Standard Deviation

This script generates figures showing the average hand path with standard deviation 
across multiple episodes for the MyoSuite CenterReachOut environment.
"""

import myosuite
import gym
import numpy as np
import os
import matplotlib.pyplot as plt
import pickle
from scipy.stats import ttest_ind

def collect_trajectory_data(env, pi, num_episodes=100, steps_per_episode=20, target_condition=None):
    """
    Collect trajectory data across multiple episodes
    
    Parameters:
    - env: gym environment
    - pi: policy
    - num_episodes: number of episodes to run
    - steps_per_episode: maximum steps per episode
    - target_condition: specific target condition to use (if None, will use first condition)
    
    Returns:
    - all_trajectories: list of trajectory arrays
    - all_velocities: list of velocity arrays
    - target_position: target position used
    - condition_y: y-coordinate of target condition
    """
    all_trajectories = []
    all_velocities = []
    target_position = None
    condition_y = target_condition
    
    # Get a consistent condition if not specified
    if condition_y is None:
        while True:
            env.reset()
            y_value = env.obs_dict['obj_pos'][1]
            if condition_y is None:
                condition_y = y_value
                target_position = np.copy(env.obs_dict['obj_pos'])
                break
    
    print(f"Collecting data for {num_episodes} episodes with target condition Y={condition_y:.3f}")
    
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
            palm_traj.append(palm_traj[-1])  # Repeat last position
            velocities.append(0.0)  # Zero velocity
        
        all_trajectories.append(np.array(palm_traj))
        all_velocities.append(np.array(velocities))
        
        if (ep + 1) % 10 == 0:
            print(f"Completed {ep + 1}/{num_episodes} episodes")
    
    return all_trajectories, all_velocities, target_position, condition_y

def calculate_trajectory_statistics(all_trajectories, all_velocities):
    """
    Calculate average trajectory and standard deviation
    
    Parameters:
    - all_trajectories: list of trajectory arrays
    - all_velocities: list of velocity arrays
    
    Returns:
    - avg_trajectory: average trajectory (steps, 3)
    - std_trajectory: standard deviation of trajectory (steps, 3)
    - avg_velocities: average velocities (steps,)
    - std_velocities: standard deviation of velocities (steps,)
    """
    trajectories_array = np.array(all_trajectories)  # shape: (episodes, steps, 3)
    velocities_array = np.array(all_velocities)      # shape: (episodes, steps)
    
    # Calculate mean and std for trajectories
    avg_trajectory = np.mean(trajectories_array, axis=0)  # shape: (steps, 3)
    std_trajectory = np.std(trajectories_array, axis=0)   # shape: (steps, 3)
    
    # Calculate mean and std for velocities
    avg_velocities = np.mean(velocities_array, axis=0)    # shape: (steps,)
    std_velocities = np.std(velocities_array, axis=0)     # shape: (steps,)
    
    return avg_trajectory, std_trajectory, avg_velocities, std_velocities

def plot_trajectory_with_std(avg_trajectory, std_trajectory, target_position, condition_y, 
                           all_trajectories, save_path=None, show_plot=True):
    """
    Plot average trajectory with standard deviation
    
    Parameters:
    - avg_trajectory: average trajectory (steps, 3)
    - std_trajectory: standard deviation of trajectory (steps, 3)
    - target_position: target position (3,)
    - condition_y: y-coordinate of target condition
    - all_trajectories: list of all trajectories for individual plotting
    - save_path: path to save the figure
    - show_plot: whether to display the plot
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    steps = avg_trajectory.shape[0]
    
    # Plot individual trajectories (faint lines)
    for i, traj in enumerate(all_trajectories):
        if i < 10:  # Only plot first 10 for clarity
            ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], 
                   color='lightblue', alpha=0.3, linewidth=0.5)
    
    # Plot average trajectory
    ax.plot(avg_trajectory[:, 0], avg_trajectory[:, 1], avg_trajectory[:, 2],
            color='blue', linewidth=3, label=f'Average Trajectory (Y={condition_y:.3f})', marker='o', markersize=4)
    
    # Plot standard deviation as error bars at key points
    key_points = np.linspace(0, steps-1, 8, dtype=int)
    for i, point_idx in enumerate(key_points):
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
    ax.set_title(f"Average Hand Path with Standard Deviation\n({len(all_trajectories)} episodes)", fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Set equal aspect ratio
    max_range = np.array([
        avg_trajectory[:, 0].max() - avg_trajectory[:, 0].min(),
        avg_trajectory[:, 1].max() - avg_trajectory[:, 1].min(),
        avg_trajectory[:, 2].max() - avg_trajectory[:, 2].min()
    ]).max() / 2.0
    
    mid_x = (avg_trajectory[:, 0].max() + avg_trajectory[:, 0].min()) * 0.5
    mid_y = (avg_trajectory[:, 1].max() + avg_trajectory[:, 1].min()) * 0.5
    mid_z = (avg_trajectory[:, 2].max() + avg_trajectory[:, 2].min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    if show_plot:
        plt.show()
    
    return fig, ax

def plot_velocity_with_std(avg_velocities, std_velocities, save_path=None, show_plot=True):
    """
    Plot average velocity with standard deviation over time
    
    Parameters:
    - avg_velocities: average velocities (steps,)
    - std_velocities: standard deviation of velocities (steps,)
    - save_path: path to save the figure
    - show_plot: whether to display the plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    time_steps = np.arange(len(avg_velocities))
    
    # Plot average velocity
    ax.plot(time_steps, avg_velocities, color='blue', linewidth=2, marker='o', markersize=4, label='Average Velocity')
    
    # Plot standard deviation as shaded area
    ax.fill_between(time_steps, 
                   avg_velocities - std_velocities,
                   avg_velocities + std_velocities,
                   alpha=0.3, color='blue', label='±1 Standard Deviation')
    
    ax.set_xlabel("Time Step", fontsize=12)
    ax.set_ylabel("Velocity Magnitude (m/s)", fontsize=12)
    ax.set_title("Average Hand Velocity Over Time with Standard Deviation", fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Velocity plot saved to {save_path}")
    
    if show_plot:
        plt.show()
    
    return fig, ax

def main():
    """Main function to run the trajectory analysis"""
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
        return
    
    # Parameters for data collection
    num_episodes = 50  # Increase for better statistics
    steps_per_episode = 20
    target_condition = None  # Set to specific value if needed
    
    # Create save directory
    save_dir = '../figures'
    os.makedirs(save_dir, exist_ok=True)
    
    try:
        # Collect data
        print("\nCollecting trajectory data...")
        all_trajectories, all_velocities, target_position, condition_y = collect_trajectory_data(
            env, pi, num_episodes, steps_per_episode, target_condition
        )
        
        # Calculate statistics
        print("\nCalculating statistics...")
        avg_trajectory, std_trajectory, avg_velocities, std_velocities = calculate_trajectory_statistics(
            all_trajectories, all_velocities
        )
        
        # Print summary statistics
        print("\n=== Summary Statistics ===")
        print(f"Number of episodes: {len(all_trajectories)}")
        print(f"Target condition Y: {condition_y:.3f}")
        print(f"Target position: [{target_position[0]:.3f}, {target_position[1]:.3f}, {target_position[2]:.3f}]")
        print(f"Average final distance to target: {np.linalg.norm(avg_trajectory[-1] - target_position):.3f} m")
        print(f"Average peak velocity: {np.max(avg_velocities):.3f} m/s")
        print(f"Trajectory duration: {len(avg_trajectory)} steps")
        
        # Create plots
        print("\nCreating trajectory plot...")
        trajectory_fig, trajectory_ax = plot_trajectory_with_std(
            avg_trajectory, std_trajectory, target_position, condition_y, all_trajectories,
            save_path=os.path.join(save_dir, 'hand_trajectory_with_std.png'),
            show_plot=False  # Don't show plot when running as script
        )
        
        print("Creating velocity plot...")
        velocity_fig, velocity_ax = plot_velocity_with_std(
            avg_velocities, std_velocities,
            save_path=os.path.join(save_dir, 'hand_velocity_with_std.png'),
            show_plot=False  # Don't show plot when running as script
        )
        
        print(f"\nAll figures saved in the '{save_dir}' directory")
        print("Trajectory analysis complete!")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        env.close()
        print("Environment closed.")

if __name__ == "__main__":
    main() 