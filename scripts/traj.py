import os
import imageio
import matplotlib.pyplot as plt
import numpy as np
import gym
from matplotlib.patches import Circle
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D)
import matplotlib.cm as cm
import myosuite
env = gym.make('CenterReachOut-v0')
# -----------------------------
# Config
# -----------------------------
ENV_ID = 'CenterReachOut-v0'
NUM_EPISODES = 1
STEPS_PER_EPISODE = 16
FREEZE_R = 0.05   # draw bubble if you use freeze shaping (meters)
env = gym.make('CenterReachOut-v0')

import pickle
# load policy

policy = "./policies/seed_42/best_policy.pickle"
pi = pickle.load(open(policy, 'rb'))


def show_video(video_path, video_width = 400):

  video_file = open(video_path, "r+b").read()

  video_url = f"data:video/mp4;base64,{b64encode(video_file).decode()}"
  return HTML(f"""<video autoplay width={video_width} controls><source src="{video_url}"></video>""")

# -----------------------------
# Helpers
# -----------------------------
def point_to_line_lateral_distance(p, s, t):
    """
    Perpendicular distance from point p to line through s->t in 3D:
        || (p - s) x (t - s) || / || t - s ||
    """
    d = t - s
    d_norm = np.linalg.norm(d)
    if d_norm < 1e-8:
        return 0.0
    return np.linalg.norm(np.cross(p - s, d)) / d_norm

def project_point_onto_line(p, s, t):
    """
    Returns the projection scalar u along s->t (0 at s, 1 at t) and the projected point.
    """
    d = t - s
    d_norm2 = np.dot(d, d)
    if d_norm2 < 1e-12:
        return 0.0, s.copy()
    u = np.dot(p - s, d) / d_norm2
    proj = s + u * d
    return u, proj

def make_freeze_sphere(center, radius, n=24):
    """Return (xs, ys, zs) for a translucent sphere surface."""
    u = np.linspace(0, 2*np.pi, n)
    v = np.linspace(0, np.pi, n)
    xs = center[0] + radius * np.outer(np.cos(u), np.sin(v))
    ys = center[1] + radius * np.outer(np.sin(u), np.sin(v))
    zs = center[2] + radius * np.outer(np.ones_like(u), np.cos(v))
    return xs, ys, zs

# -----------------------------
# Env + rollout
# -----------------------------
env = gym.make(ENV_ID)
frames = []
condition_y = None
ObjectPosition = None
all_trajectories = []
all_velocities = []
all_lateral_devs = []
all_proj_points = []
total_reward = 0.0

# Fix a consistent target y for comparability
while True:
    env.reset()
    y_value = env.obs_dict['obj_pos'][1]
    if condition_y is None:
        condition_y = float(y_value)
        ObjectPosition = np.copy(env.obs_dict['obj_pos'])
        break

for ep in range(NUM_EPISODES):
    obs = env.reset()
    while not np.isclose(env.obs_dict['obj_pos'][1], condition_y):
        obs = env.reset()

    # Start/target (we’ll recompute target every step from obs_dict just in case)
    start_pos = np.copy(env.obs_dict['palm_pos'])
    target_pos = np.copy(env.obs_dict['obj_pos'])

    palm_traj = []
    velocities = []
    lateral_devs = []
    proj_points = []   # closest points on the straight line for each palm point

    for step in range(STEPS_PER_EPISODE):
        # Current kinematics
        hand_qvel = np.copy(env.obs_dict['hand_qvel'])
        vel_mag = float(np.linalg.norm(hand_qvel))
        velocities.append(vel_mag)

        # Cache rwd breakdown (optional debug)
        # print(env.rwd_dict)

        # Render frame (optional)
        frame = env.sim.renderer.render_offscreen(width=400, height=400, camera_id=1)
        frames.append(frame)

        # Observation for policy
        o = env.get_obs()  # BaseV0 exposes this; if not, use obs from reset/step
        action, _ = pi.get_action(o)

        # Log palm before stepping
        palm_pos = np.copy(env.obs_dict['palm_pos'])
        target_pos = np.copy(env.obs_dict['obj_pos'])  # keep synced
        palm_traj.append(palm_pos)

        # Lateral deviation geometry
        ld = point_to_line_lateral_distance(palm_pos, start_pos, target_pos)
        _, proj = project_point_onto_line(palm_pos, start_pos, target_pos)
        lateral_devs.append(float(ld))
        proj_points.append(proj)

        # Step
        obs, reward, done, info = env.step(action)
        total_reward += float(reward)
        if done:
            # Capture final palm position for plotting symmetry
            palm_traj.append(np.copy(env.obs_dict['palm_pos']))
            velocities.append(float(np.linalg.norm(env.obs_dict['hand_qvel'])))
            ld = point_to_line_lateral_distance(env.obs_dict['palm_pos'], start_pos, target_pos)
            _, proj = project_point_onto_line(env.obs_dict['palm_pos'], start_pos, target_pos)
            lateral_devs.append(float(ld))
            proj_points.append(proj)
            break

    all_trajectories.append(np.array(palm_traj))
    all_velocities.append(np.array(velocities))
    all_lateral_devs.append(np.array(lateral_devs))
    all_proj_points.append(np.array(proj_points))

env.close()

# -----------------------------
# Save rollout video
# -----------------------------
os.makedirs('videos', exist_ok=True)
video_path = 'videos/test_traj.mp4'
imageio.mimsave(video_path, frames, fps=30)
print(f"Saved video to {video_path}")

# -----------------------------
# Aggregate (if multiple episodes)
# -----------------------------
# For one episode, these are just identity
traj = all_trajectories[0]              # (T, 3)
vels = all_velocities[0]                # (T,)
lats = all_lateral_devs[0]              # (T,)
projs = all_proj_points[0]              # (T, 3)

start_pos = traj[0]
target_pos = ObjectPosition

# -----------------------------
# Plot 1: 3D trajectory vs straight line + perpendicular ticks
# -----------------------------
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Straight line from start to target
ax.plot([start_pos[0], target_pos[0]],
        [start_pos[1], target_pos[1]],
        [start_pos[2], target_pos[2]],
        linewidth=2, label='Straight line (start→target)')

# Palm trajectory
ax.plot(traj[:, 0], traj[:, 1], traj[:, 2],
        marker='o', label='Palm trajectory')

# Start and target markers
ax.scatter(start_pos[0], start_pos[1], start_pos[2], s=80, c='black', marker='o', label='Start')
ax.scatter(target_pos[0], target_pos[1], target_pos[2], s=120, c='red', marker='X', edgecolor='k', label='Target')

# Draw perpendicular tick lines (palm point → closest point on line)
for p, q in zip(traj, projs):
    ax.plot([p[0], q[0]], [p[1], q[1]], [p[2], q[2]], linewidth=1)

# Freeze bubble at target (visualizes the "stop/jitter" zone)
if FREEZE_R is not None and FREEZE_R > 0:
    xs, ys, zs = make_freeze_sphere(target_pos, FREEZE_R, n=28)
    ax.plot_surface(xs, ys, zs, alpha=0.12, linewidth=0)

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("Palm Trajectory and Lateral Deviation Ticks")
ax.legend(loc='upper left')
plt.tight_layout()
plt.show()

# -----------------------------
# Plot 2: Lateral deviation over time
# -----------------------------
plt.figure(figsize=(8, 4))
plt.plot(range(len(lats)), lats, marker='o')
plt.title("Lateral Deviation (Perpendicular Distance to Line) Over Time")
plt.xlabel("Time Step")
plt.ylabel("Lateral Deviation (m)")
plt.grid(True)
plt.tight_layout()
plt.show()

# -----------------------------
# Plot 3: Velocity over time (unchanged)
# -----------------------------
plt.figure(figsize=(8, 4))
plt.plot(range(len(vels)), vels, marker='o')
plt.title("Velocity Magnitude Over Time")
plt.xlabel("Time Step")
plt.ylabel("Velocity (m/s)")
plt.grid(True)
plt.tight_layout()
plt.show()

# -----------------------------
# Debug print (last step reward dict)
# -----------------------------
try:
    print("Last-step reward dict:", env.rwd_dict)
except Exception:
    pass