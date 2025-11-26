import myosuite
import gym
import numpy as np
import os, pickle, imageio
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from base64 import b64encode
from IPython.display import HTML

# ---------- Utilities ----------
def show_video(video_path, video_width=400):
    video_file = open(video_path, "r+b").read()
    video_url = f"data:video/mp4;base64,{b64encode(video_file).decode()}"
    return HTML(f"""<video autoplay width={video_width} controls><source src="{video_url}"></video>""")

def pos_key(pos, ndigits=4):
    # Stable, hashable key for floats
    return tuple(np.round(np.array(pos, dtype=np.float64), ndigits))

def discover_unique_targets(env, want=8, max_trials=5000, atol=1e-4):
    """Reset until we collect `want` unique obj_pos from THIS env."""
    unique = []
    for _ in range(max_trials):
        env.reset()
        p = np.copy(env.obs_dict['obj_pos'])
        if not any(np.allclose(p, q, atol=atol) for q in unique):
            unique.append(p)
        if len(unique) == want:
            break
    if len(unique) < want:
        print(f"Warning: only found {len(unique)} unique targets.")
    return unique

def wait_for_target(env, target, atol=1e-4, max_tries=3000):
    """Keep resetting until env.obj_pos ~ target."""
    for _ in range(max_tries):
        env.reset()
        if np.allclose(env.obs_dict['obj_pos'], target, atol=atol):
            return True
    return False

# ---------- Your stats/plotting (slightly adapted to be reusable) ----------
def calculate_trajectory_statistics(all_trajectories, all_velocities):
    traj = np.array(all_trajectories)  # (episodes, steps, 3)
    vels = np.array(all_velocities)    # (episodes, steps)
    avg_traj = np.mean(traj, axis=0)
    std_traj = np.std(traj, axis=0)
    avg_vel = np.mean(vels, axis=0)
    std_vel = np.std(vels, axis=0)
    return avg_traj, std_traj, avg_vel, std_vel

def plot_trajectory_with_std(avg_trajectory, std_trajectory, target_position, label,
                             all_trajectories, save_path=None, show_plot=True):
    fig, ax = plt.subplots(figsize=(8, 7))
    x = avg_trajectory[:, 0]; y = avg_trajectory[:, 1]
    sx = std_trajectory[:, 0]; sy = std_trajectory[:, 1]
    s = np.maximum(sx, sy)

    ax.plot(x, y, linewidth=3, marker='o', markersize=3,
            label=f'Avg Trajectory ({label})')
    ax.fill_between(np.concatenate([x, x[::-1]]),
                    np.concatenate([y+s, (y-s)[::-1]]),
                    alpha=0.25, label='±1 SD')
    ax.scatter(target_position[0], target_position[1], c='red', marker='X',
               s=120, edgecolor='k', linewidth=1.5, label='Target')
    # 4 cm radius circle
    circ = Circle((target_position[0], target_position[1]), 0.02,
                  facecolor='none', edgecolor='black', linestyle='--', linewidth=1.2,
                  label='2 cm radius')
    ax.add_patch(circ)
    # Start position
    ax.scatter(x[0], y[0], c='green', s=80, marker='o', edgecolor='k', linewidth=1, label='Start')

    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
    ax.set_title(f"Average Hand Path – {label} ({len(all_trajectories)} eps)")
    ax.legend(loc='best')

    # Auto-bounds with margin
    all_x = np.concatenate([x, [target_position[0]]])
    all_y = np.concatenate([y, [target_position[1]]])
    x_margin = (all_x.max() - all_x.min()) * 0.3
    y_margin = (all_y.max() - all_y.min()) * 0.3
    ax.set_xlim(all_x.min()-x_margin, all_x.max()+x_margin)
    ax.set_ylim(all_y.min()-y_margin, all_y.max()+y_margin)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved {save_path}")
    if show_plot:
        plt.show()
    return fig, ax

def plot_velocity_with_std(avg_velocities, std_velocities, label, save_path=None, show_plot=True):
    fig, ax = plt.subplots(figsize=(8, 4))
    n = len(avg_velocities)
    dt_s = 0.02
    t_ms = np.arange(n) * dt_s * 1000.0  # convert timesteps to ms

    # smooth line, no point markers
    ax.plot(t_ms, avg_velocities, linewidth=2.5, label='Avg speed')
    ax.fill_between(t_ms,
                    avg_velocities - std_velocities,
                    avg_velocities + std_velocities,
                    alpha=0.25, label='±1 SD')

    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Velocity magnitude (m/s)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # optional: nicer ticks (100 ms major, 20 ms minor)
    try:
        import matplotlib.ticker as mtick
        ax.xaxis.set_major_locator(mtick.MultipleLocator(100))
        ax.xaxis.set_minor_locator(mtick.MultipleLocator(20))
    except Exception:
        pass

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved {save_path}")
    if show_plot:
        plt.show()
    return fig, ax

def plot_velocity_single(velocities, label, save_path=None, show_plot=True):
    fig, ax = plt.subplots(figsize=(8, 4))
    n = len(velocities)
    dt_s = 0.02
    t_ms = np.arange(n) * dt_s * 1000.0  # convert timesteps to ms

    # smooth line, no markers
    ax.plot(t_ms, velocities, linewidth=2.5, color='black')

    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Velocity magnitude (m/s)")
    ax.set_title(f"Velocity – {label}")
    ax.grid(True, alpha=0.3)

    # nice ticks for paper
    import matplotlib.ticker as mtick
    ax.xaxis.set_major_locator(mtick.MultipleLocator(100))
    ax.xaxis.set_minor_locator(mtick.MultipleLocator(20))

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved {save_path}")
    if show_plot:
        plt.show()
    return fig, ax

# ---------- Representative trial picker ----------
def pick_representative_trial_index(all_velocities):
    """
    Choose a 'random but representative' trial:
    - Compute peak speed per trial
    - Take indices in the IQR [Q1, Q3] to avoid outliers
    - Randomly select one from that set
    - Fallback: pick the trial whose peak is closest to the median
    """
    peaks = np.array([np.max(v) if len(v) > 0 else 0.0 for v in all_velocities], dtype=float)
    if len(peaks) == 0:
        return 0
    q1, q3 = np.percentile(peaks, [25, 75])
    iqr_idxs = np.where((peaks >= q1) & (peaks <= q3))[0]
    if len(iqr_idxs) > 0:
        return int(np.random.choice(iqr_idxs))
    # fallback to median-closest
    med = np.median(peaks)
    return int(np.argmin(np.abs(peaks - med)))

# ---------- Core: collect for ALL conditions ----------
def collect_for_target(env, pi, target_pos, episodes=20, steps_per_episode=16):
    """Run several episodes for a specific target (obj_pos ~= target_pos)."""
    all_traj, all_vel = [], []
    for ep in range(episodes):
        ok = wait_for_target(env, target_pos)
        if not ok:
            print("Failed to match target after many resets:", target_pos)
            break

        palm_traj, velocities = [], []
        for step in range(steps_per_episode):
            # measure current
            hand_qvel = np.copy(env.obs_dict['hand_qvel'])
            velocities.append(np.linalg.norm(hand_qvel))

            # act
            o = env.get_obs()
            action, _ = pi.get_action(o)

            # record palm
            palm_traj.append(np.copy(env.obs_dict['palm_pos']))

            # step
            obs, reward, done, info = env.step(action)
            if done:
                # pad if early termination
                while len(palm_traj) < steps_per_episode:
                    palm_traj.append(palm_traj[-1])
                    velocities.append(0.0)
                break

        all_traj.append(np.array(palm_traj))
        all_vel.append(np.array(velocities))
    return all_traj, all_vel

def collect_all_conditions(env, pi, episodes_per_condition=20, steps_per_episode=16, save_dir="figures"):
    """Discover 8 targets, then collect trajectories for each; return dict keyed by pos_key."""
    os.makedirs(save_dir, exist_ok=True)

    # 1) Discover actual 8 targets in THIS env
    targets = discover_unique_targets(env, want=8)
    # Sort them into a nice compass order (optional)
    center = np.mean(np.array(targets)[:, :2], axis=0)
    angles = []
    for i, p in enumerate(targets):
        dx, dy = p[0]-center[0], p[1]-center[1]
        ang = (np.arctan2(dy, dx) + 2*np.pi) % (2*np.pi)
        angles.append((ang, i))
    targets_ordered = [targets[i] for _, i in sorted(angles, key=lambda x: x[0])]

    results = {}  # key -> dict with stats/plots
    labels = ["Right","Up-right","Up","Up-left","Left","Down-left","Down","Down-right"]
    # rotate labels if needed: labels = labels[k:]+labels[:k]

    # 2) Collect per-condition
    for idx, target in enumerate(targets_ordered):
        k = pos_key(target)
        label = labels[idx % len(labels)]
        print(f"\n=== Condition {idx+1}/8: {label} | target {np.round(target,4)} ===")
        all_traj, all_vel = collect_for_target(env, pi, target,
                                               episodes=episodes_per_condition,
                                               steps_per_episode=steps_per_episode)
        if len(all_traj) == 0:
            print("No episodes collected for", label)
            continue

        avg_traj, std_traj, avg_vel, std_vel = calculate_trajectory_statistics(all_traj, all_vel)

        # 3) Plot & save (averages)
        traj_path = os.path.join(save_dir, f"traj_{idx+1:02d}_{label}.png")
        vel_path  = os.path.join(save_dir, f"vel_{idx+1:02d}_{label}.png")
        plot_trajectory_with_std(avg_traj, std_traj, target, label, all_traj, save_path=traj_path, show_plot=False)
        plot_velocity_with_std(avg_vel, std_vel, label, save_path=vel_path, show_plot=False)

        # 4) Pick and plot a representative SINGLE trial velocity (random within IQR)
        rep_idx = pick_representative_trial_index(all_vel)
        single_vel_path = os.path.join(save_dir, f"vel_single_{idx+1:02d}_{label}.png")
        plot_velocity_single(all_vel[rep_idx], label=f"{label}", save_path=single_vel_path, show_plot=False)

        # stash
        results[k] = dict(
            label=label,
            target=target,
            all_trajectories=all_traj,
            all_velocities=all_vel,
            avg_trajectory=avg_traj,
            std_trajectory=std_traj,
            avg_velocities=avg_vel,
            std_velocities=std_vel,
            traj_path=traj_path,
            vel_path=vel_path,
            representative_index=rep_idx,
            vel_single_path=single_vel_path,
        )

    print("\nDone. Saved per-condition figures in:", save_dir)
    return results

# ---------- Main ----------
if __name__ == "__main__":
    env = gym.make('CenterReachOut-v0')

    policy_path = "./models/all_Reaches/straighter_2_reach_0.08/policies/seed_42/best_policy.pickle" #Change the folder to the policy location

    try:
        pi = pickle.load(open(policy_path, 'rb'))
        print(f"Loaded policy from {policy_path}")
    except FileNotFoundError:
        print(f"Policy file not found at {policy_path}")
        pi = None

    if pi is not None:
        results = collect_all_conditions(
            env=env,
            pi=pi,
            episodes_per_condition=100,   # <- change as needed
            steps_per_episode=18,
            save_dir="figures_all_conditions"
        )

        # Optional: quick summary printout
        for k, r in results.items():
            dist_final = np.linalg.norm(r["avg_trajectory"][-1] - r["target"])
            print(f'{r["label"]:>10s}  target={np.round(r["target"],4)}  '
                  f'final_dist={dist_final:.3f} m  '
                  f'peak_v={np.max(r["avg_velocities"]):.3f} m/s  '
                  f'eps={len(r["all_trajectories"])}  '
                  f'rep_idx={r["representative_index"]}  '
                  f'single_fig={os.path.basename(r["vel_single_path"])}')

    env.close()