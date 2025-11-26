# shoulder_angle_velocity_plots.py
import myosuite
import gym
import numpy as np
import os, pickle
import matplotlib.pyplot as plt

# ======================= Config =======================
JOINT_INDEX = 2  # shoulder_elv (10=elv_angle, 11=shoulder_elv, 13=shoulder_rot)
EPISODES_PER_COND = 8
STEPS_PER_EP = 16 
SAVE_DIR = "figures_rot_angle_vel"
POLICY_PATH = "./policies/seed_42/best_policy.pickle"
ENV_NAME = "CenterReachOut-v0"
# ======================================================

# ---------- Utilities ----------
def get_dt_seconds(env):
    """Try to infer dt (sec). Your obs_dict['hand_qvel'] = qvel * dt, so undo with 1/dt."""
    # MyoSuite usually exposes env.dt; otherwise MuJoCo timestep * frame_skip.
    if hasattr(env, "dt") and isinstance(env.dt, (float, int)):
        return float(env.dt)
    try:
        timestep = float(env.model.opt.timestep)
        frameskip = int(getattr(env, "frame_skip", 1))
        return timestep * frameskip
    except Exception:
        # Fallback to 20 ms like your plotting utilities
        return 0.02

def discover_unique_targets(env, want=8, max_trials=5000, atol=1e-4):
    uniq = []
    for _ in range(max_trials):
        env.reset()
        p = np.copy(env.obs_dict['obj_pos'])
        if not any(np.allclose(p, q, atol=atol) for q in uniq):
            uniq.append(p)
        if len(uniq) == want:
            break
    if len(uniq) < want:
        print(f"[warn] only found {len(uniq)} unique targets.")
    return uniq

def wait_for_target(env, target, atol=1e-4, max_tries=3000):
    for _ in range(max_tries):
        env.reset()
        if np.allclose(env.obs_dict['obj_pos'], target, atol=atol):
            return True
    return False

# ---------- Plot like the reference figure ----------
def plot_shoulder_angle_velocity(t_ms, angle_traces, velocity_traces, title, save_path=None):
    """
    angle_traces: (n_trials, T) radians
    velocity_traces: (n_trials, T) rad/s
    """
    angle_traces = np.asarray(angle_traces, dtype=np.float64)
    velocity_traces = np.asarray(velocity_traces, dtype=np.float64)
    n_trials, T = angle_traces.shape
    assert velocity_traces.shape == (n_trials, T)
    assert len(t_ms) == T

    # nice green→red palette similar to your reference
    cmap = plt.cm.RdYlGn_r
    colors = [cmap(i / max(n_trials - 1, 1)) for i in range(n_trials)]

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6), constrained_layout=True)
    ax_ang, ax_vel = axes

    # Angle
    for i in range(n_trials):
        ax_ang.plot(t_ms, angle_traces[i], lw=2, color=colors[i])
    ax_ang.axvline(0, color='k', lw=1)
    ax_ang.axhline(0, color='k', lw=1)
    ax_ang.set_xlabel("Time After Perturbation Onset (ms)")
    ax_ang.set_ylabel("Angle (rads)")
    ax_ang.set_title("Shoulder")

    # Velocity
    for i in range(n_trials):
        ax_vel.plot(t_ms, velocity_traces[i], lw=2, color=colors[i])
    ax_vel.axvline(0, color='k', lw=1)
    ax_vel.axhline(0, color='k', lw=1)
    ax_vel.set_xlabel("Time After Perturbation Onset (ms)")
    ax_vel.set_ylabel("Angular Velocity (rads/sec)")
    ax_vel.set_title("Shoulder")

    # Clean aesthetics
    for ax in axes:
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.grid(True, alpha=0.25)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved {save_path}")
    plt.close(fig)
    return fig

# ---------- Core collection ----------
def collect_joint_for_target(env, pi, target_pos, joint_index, episodes=20, steps_per_episode=16):
    """
    For a specific target, collect angle and angular velocity traces for one joint across episodes.
    Returns: list of angle arrays (T,), list of velocity arrays (T,)
    """
    dt = get_dt_seconds(env)
    all_angles, all_vels = [], []

    for ep in range(episodes):
        if not wait_for_target(env, target_pos):
            print("[warn] failed to match target after many resets:", target_pos)
            break

        angles, vels = [], []
        for step in range(steps_per_episode):
            # Read obs
            qpos = np.copy(env.obs_dict['hand_qpos'])
            # qvel_obs = true_qvel * dt  (from your env code)
            qvel_times_dt = np.copy(env.obs_dict['hand_qvel'])

            # Store shoulder joint angle & angular velocity (rad/s)
            angle = float(qpos[joint_index])
            ang_vel = float(qvel_times_dt[joint_index] / max(dt, 1e-9))  # undo the dt scaling

            angles.append(angle)
            vels.append(ang_vel)

            # Act
            obs = env.get_obs()
            action, _ = pi.get_action(obs)

            # Step
            _, _, done, _ = env.step(action)
            if done:
                # pad if early termination
                while len(angles) < steps_per_episode:
                    angles.append(angles[-1])
                    vels.append(0.0)
                break

        all_angles.append(np.asarray(angles, dtype=np.float64))
        all_vels.append(np.asarray(vels, dtype=np.float64))

    return all_angles, all_vels, dt

def collect_and_plot_all_conditions(env, pi, joint_index=11,
                                    episodes_per_condition=20,
                                    steps_per_episode=16,
                                    save_dir="figures_shoulder_angle_vel"):

    os.makedirs(save_dir, exist_ok=True)

    targets = discover_unique_targets(env, want=8)
    ctr = np.mean(np.array(targets)[:, :2], axis=0)
    targets = sorted(targets, key=lambda p: (np.arctan2(p[1]-ctr[1], p[0]-ctr[0]) + 2*np.pi) % (2*np.pi))

    labels = ["Right","Up-right","Up","Up-left","Left","Down-left","Down","Down-right"]
    results = {}

    # For the overlay
    overlay_tms, overlay_ang, overlay_vel, overlay_labels = [], [], [], []

    for i, target in enumerate(targets):
        label = labels[i % len(labels)]
        print(f"\n=== Condition {i+1}/8: {label} | target {np.round(target,4)} ===")

        ang_traces, vel_traces, dt = collect_joint_for_target(
            env, pi, target, joint_index, episodes=episodes_per_condition, steps_per_episode=steps_per_episode
        )
        if len(ang_traces) == 0:
            print("[warn] no episodes collected for", label)
            continue

        T = len(ang_traces[0])
        t_ms = np.arange(T) * dt * 1000.0

        # Per-condition multi-trial figure (unchanged)
        out_path = os.path.join(save_dir, f"shoulder_{i+1:02d}_{label}.png")
        plot_shoulder_angle_velocity(t_ms, ang_traces, vel_traces, title="Shoulder", save_path=out_path)

        # Store for overlay (pick exactly one episode)
        one_ang = _pick_one_episode(ang_traces)
        one_vel = _pick_one_episode(vel_traces)
        if one_ang is not None and one_vel is not None:
            overlay_tms.append(t_ms)
            overlay_ang.append(one_ang)
            overlay_vel.append(one_vel)
            overlay_labels.append(label)
        else:
            print(f"[warn] skipping overlay for {label} (no valid episode)")

        results[label] = dict(
            target=np.array(target), dt=dt, t_ms=t_ms,
            angle_traces=ang_traces, velocity_traces=vel_traces,
            fig_path=out_path,
        )

        peak_speed = np.max([np.max(np.abs(v)) for v in vel_traces])
        print(f"Saved: {out_path} | episodes={len(ang_traces)} | peak |ω| ≈ {peak_speed:.2f} rad/s")

    # ---------- NEW: make the single overlay figure ----------
    if len(overlay_labels) == 8:
        all_path = os.path.join(save_dir, "shoulder_ALL_conditions_one_episode.png")
        plot_all_conditions_one_episode(overlay_tms, overlay_ang, overlay_vel, overlay_labels, all_path)
    else:
        print(f"[warn] overlay had {len(overlay_labels)} conditions; expected 8.")

    print("\nDone. Saved per-condition figures and the overlay in:", save_dir)
    return results

# ---------- NEW: one-episode-per-condition overlay ----------
def _pick_one_episode(traces):
    """Pick the first non-empty episode trace."""
    for tr in traces:
        if tr is not None and len(tr) > 0:
            return np.asarray(tr, dtype=np.float64)
    return None

def plot_all_conditions_one_episode(t_ms_list, ang_list, vel_list, labels, save_path):
    """
    Plot all conditions on the same axes with exactly one episode per condition.
    - t_ms_list: list of 1D arrays (len 8), each condition's timebase
    - ang_list, vel_list: lists of 1D arrays (len 8), one episode per condition
    - labels: list[str] (len 8)
    Colors: first 4 green, last 4 red.
    """
    # Align lengths safely (truncate to min T)
    T_min = min(len(t) for t in t_ms_list)
    t_ms = t_ms_list[0][:T_min]

    ang_mat = [a[:T_min] for a in ang_list]
    vel_mat = [v[:T_min] for v in vel_list]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    ax_ang, ax_vel = axes

    for i, lbl in enumerate(labels):
        color = 'green' if i < 4 else 'red'
        ax_ang.plot(t_ms, ang_mat[i], lw=2, color=color, label=lbl)
        ax_vel.plot(t_ms, vel_mat[i], lw=2, color=color, label=lbl)

    for ax, ylab, title in [
        (ax_ang, "Angle (rad)",  "Shoulder angle — 1 ep/cond"),
        (ax_vel, "Angular vel (rad/s)", "Shoulder velocity — 1 ep/cond"),
    ]:
        ax.axvline(0, color='k', lw=1)
        ax.axhline(0, color='k', lw=1)
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel(ylab)
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    # Single legend (avoid duplicates)
    handles, labz = ax_ang.get_legend_handles_labels()
    by_lbl = dict(zip(labz, handles))
    axes[0].legend(by_lbl.values(), by_lbl.keys(), frameon=False, ncol=2, fontsize=8)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved overlay: {save_path}")
    plt.close(fig)
    
# ---------- Main ----------
if __name__ == "__main__":
    env = gym.make(ENV_NAME)

    try:
        pi = pickle.load(open(POLICY_PATH, 'rb'))
        print(f"Loaded policy from {POLICY_PATH}")
    except FileNotFoundError:
        print(f"[error] Policy file not found: {POLICY_PATH}")
        pi = None

    if pi is not None:
        _ = collect_and_plot_all_conditions(
            env=env,
            pi=pi,
            joint_index=JOINT_INDEX,
            episodes_per_condition=EPISODES_PER_COND,
            steps_per_episode=STEPS_PER_EP,
            save_dir=SAVE_DIR
        )

    env.close()