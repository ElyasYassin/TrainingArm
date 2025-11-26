import collections
import gym
import numpy as np
import scipy.stats
from myosuite.envs.myo.base_v0 import BaseV0


class ReachEnvV1(BaseV0):
    DEFAULT_OBS_KEYS = ['hand_qpos', 'hand_qvel', 'obj_pos', 'reach_err']
    DEFAULT_RWD_KEYS_AND_WEIGHTS = {
    'reach': 1.0,     # primary driver
    'vel':   0.5,     # bell‐curve shaping (reduced from 1.0)
    'bonus': 0.4,     # small proximity bonus
   # 'act': 1,
    'penalty': 0.5,   # reduced from 1.0
    'vel_stop': 1,
    'time': 0.2,      # time penalty for faster completion (reduced from 0.5)
    'straight': 0.3   # straight reaching penalty (reduced from 0.8)
    }
    
    def __init__(self, model_path, obsd_model_path=None, seed=None, **kwargs):
        gym.utils.EzPickle.__init__(self, model_path, obsd_model_path, seed, **kwargs)
        super().__init__(model_path=model_path, obsd_model_path=obsd_model_path, seed=seed, env_credits=self.MYO_CREDIT)
        self._setup(**kwargs)

    def _setup(self,
               obj_xyz_range=None,
               obs_keys: list = DEFAULT_OBS_KEYS,
               drop_th=0.50,
               qpos_noise_range=None,
               weighted_reward_keys: dict = DEFAULT_RWD_KEYS_AND_WEIGHTS,
               **kwargs,
               ):
        
        self.prep_duration = 0.04
        self.palm_sid = self.sim.model.site_name2id("S_grasp")
        self.start_pos = self.sim.data.site_xpos[self.palm_sid].copy()
        self.object_sid = self.sim.model.site_name2id("object_o")
        self.object_bid = self.sim.model.body_name2id("Object")
        self.obj_xyz_range = obj_xyz_range 
        
        #self.start_pos = np.copy(self.sim.data.site_xpos[self.palm_sid])
        #self.target_pos = np.copy([-0.21968516789535328, -0.21249949997036174, 1.062])
        
        self.drop_th = drop_th
        self.qpos_noise_range = qpos_noise_range
        self.max_episode_time = 1.0  # Increased from 0.2 to 1.0 seconds
        self.solved = False
        self.far_th=.2
            
        self.round_robin_targets = [
         [-0.03125069216102037, -0.024064832105254508, 1.062],
         [0.10199416560603433, -0.3457434044594767, 1.062],
         [0.15718516789601672, -0.2125, 1.062],
         [0.1019938068049917, -0.07925623673839724, 1.062],
         [-0.031249807868617828, -0.40093516789591876, 1.062],
         [-0.16449389389172694, -0.3457436761747822, 1.062],
         [-0.21968516789535328, -0.21249949997036174, 1.062],
         [-0.1680450158763584, -0.07925561667782058, 1.062],]

        self.round_robin_index = 0
        
        super()._setup(obs_keys=obs_keys,
                       weighted_reward_keys=weighted_reward_keys,
                       **kwargs,
                       )
        keyFrame_id = 0 if self.obj_xyz_range is None else 1
        self.init_qpos[:] = self.sim.model.key_qpos[keyFrame_id].copy()

    def get_obs_vec(self):
        self.obs_dict['time'] = np.array([self.sim.data.time])
        self.obs_dict['hand_qpos'] = self.sim.data.qpos[:].copy()
        self.obs_dict['hand_qvel'] = self.sim.data.qvel[:].copy() * self.dt
        if self.sim.model.na > 0:
            self.obs_dict['act'] = self.sim.data.act[:].copy()
            
        # reach error
        self.obs_dict['obj_pos'] = self.sim.data.site_xpos[self.object_sid]
        self.obs_dict['palm_pos'] = self.sim.data.site_xpos[self.palm_sid]  # Fixed the bug here
        self.obs_dict['reach_err'] = np.array(self.obs_dict['palm_pos']) - np.array(self.obs_dict['obj_pos'])

        t, obs = self.obsdict2obsvec(self.obs_dict, self.obs_keys)
        return obs

    def get_obs_dict(self, sim):
        obs_dict = {}
        obs_dict['time'] = np.array([sim.data.time])
        obs_dict['hand_qpos'] = sim.data.qpos[:].copy()
        obs_dict['hand_qvel'] = sim.data.qvel[:].copy() * self.dt
        if sim.model.na > 0:
            obs_dict['act'] = sim.data.act[:].copy()

        # reach error
        obs_dict['obj_pos'] = sim.data.site_xpos[self.object_sid]
        obs_dict['palm_pos'] = sim.data.site_xpos[self.palm_sid]
        obs_dict['reach_err'] = np.array(obs_dict['palm_pos']) - np.array(obs_dict['obj_pos'])
        return obs_dict
    
    def get_reward_dict(self, obs_dict):   
        near_th = 0.08
        reach_dist = np.linalg.norm(obs_dict['reach_err'], axis=-1)
        t = float(obs_dict['time'][0])          
        T = self.max_episode_time
        τ = t / T

        # solved/done logic
        solved = (reach_dist < near_th)
        done = solved or (t >= T)
        
        σ = 0.15   # controls width
        μ = 0.3  # peak at middle of episode

        # target velocity magnitude as a bell curve (Gaussian)
        v_target = np.exp(-0.5 * ((τ - μ) / σ)**2)

        # actual velocity magnitude
        v_actual = np.linalg.norm(obs_dict['hand_qvel'], axis=-1)

        # squared error penalty from target
        vel_profile_pen = (v_actual - v_target) ** 2
        
        # Actuation penalty
        act_pen = np.linalg.norm(obs_dict['act'], axis=-1)

        # Time penalty - encourages faster completion
        time_pen = t / T  # normalized time penalty (0 to 1)

        # Path penalty (unchanged)
        start_to_target = self.round_robin_targets[self.round_robin_index] - self.start_pos
        start_to_current = obs_dict['palm_pos'] - self.start_pos
        proj = np.dot(start_to_current, start_to_target) / np.linalg.norm(start_to_target)
        closest_point = self.start_pos + (proj / np.linalg.norm(start_to_target)) * start_to_target
        deviation = np.linalg.norm(obs_dict['palm_pos'] - closest_point)

        # Straight reaching penalty - penalizes deviation from straight line
        straight_penalty = deviation

        # Persistent solved state
        if solved:
            self.solved = True
            
        # Reward dictionary 
        rwd_dict = collections.OrderedDict([
            ('reach',   float(1.0 - np.tanh(10.0 * (reach_dist)**2))),
            ('vel',     float(-0.5 * vel_profile_pen)),  # reduced from -1
            ('bonus',   float(0.1 if reach_dist < 2 * near_th else 0)),
            #('act',   float(-0.075 * act_pen)),
            ('vel_stop', float(3 if self.solved else 0)),
            ('penalty',   float((-0.5 * (reach_dist > self.far_th)))),  # reduced from -1
            ('time',   float(-0.5 * time_pen)),  # reduced from -2.0
            ('straight', float(-0.5 * straight_penalty))  # reduced from -2.0
        ])

        rwd_dict['sparse'] = -1.0 * reach_dist
        rwd_dict['solved'] = float(solved)
        rwd_dict['done'] = done

        # Weighted dense reward
        rwd_dict['dense'] = np.sum([wt * rwd_dict[key] for key, wt in self.rwd_keys_wt.items() if key in rwd_dict], axis=0)

        return rwd_dict
    
    def generate_target_pose(self):
        # Store the current index
        self.current_direction_index = self.round_robin_index

        # Select object position
        obj_pos = np.array(self.round_robin_targets[self.round_robin_index])
        self.round_robin_index = (self.round_robin_index + 1) % len(self.round_robin_targets)

        self.sim.model.body_pos[self.object_bid] = obj_pos
        self.current_object_pos = obj_pos
        self.sim.forward()

    def reset(self, reset_qpos=None, reset_qvel=None):
        self.sim.data.qvel[:] = 0
        self.sim.data.ctrl[:] = 0
        if self.sim.model.na > 0:
            self.sim.data.act[:] = 0
            
        # Reset velocity tracking for new episode
        self.max_vel_episode = 0.0
        self.solved = False
            
        self.start_pos = self.sim.data.site_xpos[self.palm_sid].copy()
        if self.qpos_noise_range is not None:
            reset_qpos_local = self.init_qpos + self.qpos_noise_range*(self.sim.model.jnt_range[:,1]-self.sim.model.jnt_range[:,0])
            reset_qpos_local[-6:] = self.init_qpos[-6:]
        else:
            reset_qpos_local = reset_qpos

        self.generate_target_pose()
        self.robot.sync_sims(self.sim, self.sim_obsd)
        obs = super().reset()
        return obs 