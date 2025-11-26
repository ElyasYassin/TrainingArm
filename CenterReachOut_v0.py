import collections
import gym
import numpy as np
import scipy.stats
from myosuite.envs.myo.base_v0 import BaseV0


class ReachEnvV0(BaseV0):
    DEFAULT_OBS_KEYS = ['time', 'obj_pos', 'reach_err', 'hand_qpos', 'hand_qvel']#['hand_qpos', 'hand_qvel', 
    DEFAULT_RWD_KEYS_AND_WEIGHTS = {
    'reach': 1.0,     # primary driver
    'vel':   1,     # bell‐curve shaping
    'act': 1,
    'penalty': 1,
    }
    DELAY = 1  # Default delay in timesteps
    
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
               delay: int = DELAY,
               **kwargs,
               ):
        
        self.prep_duration = 0.04
        self.palm_sid = self.sim.model.site_name2id("IFtip")
        self.start_pos = self.sim.data.site_xpos[self.palm_sid].copy()
        self.object_sid = self.sim.model.site_name2id("object_o")
        self.object_bid = self.sim.model.body_name2id("Object")
        self.obj_xyz_range = obj_xyz_range 

        self.drop_th = drop_th
        self.qpos_noise_range = qpos_noise_range
        self.max_episode_time = .26
        self.solved = False
        self.far_th=.1
        self.previous_reach_dist = None  # track previous step distance for progression
        
        self.round_robin_targets =[ [ 0.0613, -0.3140, 1.062], # Right 
                                   [ 0.0379, -0.2574, 1.062], # Up-right 
                                   [-0.0187, -0.2340, 1.062], # Up 
                                   [-0.0753, -0.2574, 1.062], # Up-left 
                                   [-0.0987, -0.3140, 1.062], # Left 
                                   [-0.0753, -0.3706, 1.062], # Down-left 
                                   [-0.0187, -0.3940, 1.062], # Down 
                                   [ 0.0379, -0.3706, 1.062], # Down-right ]
                                    ]
                                
        self.round_robin_index = 0
        
        super()._setup(obs_keys=obs_keys,
                       weighted_reward_keys=weighted_reward_keys,
                       **kwargs,
                       )
        keyFrame_id = 0 if self.obj_xyz_range is None else 1
        self.init_qpos[:] = self.sim.model.key_qpos[keyFrame_id].copy()
        
        # Initialize delay mechanism
        self.delay = delay
        self.obs_buffer = collections.deque(maxlen=delay + 1)  # Store up to delay+1 observations

    def get_obs_vec(self):
        self.obs_dict['time'] = np.array([self.sim.data.time])
        # self.obs_dict['hand_qpos'] = self.sim.data.qpos[:].copy()
        # self.obs_dict['hand_qvel'] = self.sim.data.qvel[:].copy() * self.dt
        # self.obs_dict['fiber_length'] = self.sim.data.actuator_length[:].copy()
        # self.obs_dict['fiber_velocity'] = self.sim.data.actuator_velocity[:].copy() * self.dt
        # if self.sim.model.na > 0:
        #     self.obs_dict['act'] = self.sim.data.act[:].copy()
            
        # reach error
        self.obs_dict['obj_pos'] = self.sim.data.site_xpos[self.object_sid]
        self.obs_dict['palm_pos'] = self.sim.data.site_xpos[self.palm_sid]
        self.obs_dict['reach_err'] = np.array(self.obs_dict['palm_pos']) - np.array(self.obs_dict['obj_pos'])

        # Get current observation vector
        t, obs_current = self.obsdict2obsvec(self.obs_dict, self.obs_keys)
        
        # Store current observation in buffer
        self.obs_buffer.append(obs_current.copy())
        
        # If we haven't accumulated enough observations (delay+1), return zeros (no feedback)
        # For delay=1: need 2 observations before giving feedback from step 1
        if len(self.obs_buffer) < self.delay + 1:
            # Return zero observation vector with same shape
            obs_delayed = np.zeros_like(obs_current)
            return obs_delayed
        
        # Return observation from 'delay' steps ago
        # The buffer stores observations in order: oldest to newest
        # buffer[0] is from delay steps ago, buffer[-1] is current
        obs_delayed = self.obs_buffer[0].copy()
        return obs_delayed

    def get_obs_dict(self, sim):
        obs_dict = {}
        obs_dict['time'] = np.array([sim.data.time])
        obs_dict['hand_qpos'] = sim.data.qpos[:].copy()
        obs_dict['hand_qvel'] = sim.data.qvel[:].copy() * self.dt
        obs_dict['fiber_length'] = sim.data.actuator_length[:].copy()
        obs_dict['fiber_velocity'] = sim.data.actuator_velocity[:].copy() * self.dt
        if sim.model.na > 0:
            obs_dict['act'] = sim.data.act[:].copy()
        # reach error
        obs_dict['obj_pos'] = sim.data.site_xpos[self.object_sid]
        obs_dict['palm_pos'] = sim.data.site_xpos[self.palm_sid]
        obs_dict['reach_err'] = np.array(obs_dict['palm_pos']) - np.array(obs_dict['obj_pos'])
        return obs_dict
    
    def get_reward_dict(self, obs_dict):   
        near_th = 0.005
        
        wx, wy, wz = 1.0, 1.0, 1.0        
        W = np.array([wx, wy, wz])
        
        reach_dist = np.linalg.norm(obs_dict['reach_err'][0][0] / W, axis=-1)

        if reach_dist < 0.03:
            reach_dist = np.linalg.norm(obs_dict['reach_err'], axis=-1)
        
        t = float(obs_dict['time'][0])          
        T = self.max_episode_time
        τ = t / T

        # solved/done logic
        solved = (reach_dist < near_th)
        done = solved or (t >= T)
        
        σ = 0.15   # controls width
        μ = 0.5  # peak at middle of episode

        # target velocity magnitude as a bell curve (Gaussian)
        v_target = np.exp(-0.5 * ((τ - μ) / σ)**2)

        # actual velocity magnitude
        v_actual = np.linalg.norm(self.sim.data.qvel[:].copy() * self.dt, axis=-1)

        # squared error penalty from target
        vel_profile_pen = (v_actual - v_target) ** 2
        
        # Actuation penalty
        act_pen = np.linalg.norm(self.sim.data.act[:].copy(), axis=-1)

        # Persistent solved state
        if solved:
            self.solved = True
            
        # Reward dictionary 
        rwd_dict = collections.OrderedDict([
            ('reach',   float(-10 * (reach_dist))),
            ('vel',     float(-1 * vel_profile_pen)),
            ('penalty',   float((-1 * (reach_dist > self.far_th)))),
            ('act', float(-0.03 * (act_pen))),  # actuation effort penalty
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
        self.previous_reach_dist = None  # reset step-by-step progression tracking
        
        # Reset delay buffer for new episode
        self.obs_buffer.clear()
            
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