import warnings
warnings.filterwarnings('ignore')
import argparse
import gym
import os
from mjrl.utils.gym_env import GymEnv
from mjrl.policies.gaussian_mlp import MLP, RNN
from mjrl.baselines.mlp_baseline import MLPBaseline
from mjrl.algos.ppo_clip import PPO
from mjrl.algos.npg_cg import NPG
from mjrl.utils.train_agent import train_agent
import myosuite
import torch


device = "cpu"
print(f"Using device: {device}")

# -------------------------------
# Parse CLI arguments
# -------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42, help="Random seed for training")
args = parser.parse_args()
seed = args.seed

# -------------------------------
# Setup directories
# -------------------------------
log_dir = f"logs/seed_{seed}"
model_dir = f"policies/seed_{seed}"
os.makedirs(log_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)


env = gym.make('CenterReachOut-v0')
env.reset()


#Policy hyperparameters
policy_size = (64, 64)  # Size of the policy network
seed = 1  # Random seed used for reproducibility
rl_step_size = 0.05
learnrate = 0.00025
entcoeff = 0.05
batchsize = 16
num_epoch = 10

#Value Function hyperparameters
vf_hidden_size = (256, 256) # Value Function Network size
seed = 1  # Random seed used for reproducibility
rl_step_size = 0.1
learnrate = 0.000325
entcoeff = 0.1
batchsize = 32
num_epoch = 10
regcoef = 1e-3

#Training hyperparameters
iterations = 1000
gamma = 0.995
gae_lambda = 0.97


# Wrap the environment
e = GymEnv(env)

policy = MLP(e.spec, hidden_sizes=policy_size, seed=seed, init_log_std=-0.5, min_log_std=-4.0)
baseline = MLPBaseline(e.spec, reg_coef=regcoef, batch_size=batchsize , hidden_sizes=vf_hidden_size, epochs=num_epoch, learn_rate=learnrate)


print(e.observation_dim) #For CenterReachOut-v0: 157 
print(e.action_dim) #For CenterReachOut-v0: 63 

agent = NPG(e, policy, baseline, epochs=10, normalized_step_size=rl_step_size, seed=seed, save_logs=True,ent_coeff=entcoeff ,tensorboard_log="./ppo_objhold_tensorboard/")


print("========================================")
print("Starting policy learning")
print("========================================")

# Train the agent
final_score = train_agent(job_name='.',
            agent=agent,
            seed=seed,
            niter=1000,
            gamma=gamma,
            gae_lambda=gae_lambda,
            policy_dir=model_dir,
            logs_dir=log_dir,
            sample_mode = "trajectories",
            num_traj=25,
            num_samples=2048,
            save_freq=50,
            evaluation_rollouts=10)

print("========================================")
print("Job Finished.")
print("========================================")
