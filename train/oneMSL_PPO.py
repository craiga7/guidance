"""
Training one missile engage one target
- Both have constant velocity
- Target has constant heading

Created by: Aaron Craig
Affiliation: University of Cincinnati
"""
import os
import sys
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
import ray
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPOConfig
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Local Modules for Env
sys.path.append('..')# one directory up
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)# one directory up
os.environ['PYTHONPATH'] = '..' # must point to path with custom gym env (where quadenv.py is located)
from environments.singleagent.oneTGT_oneMSL import oneMSL
from parameters_training import params

# Used training parameters
n_iter = params["n_iter"]
num_rollout_workers = params["num_rollout_workers"]
num_gpus = params["num_gpus"]
train_batch_size = params["train_batch_size"]
lr = params["lr"]
lambda_ = params["lambda"]
kl_coeff = params["kl_coeff"]
sgd_minibatch_size = params["sgd_minibatch_size"]
num_sgd_iter = params["num_sgd_iter"]
entropy_coeff = params["entropy_coeff"]
clip_param = params["clip_param"]
gamma = params["gamma"]
model = params["model"]


# RLlib ALGORITHM CONFIGURATION
config = (
    PPOConfig()
    .rollouts(num_rollout_workers=num_rollout_workers)
    .resources(num_gpus=num_gpus)
    .training(
        train_batch_size=train_batch_size,
        lr=lr,
        lambda_=lambda_,
        kl_coeff=kl_coeff,
        sgd_minibatch_size=sgd_minibatch_size,
        num_sgd_iter=num_sgd_iter,
        entropy_coeff=entropy_coeff,
        clip_param=clip_param,
        gamma=gamma,
        model=model,
    )
    .environment(env=oneMSL,
                    env_config=params,
                    disable_env_checking=False,
                    render_env=False)
)
algo = config.build()

# STUFF TO PLOT TRAINING
fig = plt.figure(1)
plt.title("Learning Progress")
plt.ylabel("Mean Reward")
plt.xlabel("Episode")
reward_data = []
i_data = []
init = True
print("\n")

# TRAINING LOOPS
for i in range(n_iter):
    result = algo.train()

    # save after first iter
    if i == 0:
            checkpoint_dir = algo.save()
            print(f"Checkpoint saved in directory {checkpoint_dir}")
    # print a few important metrics from result dict
    for param in ['training_iteration', 'episode_reward_max', 'episode_reward_mean', 'episode_reward_min', 'episode_len_mean', 'time_this_iter_s']:
        print(str(param)+": " + str(result[param]))
    # save every x iterations
    if i % 10 == 0:
        checkpoint_dir = algo.save()
        print(f"Checkpoint saved in directory {checkpoint_dir}")
        plt.savefig(checkpoint_dir+"/reward_graph.png")
    
    # update reward plot
    reward_data.append(result["episode_reward_mean"])
    i_data.append(i+1)
    plt.plot(i_data,reward_data,'r.-')
    plt.plot([i+1]*len(result["hist_stats"]["episode_reward"]), result["hist_stats"]["episode_reward"], 'g*', markersize=1)
    plt.pause(0.001)
    print("\n")

print("Finished Training")
checkpoint_dir = algo.save()
print(f"Checkpoint saved in directory {checkpoint_dir}")
plt.savefig(checkpoint_dir+"/reward_graph.png")
plt.show()
