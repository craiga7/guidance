"""
Results of one missile engage one target
- Both have constant velocity
- Target has constant heading

Created by: Aaron Craig
Affiliation: University of Cincinnati
"""
import os
import sys
import datetime
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
os.environ['PYTHONPATH'] = '..' # must point to path with custom gym env (where msl.py is located)
from environments.singleagent.oneTGT_oneMSL import oneMSL
from parameters_training import params
    
"""
Trained Runs:
# Acceleration
"/home/risclab/ray_results/PPO_oneMSL_2023-11-29_16-11-06xugt6qrb/checkpoint_000500"
/home/risclab/ray_results/PPO_oneMSL_2023-12-05_10-57-26cbadccvf/checkpoint_000361
/home/risclab/ray_results/PPO_oneMSL_2023-12-05_11-15-107v16lwel/checkpoint_000311
/home/risclab/ray_results/PPO_oneMSL_2023-12-05_11-52-40yhpfrh0t/checkpoint_000500
/home/risclab/ray_results/PPO_oneMSL_2023-12-05_12-59-10yvz0kp2j/checkpoint_000500
/home/risclab/ray_results/PPO_oneMSL_2023-12-05_13-38-297_ki1l_i/checkpoint_000500

# PNC
/home/risclab/ray_results/PPO_oneMSL_2023-11-30_11-42-28bit5i__n/checkpoint_000500
/home/risclab/ray_results/PPO_oneMSL_2023-11-30_14-25-49d12lr8wi/checkpoint_000500
/home/risclab/ray_results/PPO_oneMSL_2023-11-30_14-58-00hf8as9y0/checkpoint_000750
 /home/risclab/ray_results/PPO_oneMSL_2023-12-05_10-05-17j7fnzzvi/checkpoint_000500
"""

# CHANGE THIS PARAMETERS AS DESIRED
savevideo = False
if savevideo: episodes = 1
else: episodes = 10

# paths for restore
controller = "PN_act"
path = "/home/risclab/ray_results/PPO_oneMSL_2023-12-05_13-38-297_ki1l_i/checkpoint_000500" 
path_fig = "/home/risclab/ray_results/PPO_oneMSL_05DEC23"
act_type = params["act_type"]
obs_type = params["obs_type"]

# Used training parameters
n_iter = params["n_iter"]
num_rollout_workers = params["num_rollout_workers"]
num_gpus = params["num_gpus"]
train_batch_size = params["train_batch_size"]
lr = params["lr"]
lambda_ = params["lambda"]
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

# restore algorithm from checkpoint
algo.restore(path)

env = gym.make("oneMSL-v0",params=params)

test_case = "\n Iter: "+str(n_iter)+"; # MSL: 1; # TGT: 1; Act: " + act_type + "; Obs: " + obs_type

fig7, (ax70,ax71) = plt.subplots(2,1,layout="constrained")
fig7.suptitle("Vehicle Collisions "+controller)
ax70.set_xlabel("episodes")
ax70.set_ylabel("Collision")
ax71.set_ylabel("Acceleration Total (m*rad/s^2)")
collisions = []
acc_total = []
episodes_t = []

# decision loop
for episode in range(1,episodes+1):
    time_now = datetime.datetime.now()
    time_cur = str(time_now.timestamp())
    i = 0
    obs_dict, info_dict = env.reset()
    terminated = False 

    score = 0
    if savevideo: plt.savefig(os.path.dirname(__file__) + "/gifmaker/frame_" + str(i).zfill(5) + ".png")
    
    # Stuff to plot the training
    fig2, (ax1,ax2) = plt.subplots(2,1,layout="constrained")
    fig2.suptitle("Vehicle Positions "+ test_case)
    ax2.set_xlabel("time (sec)")
    ax1.set_ylabel("Pos x (m)")
    ax2.set_ylabel("Pos y (m)")
    states_target_0_x = []
    states_target_0_y = []
    states_missile_0_x = []
    states_missile_0_y = []
    states_virtgt_0_x = []
    states_virtgt_0_y = []

    fig3, (ax4,ax5,ax6) = plt.subplots(3,1,layout="constrained")
    fig3.suptitle("Vehicle Attitudes "+ test_case)
    ax5.set_xlabel("time (sec)")
    ax5.set_ylabel("Lat_Acc")
    ax4.set_ylabel("Head (rad)")
    ax6.set_ylabel("Velocity (m/s)")
    states_target_0_head = []
    states_target_0_vel =[]
    states_missile_0_head = []
    states_missile_0_vel = []
    lat_acc_missile_0 = []
    states_virtual_0_vel = []
    states_virtual_0_head = []
    
    fig4, (ax7,ax8,ax9) = plt.subplots(3,1,layout="constrained")
    fig4.suptitle("Vehicle Ranges "+ test_case)
    ax9.grid()
    ax8.grid()
    ax7.grid()
    ax9.set_xlabel("time (sec)")
    ax7.set_ylabel("X Dist (m)")
    ax8.set_ylabel("Y Dist (m)")
    ax9.set_ylabel("Range (m)")
    error_missile_0_px = []
    error_missile_0_py = []
    range_missile_0 = []

    fig5, (ax10) = plt.subplots(1,1,layout="constrained")
    fig5.suptitle("Vehicle Real Space Positions "+ str(n_iter) + test_case)
    ax10.grid()
    ax10.set_xlabel("x (m)")
    ax10.set_ylabel("y (m)")

    fig6, (ax11) = plt.subplots(1,1,layout="constrained")
    fig6.suptitle("Actions, iter: "+ str(n_iter) + test_case)
    ax11.grid()
    ax11.set_xlabel("time (sec)")
    actions = []

    time_data = []
    
    while not terminated:
        i += 1
        action = algo.compute_single_action(obs_dict)
        # action = 3

        obs_dict, reward, terminated, truncated, info_dict = env.step(action)

        if savevideo: plt.savefig(os.path.dirname(__file__) + "/gifmaker/frame_" + str(i).zfill(5) + ".png")
        score += reward

        if not truncated:
            # Update State Information for Plotting
            time_data.append((i+1)*params['timestep'])

            actions.append(action)

            error_missile_0_px.append(info_dict["Target_0"][0] - info_dict["Missile_0"][0])
            error_missile_0_py.append(info_dict["Target_0"][1] - info_dict["Missile_0"][1])
            range_missile_0.append(info_dict["R_LOS"])

            states_target_0_head.append(info_dict["Target_0"][2])
            states_target_0_vel.append(info_dict["Vel_T"])
            states_missile_0_head.append(info_dict["Missile_0"][2])
            states_missile_0_vel.append(info_dict["Vel_M"])
            lat_acc_missile_0.append(info_dict["Acc_M"])
            states_virtual_0_vel.append(info_dict["Vel_M"])
            states_virtual_0_head.append(info_dict["Virtual_0"][2])

            states_target_0_x.append(info_dict["Target_0"][0])
            states_target_0_y.append(info_dict["Target_0"][1])
            states_missile_0_x.append(info_dict["Missile_0"][0])
            states_missile_0_y.append(info_dict["Missile_0"][1])
            states_virtgt_0_x.append(info_dict["Virtual_0"][0])
            states_virtgt_0_y.append(info_dict["Virtual_0"][1])

    # Save Vehicle State Figure
    ax10.plot(
        states_target_0_x,
        states_target_0_y,
        'k--',label='Target_0'
    )
    ax10.plot(
        states_missile_0_x,
        states_missile_0_y,
        'r--',label='Missile_0'
    )
    # ax10.plot(
    #     states_virtgt_0_x,
    #     states_virtgt_0_y,
    #     'b.',label='Virtual_0'
    #     )
    # ax10.set_xlim([0,6000])
    # ax10.set_ylim([0,8000])
    fig5.legend(loc="outside lower center", ncols=2, fontsize=14)
    fig5.savefig(path_fig+"/Vehicle_Positions_Real_"+act_type+obs_type+time_cur+".png")

    # Save Action State Figure
    if params["act_type"] == "ACC":
        ax11.plot(
            time_data,
            actions,
            'r-',label='Acc_M'
        )
    elif params["act_type"] == "PNC":
        ax11.plot(
            time_data,
            actions,
            'r.',label='PN_Gain'
        )
    fig6.legend(loc="outside lower center", ncols=2, fontsize=14)
    fig6.savefig(path_fig+"/Actions_"+act_type+obs_type+time_cur+".png")

    ax7.plot(time_data,error_missile_0_px,'r--',label='Missile_0')
    ax8.plot(time_data,error_missile_0_py,'r--')
    ax9.plot(time_data,range_missile_0,'r--')
    fig4.legend(loc="outside lower center", ncols=2, fontsize=14)
    fig4.savefig(path_fig+"/Vehicle_Ranges_"+act_type+obs_type+time_cur+".png")

    ax4.plot(time_data,states_target_0_head,'k--',label='Target_0')
    ax4.plot(time_data,states_missile_0_head,'r--',label='Missile_0')
    # ax4.plot(time_data,states_virtual_0_head,'b--',label='Virtual_0')
    ax5.plot(time_data,lat_acc_missile_0,'r-')
    ax6.plot(time_data,states_target_0_vel,'k-')
    ax6.plot(time_data,states_missile_0_vel,'r-')
    # ax6.plot(time_data,states_virtual_0_vel,'b-')
    # ax5.set_ylim([-100,100])
    fig3.legend(loc="outside lower center", ncols=3,fontsize=14)
    fig3.savefig(path_fig+"/Vehicle_Attitude_"+act_type+obs_type+time_cur+".png")

    ax1.plot(time_data,states_target_0_x,'k--',label='Target_0')
    ax1.plot(time_data,states_missile_0_x,'r--',label='Missile_0')
    # ax1.plot(time_data,states_virtgt_0_x,'b--',label='Virtual_0')
    ax2.plot(time_data,states_target_0_y,'k--')
    ax2.plot(time_data,states_missile_0_y,'r--')
    # ax2.plot(time_data,states_virtgt_0_y,'b--')
    # ax1.set_ylim([0,20000])
    # ax2.set_ylim([0,8000])
    fig2.legend(loc="outside lower center", ncols=3,fontsize=14)
    fig2.savefig(path_fig+"/Vehicle_Position_"+act_type+obs_type+time_cur+".png")

    # Save Animation Figure
    fig1 = plt.figure(1)
    fig1.savefig(path_fig+"/Animation_"+act_type+obs_type+time_cur+".png")

    Acc_T = np.sum(np.abs(lat_acc_missile_0))

    print(f"Episode {episode}, Score: {score}, Trunc: {truncated}, Term: {terminated}, Total_Acc_M: {Acc_T}")

    plt.close(1)
    plt.close(2)
    plt.close(3)
    plt.close(4)
    plt.close(5)
    plt.close(6)

    episodes_t.append(episode)
    acc_total.append(Acc_T)
    if truncated==True:
        collisions.append(1)
    else:
        collisions.append(0)

print("Total Collisions: ",np.sum(collisions)," Average Acceleration: ", np.average(acc_total))

ax70.plot(episodes_t,collisions,'k-')
ax71.plot(episodes_t,acc_total,'r-')
# ax1.set_ylim([0,20000])
# ax2.set_ylim([0,8000])
# fig7.legend(loc="outside lower center", ncols=3,fontsize=14)
fig7.savefig(path_fig+"/Coll_accel "+controller+obs_type+time_cur+".png")
