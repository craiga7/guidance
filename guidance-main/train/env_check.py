#!/usr/bin/env python
# encoding: utf-8
import os
import sys
import numpy as np
import gymnasium as gym
from ray.rllib.utils.pre_checks import env as e
from ray.tune.registry import register_env
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
# Local Modules for Env
sys.path.append('../guidance')# one directory up
from environments.singleagent.oneTGT_oneMSL import oneMSL

# /home/acraig/guidance/environments/oneTGT_oneMSL.py
# /home/risclab/guidance/environments/singleagent
# /home/risclab/guidance/environments/singleagentMSLenv.py

params = {
        "agents": { 
                    "Targets": { 
                        "0":{
                        "state0": np.reshape(np.array([0.0, 5000.0, 0.0]),(3,1)),
                        "vel_T": 686,       # m/s for mach 2.0
                        "omega_T": 0,
                        "color": "k",
                        "type": "target",
                        },
                    },
                    "Missiles": {
                        "0": {
                            "state0": np.reshape(np.array([0.0, 0.0, 0.0]),(3,1)),
                            "vel_M": 857.5,     # m/s for mach 2.5
                            "PN_Gain": 3,                       # PN Gain set to 3 default
                            "color": "b",
                            "type": "missile",
                        }
                    },
                    "Virtuals": {
                        "0": {
                            "state0": np.reshape(np.array([0.0, 0.0, 0.0]),(3,1)),
                            "vel_M": 2.5,
                            "PN_Gain": 3,                       # PN Gain set to 3 default
                            "color": "g",
                            "type": "virtual",
                        }
                    },
            },
            "render_mode": None,                # Used to start a render (str)
            "render_frames": 1,                 # Defines render frames (int)
            "timestep": 1/100,                  # Time step for step call, Hz in denominator (sec) (float)
            "g": 9.806650,                      # Constant gravity (m/s^2) (float)
            "num_targets": 1,                   # Number of targets (int)
            "num_missiles": 1,                  # Number of missiles (int)
            "engagement_radius": 10.0,          # Crash Radius of a missile (float)
            "end_time": 120,                    # Defines simulation end time (sec) (int)
            "engagement_time": 45.0,            # Defines drone response time (sec) (float)
            "dynamic_steps" : 1,                # Dynamic physic steps in step call (int)
            "obs_type": "LOS",                  # STA or LOS (States or LOS Measurements) (str)
            "act_type": "VIR",                  # ACC, VIR, POL, TGT (str)
                                                # -Acceleration: ACC
                                                # -Virtual Target: VIR
                                                # -Missile Reference: POL
                                                # -Target Reference Sphere: TGT
            }     

test = oneMSL(params)
print(test.reset())

def env_creator(params):
    return oneMSL(params)

myenv = env_creator(params)
register_env("my_env",env_creator)

# Test iterations
n_iter = 100

for i in range(n_iter):

    action_dict =  np.array([685,-10])
        
    observation, reward, terminated, truncated, info = myenv.step(action_dict)

    print('Observation',observation)
    print('Reward',reward)
    print('Info',info)