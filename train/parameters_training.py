import numpy as np

params = {
        "agents": { 
                    "Targets": { 
                        "0":{
                            "state0": np.array([[5000.0], [5000.0], [0.0]]),
                            "pos_rand": "RAND",
                            "vel_T": 50.0,       # m/s 
                            "alpha_T": "RAND",      # initial angle random (Default: "RAND")
                            "omega_T": 0,
                            "color": "k",
                            "type": "target",
                        },
                    },
                    "Missiles": {
                        "0": {
                            "state0": np.array([[0.0], [0.0], [np.pi/2]]),
                            "pos_rand": "RAND",
                            "vel_M": 200.5,     # m/s 
                            "alpha_M": "RAND",                 # inital heading random (Default: "RAND")
                            "PN_Gain": 3,                       # PN Gain (Default: 3) for non PNC/ACC actions
                            "color": "b",
                            "type": "missile",
                        }
                    },
                    "Virtuals": {
                        "0": {
                            "state0": np.reshape(np.array([0.0, 0.0, 0.0]),(3,1)),
                            "vel_M": 200.0,
                            "alpha_V": 0,                 # inital heading
                            "PN_Gain": 3,                       # PN Gain set to 3 default
                            "color": "g",
                            "type": "virtual",
                            "dist": 500,
                        }
                    },
            },
            "render_mode": None,                # Used to start a render (str)
            "render_frames": 1,                 # Defines render frames (int)
            "timestep": 1/100,                  # Time step for step call, Hz in denominator (sec) (float)
            "g": 9.806650,                      # Constant gravity (m/s^2) (float)
            "num_targets": 1,                   # Number of targets (int)
            "num_missiles": 1,                  # Number of missiles (int)
            "engagement_radius": 50.0,         # Crash Radius of a missile (float)
            "end_time": 40.0,                   # Defines simulation end time (sec) (int)
            "engagement_time": 30.0,            # Defines drone response time (sec) (float)
            "dynamic_steps" : 1,                # Dynamic physic steps in step call (int)
            "obs_type": "LOS",                  # STA, LOS, COM, FUL (str)
                                                # STA: States 
                                                # LOS: Hardware Measurements
                                                # COM: Combined LOS and STA
                                                # FUL: All information passed to PN Controller
            "act_type": "ACC",                  # ACC, VIR, POL, TGT, PNC(str)
                                                # -Acceleration: ACC
                                                # -Virtual Target with position: VIR_pos
                                                # -Virtual Target vehicle: VIR_vic
                                                # -Virtual Target with rdot = 0: VIR_opt
                                                # -Missile Reference: MSL
                                                # -Missile Reference with rho: MSLplus
                                                # -Target Reference Sphere: TGT
                                                # -Target Reference Sphere with distance: TGTplus
                                                # -PN Gain: PNC
            "n_iter": 500,                      # Training iterations
            "num_rollout_workers": 20,           # Rollout workers for policy (Default: 2)
            "train_batch_size": 20000,           # Training batch size (Default: 4000)
            "lr": 1e-4,                          # learning rate (Default: 5e-5) 
            "_disable_preprocessor_api": False, # preprocessing api's (Default: False)
            "num_gpus": 1,                       # total GPUs (Default: 0)
            "use_critic": True,                 # Critic (Default: True) must be true for updates
            "use_gae": True,                    # Generalized Advantage Estimator (Default: True) with a value function
            "lambda": 0.95,                      # The GAE (lambda) parameter (Default: 1.0) 
            "use_kl_loss": True,                # KL-term in the loss function (Default: True)
            "kl_coeff": 0.5,                     # Alpha: Initial coefficient for KL divergence (Default: 0.2)
            "kl_target": 0.01,                  # Target value for KL divergence (Default: 0.01)
            "sgd_minibatch_size": 700,           # Total SGD batch size (Default: 128) across all devices for SGD. This defines the minibatch size within each epoch
            "num_sgd_iter": 20,                  # Epochs: Number of SGD iterations (Default: 30) in each outer loop (i.e., number of epochs to execute per train batch).
            "shuffle_sequences": True,          # Whether to shuffle sequences (Default: True) in the batch when training
            "vf_loss_coeff": 1.0,               # Coefficient of the value function loss (Default: 1.0). IMPORTANT: you must tune this if you set vf_share_layers=True inside your model’s config
            "entropy_coeff": 0.01,               # Beta: Coefficient of the entropy regularizer. (Default: 0.0)
            "entropy_coeff_schedule": None,     # Decay schedule for the entropy regularizer. (Default: None)
            "clip_param": 0.1,                  # The PPO clip parameter. (Default: 0.3)
            "vf_clip_param": 10.0,              # Clip param for the value function. (Default: 10.0) Note that this is sensitive to the scale of the rewards. If your expected V is large, increase this.
            "grad_clip": None,                   # If specified (Default: None), clip the global norm of gradients by this amount

            "gamma": 0.995,                      # gamma discount factor of MDP (Default: 0.9)
            "model": {
                "fcnet_hiddens": [64, 64],      # neural layers (Default: [256,256])
                "fcnet_activation": "tanh",     # activation function for all layers (Default: tanh)
                "vf_share_layers": False,       # Value Function Sharing (Default: False) if True must tune "vf_loss_coeff"  
            }
            }  
