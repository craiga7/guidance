"""
Environment base for missile guidance

Created by: Aaron Craig
Affiliation: University of Cincinnati
"""
import sys
import gymnasium as gym
import numpy as np
import math

# Local Modules for Env
sys.path.append('..')# one directory up
from dynamics.dubins_2D import Dubins2D
from control.PN_controller import pn_control
from viewers.dataPlotter import dataPlotter
from viewers.trajPlotter import trajPlotter

class BaseMSLenv(gym.Env):
    """ 
    Environment that feeds into agents. In accordance with Ray rllib agents are
    identified by string id's. 
    Created By: Aaron Craig
    """
    metadata = {"render.modes": ["3d","action"]}

    def __init__(self, params=None):
        # Initialize the parameters of agents and simulation space
        """
        config = {
            "agents": { 
                    "Targets": { 
                        "0":{
                        "state0": np.reshape(np.array([0.0, 5000.0, 0.0]),(3,1)),
                        "vel_T": 2.0,
                        "omega_T": 0,
                        "color": "k",
                        "type": "target",
                        },
                    },
                    "Missiles": {
                        "0": {
                            "state0": np.reshape(np.array([0.0, 0.0, 0.0]),(3,1)),
                            "vel_M": 2.5,
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
            "timestep": 1/40,                   # Time step for step call, Hz in denominator (sec) (float)
            "g": 9.806650,                      # Constant gravity (m/s^2) (float)
            "num_targets": 1,                   # Number of targets (int)
            "num_missiles": 1,                  # Number of missiles (int)
            "engagement_radius": 0.75,          # Crash Radius of missile (float)
            "end_time": 120,                    # Defines simulation end time (sec) (int)
            "engagement_time": 45.0,            # Defines drone response time (sec) (float)
            "dynamic_steps" : 4,                # Dynamic physic steps in step call (int)
            "obs_type": "LOS",                  # STA or LOS (States or LOS Measurements) (str)
            "act_type": "VIR",                  # ACC, VIR, POL, TGT (str)
                                                # -Acceleration: ACC
                                                # -Virtual Target: VIR
                                                # -Missile Reference: POL
                                                # -Target Reference Sphere: TGT
            }
        """
        # DEFINE CONSTANTS
        self.render_mode = params["render_mode"]
        self.render_frames = params["render_frames"]
        self.ts = params["timestep"]
        self.g = params["g"]

        # DEFINE EXPERIMENT PARAMETERS
        self.num_targets = params["num_targets"]
        self.num_missiles = params["num_missiles"]
        self.engagement_radius= params["engagement_radius"] 
        self.end_time = params["end_time"]
        self.engagement_time = params["engagement_time"]
        self.dynamic_steps = params["dynamic_steps"] 
        self.obs_type = params["obs_type"]
        self.act_type = params["act_type"]

        # DEFINE DRONE PARAMETERS
        self.agents = params["agents"]
        self.Targets = self.agents["Targets"]
        self.Missiles = self.agents["Missiles"]
        self.Virtuals = self.agents["Virtuals"]

        # DEFINE A WRAP FOR ANGLES
        self.wrap = lambda phases :(phases + np.pi) % (2 * np.pi) - np.pi

        # CREATE ACTION AND OBSERVATION SPACES
        self.action_space = self._actionSpace()
        self.observation_space = self._observationSpace()
        
        # ZERO OUT VARIABLES AND PARAMETERS IN INIT
        self._master_reset()


    def reset(self, *, seed=None, options=None):
        """Call super reset() method to set np_random with value of seed. 
        Inputs: seed, options
        
        Returns: 
        observation | dict
            initial observation of specific implementation of _computeObs()
        info | dict
            sensor information from vehicles in initial states
        """
        super().reset(seed=seed)

        # Reset variables and parameters of agents
        self._master_reset()

        # Render Reset
        self._render_reset()

        # Compute Returns from Reset: Initial observation and information
        self.info = self._computeInfo()
        self.observation = self._computeObs()
        self.info_0 = self.info
        self.observation_0 = self.observation
        # print("BaseMSLenv_reset_info",self.info)

        return self.observation, self.info

    def step(self, action):
        """Step that inputs action to Guidance controller
        Inputs:
        action | ndarray
            Current actions based on type (TGT or VIR)    
        
        Returns:
        observation | dict
            current observation of implementation of _computeObs()
        info | dict
            sensor information from vehicles in current states
        reward | float
            current reward of step with _computeReward()
        terminated | bool
            is current episode over for simulation
        truncated | bool
            are any conditions met that are violated (safety boundaries)
        """
        self.action_dict = action

        for i in range(self.dynamic_steps):
            
            # for agt in self.Targets.keys():
            target  = self.Targets["0"]["obj"]
            vel_T   = self.Targets["0"]["vel_T"] 
            omega_T = self.Targets["0"]["omega_T"]

            # Update states of Targets
            target.update(vel_T,omega_T)
            alpha_T = target._state[2][0]
            
            # for agt in self.Missiles.keys():
            missile     = self.Missiles["0"]["obj"]
            vel_M       = self.Missiles["0"]["vel_M"]
            PN_gain     = self.Missiles["0"]["PN_Gain"]
            x_M = missile._state[0][0]
            y_M = missile._state[1][0]
            alpha_M = missile._state[2][0]

            virtual     = self.Virtuals["0"]["obj"]
    
            parameters_T = np.array([vel_T,omega_T])   
            parameters_M = np.array([PN_gain,vel_M])       

            if self.act_type == "VIR_pos":
            # x_V, y_V
                x_V = action[0]
                y_V = action[1]
                psi_V = self.Targets["0"]["state0"][2][0]
                omega_V = omega_T

                state_vir = np.reshape(np.array([x_V, y_V, psi_V]),(3,1))
                vel_V = vel_T
                parameters_V = np.array([vel_V,omega_V]).T   
                virtual._state = state_vir
                acc_M = self._control(virtual._state,missile._state,parameters_V,parameters_M)  
            
            elif self.action_type == "VIR_obj":
                # vel_V, alpha_V
                virtual._update(vel_V,alpha_V)

            elif self.action_type == "VIR_opt":
                # alpha_V
                alpha_V = action[0]

                vel_V = vel_M*np.cos(alpha_M-theta_MV)/np.cos(alpha_V-theta_MV)
                alpha_dot_V = (alpha_V - self.alpha_V0)/self.ts
                alpha_V = alpha_dot_V*vel_V
                virtual._update(vel_V,alpha_V)
                parameters_V = np.array(vel_V,alpha_V)
                
                x_V = virtual._state[0][0]
                y_V = virtual._state[1][0]
                alpha_V=virtual._state[2][0]
                
                del_x = x_V - x_M
                del_y = y_V - y_M
                
                R_MV     = math.sqrt(del_x**2 + del_y**2)    # LOS distance (m)
                theta_MV = math.atan2(del_y,del_x)         # LOS angle (rad)

                V_theta = vel_T*math.sin(alpha_T-theta_MV)-vel_M*math.sin(alpha_M-theta_MV)
                thetadot = V_theta/R_MV
                acc_M = PN_gain*vel_M*thetadot               # commanded acceleration (m/s^2)

                theta_MV_dot = 1/R_MV*(vel_V*np.sin(alpha_V-theta_MV) - vel_M*np.sin(alpha_V-theta_MV))
                
                self.alpha_V0 = alpha_V
                acc_M = PN_gain*vel_M*theta_MV_dot
     
            elif self.act_type == "ACC":
                acc_M = action[0]

            elif self.act_type == "MSL":
                # phi_M, alpha_M
                rho_MV = 200
                phi_MV = action[0]
                alpha_V = action[1]
                vel_V = vel_M - 1
                x_V = missile._state[0][0]+rho_MV*np.cos(phi_MV)
                y_V = missile._state[1][0]+rho_MV*np.sin(phi_MV)
                state_vir = np.reshape(np.array([x_V, y_V, alpha_V]),(3,1))
                virtual._state = state_vir
                parameters_V = np.array([vel_M,alpha_V])
                acc_M = self._control(virtual._state,missile._state,parameters_V,parameters_M)  

            elif self.act_type == "MSLplus":
                rho_MV = action[0]
                phi_MV = action[1]
                alpha_V = action[2]
                vel_V = vel_M
                x_V = missile._state[0][0]+rho_MV*np.cos(phi_MV)
                y_V = missile._state[1][0]+rho_MV*np.sin(phi_MV)
                state_vir = np.reshape(np.array([x_V, y_V, alpha_V]),(3,1))
                virtual._state = state_vir
                parameters_V = np.array([vel_M,alpha_V])
                acc_M = self._control(virtual._state,missile._state,parameters_V,parameters_M)  

            elif self.act_type == "TGT":
                rho_TV = 100
                phi_TV = action[0]
                alpha_V = action[1]
                vel_M = vel_T
                x_V = target._state[0][0]+rho_TV*np.cos(phi_TV)
                y_V = target._state[1][0]+rho_TV*np.sin(phi_TV)
                state_vir = np.reshape(np.array([x_V, y_V, alpha_V]),(3,1))
                virtual._state = state_vir
                parameters_V = np.array([vel_M,alpha_V])
                acc_M = self._control(virtual._state,missile._state,parameters_V,parameters_M)  

            elif self.act_type == "TGTplus":
                rho_TV = action[0]
                phi_TV = action[1]
                alpha_V = action[2]
                vel_M = vel_T
                x_V = target._state[0][0]+rho_TV*np.cos(phi_TV)
                y_V = target._state[1][0]+rho_TV*np.sin(phi_TV)
                state_vir = np.reshape(np.array([x_V, y_V, alpha_V]),(3,1))
                virtual._state = state_vir
                parameters_V = np.array([vel_M,alpha_V])
                acc_M = self._control(virtual._state,missile._state,parameters_V,parameters_M)  

            elif self.act_type == "PNC":
                PN_gain_act = action[0]
                parameters_M = np.array([PN_gain_act,vel_M])      
                acc_M = self._control(target._state,missile._state,parameters_T,parameters_M)  

            # Update states of missiles
            missile.update(vel_M,acc_M)
            missile._state[2][0] = self.wrap(missile._state[2][0])

            self.sim_time += self.ts
            self.sim_steps += 1

            # Render Simulation
            if self.sim_steps % self.render_frames == 0 and self.render_mode != None:
                self._render_frame(action) 

            # Pass data into functions
            self.Acc_M = acc_M

        self.observation = self._computeObs()
        self.info = self._computeInfo()
        self.reward = self._computeReward()
        self.truncated = self._computeTruncated()
        self.terminated = self._computeTerminated()

        return self.observation, self.reward, self.terminated, self.truncated, self.info

    def _control(self,state_T,state_M,parameters_T,parameters_M):
        # Define the individual states
        # state_T:  x(m), y(m), psi(rad)
        x_T = state_T[0]
        y_T = state_T[1]
        psi_T=state_T[2]
        # state_M: x(m), y(m), psi(rad)
        x_M = state_M[0]
        y_M = state_M[1]
        psi_M=state_M[2]

        # Define the constants needed from parameters
        # parameters_T = [vel_T, omega_T]
        # parameters_M = [PN Gain, vel_M]
        vel_T  = parameters_T[0]
        N_gain = parameters_M[0]
        vel_M  = parameters_M[1]

        # Command Equations
        del_x = x_T - x_M
        del_y = y_T - y_M
        
        LOS_R     = math.sqrt(del_x**2 + del_y**2)    # LOS distance (m)
        theta_LOS = math.atan2(del_y,del_x)         # LOS angle (rad)

        V_R = vel_T*math.cos(psi_T-theta_LOS) - vel_M*math.cos(psi_M-theta_LOS)
        Rdot = V_R
        V_theta = vel_T*math.sin(psi_T-theta_LOS) - vel_M*math.sin(psi_M-theta_LOS)

        if LOS_R == 0:
            LOS_R = 1
        thetadot = V_theta/LOS_R
        acc_M = N_gain*vel_M*thetadot               # commanded acceleration (m/s^2)

        return acc_M

    def render(self):
        """
        External render definiton
        """
        self._render_frame()
    
    def _render_frame(self, action=None):
        """
        If rendering is enabled, update figure
        """

        if "2d" in self.render_mode:
            self.trajPlot.update(
                self.Missiles["0"]["obj"]._state[0], # xm
                self.Missiles["0"]["obj"]._state[1], # ym
                self.Targets["0"]["obj"]._state[0], # xt,
                self.Targets["0"]["obj"]._state[1], # yt
                self.Virtuals["0"]["obj"]._state[0], # xv
                self.Virtuals["0"]["obj"]._state[1], # yv
            )
        if "plot" in self.render_mode:
            info = self._computeInfo()
            self.dataPlot.update(
                self.sim_time,
                info["LOS_R"],
                info["LOS_theta"],
                info["Rdot"],
                info["thetadot"],
                info["acc_M"]
            )

    def _anim_agents(self):
        """
        constructs multi-agent dict for animation function to use
        """
        agents_ref = {
            "Targets": self.Targets,
            # "Vir_Targets": self.Vir_Targets,
            "Missiles": self.Missiles,
        }
        agents = {}

        for type in agents_ref.keys():
            for agt in agents_ref[type].keys():
                vic = agents_ref[type][agt]["obj"]
                agents |= {
                    str(agt):
                        {
                            "x": float(vic._state[0]),
                            "y": float(vic._state[1]),
                            "psi": float(vic._state[2]),
                            "color": agents_ref[type][agt]["color"]
                        } 
                }
                         
        return agents  
    
    def _render_reset(self):
        """
        Used to reset render modes
        """
        if self.render_mode == "2d" or self.render_mode == "plot2d": 
            self.trajPlot = trajPlotter(self.Missiles["0"]["PN_Gain"])
        if self.render_mode == "plot" or self.render_mode == "plot2d":
            self.dataPlot = dataPlotter(self.Missiles["0"]["PN_Gain"]) 

    def _master_reset(self):
        """
        Master Reset of all variables in learning and parameters of objects
        """
        # Initialize time and steps
        self.step_counter   = 0 # Used for logging later
        self.sim_time       = 0
        self.sim_steps      = 0
        self.action_dict    = []
        self.Acc_M          = 0

        # Create agent ids for dictionaries
        self._agent_ids = ["missile_" + str(r) for r in range(self.num_missiles)]
        self._target_ids = ["target_" + str(r) for r in range(self.num_targets)]

        for agt in self.Targets.keys():
            self.targetpos_low_x = 4500
            self.targetpos_high_x = 5000
            self.targetpos_low_y = 4500
            self.targetpos_high_y = 5000
            self.alpha_set_T = [ -3*np.pi/4, np.pi, 3*np.pi/4]
            prob = [0.3, 0.4, 0.3]
            if self.Targets[agt]["alpha_T"] == "RAND":
                alpha_T = np.random.choice(self.alpha_set_T,1,p=prob)
            else:
                alpha_T = self.Targets[agt]["state0"][2]
            if self.Targets[agt]["pos_rand"] == "RAND":
                pos_x_rand_t = np.random.randint(low=self.targetpos_low_x,high= self.targetpos_high_x,size=(1,1))
                pos_y_rand_t = np.random.randint(low=self.targetpos_low_y,high= self.targetpos_high_y,size=(1,1))
                pos_t = [pos_x_rand_t,pos_y_rand_t]
            else:
                pos_t = self.Targets[agt]["state0"][0:2]
            
            self.Targets[agt]["state0"][0:2] =  pos_t
            self.Targets[agt]["state0"][2] = alpha_T
            self.Targets[agt]["obj"] = Dubins2D(self.ts, self.Targets[agt]["state0"])

        for agt in self.Missiles.keys():
            self.missilepos_low = 0.0
            self.missilepos_high_x= 10.0
            self.missilepos_high_y= 10.0
            self.alpha_set_M = [ 0, np.pi/4, np.pi/2]
            prob = [0.3, 0.4, 0.3]
            if self.Missiles[agt]["alpha_M"] == "RAND":
                alpha_M = np.random.choice(self.alpha_set_M,1,p=prob)
            else:
                alpha_M = self.Missiles[agt]["state0"][2]
            if self.Missiles[agt]["pos_rand"] == "RAND":
                pos_x_randint = np.random.randint(low=self.missilepos_low,high= self.missilepos_high_x,size=(1,1))
                pos_y_randint = np.random.randint(low=self.missilepos_low,high= self.missilepos_high_y,size=(1,1))
                pos_m = [pos_x_randint,pos_y_randint]
            else:
                pos_m = self.Missiles[agt]["state0"][0:2]

            self.Missiles[agt]["state0"][0:2] = pos_m
            self.Missiles[agt]["state0"][2] = alpha_M
            self.Missiles[agt]["obj"] = Dubins2D(self.ts, self.Missiles[agt]["state0"])
    
        for agt in self.Virtuals.keys():
            if self.act_type == 'VIR':
                self.virtual_low = self.missilepos_low+500
                self.virtual_high = self.missilepos_high+500
                pos_randint = np.random.randint(low=self.virtual_low,high=self.virtual_high,size=(2,1))
            elif self.act_type == 'VIRopt':
                R_MV = self.Virtuals[agt]["dist"]
                state_M = self.Missiles["0"]["state0"][0:3]
                pos_randint = np.array([state_M[0]+np.sqrt(R_MV)/2, state_M[1]+np.sqrt(R_MV)/2])
                self.Virtuals[agt]["state0"][2] = 0
                self.Virtuals[agt]["vel_M"] = self.Missiles["0"]["vel_M"]
            elif self.act_type == 'MSL':
                self.virtual_low = self.missilepos_low+100
                self.virtual_high = self.missilepos_high+100
                pos_randint = np.random.randint(low=self.virtual_low,high=self.virtual_high,size=(2,1))
            elif self.act_type == 'TGT':
                pos_randint = np.random.randint(low=self.targetpos_low_x,high=self.targetpos_high_x,size=(2,1))
            else:
                pos_randint = np.random.randint(low=self.targetpos_low_x,high=self.targetpos_high_x,size=(2,1))
            self.Virtuals[agt]["state0"][0] = pos_randint[0]
            self.Virtuals[agt]["state0"][1] = pos_randint[1]
            self.alpha_set_V = [ 0, np.pi/4, np.pi/2]
            prob = [0.3, 0.4, 0.3]
            if self.Virtuals[agt]["alpha_V"] == "RAND":
                alpha_V = np.random.choice(self.omega_set_V,1,p=prob)
            else:
                alpha_V = self.Virtuals[agt]["alpha_V"]       
            self.Virtuals[agt]["state0"][2] = alpha_V    
            self.Virtuals[agt]["obj"] = Dubins2D(self.ts, self.Virtuals[agt]["state0"])
       
    def _VehicleStates(self, target,missile):
        """
        Returns a vector of the vehicles current states (3,1)

        Inputs:
        agents | obj
            current agent as object

        Returns:
        states | ndarray - Array of floats
            x, y, psi
        """
        # state_T:  x(m), y(m), psi(rad)
        target = target._state
        x_T     = target[0]
        y_T     = target[1]
        psi_T   = target[2]
        # state_M: x(m), y(m), psi(rad)
        missile = missile._state
        x_M     = missile[0]
        y_M     = missile[1]
        psi_M   = missile[2]

        states = np.concatenate([x_T,y_T,psi_T,x_M,y_M,psi_M])
        return states.T
    
    def _SensorStates(self,target,missile):
        """
        Returns the vector of sensor states

        Inputs:
        Target | obj
            must pass through vehicle states to flatten array
        Missile | obj
            must pass through vehicle states to flatten array

        Returns:
        sensors_states | dict - (2,1)(float):
            LOS_R,  LOS_theta
        """             
        LOS_R       = self._range_sensor(target._state,missile._state)
        LOS_theta   = self._LOSangle_sensor(target._state,missile._state)
        sensor_states = np.array([LOS_R, LOS_theta],dtype=np.float64)
        
        return sensor_states
    
    def _range_sensor(self, target, missile):
        """
        Inputs: 
         - target states: [x_tgt, y_tgt, psi_tgt]
         - missile states: [x_msl, y_msl, psi_msl]
        Output:
         - range from missile to target
        """
        # state_T:  x(m), y(m), psi(rad)
        x_T = target[0]
        y_T = target[1]
        # state_M: x(m), y(m), psi(rad)
        x_M = missile[0]
        y_M = missile[1]
        # Inputes to theta
        del_x = x_T - x_M
        del_y = y_T - y_M
        
        LOS_R = math.sqrt(del_x**2 + del_y**2)    # LOS distance (m)
               
        return LOS_R
    
    def _LOSangle_sensor(self, target, missile):
        """
        Inputs: 
         - target states: [x_tgt, y_tgt, psi_tgt]
         - missile states: [x_msl, y_msl, psi_msl]
        Output:
         - heading angle from missile to target
        """
        # state_T:  x(m), y(m), psi(rad)
        x_T = target[0]
        y_T = target[1]
        # state_M: x(m), y(m), psi(rad)
        x_M = missile[0]
        y_M = missile[1]
        # Inputes to theta
        del_x = x_T - x_M
        del_y = y_T - y_M

        LOS_theta = math.atan2(del_y,del_x)         # LOS angle (rad)

        return LOS_theta
    
    def _observationSpace(self):
        """
        Returns observation space of env, implement in single or multi agent class
        """
        raise NotImplementedError   

    def _actionSpace(self):
        """
        Returns action space of env, implement in single or multi agent class
        """
        raise NotImplementedError     

    def _filterActions(self, action):
        """
        Used to take action of any type (NED, Lat/Lon) 
        and make into missile specific control inputs

        Inputs:
        action | ndarry 

        Returns:
        action | ndarray (4,1)
            x_vtgt, y_vtgt, z_vtgt
        """
        raise NotImplementedError

    def _computeObs(self):
        """
        Computes current observation, implement in single or multi agent class
        """
        raise NotImplementedError
    
    def _computeReward(self):
        """
        Computes current reward, implement in single or multi agent class
        """
        raise NotImplementedError        
    
    def _computeTerminated(self):
        """
        Computes if done condition met, implement in single or multi agent class
        """
        raise NotImplementedError    
    
    def _computeInfo(self):
        """
        Computes information, implement in single or multi agent class
        """
        raise NotImplementedError 
    
    def _computeTruncated(self):
        """
        Computes if truncated condition met, implement in single or multi agent class
        """
        raise NotImplementedError 
    
