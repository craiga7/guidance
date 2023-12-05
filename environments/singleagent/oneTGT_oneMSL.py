"""
Environment that trains one Missile to Engage one Target

Created by: Aaron Craig
Affiliation: University of Cincinnati
"""
import sys
import numpy as np
import math 

# Local Modules for Env
sys.path.append('./guidance')# one directory up
from control.PN_controller import pn_control
from environments.singleagentMSLenv import BaseSingleAgentMSL

class oneMSL(BaseSingleAgentMSL):
    """
    Single missile agent RL to engage at target position
    """

    def __init__(self, params):
        """
        Initialize the environment for single agent

        Uses BaseSingleAgentMSL class
        Inputs:
        params | dict
            dictionary of inputs, see BaseSingleAgentMSL for details
        obs | ObservationType
            (Sensor or States)
        act | ActionType
            TGT or VIR
        """
        super().__init__(params=params)

    def _computeReward(self):
        """
        Define a penalty until the missile is within target radius
        
        Return:
        reward | float
        """
        # Obtain LOS Range from SensorStates
        # sensor = self._SensorStates(self.Targets["0"]["obj"],self.Missiles["0"]["obj"])
        sensor = self.observation

        # Use information directory states
        info_plus = self.info
        info_0 = self.info_0

        # Current States
        R_LOS = sensor[0]

        # Initial States
        R_LOS_0 = info_0["R_LOS"]

        """
        # define coefficients
        # Arguement used in Ben Asher Book reversed for distance penalty increasing with time
        # k_dist = np.exp(self.sim_time - self.engagement_time) 
# TODO: gains on 01NOV23
        k_dist = 1e-2
        k_cont = 1e-2
        k_rdot = 1e-2
        k_head = 1e-2
        k_thetadot = 1e-2
        k_engage = 10

        # Initialize penalties
        dist_pen = 0
        time_pen = 0
        cont_pen = 0
        
        # Initialize advantage
        rdot_adv = 0
        collide_adv = 0

        # Conditions for advantages (cost)
        if sensor[0] < self.engagement_radius:
            collide_adv = k_engage
        if sensor_plus["Rdot"] < 0:
            rdot_adv = k_rdot

        # Conditions for penalties
        dist_pen        = -k_dist*(R_LOS/R_LOS_0)
        cont_pen        = -k_cont*(np.abs(sensor_plus["Acc_M"])/100) 
        head_pen        = -k_head*np.abs(sensor_plus["theta_LOS"]-sensor_plus["Missile_0"][0])/(2*np.pi)

        # Sum of all costs
        penalty = dist_pen  + cont_pen 
        reward  = collide_adv + rdot_adv
        cost = reward + penalty
        """
        
        
        # Initialize rewards
        pen_dist = 0
        adv_dist = 0
        pen_time = 0
        pen_FOV  = 0
        pen_cont = 0
        adv_cold = 0
        
        # Define coefficients
        k_dist  = 1e-2
        k_cont  = 1e-4
        k_time  = 5e-4
        k_cold  = 1
        k_FOV   = 1e-3

        # Conditions for terminal rewards 
        if R_LOS <= self.engagement_radius:
            adv_cold = k_cold
            adv_dist = (1/sensor[0])      
        if self.sim_time >= (self.end_time-self.ts):
            pen_dist = -k_dist*(R_LOS/R_LOS_0)
        # if self.sim_time >= (self.end_time-self.ts) and np.abs(sensor[1] - info_plus["Missile_0"][2]) > np.pi/2:
        #     pen_FOV = -k_FOV
        #     pen_dist = -k_dist*(R_LOS/R_LOS_0)

        # Conditions for shaping rewards
        if R_LOS > self.engagement_radius:
            pen_cont = -k_cont*(info_plus["Acc_M"]/100)**2 
            pen_time = -k_time*(self.ts)

            if np.abs(sensor[1] - info_plus["Missile_0"][2]) > np.pi/2:
                pen_FOV = -k_FOV

        if sensor[0] < self.engagement_radius or self.sim_time > self.end_time:
            print("oneTGT_oneMSL_adv_cold",adv_cold)
            print("oneTGT_oneMSL_adv_dist",adv_dist)
            print("oneTGT_oneMSL_pen_dist",pen_dist)
            print("oneTGT_oneMSL_pen_cont",pen_cont)
            print("oneTGT_oneMSL_pen_time",pen_time)
            print("oneTGT_oneMSL_pen_FOV",pen_FOV)

        # Sum of all rewards (cost)
        r_term = adv_cold + adv_dist + pen_dist + pen_FOV
        r_shape = pen_cont + pen_time 
        cost = r_term + r_shape

        return cost
    
    def _computeTruncated(self):
        """
        Define the truncated condition(s)

        Return:
        truncated | boolean
        """
        self.truncated = False
        sensor = self.observation
        if sensor[0] < self.engagement_radius:
            self.truncated = True
            
        return self.truncated
    
    def _computeTerminated(self):
        """
        Define the terminated condition(s)

        Return:
        terminated | boolean
        """
        self.terminated = False

        if self.sim_time >= self.end_time:
            self.terminated = True
        elif self.truncated == True:
            self.terminated = True

        return self.terminated
        
    def _computeObs(self):
        """
        Return current observation from environment passes to reset and step
        Inputs:
        target | ndarray
            Current target states
        missile | ndarray
            Current missile states

        Return:
        obs | dict - Dict of agt states or other when type (sensors,) is defined
        """
        obs_dict = {}

        # States (6,1): x_tgt, y_tgt, psi_tgt, x_msl, y_msl, psi_msl
        if self.observation_type == "STA":
            # for agt in self._agent_ids:
            obs_dict = self._VehicleStates(self.Targets["0"]["obj"],self.Missiles["0"]["obj"])
        
        # LOS Sensor (5,1): LOS_R, LOS_theta, R_dot, Theta_dot, Acc_M
        elif self.observation_type == "LOS":
            # for agt in self._agent_ids:
            obs_dict = np.array([self.info["R_LOS"],
                        self.info["theta_LOS"],
                        self.info["Rdot"],
                        self.info["thetadot"],
                        self.info["Acc_M"],
            ])

        # COM (10 , 1): LOS_R, LOS_theta, R_dot, Theta_dot, x_tgt, y_tgt, psi_tgt, x_msl, y_msl, psi_msl
        elif self.observation_type == "COM":
            obs_dict = np.array([self.info["R_LOS"],
                        self.info["theta_LOS"],
                        self.info["Rdot"],
                        self.info["thetadot"],
                        self.info["Acc_M"],
                        self.info["Target_0"][0],
                        self.info["Target_0"][1],
                        self.info["Target_0"][2],
                        self.info["Missile_0"][0],
                        self.info["Missile_0"][1],
                        self.info["Missile_0"][2]
            ])

        # FUL (10,1): x_tgt, y_tgt, psi_tgt, x_msl, y_msl, alpha_msl, vel_tgt, alpha_tgt, pn_msl, vel_msl
        elif self.observation_type == "FUL":
            obs_MSL = self.Missiles["0"]["obj"]._state
            obs_TGT = self.Targets["0"]["obj"]._state            

            # for agt in self.Targets.keys():
            vel_T   = self.Targets["0"]["vel_T"] 
            omega_T = self.Targets["0"]["omega_T"]
            
            # for agt in self.Missiles.keys():
            vel_M       = self.Missiles["0"]["vel_M"]
            PN_gain     = self.Missiles["0"]["PN_Gain"]
            # print(obs_TGT.T[0],obs_MSL.T[0],[vel_T],[alpha_T],[PN_gain],[vel_M])
            obs_dict = np.concatenate((obs_TGT.T[0],
                                       obs_MSL.T[0],
                                       [vel_T],
                                       [omega_T],
                                       [PN_gain],
                                       [vel_M]),
                                       axis=0
                                    )   
        
        return obs_dict
    
    def _computeInfo(self):
        """
        Define the information for reward penalties

        Return:
        info | dict
            "Target_0": states_target,
            "Missile_0": states_missile,
            "Virtual_0": states_virtual,
            "Vel_T": vel_T,
            "Omega_T": omega_T, 
            "Vel_M": vel_M,
            "Acc_M": acc_M,
            "R_LOS": R_LOS, 
            "theta_LOS": theta_LOS,
            "Rdot": Rdot, 
            "thetadot": thetadot, 
        """
        info_dict = {}
        Acc_M = self.Acc_M
        target = self.Targets["0"]["obj"]
        missile= self.Missiles["0"]["obj"]
        virtual= self.Virtuals["0"]["obj"]
        vel_T   = self.Targets["0"]["vel_T"] 
        omega_T = self.Targets["0"]["omega_T"]
        vel_M       = self.Missiles["0"]["vel_M"]
        PN_gain     = self.Missiles["0"]["PN_Gain"]
        parameters_T = np.array([[vel_T,omega_T]]).T   
        parameters_M = np.array([[PN_gain,vel_M]]).T  
        control_out = pn_control.control(target._state,missile._state,parameters_T,parameters_M)

        vel_T = control_out[0]
        omega_T = control_out[1]
        vel_M = control_out[2]
        acc_M = control_out[3] # Not accurate acc_M
        R_LOS = control_out[4]
        theta_LOS = control_out[5]
        Rdot = control_out[6]
        thetadot = control_out[7]

        states_target = np.array([target._state[0][0],target._state[1][0],target._state[2][0]])
        states_missile = np.array([missile._state[0][0],missile._state[1][0],missile._state[2][0]])
        state_virtual = np.array([virtual._state[0][0],virtual._state[1][0],virtual._state[2][0]])
        if self.action_dict == []:
            states_action = np.zeros(shape=self.action_space.shape)

        states_action = self.action_dict

        info_dict = {
            "Action_0": states_action,
            "Target_0": states_target,
            "Missile_0": states_missile,
            "Virtual_0": state_virtual,
            "Vel_T": vel_T,
            "omega_T": omega_T, 
            "Vel_M": vel_M,
            "Acc_M": Acc_M,
            "R_LOS": R_LOS, 
            "theta_LOS": theta_LOS,
            "Rdot": Rdot, 
            "thetadot": thetadot, 
        }
        return info_dict
    
