"""
Environment that trains one Missile 

Created by: Aaron Craig
Affiliation: University of Cincinnati
"""
import sys
import numpy as np
from gymnasium import spaces
from enum import Enum

# Local Modules for Env
sys.path.append('./guidance')# one directory up
from environments.BaseMSLenv import BaseMSLenv

class BaseSingleAgentMSL(BaseMSLenv):
    """
    Base singe MSL environment for RL (PPO). Can be used in Multi-Agent Environment call.
    """

    def __init__(
            self, 
            params
            ):
        """
        Initialize the singe agent for RL environment. 
        
        Parameters:
        stuff | int - Set to 1 for single agent
        """
        self.observation_type = params["obs_type"]
        self.action_type = params["act_type"]

        # Initialize env back to parent class using super().__init(stuff)
        super().__init__(params=params)

    def _actionSpace(self):
        """
        Return action spaces of environment from config type

        Returns:
        action | ndarray - Action space defined from config type
        ACC, VIR, MSL, POL, TGT (str)
             -Acceleration: ACC
             -Virtual Target: VIR
             -Missile States: MSL
             -Polar Virtual: POL
             -Target Reference Sphere: TGT
        """
        if self.action_type == "ACC":
            # acc_M
            action_spaces = spaces.Box(
                low=np.array([-100]),
                high=np.array([100]),
                shape=(1,),
                dtype=np.float64
            ) 

        elif self.action_type == "VIR_pos":
            # x_V, y_V
            action_spaces = spaces.Box(
                low=np.array([5000, 4000]),
                high=np.array([15000, 6000]),
                shape=(2,),
                dtype=np.float64
            )    
        
        elif self.action_type == "VIR_vic":
            # vel_V, omega_V
            action_spaces = spaces.Box(
                low=np.array([self.Targets["0"]["vel_T"], -100]),
                high=np.array([self.Missiles["0"]["vel_M"], 100]),
                shape=(2,),
                dtype=np.float64
            )            

        elif self.action_type == "VIR_opt":
            # psi_V
            action_spaces = spaces.Box(
                low=np.array([-np.pi/4]),
                high=np.array([np.pi/4]),
                shape=(1,),
                dtype=np.float64
            )        
            
        elif self.action_type == "MSL":
            # phi_M, psi_M
            action_spaces = spaces.Box(
                low=np.array([-np.pi/4, -np.pi/4]),
                high=np.array([np.pi/4,  np.pi/4]),
                shape=(2,),
                dtype=np.float64
            )

        elif self.action_type == "MSLplus":
            # rho_M, phi_M, psi_M
            action_spaces = spaces.Box(
                low=np.array([200, -np.pi/2, -np.pi/4]),
                high=np.array([500, np.pi/2, np.pi/4]),
                shape=(3,),
                dtype=np.float64
            )

        elif self.action_type == "TGT":
            # phi_T, psi_M
            action_spaces = spaces.Box(
                low=np.array([-np.pi, -np.pi]),
                high=np.array([np.pi, np.pi]),
                shape=(2,),
                dtype=np.float64
            )

        elif self.action_type == "TGTplus":
            # rho_T, phi_T, psi_M
            action_spaces = spaces.Box(
                low=np.array([1, -np.pi, -np.pi]),
                high=np.array([100, np.pi, np.pi]),
                shape=(3,),
                dtype=np.float64
            )

        elif self.action_type == "PNC":
            # PN Gain
            # action_spaces = spaces.Discrete(n=5,start=1)
            action_spaces = spaces.Box(
                low=np.array([1]),
                high=np.array([5]),
                shape=(1,),
                dtype=np.int0
            )

        else:
            print("Action space is not defined correctly") 
            
        return action_spaces
    
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
    
    def _observationSpace(self):
        """
        Uses the base UAV in order to return obs

        Returns:
        obs | dict - Box of (states,) or other when type (sensors,) is defined

        States (6,1):
            x_tgt, y_tgt, psi_tgt, x_msl, y_msl, psi_msl

        LOS Sensor (5,1):
            LOS_R, LOS_theta, R_dot, Theta_dot, acc_M

        COM (11 , 1)
            LOS_R, LOS_theta, R_dot, Theta_dot, acc_M, x_tgt, y_tgt, psi_tgt, x_msl, y_msl, psi_msl

        FUL (10,1)
            x_tgt, y_tgt, psi_tgt, x_msl, y_msl, psi_msl, vel_tgt, omega_tgt, pn_msl, vel_msl
        """
        if self.observation_type == "STA":
            obs = spaces.Box(
                low=np.array([-np.inf, -np.inf,-np.pi, -np.inf, -np.inf, -np.pi]),
                high=np.array([np.inf, np.inf, np.pi,  np.inf,  np.inf,  np.pi]),
                shape=(6,),
                dtype=np.float64
            )
        elif self.observation_type == "LOS":
            obs =  spaces.Box(
                low=np.array([0.0, -np.pi, -np.inf, -np.inf, -np.inf]),
                high=np.array([np.inf,  np.pi, np.inf, np.inf, np.inf]),
                shape=(5,),
                dtype=np.float64
            )
        elif self.observation_type == "COM":
            obs =  spaces.Box(
                low=np.array([0.0, -np.pi, -np.inf, -2*np.pi, -np.inf, -np.inf, -np.inf, -np.pi,-np.inf, -np.inf, -np.pi]),
                high=np.array([np.inf,  np.pi, np.inf, 2*np.pi, np.inf, np.inf,  np.inf,  np.pi,np.inf,  np.inf,  np.pi]),
                shape=(11,),
                dtype=np.float64
            )
        elif self.observation_type == "FUL":
            obs =  spaces.Box(
                low=np.array([0.0, -np.inf, -np.pi, -np.inf, -np.inf, -np.pi, 0.0, -np.pi, 0.0, 0.0]),
                high=np.array([np.inf, np.inf, np.pi, np.inf, np.inf, np.pi, np.inf, np.pi, 10.0, np.inf]),
                shape=(10,),
                dtype=np.float64
            )
        else: 
            return print("Error in Observation Space type")

        return obs



