import sys
sys.path.append('.')
import numpy as np
import math

### This file acts as a Pure Proportional Navigation Guidance Law 
# Code here is written in reference to Melody Mayle and paper from Ashwini Ratnoo

class pn_control:
    def __init__(self,state_T,state_M,parameters_T,parameters_M):
        nothing = 0

    def control(state_T,state_M,parameters_T,parameters_M):
    # Define the individual states
        # state_T:  x(m), y(m), psi(rad)
        x_T = state_T[0]
        y_T = state_T[1]
        alpha_T=state_T[2]
        # state_M: x(m), y(m), psi(rad)
        x_M = state_M[0]
        y_M = state_M[1]
        alpha_M=state_M[2]

        # Define the constants needed from parameters
        # parameters_T = [vel_T, omega_T]
        # parameters_M = [PN Gain, vel_M]
        vel_T  = parameters_T[0]
        omega_T= parameters_T[1]
        N_gain = parameters_M[0]
        vel_M  = parameters_M[1]

    # Command Equations
        del_x = x_T - x_M
        del_y = y_T - y_M
        
        LOS_R     = math.sqrt(del_x**2 + del_y**2)    # LOS distance (m)
        theta_LOS = math.atan2(del_y,del_x)         # LOS angle (rad)

        V_R = vel_T*math.cos(alpha_T-theta_LOS) - vel_M*math.cos(alpha_M-theta_LOS)
        Rdot = V_R
        V_theta = vel_T*math.sin(alpha_T-theta_LOS)-vel_M*math.sin(alpha_M-theta_LOS)
        thetadot = V_theta/LOS_R
        acc_M = N_gain*vel_M*thetadot               # commanded acceleration (m/s^2)

    # True PN, lateral acceleration of missile is normal to velocity of missile
    # Outputs
    #   [vel_T,omega_T,vel_M,acc_M,LOS_R,theta_LOS,Rdot,thetadot].T

        u = np.array([vel_T,omega_T,vel_M,acc_M]).T
        xr= np.array([[LOS_R,theta_LOS,Rdot[0],thetadot[0]]])

    # Assemble the control, states, and conditional criteria
        control_out = np.concatenate((u,xr),axis=None)
        return control_out