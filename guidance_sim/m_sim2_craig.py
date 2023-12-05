import sys
import numpy as np
import math
import matplotlib.pyplot as plt
sys.path.append('./guidance')
import parameters.simulation_parameters_craig as SIM
from dynamics.dubins_2D_craig import Dubins2D
from control.PN_controller import pn_control
from viewers.dataPlotter import dataPlotter
from viewers.trajPlotter import trajPlotter

Ts = SIM.ts_simulation

# instantiate initial states of vehicles
M_state=np.array([[0,0,0]],dtype=float).T       # missile_0 state x(m), y(m), psi(rad)
T_state=np.array([[0,5000,0]],dtype=float).T    # target_0 state  x(m), y(m), psi(rad)

# Constants for missile
vel_M   = 300     # missile velocity (m/s)
N_gain  = np.array([[5]]).T
N_color = np.array([[1, 0, 0],[0, 0, 1],[0, 1, 0],[1, 1, 0]])
collision_dist = 4 # (m)

# Constants for target
vel_T   = 200     # target velocity (m/s)
omega_T = 0       # heading rate of target (rad/s)

# main simulation loop
# Run the simulation for different N_gain values
for i in range(0,len(N_gain)):
    PN_gain = N_gain[i]
    print('PN_Gain',str(PN_gain[0]))

    # Make vehicles into objects
    missile = Dubins2D(Ts,M_state)    # input: time_step, state; output: missile object
    target = Dubins2D(Ts,T_state)     # input: time_step, state; output: target object

    # Initialize the missile(s) and target(s)
    # instantiate the simulation plots and animation
    t = SIM.start_time # time starts at t_start
    del_x = target._state[0,0]-missile._state[0,0]
    del_y = target._state[1,0]-missile._state[1,0]
    del_total = (del_x**2 + del_y**2)
    R_LOS = math.sqrt(del_total)    # LOS distance (m)

    dataPlot = dataPlotter(PN_gain[0])
    trajPlot = trajPlotter(PN_gain[0])

    while R_LOS > collision_dist :  
        # state_M: x(m), y(m), psi(rad)
        xm=missile._state[0,0]
        ym=missile._state[1,0]
        psim=missile._state[2,0]
        # state_T:  x(m), y(m), psi(rad)
        xt=target._state[0,0]
        yt=target._state[1,0]
        psit=target._state[2,0]
            
        parameters_T = np.array([[vel_T,omega_T]]).T      # parameters_T = [vel_T, omega_T]
        parameters_M = np.array([[PN_gain[0],vel_M]]).T   # parameters_M = [PN Gain, vel_M]
        control_out = pn_control.control(target._state,missile._state,parameters_T,parameters_M)

        # Use control outputs for vehicle updates
        # [vel_T,omega_T,vel_M,acc_M,R_LOS,theta_LOS,Rdot,thetadot].T
        vel_T = control_out[0]
        omega_T = control_out[1]
        vel_M = control_out[2]
        acc_M = control_out[3]
        R_LOS = control_out[4]
        theta_LOS = control_out[5]
        Rdot = control_out[6]
        thetadot = control_out[7]

        missile.update(vel_M,acc_M)
        target.update(vel_T,omega_T)
        dataPlot.update(t,R_LOS,theta_LOS,Rdot,thetadot,acc_M)

# TODO: Check states into trajPlot
        print('states',xm,ym,xt,yt)

        trajPlot.update(xm,ym,xt,yt)
        t=t+Ts
    plt.pause(3)

print('Press key to close')
plt.waitforbuttonpress()
 


