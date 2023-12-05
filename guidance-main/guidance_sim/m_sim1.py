import sys
sys.path.append('.')
import numpy as np
import parameters.simulation_parameters as SIM

import matplotlib.pyplot as plt

from dynamics.dubins_2D import Dubins2D
from viewers.dataPlotter import dataPlotter
from viewers.trajPlotter import trajPlotter
Ts=0.1
# instantiate pendulum, controller, and reference classes
m_state=np.array([[0,0,0]],dtype=float).T # missile state
t_state=np.array([[0,100,0]],dtype=float).T # target

missile=Dubins2D(Ts,m_state)
target=Dubins2D(Ts,t_state)
dataPlot = dataPlotter()
trajPlot=trajPlotter()
vm=20.0
vt=5.0
am_lat=0.0
at_lat=0.0

# instantiate the simulation plots and animation

t = SIM.start_time # time starts at t_start
rho=400
while rho > 4 :  # main simulation loop
    
    xm=missile._state[0,0]
    ym=missile._state[1,0]
    psim=missile._state[2,0]

    xt=target._state[0,0]
    yt=target._state[1,0]
    psit=target._state[2,0]

    rho= np.sqrt((xt-xm)**2+(yt-ym)**2)
    theta=np.arctan2(yt-ym,xt-xm)
    rho_dot=vt*np.cos(psit-theta)-vm*np.cos(psim-theta)
    theta_dot=(vt*np.sin(psit-theta)-vm*np.sin(psim-theta))/rho
    am_lat=3*vm*theta_dot

    missile.update(vm,am_lat)
    #print('rho=',rho)
    target.update(vt,at_lat)
    #print('thetadot=',theta)
    dataPlot.update(t,rho,theta,rho_dot,theta_dot,am_lat)
    trajPlot.update(xm,ym,xt,yt)
    t=t+Ts
    plt.pause(0.0001)
print('Press key to close')
plt.waitforbuttonpress()
plt.close()    


