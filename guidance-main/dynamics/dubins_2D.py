import sys
sys.path.append('.')
import numpy as np
# load message types
from math import cos, sin, tan

class Dubins2D:
    def __init__(self, Ts, state0):
        self.ts_simulation = Ts
        # set initial states based on parameter file
        # _state is the 3x1 internal state of the vehicle that is propagated:
        # _state = [x, y, alpha]
        self._state=state0
        # self._state = np.array([[vehicle.x],      # (0)
        #                        [vehicle.y],       # (1)
        #                        [vehicle.alpha]])    # (2)
        
    ###################################
    # public functions
    def update(self, v, a_lat):
        '''
            Integrate the differential equations defining dynamics. 
            Inputs are the forces and moments on the vehicle.
            Ts is the time step between function calls.
        '''

        # Integrate ODE using Runge-Kutta RK4 algorithm
        time_step = self.ts_simulation
        k1 = self._derivatives(self._state, v, a_lat)
        k2 = self._derivatives(self._state + time_step/2.*k1, v, a_lat)
        k3 = self._derivatives(self._state + time_step/2.*k2, v, a_lat)
        k4 = self._derivatives(self._state + time_step*k3, v, a_lat)
        self._state += time_step/6 * (k1 + 2*k2 + 2*k3 + k4)
       

    ###################################
    # private functions
    def _derivatives(self, state, v, a_lat):
        """
        for the dynamics xdot = f(x, u), returns f(x, u)
        """
        # extract the states
        # _state = [x, y, alpha]
        # three states x, y, alpha
        x=state[0,0]
        y=state[1,0]
        alpha=state[2,0]

        # collect the derivative of the states
        if v == 0:
            x_dot = np.array([[v*np.cos(alpha), v*np.sin(alpha), 0]]).T
        else:
            x_dot = np.array([[v*np.cos(alpha), v*np.sin(alpha), a_lat/v]]).T
        #print(x_dot)
        return x_dot

    
