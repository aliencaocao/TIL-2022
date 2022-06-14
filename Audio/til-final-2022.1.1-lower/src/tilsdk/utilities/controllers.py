import time
import numpy as np
from numpy.typing import ArrayLike

class PIDController:
    '''PID Controller Implementation'''

    def __init__(self, Kp:ArrayLike, Kd:ArrayLike, Ki:ArrayLike, state:ArrayLike=None, t:float=time.perf_counter()):
        '''
        Parameters
        ----------
        Kp : ArrayLike
            P-gain values. Same dimension as state.
        Kd : ArrayLike
            D-gain values. Same dimension as state.
        Ki : ArrayLike
            I-gain values. Same dimension as state.
        state : ArrayLike
            Initial state.
        t : float
            Initial time.
        '''
        self.Kp = np.array(Kp)
        self.Kd = np.array(Kd)
        self.Ki = np.array(Ki)

        self.state = state

        if state is not None:
            self.d_state = np.zeros(state.shape)
            self.i_state = np.zeros(state.shape)
            self.prev_time = t
        else:
            self.d_state = None
            self.i_state = None
            self.prev_time = None

    def update(self, state:ArrayLike, t:float=time.perf_counter()) -> ArrayLike:
        '''Update the controller with new state.

        Parameters
        ----------
        state : ArrayLike
            State update.
        t : float
            Time associated with state update.
        
        Returns
        -------
        output : ArrayLike
            Controller output.
        '''
        state = np.array(state)

        if self.state is not None:
            dt = t - self.prev_time
            if dt > 1e-10:
                self.d_state = (state - self.state)/dt
                self.i_state += dt*(state - self.state)/2
            else:
                # too little time passed, assume no change
                self.d_state = np.zeros(state.shape)
                self.i_state += np.zeros(state.shape)
        else:
            # initialize
            self.d_state = np.zeros(state.shape)
            self.i_state = np.zeros(state.shape)

        self.state = state
        self.prev_time = t

        return self.Kp*self.state + self.Kd*self.d_state + self.Ki*self.i_state

    def reset(self):
        '''Reset the controller.
        
        Control gains are preserved.
        '''
        self.state = None
        self.d_state = None
        self.i_state = None
        self.prev_time = None