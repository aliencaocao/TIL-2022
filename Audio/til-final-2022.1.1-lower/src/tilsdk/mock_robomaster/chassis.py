import json

class Chassis:
    def __init__(self, robot):
        self.url = robot.url
        self.manager = robot.manager

    def drive_speed(self, x:float=0.0, y:float=0.0, z:float=0.0):
        '''Command robot to drive at given velocity.
        
        Parameters
        ----------
        x : float
            Forward velocity in m/s.
        y : float
            Rightwards velocity in m/s.
        z : float
            Clockwise angular velocity in deg/s.
        '''
        response = self.manager.request(method='POST',
                                        url=self.url+'/cmd_vel',
                                        headers={'Content-Type': 'application/json'},
                                        body=json.dumps({
                                            'vel': {
                                                'x': x,   # arena frame
                                                'y': -y,  # arena frame
                                                'z': -z   # arena frame
                                            }
                                        }))