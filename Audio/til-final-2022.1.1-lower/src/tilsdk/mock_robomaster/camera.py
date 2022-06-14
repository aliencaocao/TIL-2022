import numpy as np

class Camera:
    '''Mock robomaster camera.'''

    def __init__(self, robot):
        self.url = robot.url
        self.manager = robot.manager
        self._is_initialized = False

    def read_cv2_image(self, timeout:float=3, strategy:str='pipeline'):
        '''Read image from robot camera.
        
        For mock, gets image from simulator.

        Parameters
        ----------
        timeout
            Timeout value.

        strategy
            Image acquisition strategy. For challenge, 'newest' should be used.

        Returns
        -------
        img : ndarray
            cv2 image.
        '''
        if not self._is_initialized:
            raise Exception('Camera stream not started.')

        response = self.manager.request(method='GET',
                                        url=self.url+'/camera')

        img = np.frombuffer(response.data, np.uint8)
        img = img.reshape((720, 1280, 3))
        
        return img

    def start_video_stream(self, display:bool=True, resolution='720p'):
        self._is_initialized = True