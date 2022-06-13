import urllib3
from .types import *
import json
import base64
import matplotlib.pyplot as plt
import io
from typing import List, Tuple
import logging

class LocalizationService:
    '''Communicates with localization server to obtain map, pose and clues.
    '''

    def __init__(self, host:str='localhost', port:int=5566):
        '''
        Parameters
        ----------
        host : str
            Hostname or IP address of localization server.
        port: int
            Port number of localization server.
        '''
        self.url = 'http://{}:{}'.format(host, port)
        self.manager = urllib3.PoolManager()

    def get_map(self) -> SignedDistanceGrid:
        '''Get map as occupancy grid.
        
        Returns
        -------
        grid : SignedDistanceGrid
            Signed distance grid.
        '''
        response = self.manager.request(method='GET',
                                        url=self.url+'/map')

        data = json.loads(response.data)

        grid = base64.decodebytes(data['map']['grid'].encode('utf-8'))

        img = plt.imread(io.BytesIO(grid))
        grid = SignedDistanceGrid.from_image(img, data['map']['scale'])

        return grid

    def get_pose(self) -> Tuple[RealPose, List[Clue]]:
        '''Get real-world pose of robot.
        
        Returns
        -------
        pose : RealPose
            Pose of robot.
        clues : List[Clue]
            Clues available at robot's location.
        '''

        response = self.manager.request(method='GET',
                                        url=self.url+'/pose')

        if response.status != 200:
            logging.getLogger('Localization Service').debug('Could not get pose.')
            return None, None

        data = json.loads(response.data)

        pose = RealPose(
            x=data['pose']['x'],
            y=data['pose']['y'],
            z=data['pose']['z']
        )
        clues = []
        
        for clue in data['clues']:
            clue_id = clue['clue_id']
            location = RealLocation(clue['location']['x'], clue['location']['y'])
            audio = base64.decodebytes(clue['audio'].encode('utf-8'))
            clues.append(Clue(clue_id, location, audio))

        return pose, clues