import urllib3
import json
import base64
import logging 
import cv2
from typing import List, Any
from tilsdk.cv.types import DetectedObject
from tilsdk.localization.types import RealPose

class ReportingService:
    '''Communicates with reporting server to submit target reports.'''

    def __init__(self, host:str='localhost', port:int=5000):
        '''
        Parameters
        ----------
        host
            Hostname or IP address of reporting server.
        port
            Port number of reporting server.
        '''

        self.url = 'http://{}:{}'.format(host, port)
        self.manager = urllib3.PoolManager()

    def report(self, pose:RealPose, img:Any, targets:List[DetectedObject]):
        '''Report targets.

        Parameters
        ----------
        pose
            Robot pose where targets were seen.
        img
            OpenCV image from which targets were detected.
        targets
            Detected targets.
        '''
        
        _, encoded_img = cv2.imencode('.png',img)
        base64_img = base64.b64encode(encoded_img).decode("utf-8")

        response = self.manager.request(method='POST',
                                        url=self.url+'/report',
                                        headers={'Content-Type': 'application/json'},
                                        body=json.dumps({
                                            'pose': pose._asdict(),
                                            'image': base64_img,
                                            'targets': [{
                                                'id': t.id,
                                                'cls': t.cls,
                                                'bbox': {
                                                    'x': t.bbox[0],
                                                    'y': t.bbox[1],
                                                    'w': t.bbox[2],
                                                    'h': t.bbox[3],
                                                }
                                            } for t in targets]
                                        }))

        return response

    def start_run(self):
        response = self.manager.request(method='GET',
                                        url=self.url+'/start_run')