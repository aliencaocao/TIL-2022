import urllib3
from .camera import Camera

from .chassis import Chassis

class Robot:
    def __init__(self, host:str='localhost', port:int=5566):
        self.url = 'http://{}:{}'.format(host, port)
        self.manager = urllib3.PoolManager()

        self.chassis = Chassis(self)
        self.camera = Camera(self)

    def initialize(self, conn_type='ap', proto_type='udp'):
        '''Does nothing for mock.'''
        pass