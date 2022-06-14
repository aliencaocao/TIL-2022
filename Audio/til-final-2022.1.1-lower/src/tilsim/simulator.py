import argparse
from collections import namedtuple
import yaml
import base64
import logging
import time
from pathlib import Path
from threading import Lock, Thread
import cv2

import flask
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import default_rng
from matplotlib.colors import to_rgba
from werkzeug.serving import WSGIRequestHandler

from tilsdk.localization import *

map_log_level = {
    'debug': logging.DEBUG,
    'info': logging.INFO,
    'warn': logging.WARNING,
    'warning': logging.WARNING,
    'error': logging.ERROR,
    'critical': logging.CRITICAL
}

Rot = lambda t: np.array([[np.cos(t), -np.sin(t)],
                          [np.sin(t), np.cos(t)]], dtype=float)

# Flask defaults to HTTP 1.0. Use HTTP 1.1 to keep connections alive for high freqeuncy requests.
WSGIRequestHandler.protocol_version = 'HTTP/1.1'

##### Flask server #####
app = flask.Flask(__name__)

@app.route('/map', methods=['GET'])
def get_map():
    f = open(config.map_file, 'rb')
    grid = f.read()
    f.close()
    
    return {
        'map': {
            'scale': config.map_scale,
            'grid': base64.encodebytes(grid).decode('utf-8')
        }
    }

@app.route('/pose', methods=['GET'])
def get_pose():
    global robot

    real_pose = robot.pose
    pose = robot.noisy_pose if config.use_noisy_pose else real_pose
    clues = []

    for clue in config.clues:
        if np.linalg.norm(real_pose[:2] - (clue['trigger']['x'], clue['trigger']['y'])) < clue['trigger']['r']:
            audio_bytes = Path(clue['audio_file']).read_bytes()
            
            clues.append({
                'clue_id': clue['clue_id'],
                'location': clue['location'],
                'audio': base64.encodebytes(audio_bytes).decode('utf-8')
            })

    return {
        'pose': {
            'x': float(pose[0]),
            'y': float(pose[1]),
            'z': float(pose[2])
        },
        'clues': clues
    }

@app.route('/cmd_vel', methods=['POST'])
def post_cmd_vel():
    global robot

    data = flask.request.get_json(silent=True)

    if not data:
        logging.getLogger('/cmd_vel').warning('Unknown request, ignoring...')
        return 'Bad request.', 400

    vel = (data['vel']['x'], data['vel']['y'], data['vel']['z'])
    robot.vel = vel

    logging.getLogger('/cmd_vel').info('Velocity set to: {}'.format(vel))

    return 'OK'

@app.route('/report', methods=['POST'])
def post_report():
    data =  flask.request.get_json()
    num_targets = len(data['targets'])
    logging.getLogger('/report').info('Report received. {} targets.'.format(num_targets))
    
    return 'OK'

@app.route('/camera', methods=['GET'])
def get_camera():
    global config, robot

    logging.getLogger('/camera').info('Camera image requested.')

    for target in config.targets:
        loc = RealLocation(target['trigger']['x'], target['trigger']['y'])
        if euclidean_distance(robot.pose, loc) <= target['trigger']['r']:
            img = cv2.imread(target['image_file'], cv2.IMREAD_COLOR)
            img = cv2.resize(img, (1280, 720))
            buf = img.tobytes()
            return buf, 200

    return np.zeros((1280, 720, 3), dtype=np.uint8).tobytes(), 200

def start_server():
    global config
    app.run(host=config.host, port=config.port)

##### Robot classes #####

class SimRobot:
    '''Simulated robot.'''

    def __init__(self, pose=(0,0,0), vel=(0,0,0), timeout:float=0.5):
        '''
        Parameters
        ----------
        pose
            Initial pose.
        vel
            Initial velocity.
        '''
        self._pose = np.array(pose, dtype=float)
        self._pose_lock = Lock()
        self._vel = np.array(vel, dtype=float)
        self._last_changed = time.time()
        self._vel_lock = Lock()
        self.timeout = timeout

        self.rng = default_rng()

    def step(self, dt:float) -> None:
        '''Step the simulation.
        
        Parameters
        ----------
        dt : float
            Time since last simulation step.
        '''
        with self._pose_lock, self._vel_lock:
            vel = np.array([*(Rot(np.radians(self._pose[2]))@self._vel[:2]), self._vel[2]])
            self._pose += vel*dt

    @property
    def pose(self):
        with self._pose_lock:
            return self._pose

    @pose.setter
    def pose(self, value):
        with self._pose_lock:
            self._pose = np.array(value, dtype=float)
                
    @property
    def vel(self):
        with self._vel_lock:
            return self._vel

    @vel.setter
    def vel(self, value):
        with self._vel_lock:
            self._vel = np.array(value, dtype=float)
            self._last_changed = time.perf_counter()

    @property
    def last_changed(self) -> float:
        with self._vel_lock:
            return self._last_changed

    @property
    def noisy_pose(self):
        self._pose_lock.acquire()
        pose = self._pose
        self._pose_lock.release()

        # back out front and rear tag positions
        angle = np.radians(pose[2])
        half_dir_vec = config.robot_phy_length/2*np.array((np.cos(angle), np.sin(angle)))
        front = pose[:2] + half_dir_vec
        back = pose[:2] - half_dir_vec
        
        # add noise
        front += self.rng.normal(0, config.position_noise_stddev, size=2)
        back += self.rng.normal(0, config.position_noise_stddev, size=2)
        
        noisy_dir_vec = front-back
        noisy_angle = np.degrees(np.arctan2(noisy_dir_vec[1], noisy_dir_vec[0]))
        return  np.array([*((front+back)/2), noisy_angle])


class ActualRobot:
    '''Passthrough for actual robot.
    
    Uses pose information from a localization service
    instance and does not perform simulation.
    '''
    def __init__(self, host:str='localhost', port:int=5567):
        '''
        Parameters
        ----------
        host : str
            Localization service host.
        port : int
            Localization service port.
        '''
        self.loc_service = LocalizationService(host, port)
        self._pose = np.array((0,0,0))
        self._pose_lock = Lock()
        self.step(0) # initialize pose

    def step(self, dt:float) -> None:
        '''Step the simulation.
        
        For ActualRobot this gets latest pose from localization service.

        Parameters
        ----------
        dt : float
            Time since last simulation step.
        '''
        pose, _ = self.loc_service.get_pose()
        if pose:
            with self._pose_lock:
                self._pose = np.array(pose)

    @property
    def pose(self):
        with self._pose_lock:
            return self._pose

##### Visualization #####

def draw_robot(ax, refs=None, draw_noisy=False):
    '''Draw robot on given axes.
    
    Parameters
    ----------
    refs
        Matplotlib refs to previously draw robot.
    draw_noisy : bool
        Draw robot with simulated noise.

    Returns
    -------
    new_refs
        Matplotlib refs to drawn robot.
    '''
    global robot, config

    pose = robot.pose
    grid_loc = real_to_grid_exact(pose[:2], config.map_scale)
    angle = np.radians(pose[2])

    if refs:
        for ref in refs:
            ref.remove()

    new_refs = []
    # draw actual robot
    circle = mpatches.Circle(grid_loc, radius=config.robot_radius, color='red')
    new_refs.append(ax.add_artist(circle))
    arrow = mpatches.Arrow(*grid_loc, config.robot_radius*np.cos(angle), config.robot_radius*np.sin(angle), width=config.robot_radius/2, color='blue')
    new_refs.append(ax.add_artist(arrow))

    # draw noisy robot
    if draw_noisy:
        pose_noisy = robot.noisy_pose
        grid_loc_noisy = real_to_grid_exact(pose_noisy[:2], config.map_scale)
        angle_noisy = np.radians(pose_noisy[2])

        circle_noisy = mpatches.Circle(grid_loc_noisy, radius=config.robot_radius, color=to_rgba('green', alpha=0.3))
        new_refs.append(ax.add_artist(circle_noisy))
        arrow_noisy = mpatches.Arrow(*grid_loc, config.robot_radius*np.cos(angle_noisy), config.robot_radius*np.sin(angle_noisy), width=config.robot_radius/2, color=to_rgba('blue', alpha=0.3))
        new_refs.append(ax.add_artist(arrow_noisy))

    return new_refs

def draw_clues(ax):
    global config

    for clue in config.clues:
        trigger_loc = RealLocation(clue['trigger']['x'], clue['trigger']['y'])
        r = clue['trigger']['r']

        trigger_loc = real_to_grid_exact(trigger_loc, config.map_scale)
        r /= config.map_scale

        circle = mpatches.Circle(trigger_loc, radius=r, color=to_rgba('yellow', alpha=0.2))
        ax.add_artist(circle)

        dest_loc = RealLocation(clue['location']['x'], clue['location']['y'])
        dest_loc = real_to_grid_exact(dest_loc, config.map_scale)

        ax.scatter(dest_loc[0], dest_loc[1], marker='x', color='yellow')

        ax.annotate(clue['clue_id'], trigger_loc, color='yellow')
        ax.annotate(clue['clue_id'], dest_loc, color='yellow')


def draw_targets(ax):
    global config

    for target in config.targets:
        logging.getLogger('draw_targets').debug(target)
        loc = RealLocation(target['trigger']['x'], target['trigger']['y'])
        r = target['trigger']['r']

        loc = real_to_grid_exact(loc, config.map_scale)
        r /= config.map_scale

        circle = mpatches.Circle(loc, radius=r, color=to_rgba('green', alpha=0.2))
        ax.add_artist(circle)
        ax.annotate(target['target_id'], loc, color='green')


def main():
    ##### Parse Args #####
    parser = argparse.ArgumentParser(description='Robot simulator server for TIL2022 Robotics Challenge.')
    
    grp_map = parser.add_argument_group('Map data (Required)')
    grp_map.add_argument('-mf', '--map_file', metavar='file', type=str, required=False, help='Map image filename.')
    grp_map.add_argument('-ms', '--map_scale', metavar='scale', type=float, required=False, help='Map scale, i.e. ratio of real-world unit to grid/px unit.')

    grp_sim = parser.add_argument_group('Simulation configuration')
    grp_sim.add_argument('-s', '--start_pose', metavar=('x', 'y', 'z'), nargs=3, type=float, required=False, help='Start pose of robot in real-world units. Ignored if proxying pose. (Default: 0.0 0.0 0.0)')
    grp_sim.add_argument('-n', '--use_noisy_pose', action='store_true', help='Simulate localization noise. Ignored if proxying pose.')
    grp_sim.add_argument('-nl', '--robot_phy_length', metavar='length', type=float, required=False, help='Physical length of robot for position noise simulation. Ignored if proxying pose or use_noisy_pose if not set. (Default: 0.32)')
    grp_sim.add_argument('-ns', '--pos_noise_stddev', metavar='s', type=float, required=False, help='Standard deviation of position noise. Ignored if proxying pose or use_noisy_pose is not set. Default: 0.05')

    grp_disp = parser.add_argument_group('Visualization display configuration')
    grp_disp.add_argument('-r', '--robot_radius', metavar='radius', type=float, required=False, help='Radius of marker for robot visualization in px. (Default: 10)')

    grp_net = parser.add_argument_group('Network configuration')
    grp_net.add_argument('-i', '--host', metavar='host', type=str, required=False, help='Server hostname or IP address. (Default: 0.0.0.0)')
    grp_net.add_argument('-p', '--port', metavar='port', type=int, required=False, help='Server port number. (Default: 5566)')

    grp_proxy = parser.add_argument_group('Pose proxy configuration', description='Allow passthrough of robot pose from a localization server.')
    grp_proxy.add_argument('-q', '--proxy_real_robot', action='store_true', help='Proxy real robot pose.')
    grp_proxy.add_argument('-qi', '--proxy_host', metavar='host', type=str, required=False, help='Localization server hostname or IP address. (Default: "localhost")')
    grp_proxy.add_argument('-qp', '--proxy_port', metavar='port', type=int, required=False, help='Localization server port number. (Default: 5567)')

    grp_log = parser.add_argument_group('Logging configuration')
    grp_log.add_argument('-ll', '--log', dest='log_level', metavar='level', type=str, required=False, help='Logging level. Default: "info"')

    grp_conf = parser.add_argument_group('Configuration file')
    grp_conf.add_argument('-c', '--config', metavar='config', type=str, required=False, help='Config YAML file. If provided config file supersedes command line options.')

    args = parser.parse_args()

    ##### Set Config #####
    global config

    # default
    config = {
        'host': '0.0.0.0',
        'port': 5566,
        'robot_radius': 10,
        'start_pose': (2.0, 2.0, 0.0),
        'use_noisy_pose': True,
        'robot_phy_length': 0.32,
        'pos_noise_stddev': 0.05,
        'proxy_real_robot': False,
        'proxy_host': 'localhost',
        'proxy_port': 5567,
        'log_level': 'info',
        'clues': [],
        'targets': [],
    }

    if args.config:
        # load yaml first
        with open(args.config, 'r') as f:
            config_ = yaml.safe_load(f)
            config.update(config_)
            
            # handle pose specially to make a tuple
            if 'start_pose' in config_:
                config['start_pose'] = (config['start_pose']['x'], config['start_pose']['y'], config['start_pose']['z'])

    # update with args
    for key, value in vars(args).items():
        if (value is not None) and (key not in config.keys()):
            config[key] = value

    config = namedtuple('Config', config.keys())(*config.values())

    ##### Setup logging #####
    logging.basicConfig(level=map_log_level[config.log_level],
                    format='[%(levelname)5s][%(asctime)s][%(name)s]: %(message)s',
                    datefmt='%H:%M:%S')


    ##### Setup robot #####
    global robot
    if config.proxy_real_robot:
        robot = ActualRobot(config.proxy_host, config.proxy_port)
    else:
        robot = SimRobot(pose=config.start_pose)

    ##### Setup visualization #####
    fig, ax = plt.subplots()
    map_img = plt.imread(config.map_file)
    plt.imshow(map_img)
    draw_clues(ax)
    draw_targets(ax)
    draw_refs = draw_robot(ax)
    plt.ion()
    plt.draw()

    ##### Setup server #####
    server_thread = Thread(target=start_server, daemon=True)
    server_thread.start()

    ##### Main loop #####
    before = time.perf_counter()

    while True:
        now = time.perf_counter()
       
        if not config.proxy_real_robot:
            # safety timeout
            if now - robot.last_changed >= robot.timeout:
                robot.vel = (0., 0., 0.)

        robot.step(now-before)
        before = now

        draw_refs = draw_robot(ax, draw_refs, config.use_noisy_pose)

        plt.draw()
        plt.pause(0.01)


if __name__ == '__main__':
    main()