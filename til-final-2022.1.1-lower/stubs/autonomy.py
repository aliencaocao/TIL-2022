import subprocess


def prep_env():
    subprocess.check_output(['pip', 'install', './pyastar2d'])
    subprocess.check_output(['pip', 'install', 'mmcv_full-1.5.0-cp38-cp38-manylinux1_x86_64.whl'])
    subprocess.check_output(['pip', 'install', './UniverseNet'])
    subprocess.check_output(['pip', 'install', '-U', 'torch-1.11.0+cu115-cp38-cp38-linux_x86_64.whl'])
    subprocess.check_output(['pip', 'install', '-U', 'torchaudio-0.11.0+cu115-cp38-cp38-linux_x86_64.whl'])
    subprocess.check_output(['pip', 'install', '-U', 'torchvision-0.12.0+cu115-cp38-cp38-linux_x86_64.whl'])


def clean_up_env():
    subprocess.check_output(['pip', 'uninstall', 'pyastar2d'])
    subprocess.check_output(['pip', 'uninstall', 'mmcv-full'])
    subprocess.check_output(['pip', 'uninstall', 'mmdet'])


# prep_env()  # only for laptop

import time
import logging
from typing import List

from tilsdk import *  # import the SDK
from tilsdk.utilities import PIDController, SimpleMovingAverage  # import optional useful things
from tilsdk.mock_robomaster.robot import Robot  # Use this for the simulator
#from robomaster.robot import Robot  # Use this for real robot

# Import your code
from cv_service import CVService, MockCVService
from nlp_service import NLPService, MockNLPService
from planner import MyPlanner

# Setup logging in a nice readable format
logging.basicConfig(level=logging.INFO,
                    format='[%(levelname)5s][%(asctime)s][%(name)s]: %(message)s',
                    datefmt='%H:%M:%S')

# Define config variables in an easily accessible location
# You may consider using a config file
REACHED_THRESHOLD_M = 0.3  # TODO: Participant may tune, in meters
ANGLE_THRESHOLD_DEG = 20.0  # TODO: Participant may tune.
ROBOT_RADIUS_M = 0.17  # TODO: Participant may tune. 0.390 * 0.245 (L x W)
NLP_PREPROCESSOR_DIR = 'finals_audio_model'
NLP_MODEL_DIR = 'model.onnx'
CV_CONFIG_DIR = 'universenet_custom_config.py'
CV_MODEL_DIR = 'epoch_26.pth'


# Convenience function to update locations of interest.
def update_locations(old: List[RealLocation], new: List[RealLocation]) -> None:
    '''Update locations with no duplicates.'''
    if new:
        for loc in new:
            if loc not in old:
                logging.getLogger('update_locations').info('New location of interest: {}'.format(loc))
                old.append(loc)

prev_img_rpt_time = time.time()                
def do_cv():
    global prev_img_rpt_time
    if not prev_img_rpt_time or time.time() - prev_img_rpt_time >= 1:  # throttle to 1 submission per second, and only read img if necessary
    img = robot.camera.read_cv2_image(strategy='newest')

    # Process image and detect targets
    targets = cv_service.targets_from_image(img)

    # Submit targets
    if targets:
        prev_img_rpt_time = time.time()
        logging.getLogger('Main').info('{} targets detected.'.format(len(targets)))
        # logging.getLogger('Reporting').info(rep_service.report(pose, img, targets))  # Only for real robot

                

def main():
    # Initialize services
    cv_service = CVService(config_file=CV_CONFIG_DIR, checkpoint_file=CV_MODEL_DIR)
    # cv_service = MockCVService(model_dir=CV_MODEL_DIR)
    nlp_service = NLPService(preprocessor_dir=NLP_PREPROCESSOR_DIR, model_dir=NLP_MODEL_DIR)
    loc_service = LocalizationService(host='192.168.20.56', port=5522)  # for real robot
    # loc_service = LocalizationService(host='localhost', port=5566)  # for simulator
    # rep_service = ReportingService(host='localhost', port=5501)  # only avail on simulator
    robot = Robot()
    robot.initialize(conn_type="sta", sn="3JKDH2T001U0H4")
    robot.camera.start_video_stream(display=False, resolution='720p')

    # Start the run
    # rep_service.start_run()  # only avail on simulator

    # Initialize planner
    map_: SignedDistanceGrid = loc_service.get_map()
    map_ = map_.dilated(1.5 * ROBOT_RADIUS_M / map_.scale)
    planner = MyPlanner(map_, waypoint_sparsity=0.4, optimize_threshold=3, consider=4, biggrid_size=0.8)

    # Initialize variables
    seen_clues = set()
    curr_loi: RealLocation = None
    path: List[RealLocation] = []
    lois: List[RealLocation] = []
    curr_wp: RealLocation = None

    # Tune here
    tracker = PIDController(Kp=(0.35, 0.2), Ki=(0.15, 0.1), Kd=(0.3, 0.2))

    # Initialize pose filter
    pose_filter = SimpleMovingAverage(n=10)

    # Define filter function to exclude clues seen before   
    new_clues = lambda c: c.clue_id not in seen_clues
    prev_img_rpt_time = 0
    # Main loop
    while True:
        # Get new data
        pose, clues = loc_service.get_pose()
        pose = pose_filter.update(pose)
        if not pose:
            # no new data, continue to next iteration.
            continue

        # Set this location as visited in the planner (so no need to visit here again if there are no clues)
        planner.visit(pose[:2])

        # Filter out clues that were seen before
        clues = list(filter(new_clues, clues))

        # Process clues using NLP and determine any new locations of interest
        if clues:
            new_lois, maybe_useful_lois = nlp_service.locations_from_clues(clues)  # new locations of interest  TODO: use maybe useful lois
            if len(new_lois):
                logging.getLogger('Main').info('New location(s) of interest: {}.'.format(new_lois))
            update_locations(lois, new_lois)
            seen_clues.update([c.clue_id for c in clues])

        do_cv()

        if not curr_loi:
            if len(lois) == 0:
                logging.getLogger('Main').info('No more locations of interest.')
                explore_next = planner.get_explore(pose[:2])
                if explore_next is None:
                    break
                lois.append(explore_next)
                # break

            # Get new LOI
            lois.sort(key=lambda l: euclidean_distance(l, pose), reverse=True)
            curr_loi = lois.pop()
            logging.getLogger('Main').info('Current LOI set to: {}'.format(curr_loi))

            # Plan a path to the new LOI
            logging.getLogger('Main').info('Planning path to: {}'.format(curr_loi))

            path = planner.plan(pose[:2], curr_loi, display=False)
            if path is None:
                logging.getLogger('Main').info('No possible path found, location skipped')
                # print("lois",lois)
                # print("seen:",seen_clues)
                curr_loi = None
            else:
                path.reverse()  # reverse so closest wp is last so that pop() is cheap , waypoint
                curr_wp = None
                logging.getLogger('Main').info('Path planned.')
        else:
            # There is a current LOI objective.
            # Continue with navigation along current path.
            if path:
                # Get next waypoint
                if not curr_wp:
                    curr_wp = path.pop()
                    logging.getLogger('Navigation').info('New waypoint: {}'.format(curr_wp))

                # Calculate distance and heading to waypoint
                dist_to_wp = euclidean_distance(pose, curr_wp)
                ang_to_wp = np.degrees(np.arctan2(curr_wp[1] - pose[1], curr_wp[0] - pose[0]))
                ang_diff = -(ang_to_wp - pose[2])  # body frame

                # ensure ang_diff is in [-180, 180]
                if ang_diff < -180:
                    ang_diff += 360

                if ang_diff > 180:
                    ang_diff -= 360

                # logging.getLogger('Navigation').info('ang_to_wp: {}, hdg: {}, ang_diff: {}'.format(ang_to_wp, pose[2], ang_diff))
                # logging.getLogger('Navigation').info('Pose: {}'.format(pose))

                # Consider waypoint reached if within a threshold distance
                if dist_to_wp < REACHED_THRESHOLD_M:
                    logging.getLogger('Navigation').info('Reached wp: {}'.format(curr_wp))
                    tracker.reset()
                    curr_wp = None
                    continue

                # Determine velocity commands given distance and heading to waypoint
                vel_cmd = tracker.update((dist_to_wp, ang_diff))

                # logging.getLogger('Navigation').info('dist: {} ang:{} vel:{}'.format(dist_to_wp,ang_diff,vel_cmd))

                # reduce x velocity
                vel_cmd[0] *= np.cos(np.radians(ang_diff))

                # If robot is facing the wrong direction, turn to face waypoint first before
                # moving forward.
                if abs(ang_diff) > ANGLE_THRESHOLD_DEG:
                    vel_cmd[0] = 0.0

                # Send command to robot
                robot.chassis.drive_speed(x=vel_cmd[0], z=vel_cmd[1])

        else:
            logging.getLogger('Navigation').info('End of path. Spinning now.')
            curr_loi = None
            
            starting_angle = pose[2]
            starting_angle %= 360
            first_turn_angle = starting_angle%90
            
            robot.chassis.drive_speed(x=0, z=first_turn_angle)
            time.sleep(1)
            robot.chassis.drive_speed(x=0, z=0)
            
            current_angle = (starting_angle-first_turn_angle)%360
            
            if planner.wall_within_1m(pose,current_angle):
                do_cv(debug=True)
            
            for spinning in range(3):
                robot.chassis.drive_speed(x=0, z=90)
                time.sleep(1)
                robot.chassis.drive_speed(x=0, z=0)
                current_angle = (current_angle-90)%360
                if planner.wall_within_1m(pose,current_angle):
                    do_cv(debug = True)
                
            logging.getLogger('Navigation').info('Done spinning. Moving on.')
            continue

    robot.chassis.drive_speed(x=0.0, y=0.0, z=0.0)  # set stop for safety
    logging.getLogger('Main').info('Mission Terminated.')   


if __name__ == '__main__':
    main()

# clean_up_env()  # only for laptop
