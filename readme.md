# DSTA BrainHack Today I Learnt AI Camp 2022
### Team: 8000SGD_CAT
### School: [Hwa Chong Institution Infocomm & Robotics Society (HCIRS)](https://github.com/hcirs) Machine Learning section
### Achievement: Champion in JC/Poly/ITE/IP category, highest score among all teams in the competition
For more details, see the scores section below.

### Team members
* [Billy Cao](https://github.com/aliencaocao) (L): Audio/CV/Robot
* [Marcus Wee](https://github.com/Marcushadow): CV
* [Ho Wing Yip](https://github.com/HoWingYip): CV/Robot
* [Huang Qirui](https://github.com/hqrui): Audio/CV/Robot
* [Ooi Xuan Shan](https://github.com/ooixs): Audio/CV
* Special thanks to [Theodore Lee](https://github.com/TheoLeeCJ) for providing cloud-related technical support


## Finals leaderboard
### JC/Poly/ITE/IP

|    Team     | Score | Time taken/min | Correct Targets found per 5min | Penalties | Tiebreaker tiles |
|:-----------:|:-----:|:--------------:|:------------------------------:|:---------:|:----------------:|
| 8000SGD_CAT |  3.0  |      9.93      |              1.76              |   -0.5    |        -         |
| ACEOFSPADES |  0.5  |      2.98      |              1.68              |   -0.5    |        -         |
| TRACK_ROBOT |  0.5  |      8.88      |              1.13              |   -1.5    |        -         |
|  4APOSTLES  | -2.5  |      7.45      |              1.68              |    -3     |        -         |
| ALGOMATRIX  | -4.5  |       15       |               0                |   -0.5    |        -         |
| T3RM1N4TOR  | (DNF) |       15       |               0                |     0     |        5         |

### University
|     Team     | Score | Time taken/min | Correct Targets found per 5min | Penalties | Tiebreaker tiles |
|:------------:|:-----:|:--------------:|:------------------------------:|:---------:|:----------------:|
| KEIWUAI200FF |   0   |       15       |               0                |     0     |        17        |
| T0X1C_V4P0R  |   0   |       15       |               0                |     0     |        13        |
|   PALMTREE   |  -1   |       15       |               0                |    -1     |        2         |
|    TEAMU4    | -0.5  |       15       |               0                |   -0.5    |        -         |
|  STRONGZERO  | -1.5  |       15       |               0                |   -1.5    |        -         |
|  200SUCCESS  | (DNF) |       15       |               0                |     0     |        1         |


## Directory structure

* [Audio](Audio): Code for NLP task
* [CV](CV): Code for CV task
* [pyastar2d](https://github.com/aliencaocao/pyastar2d): Our modified fork of PyAstar2D implementation
* [til-final-v2](til-final-v2): Code for finals robot driving
* [til2022-final](https://github.com/DinoHub/til2022-final): DSTA provided simulator and SDK library for controlling the robot for finals

## Technical writeup

### NLP - Speech Emotion Recognition
We used Facebook's pretrained [Wav2Vec2.0 XSLR-53 model](https://huggingface.co/facebook/wav2vec2-large-xlsr-53) fine-tuned by [harshit345](https://huggingface.co/harshit345) for speech emotion recognition. Pretrained model: [https://huggingface.co/harshit345/xlsr-wav2vec-speech-emotion-recognition](https://huggingface.co/harshit345/xlsr-wav2vec-speech-emotion-recognition)

We fine-tuned it further on a combination of multiple datasets. For details, please see [audio data citations](Audio/data/citations.txt)

For finals, we modified the 5-class classifier model into a binary one, since the finals only requre classification of audio between angry/sad and happy/neutral. Our binary classification model reached 99.1% accuracy on a 20% split validation set. Our model was also converted to ONNX format with graph optimization and FP16 precision.

### Computer Vision - Human standing/fallen detection
For qualifiers, we used Cascade-RCNN and YOLOv5x pretrained on the [VFP290K](https://github.com/DASH-Lab/VFP290K) dataset, and did weighted box fusion to ensemble them.

For finals, we used the [VarifocalNet X-101-64x4d](https://github.com/open-mmlab/mmdetection/blob/master/configs/vfnet/README.md). Our deployed model on the robot is optimized with FP16 precision and convolutional and batch-normalization fusion.

We apply additional preprocessing and post-processing for deployment on the robot to increase model accuracy and reduce false positives:
* Zoom-in the camera feed: 30% cropped from top, 10% cropped from other 3 sides.
* Confidence threshold of 0.4, NMS IoU threshold of 0.5, and only allow detection while the robot is stationary.

### Robotics
#### Pathfinding and planning
We used the [A* pathfinding algorithm](https://en.wikipedia.org/wiki/A*_search_algorithm) combined with a distance transformed potential map to do pathfinding. We used a modified version of the [PyAstar2D implementation](https://github.com/hjweide/pyastar2d) as it is written in C++ with Python bindings which is about 9 times faster than a pure-python implementation according to our testing. Our modified version correctly handles the cost of traversing diagonal paths in an omnidirectional connected graph, it can be found [here](https://github.com/aliencaocao/pyastar2d). Our distance-transformed potential map is also customized such that area further than a certain distance (40cm pre-dilation, 14.5cm post-dilation) from the wall are all treated with equal potential. This prevents the robot from being forced to take a path in the middle of 2 very far away walls, which is suboptimal and causes some targets to not be detected by the robot.

We further optimize the robot's waypoints returned by the path finding algorithm by removing unnecessary waypoints that lie on a reasonable straight line between two waypoints. This allow the robot to travel to waypoints faster and reduces excessive self-correction of paths and improve stability of navigation.

When the robot is out of locations of interest (LOIs) to explore, it will reuse previously received clues that are classified as 'maybe useful'. If those clues are exhausted, the robot will start to explore the rest of the area by dividing the map into 80cm x 80cm grids and visiting the closest unvisited grids.

#### Controller
We used the [proportional–integral–derivative (PID) controller](https://en.wikipedia.org/wiki/PID_controller) with parameters (Velocity, angle) Kp=(0.35, 0.2), Ki=(0.1, 0.0), Kd=(0, 0) to control the movement of the robot.

For any questions regarding our codebase, please open a Discussion.