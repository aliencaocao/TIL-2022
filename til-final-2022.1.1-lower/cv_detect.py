from mmdet.apis import init_detector, inference_detector
from mmcv.visualization import imshow_det_bboxes
import mmcv

"""
Usage example:

from cv_detect import detection_init, detect

detection_init(
  config_file="configs/universenet/universenet101_2008d_fp16_4x4_mstrain_480_960_20e_coco.py",
  checkpoint_file="/content/drive/MyDrive/TIL2022/CV/universenet_checkpoints/universenet101_2008d_fp16_4x4_mstrain_480_960_20e_coco.py_20220614_032652/epoch_12.pth"
)

detections = detect(camera.read_cv2_image(...))
"""

model = None

# Call this function only once. It is quite expensive!
def detection_init(config_file, checkpoint_file):
  global model # I'm sorry.
  model = init_detector(config_file, checkpoint_file, device='cuda:0')

def detect(img_array):
  result = inference_detector(model, img_array)
  detections = []
  
  current_detection_id = 1
  for class_id, this_class_detections in enumerate(result):
    for detection in this_class_detections:
      x1, y1, x2, y2, confidence = [float(x) for x in detection]
      detections.append({
        "id": current_detection_id,
        "cls": "fallen" if class_id == 0 else "standing",
        "bbox": {
          "x": x1,
          "y": y1,
          "w": x2 - x1,
          "h": y2 - y1,
        }
      })
      current_detection_id += 1

  return detections