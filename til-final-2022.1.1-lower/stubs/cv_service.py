from typing import List, Any
from tilsdk.cv.types import *
from tilsdk.cv import DetectedObject, BoundingBox
from mmdet.apis import init_detector, inference_detector
from mmcv.cnn import fuse_conv_bn
from mmcv.runner import wrap_fp16_model

img_id = 0  # for debugging only


class CVService:
    def __init__(self, config_file, checkpoint_file):
        '''
        Parameters
        ----------
        config_file : str
            Path to mmdet config file.
        checkpoint_file : str
            Path to model checkpoint.
        '''
        print('Initializing CV service...')
        self.model = init_detector(config_file, checkpoint_file, device="cuda:0")
        # TODO: below 2 lines need test
        wrap_fp16_model(self.model)
        self.model = fuse_conv_bn(self.model)
        print('CV service initialized.')

    def targets_from_image(self, img) -> List[DetectedObject]:
        '''Process image and return targets.
        
        Parameters
        ----------
        img : Any
            Input image.
        
        Returns
        -------
        results  : List[DetectedObject]
            Detected targets.
        '''
        global img_id
        result = inference_detector(self.model, img)
        detections = []

        current_detection_id = 1
        for class_id, this_class_detections in enumerate(result):
            for detection in this_class_detections:
                x1, y1, x2, y2, _confidence = [float(x) for x in detection]
                detections.append(DetectedObject(
                    id=current_detection_id,
                    cls=1 - class_id,
                    bbox=BoundingBox(x=(x1+x2)/2, y=(y1+y2)/2, w=x2-x1, h=y2-y1),
                ))
                print(f'Detected {"fallen" if class_id == 0 else "standing"}, conf {_confidence}')
                current_detection_id += 1
                img_id += 1
                self.model.show_result(img, result, out_file=f'results/result{img_id} - {current_detection_id}.jpg')

        return detections


class MockCVService:
    '''Mock CV Service.
    
    This is provided for testing purposes and should be replaced by your actual service implementation.
    '''

    def __init__(self, model_dir: str):
        '''
        Parameters
        ----------
        model_dir : str
            Path of model file to load.
        '''
        # Does nothing.
        pass

    def targets_from_image(self, img: Any) -> List[DetectedObject]:
        '''Process image and return targets.
        
        Parameters
        ----------
        img : Any
            Input image.
        
        Returns
        -------
        results  : List[DetectedObject]
            Detected targets.
        '''
        # dummy data
        bbox = BoundingBox(100, 100, 300, 50)
        obj = DetectedObject("1", "fallen", bbox)
        return [obj]
