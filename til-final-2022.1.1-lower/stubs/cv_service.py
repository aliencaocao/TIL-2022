from typing import List, Any
from tilsdk.cv.types import *
from tilsdk.cv import DetectedObject, BoundingBox
from mmdet.apis import init_detector, inference_detector


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

        self.model = init_detector(config_file, checkpoint_file, device="cuda:0")

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

        result = inference_detector(self.model, img)
        detections = []

        current_detection_id = 1
        for class_id, this_class_detections in enumerate(result):
            for detection in this_class_detections:
                x1, y1, x2, y2, _confidence = [float(x) for x in detection]
                detections.append(DetectedObject(
                    id=current_detection_id,
                    cls="fallen" if class_id == 0 else "standing",
                    bbox=BoundingBox(x=x1, y=y1, w=x2-x1, h=y2-y1),
                ))
                current_detection_id += 1

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
        obj = DetectedObject("1", "1", bbox)
        return [obj]
