from typing import Iterable, List
from tilsdk.localization.types import *
import onnxruntime as ort
import librosa
import numpy as np
import sys
import io
sys.path.append('../Audio/')  # for local import, in real finals put the transformers folder in working dir and remove this
from transformers import Wav2Vec2FeatureExtractor  # local import


class NLPService:
    def __init__(self, preprocessor_dir: str, model_dir: str):
        '''
        Parameters
        ----------
        model_dir : str
            Path of model file to load.
        '''
        print('Initializing NLP service...')
        self.sess_options = ort.SessionOptions()
        self.sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        print(f'Available runtime providers: {ort.get_available_providers()}')
        print('Loading preprocessor...')
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(preprocessor_dir)
        print('Loading model...')
        self.model = ort.InferenceSession(model_dir, sess_options=sess_options, providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider'])  # try tensorrt provider first
        self.id2label = {0: True, 1: False}  # 0: "angry_sad", 1: "happy_neutral", useful or not
        print('NLP service initialized.')

    def predict(self, wav: bytes) -> bool:
        try:
            speech_array, sr = librosa.load(io.BytesIO(wav), sr=16000, mono=True)
            features = processor(speech_array, sampling_rate=16000, return_tensors="np")
            onnx_outputs = model.run(None, {model.get_inputs()[0].name: features.input_values})[0]
            prediction = np.argmax(onnx_outputs, axis=-1)
            return self.id2label[prediction.squeeze().tolist()]  # returns useful or not
        except Exception as e:
            print(f'Error while predicting: {e}')
            return False

    def locations_from_clues(self, clues: Iterable[Clue]) -> Lists[RealLocation]:
        '''Process clues and get locations of interest.
        
        Parameters
        ----------
        clues
            Clues to process.

        Returns
        -------
        lois
            Locations of interest.
        '''
        print(f'Predicting {len(clues)} clues...')
        useful_locs, maybe_useful_locs = [], []
        for clue in clues:
            if self.predict(clue.audio):
                useful_locs.append(clue.location)
            else:
                maybe_useful_locs.append(clue.location)
        print(f'Found {len(useful_locs)} useful and {len(maybe_useful_locs)} maybe useful.')
        return useful_locs, maybe_useful_locs


class MockNLPService(NLPService):
    '''Mock NLP Service.
    
    This is provided for testing purposes and should be replaced by your actual service implementation.
    '''