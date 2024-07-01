import json
from tqdm import trange
import numpy as np
from essentia.standard import (
    MonoLoader, 
    TensorflowPredictMusiCNN,
    TensorflowPredictVGGish, 
    KeyExtractor, 
    TensorflowPredictEffnetDiscogs,
    TensorflowPredict2D
)

from BeatNet.BeatNet import BeatNet



class MIR:
    def __init__(self, weights_path):
        self.sr = 16000
        self.key_extractor = KeyExtractor()
        self.effnet_embedding_model = TensorflowPredictEffnetDiscogs(
            graphFilename=f"{weights_path}/discogs-effnet-bs64-1.pb", 
            output="PartitionedCall:1"
        )

        self.genre_classes = json.load(open(f'{weights_path}/mtg_jamendo_genre-discogs-effnet-1.json', 'r'))['classes']
        self.genre_head = TensorflowPredict2D(graphFilename=f"{weights_path}/mtg_jamendo_genre-discogs-effnet-1.pb")
        self.inst_classes = json.load(
            open(f'{weights_path}/mtg_jamendo_instrument-discogs-effnet-1.json', 'r')
        )['classes']
        self.inst_head = TensorflowPredict2D(graphFilename=f"{weights_path}/mtg_jamendo_instrument-discogs-effnet-1.pb")

        self.musicnn_embedding_model = TensorflowPredictMusiCNN(
            graphFilename=f"{weights_path}/msd-musicnn-1.pb", output="model/dense/BiasAdd"
        )
        self.emo_head = TensorflowPredict2D(
            graphFilename=f"{weights_path}/emomusic-msd-musicnn-2.pb", output="model/Identity"
        )
        self.beat_estimator = BeatNet(1, mode='offline', inference_model='DBN', thread=False)

        


    def recognize_file(self, file_path):
        audio = MonoLoader(sampleRate=self.sr, filename=file_path)()
        length = audio.shape[0] / self.sr
        
        emb = self.effnet_embedding_model(audio)
        genre_preds = self.genre_head(emb)
        inst_preds = self.inst_head(emb)

        emb = self.musicnn_embedding_model(audio)
        emo_preds = self.emo_head(emb)
        key_pred = self.key_extractor(audio)
        beats = self.beat_estimator.process(audio_path=file_path)
        return {
            'genre': genre_preds, 
            'instrument': inst_preds, 
            'emotion': emo_preds, 
            'key': '-'.join(key_pred[:2]),
            'beats' : beats,
            'tempo' : beats.shape[0] * 60 // length
        }
        
        
        