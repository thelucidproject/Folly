import json
from tqdm import trange
import numpy as np
import librosa

from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.image import grid_to_graph
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
    def __init__(self, weights_path, distance_threshold=0.1):
        self.sr = 16000
        self.threshold = distance_threshold
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


    def _extract_segments(self, audio):
        mel = librosa.amplitude_to_db(
            librosa.feature.melspectrogram(y=audio, sr=self.sr)
        )
        mel = mel / np.linalg.norm(mel)
        clusterer = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=self.threshold,
            connectivity=grid_to_graph(n_x=mel.shape[1], n_y=1, n_z=1)
        )

        segs = librosa.segment.agglomerative(data=mel, k=None, clusterer=clusterer)
        segs = librosa.frames_to_time(segs, sr=self.sr)
        return segs

        


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
        segments = self._extract_segments(audio)
        return {
            'genre': genre_preds, 
            'instrument': inst_preds, 
            'emotion': emo_preds, 
            'key': '-'.join(key_pred[:2]),
            'beats' : beats,
            'tempo' : beats.shape[0] * 60 // length,
            'segments': segments
        }
        
        
        