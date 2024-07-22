import json
from tqdm.autonotebook import trange
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



class MusicInformationRetreiver:
    def __init__(self, weights_path, segment_threshold=0.1):
        self.sr = 16000
        self.threshold = segment_threshold

        # self.beat_estimator = BeatNet(1, mode='offline', inference_model='DBN', thread=False)
        
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


    def _extract_segments(self, audio):
        S = np.abs(librosa.stft(audio))
        clusterer = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=self.threshold,
            connectivity=grid_to_graph(n_x=S.shape[1], n_y=1, n_z=1)
        )

        segs = librosa.segment.agglomerative(data=S / np.linalg.norm(S), k=None, clusterer=clusterer)
        segs = librosa.frames_to_time(segs, sr=self.sr)
        return segs


    def _predict_chunk(self, x):
        emb = self.effnet_embedding_model(x)
        genre_pred = self.genre_head(emb)
        inst_pred = self.inst_head(emb)

        emb = self.musicnn_embedding_model(x)
        emo_pred = self.emo_head(emb)
        emo_pred = (emo_pred - 1) / 8
        return genre_pred, inst_pred, emo_pred, 
        


    def recognize_file(self, file_path, verbose=True):
        audio, _ = librosa.load(file_path, sr=self.sr)
        length = audio.shape[0] / self.sr
        
        ## segment boundaries
        segments = self._extract_segments(audio)
        
        ## Beat and Tempo
        tempo, beats = librosa.beat.beat_track(y=audio, sr=self.sr)
        beats = librosa.frames_to_time(beats, sr=self.sr)
        
        # beats = self.beat_estimator.process(audio_path=file_path)[:, 0]

        m = int(30 * self.sr)
        k = audio.shape[0] // m
        prog = trange if verbose else range
        genre_preds = []
        inst_preds = []
        emo_preds = []
        for i in prog(k):
            res = self._predict_chunk(audio[i*m : (i+1)*m])
            genre_preds += [res[0]]
            inst_preds += [res[1]]
            emo_preds += [res[0]]
        emo_preds = np.concatenate(emo_preds, axis=0)
        
        return {
            'beats' : beats,
            'length': length,
            'segments': segments[1:],
            'tempo' : int(tempo[0]),
            'genre': np.concatenate(genre_preds, axis=0), 
            'instrument': np.concatenate(inst_preds, axis=0), 
            'valence': emo_preds[:, 0], 
            'arousal': emo_preds[:, 1]
        }
        
        
        