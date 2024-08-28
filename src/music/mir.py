import json
from tqdm.autonotebook import tqdm
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




class MusicInformationRetreiver:
    def __init__(self, weights_path, fps=20):
        self.sr = 16000
        self.hop_length = int(self.sr / fps)
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


    def _extract_segment_bounds(self, audio, threshold=0.1):
        S = np.abs(librosa.stft(audio, hop_length=self.hop_length))
        clusterer = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=threshold,
            connectivity=grid_to_graph(n_x=S.shape[1], n_y=1, n_z=1)
        )

        segs = librosa.segment.agglomerative(data=S / np.linalg.norm(S), k=None, clusterer=clusterer)
        segs = librosa.frames_to_time(segs, sr=self.sr)
        return segs.tolist()


    def _predict_segment(self, seg, genre_threshold=0.3, inst_threshold=0.3):
        diff = self.sr * 6 - seg.shape[0]
        if diff > 0:
            seg = np.pad(seg, (0, diff))
        emb = self.musicnn_embedding_model(seg)
        emo_pred = self.emo_head(emb)
        emo_pred = (emo_pred - 1) / 8
        valence, arousal = emo_pred[:, 0], emo_pred[:, 1]

        emb = self.effnet_embedding_model(seg)
        genre_pred = self.genre_head(emb)
        inst_pred = self.inst_head(emb)

        genre_ids = np.where(genre_pred > genre_threshold)[1]
        genre = set([self.genre_classes[gid] for gid in genre_ids])
        
        inst_ids = np.where(inst_pred > inst_threshold)[1]
        instrument = set([self.inst_classes[iid] for iid in inst_ids])
        return {
            'valence' : valence.tolist(), 
            'arousal': arousal.tolist(),
            'genre': list(genre), 
            'instrument': list(instrument)
        }
        


    def __call__(
        self, 
        file_path, 
        segment_threshold=0.1, 
        add_segment_info=False,
        genre_threshold=0.3, 
        inst_threshold=0.3,
        progress_bar=None
    ):
        
        audio, _ = librosa.load(file_path, sr=self.sr)
        length = audio.shape[0] / self.sr
        
        bounds = self._extract_segment_bounds(audio, segment_threshold) + [length]
        segments = []

        prog = tqdm if progress_bar is None else progress_bar
        for i in prog(range(1, len(bounds))):
            seg = {
                'start': np.round(bounds[i-1], 3),
                'end' : np.round(bounds[i], 3), 
                'duration' : np.round(bounds[i] - bounds[i-1], 3)
            }
            if add_segment_info:
                start = int(seg['start'] * self.sr)
                end = int(seg['end'] * self.sr)
                seg |= self._predict_segment(audio[start:end], genre_threshold, inst_threshold)
            segments += [seg]
        return segments
        