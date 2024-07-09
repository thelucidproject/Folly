import numpy as np
import torch
import math
import librosa

from speechbrain.inference.interfaces import foreign_class
from tqdm.autonotebook import trange

from .emo_dim import EmotionDimensionModel



class SER:
    def __init__(self, chunk_size=8000, device='cpu'):
        self.chunk_size = chunk_size
        self.sr = 16000
        self.device = device
        
        self.cls_model = foreign_class(
            source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP", 
            pymodule_file="custom_interface.py", 
            classname="CustomEncoderWav2vec2Classifier"
        ).to(device)

        self.reg_model = EmotionDimensionModel.from_pretrained(
            'audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim'
        ).to(device)


    def recognize_chunk(self, new_chunk, normalize=True):
        audio = new_chunk.astype(np.float32)
        if normalize:
            audio = audio / 32768.0
        audio = torch.tensor(audio).unsqueeze(0).to(self.device)

        _, _, _, label = self.cls_model.classify_batch(audio)
        dims = self.reg_model(audio).detach()[0]
        return label[0], dims[0].item(), dims[1].item(), dims[2].item()
    
    def recognize_file(self, file_path, verbose=True):
        x, _ = librosa.load(file_path, sr=self.sr)
        m = int(self.chunk_size * self.sr / 1000)
        k = math.ceil(x.shape[0] / m)
        prog = trange if verbose else range
        labels = []
        dims = []
        for i in prog(k):
            res = self.recognize_chunk(x[i*m : (i+1)*m], normalize=False)
            labels += [res[0]]
            dims += [res[1:]]
        dims = np.stack(dims).T
        return {'labels': labels, 'arousal' : dims[0], 'dominance': dims[1], 'valence' : dims[2]}