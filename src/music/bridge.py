import numpy as np
from collections import Counter

class MusicBridge:
    def __init__(self, genre_classes, inst_classes):
        self.genre_classes = genre_classes
        self.inst_classes = inst_classes

    def extract_segments(self, music_res, class_threshold=0.3):
        prev = 0
        bounds = np.append(music_res['segments'], music_res['length'])
        segments = []
        n_music_emo = len(music_res['valence'])
        n_music_tag = len(music_res['genre'])
        for end in bounds:
            seg = {'start' : prev, 'end': end, 'duration': end - prev}
            idx1 = int(prev * n_music_emo // music_res['length']) + 1
            idx2 = int(end * n_music_emo // music_res['length'])
            for dim in ['valence', 'arousal']:
                seg[dim] = music_res[dim][idx1:idx2].mean()
        
            idx1 = int(prev * n_music_emo // music_res['length']) + 1
            idx2 = int(end * n_music_emo // music_res['length'])
            
            genre = music_res['genre'][idx1:idx2].mean(axis=0)
            genre_ids = np.where(genre > class_threshold)[0]
            genre = [self.genre_classes[gid] for gid in genre_ids]
            
            instrument = music_res['instrument'][idx1:idx2].mean(axis=0)
            inst_ids = np.where(instrument > class_threshold)[0]
            instrument = [self.inst_classes[iid] for iid in inst_ids]

            seg['text'] = ', '.join(genre + instrument)
            
            segments += [seg]
            prev = end
        return segments