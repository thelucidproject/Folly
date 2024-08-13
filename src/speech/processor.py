import numpy as np
from copy import deepcopy

from keybert import KeyBERT



class PostProcessor:
    def __init__(self, num_keywords=5):
        self.kw_model = KeyBERT()
        self.num_keywords = num_keywords

    def _extract_keywords(self, segments):
        def extract_kw(text):
            keywords = self.kw_model.extract_keywords(text, top_n=self.num_keywords)
            return [l[0] for l in keywords]

        res = []
        for s in segments:
            s['text'] = ', '.join(extract_kw(s['text']))
            res += [s]
        return res

    def _refine_segments(self, segments, add_emotion_info=False, max_dist=1.):
        res = []
        i = 0
        while i < len(segments):
            j = i + 1
            while j < len(segments) and segments[j]['start'] - segments[j-1]['end'] < max_dist:
                j += 1
            seg = {
                'text' : ''.join([s['text'] for s in segments[i:j]]),
                'start' : segments[i]['start'],
                'end' : segments[j-1]['end'],
                'duration' : segments[j-1]['end'] - segments[i]['start'],
            }
            if add_emotion_info:
                seg['arousal'] = sum([s['arousal'] for s in segments[i:j]]) / (j - i)
                seg['valence'] = sum([s['valence'] for s in segments[i:j]]) / (j - i)
                seg['dominance'] = sum([s['dominance'] for s in segments[i:j]]) / (j - i)
            res += [seg]
            i = j
        return res
        

    def __call__(self, asr_res, add_emotion_info=False, max_dist=1., extract_kw=False):
        va = np.array([0 if r == '' else 1 for r in asr_res['text']])
        bin = asr_res['length'] / len(asr_res['text'])
        times = np.arange(0, asr_res['length'], bin)
        assert len(times) == len(asr_res['text'])

        speech_on = np.where(
            va > np.pad(va[:-1], (1,0))
        )[0]
        speech_off = np.where(
            va > np.pad(va[1:], (0,1))
        )[0] + 1

        segments = []
        for on, off in zip(speech_on, speech_off):
            seg = {
                'text' : ''.join(asr_res['text'][on:off]),
                'start' : times[on],
                'end' : times[off],
                'duration' : times[off] - times[on],
            }
            if add_emotion_info:
                seg['arousal'] = asr_res['arousal'][on:off].mean()
                seg['valence'] = asr_res['valence'][on:off].mean()
                seg['dominance'] = asr_res['dominance'][on:off].mean()
            segments += [seg]
            
        segments = self._refine_segments(segments, add_emotion_info=add_emotion_info, max_dist=max_dist)
        if extract_kw:
            segments = self._extract_keywords(segments)
        return segments
            
