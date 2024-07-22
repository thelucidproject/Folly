import numpy as np
from copy import deepcopy

from keybert import KeyBERT



class SpeechBridge:
    def __init__(self, num_keywords=5):
        self.kw_model = KeyBERT()
        self.num_keywords = num_keywords

    def _extract_keywords_from_events(self, events):
        def extract_kw(text):
            keywords = self.kw_model.extract_keywords(text, top_n=self.num_keywords)
            return [l[0] for l in keywords]

        res = []
        for e in events:
            e['text'] = ', '.join(extract_kw(e['text']))
            res += [e]
        return res

    def _refine_events(self, events, max_dist=1.):
        res = []
        i = 0
        while i < len(events):
            j = i + 1
            while j < len(events) and\
                events[j]['start'] - events[j-1]['end'] < max_dist:
                j += 1
            res += [{
                'text' : ''.join([e['text'] for e in events[i:j]]),
                'start' : events[i]['start'],
                'end' : events[j-1]['end'],
                'duration' : events[j-1]['end'] - events[i]['start'],
                'arousal' : sum([e['arousal'] for e in events[i:j]]) / (j - i),
                'valence' : sum([e['valence'] for e in events[i:j]]) / (j - i),
                'dominance' : sum([e['dominance'] for e in events[i:j]]) / (j - i)
            }]
            i = j
        return res
        

    def to_events(self, asr_res, max_dist=1., extract_kw=False):
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

        events = []
        for on, off in zip(speech_on, speech_off):
            events += [{
                'text' : ''.join(asr_res['text'][on:off]),
                'start' : times[on],
                'end' : times[off],
                'duration' : times[off] - times[on],
                'arousal' : asr_res['arousal'][on:off].mean(),
                'valence' : asr_res['valence'][on:off].mean(),
                'dominance' : asr_res['dominance'][on:off].mean()
            }]
            
        events = self._refine_events(events, max_dist)
        if extract_kw:
            events = self._extract_keywords_from_events(events)
        return events
            
