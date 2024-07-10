import copy
import numpy as np
import torch
import math
import librosa
from omegaconf import OmegaConf, open_dict
from tqdm.autonotebook import trange

from speechbrain.inference.interfaces import foreign_class
from .emo_dim import EmotionDimensionModel

import nemo.collections.asr as nemo_asr
from nemo.collections.asr.models.ctc_bpe_models import EncDecCTCModelBPE
from nemo.collections.asr.parts.utils.streaming_utils import CacheAwareStreamingAudioBuffer
from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis

class SpeechInformationRetreiver:
    def __init__(self, model_name, decoder_type='ctc', lookahead_size=80, device='cpu'):
        self.sr = 16000
        self.device = device
        self.model_name = model_name
        self.decoder_type = decoder_type
        self.encoder_step_length = 80  # in milliseconds
        self.lookahead_size = lookahead_size 
        self.chunk_size = self.lookahead_size + self.encoder_step_length

        ## SER
        self.emo_cls_model = foreign_class(
            source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP", 
            pymodule_file="custom_interface.py", 
            classname="CustomEncoderWav2vec2Classifier"
        ).to(device)

        self.emo_reg_model = EmotionDimensionModel.from_pretrained(
            'audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim'
        ).to(device)

        
        self._setup_asr_model()
        self._setup_preprocessor()
        self._reset_asr_parameters()

        
    def _setup_asr_model(self):
        self.asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name=self.model_name)
        self.asr_model.change_decoding_strategy(decoder_type=self.decoder_type)

        decoding_cfg = self.asr_model.cfg.decoding
        with open_dict(decoding_cfg):
            # save time by doing greedy decoding and not trying to record the alignments
            decoding_cfg.strategy = "greedy"
            decoding_cfg.preserve_alignments = False
            if hasattr(self.asr_model, 'joint'):  # if an RNNT model
                # restrict max_symbols to make sure not stuck in infinite loop
                decoding_cfg.greedy.max_symbols = 10
                # sensible default parameter, but not necessary since batch size is 1
                decoding_cfg.fused_batch_size = -1
            self.asr_model.change_decoding_strategy(decoding_cfg)
            self.asr_model.eval().to(self.device)
    
    def _setup_preprocessor(self):
        cfg = copy.deepcopy(self.asr_model._cfg)
        OmegaConf.set_struct(cfg.preprocessor, False)

        # some changes for streaming scenario
        cfg.preprocessor.dither = 0.0
        cfg.preprocessor.pad_to = 0
        cfg.preprocessor.normalize = "None"
        
        self.preprocessor = EncDecCTCModelBPE.from_config_dict(cfg.preprocessor).to(self.device)

    
    def _reset_asr_parameters(self):
        # get parameters to use as the initial cache state
        self.cache_last_channel, self.cache_last_time, self.cache_last_channel_len = self.asr_model.encoder.get_initial_cache_state(
            batch_size=1
        )
        self.previous_hypotheses = None
        self.pred_out_stream = None
        
        # cache-aware models require some small section of the previous processed_signal to
        # be fed in at each timestep - we initialize this to a tensor filled with zeros
        # so that we will do zero-padding for the very first chunk(s)
        self.num_channels = self.asr_model.cfg.preprocessor.features
        self.pre_encode_cache_size = self.asr_model.encoder.streaming_cfg.pre_encode_cache_size[1]
        self.cache_pre_encode = torch.zeros(
            (1, self.num_channels, self.pre_encode_cache_size), 
            device=self.device
        )

    
    # helper function for extracting transcriptions
    def _extract_transcriptions(self, hyps):
        """
            The transcribed_texts returned by CTC and RNNT models are different.
            This method would extract and return the text section of the hypothesis.
        """
        if isinstance(hyps[0], Hypothesis):
            transcriptions = []
            for hyp in hyps:
                transcriptions.append(hyp.text)
        else:
            transcriptions = hyps
        return transcriptions
    

    def transcribe_chunk(self, new_chunk, normalize=True):
        # new_chunk is provided as np.int16, so we convert it to np.float32
        # as that is what our ASR models expect
        audio_data = new_chunk.astype(np.float32)
        if normalize:
            audio_data = audio_data / 32768.0

        # get mel-spectrogram signal & length
        audio_signal = torch.from_numpy(audio_data).unsqueeze_(0).to(self.device)
        audio_signal_len = torch.Tensor([audio_data.shape[0]]).to(self.device)
        processed_signal, processed_signal_length = self.preprocessor(
            input_signal=audio_signal, length=audio_signal_len
        )
        
        # prepend with cache_pre_encode
        processed_signal = torch.cat([self.cache_pre_encode, processed_signal], dim=-1)
        processed_signal_length += self.cache_pre_encode.shape[1]
        
        # save cache for next time
        self.cache_pre_encode = processed_signal[:, :, -self.pre_encode_cache_size:]
        
        with torch.no_grad():
            (
                self.pred_out_stream,
                transcribed_texts,
                self.cache_last_channel,
                self.cache_last_time,
                self.cache_last_channel_len,
                self.previous_hypotheses,
            ) = self.asr_model.conformer_stream_step(
                processed_signal=processed_signal,
                processed_signal_length=processed_signal_length,
                cache_last_channel=self.cache_last_channel,
                cache_last_time=self.cache_last_time,
                cache_last_channel_len=self.cache_last_channel_len,
                keep_all_outputs=False,
                previous_hypotheses=self.previous_hypotheses,
                previous_pred_out=self.pred_out_stream,
                drop_extra_pre_encoded=None,
                return_transcription=True,
            )
        
        return self._extract_transcriptions(transcribed_texts)[0]

    
    def recognize_chunk(self, new_chunk, normalize=True):
        audio = new_chunk.astype(np.float32)
        if normalize:
            audio = audio / 32768.0
        audio = torch.tensor(audio).unsqueeze(0).to(self.device)

        _, _, _, label = self.emo_cls_model.classify_batch(audio)
        dims = self.emo_reg_model(audio).detach()[0]
        return label[0], dims[0].item(), dims[1].item(), dims[2].item()
    

    def recognize_file(self, file_path, verbose=True):
        self._reset_asr_parameters()
        x, _ = librosa.load(file_path, sr=self.sr)
        m = int(self.chunk_size * self.sr / 1000)
        k = x.shape[0] // m
        prog = trange if verbose else range
        emo_labels = []
        emo_dims = []
        for i in prog(k):
            asr_res = self.transcribe_chunk(x[i*m : (i+1)*m], normalize=False)
            emo_res = self.recognize_chunk(x[i*m : (i+1)*m], normalize=False)
            emo_labels += [emo_res[0]]
            emo_dims += [emo_res[1:]]

        emo_dims = np.stack(emo_dims).T
        return {
            'text': asr_res, 
            'length' : x.shape[0] // self.sr,
            'emotion_labels': emo_labels, 
            'arousal' : emo_dims[0], 
            'dominance': emo_dims[1], 
            'valence' : emo_dims[2]
        }

            