import torch
import numpy as np
import librosa
import cv2
import subprocess
from tqdm.auto import tqdm


from pydub import AudioSegment

from diffusers import StableDiffusionPipeline
from diffusers import DPMSolverMultistepScheduler

def get_audio_amplitudes(path, fps, percussive=False):
    audio, sr = librosa.load(path)
    length = audio.shape[0] / sr
    if percussive:
        _, audio = librosa.effects.hpss(audio, margin=(1.0,2.0))
        
    spec_raw = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=512, hop_length=sr // fps)
    spec_max = np.amax(spec_raw, axis=0)
    spec_norm = (spec_max - np.min(spec_max)) / np.ptp(spec_max)
    return spec_norm, length

def slerp(v0, v1, t, DOT_THRESHOLD=0.9995):
    dot = torch.sum(v0 * v1) / (torch.norm(v0) * torch.norm(v1))
    if torch.abs(dot) > DOT_THRESHOLD:
        v2 = (1 - t) * v0 + t * v1
    else:
        theta_0 = torch.acos(dot)
        sin_theta_0 = torch.sin(theta_0)
        theta_t = theta_0 * t
        sin_theta_t = torch.sin(theta_t)
        s0 = torch.sin(theta_0 - theta_t) / sin_theta_0
        s1 = sin_theta_t / sin_theta_0
        v2 = s0 * v0 + s1 * v1
    return v2


class VideoGenerator:
    def __init__(
        self, 
        model_name="CompVis/stable-diffusion-v1-4", 
        width=512, 
        height=512, 
        seed=7, 
        negative_prompt=None
    ):
        
        self.pipeline = StableDiffusionPipeline.from_pretrained(
            model_name, torch_dtype=torch.float16, safety_checker=None
        ).to("cuda")
        self.pipeline.scheduler = DPMSolverMultistepScheduler.from_config(self.pipeline.scheduler.config)
        self.pipeline.enable_xformers_memory_efficient_attention()

        self.width = width
        self.height = height
        self.noise_shape = (1, self.pipeline.unet.config.in_channels, height // 8, width // 8)
        self.rand_generator = torch.Generator(device='cuda').manual_seed(seed)

        self.negative_prompt = negative_prompt

    def _refine_prompt(self, text, style):
        return f'an image of {text} in the style of {style}, '

    def _generate_noise(self, music_segments, music_amps, fps):
        N = len(music_segments)
        segment_latents = [
            torch.randn(
                self.noise_shape, 
                generator=self.rand_generator,
                device='cuda',
                dtype=torch.float16
            )
            for _ in range(N + 1)
        ]
        num_frames = [int(np.round(s['duration'] * fps)) for s in music_segments]
        latents = []
        for i in range(N):
            noise_a = segment_latents[i]
            noise_b = segment_latents[i+1]
            idx = sum(num_frames[:i])
            T = np.cumsum(music_amps[idx:idx+num_frames[i]])
            T /= T[-1]
            T[0] = 0.0
            for t in T:
                latents += [slerp(noise_a, noise_b, t)]
        return torch.cat(latents, dim=0)

    def _generate_text_embedding(self, speech_segments, speech_amps, fps, max_length):
        speech_segments_aug = [
            {'start':0, 'end':0, 'duration':0, 'text':speech_segments[0]['text'], 'style':speech_segments[0]['style']}
        ] + speech_segments + [
            {'start':max_length, 'end':max_length, 'duration':0, 'text':speech_segments[-1]['text'], 'style':speech_segments[-1]['style']}
        ]

        num_frames = [int(np.round(s['duration'] * fps)) for s in speech_segments_aug]
        text_input = self.pipeline.tokenizer(
            [self._refine_prompt(s['text'], s['style']) for s in speech_segments_aug],
            padding="max_length",
            max_length=self.pipeline.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            segment_embeddings = self.pipeline.text_encoder(text_input.input_ids.to('cuda'))[0]

        text_embeddings = []
        for i, seg in enumerate(speech_segments_aug[:-1]):
            emb_a = segment_embeddings[i]
            emb_b = segment_embeddings[i+1]
            text_embeddings += ([emb_a] * num_frames[i])
            
            gap_duration = speech_segments_aug[i+1]['start'] - seg['end']
            gap_frames = int(np.round(gap_duration * fps))
            idx = len(text_embeddings)
            T = np.cumsum(speech_amps[idx:idx+gap_frames])
            if T[-1] > 0.:
                T /= T[-1]
            T[0] = 0.0
            for t in T:
                text_embeddings += [torch.lerp(emb_a, emb_b, t)]
        return torch.stack(text_embeddings)

    
    def _save_video(self, save_path, frames, speech_filepath, music_filepath, duration, fps):
        ## Save video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # For MP4
        video_writer = cv2.VideoWriter(
            save_path[:-4] + '-video.mp4', fourcc, fps, (self.width, self.height)
        )
        for frame in frames:
            frame = np.array(frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            video_writer.write(frame)
        video_writer.release()

        ## Save Audio
        audio1 = AudioSegment.from_file(speech_filepath)[:duration * 1000]
        audio2 = AudioSegment.from_file(music_filepath)[:duration * 1000]
        mixed_audio = audio1.overlay(audio2)
        mixed_audio.export(save_path[:-4] + '-audio.mp3', format="mp3")

        ## Mix video and Audio
        command = [
            'ffmpeg',
            '-y',
            '-i', save_path[:-4] + '-video.mp4',
            '-i', save_path[:-4] + '-audio.mp3',
            '-c:v', 'copy',
            '-c:a', 'aac',
            '-strict', 'experimental',
            '-b:a', '192k',
            save_path
        ]
        try:
            subprocess.run(command, check=True)
            print(f"Successfully merged video_file and audio_file into {save_path}")
        except subprocess.CalledProcessError as e:
            print(f"An error occurred: {e}")


    def generate(
        self, 
        speech_segments,
        music_segments, 
        speech_filepath, 
        music_filepath, 
        percussive_reactivity=True,
        batch_size=32, 
        duration=None, 
        guidance_scale=7.,
        num_inference_steps=50,
        fps=20,
        save_path=None
    ):
        
        speech_amps, _ = get_audio_amplitudes(speech_filepath, fps=fps)
        music_amps, l = get_audio_amplitudes(music_filepath, fps=fps, percussive=percussive_reactivity)
        print('Audio activities captured')

        noise_latents = self._generate_noise(music_segments, music_amps, fps)
        text_embeddings = self._generate_text_embedding(speech_segments, speech_amps, fps, max_length=l)
        print('latent vectors computed')

        frames = []
        batch_size = 32
        if duration is None:
            n = min(text_embeddings.shape[0], noise_latents.shape[0])
        else:
            n = duration * fps

        print('starting generation...')
        self.pipeline.set_progress_bar_config(disable=True)
        for i in tqdm(range(n // batch_size + 1)):
            noise = noise_latents[i*batch_size : (i+1)*batch_size]
            text_emb = text_embeddings[i*batch_size : (i+1)*batch_size]
            neg_prompt = None if self.negative_prompt is None else [self.negative_prompt]*noise.shape[0]
            frames += self.pipeline(
                height=self.height, 
                width=self.width, 
                latents=noise, 
                prompt_embeds=text_emb, 
                guidance_scale=guidance_scale,
                negative_prompt=neg_prompt,
                num_inference_steps=num_inference_steps,
            ).images
            torch.cuda.empty_cache()

        if save_path is not None:
            self._save_video(save_path, frames, speech_filepath, music_filepath, duration, fps)
            
        return frames