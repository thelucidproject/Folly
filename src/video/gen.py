import librosa
from tqdm.auto import tqdm
from muse import PipelineMuse

class VideoGenerator:
    def __init__(self, fps=20):
        self.fps = fps
        self.pipe = PipelineMuse.from_pretrained("openMUSE/muse-512-finetuned")
        # self.pipe.transformer.enable_xformers_memory_efficient_attention()


    def generate_main_frames(self, events):
        frames = []
        for e in events:
            img = pipe(
                e.prompt, 
                timesteps=10, 
                guidance_scale=20, 
                transformer_seq_len=256, 
                use_fp16=True
            )[0]
            frames += [img]
        return frames
            

    def interpolate_frames(self, frames, energies):
        pass

    def save(self, mix_audio_path, output_path):
        clip = ImageSequenceClip(frames, fps=2)
        clip.write_videofile(output_path, codec="libx264", fps=self.fps)
        audio = AudioFileClip(mix_audio_path)
        video = VideoFileClip(output_path)
        video.set_audio(audio)
        video.write_videofile(output_path, codec="libx264", fps=self.fps)

    def make_video(self, events, mix_audio_path):
        x, sr = librosa.load(mix_audio_path)
        hop = sr // self.fps
        enenrgy = librosa.feature.rms(y=x, sr=sr, hop_length=hop)

        main_frames = self.generate_main_frames(events)
        all_frames = self.interpolate_frames(main_frames, enenrgy)
        