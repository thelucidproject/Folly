import os, logging, warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger("nemo_logger").setLevel(logging.CRITICAL)
warnings.filterwarnings("ignore")

import gradio as gr
import gc
import matplotlib.pyplot as plt
import numpy as np
import json

from src.music.mir import MusicInformationRetreiver
from src.speech.asr import SpeechInformationRetreiver
from src.video.gen import VideoGenerator


class Demo:
    def __init__(self, mir, sir, gen):
        self.mir = mir
        self.sir = sir
        self.gen = gen
        self.frames = None
        self.speech_segments = None

        self.music_interface = self.setup_music_interface()
        self.speech_interface = self.setup_speech_interface()
        self.video_interface = self.setup_video_interface()
        self.interface = gr.TabbedInterface(
            [self.music_interface, self.speech_interface, self.video_interface],
            ['Music', 'Voice Over', 'Video'],
            title='Folly Demo'
        )

    def analyse_music(self, music_path, segment_threshold, instrument_threshold, genre_threshold):
        segments = self.mir(
            music_path, 
            segment_threshold=segment_threshold, 
            genre_threshold=genre_threshold, 
            inst_threshold=instrument_threshold, 
            add_segment_info=True,
            progress_bar=gr.Progress().tqdm
        )
        seg_plot = self.plot_segments(segments)
        emo_plot = self.plot_emotions(segments, keys=['valence', 'arousal'])
        durations = [seg['duration'] for seg in segments]
        return seg_plot, emo_plot, str(durations)

    def preprocess_speech(self, speech_path):
        self.speech_segments = self.sir(
            speech_path, 
            add_emotion_info=True, 
            progress_bar=gr.Progress().tqdm
        )
        seg_plot = self.plot_segments(self.speech_segments)
        emo_plot = self.plot_emotions(self.speech_segments, keys=['valence', 'arousal', 'dominance'])
        text = [seg['text'] for seg in self.speech_segments]
        text = [f'Segment {i+1}: {seg["text"]}' for i,seg in enumerate(self.speech_segments)]
        return seg_plot, emo_plot, '\n\n'.join(text)
        

    def analyse_speech(self, max_dist, extract_kw, num_keywords):
        segments = self.sir.post_process(
            self.speech_segments, 
            add_emotion_info=True, 
            max_dist=max_dist, 
            extract_kw=extract_kw, 
            num_keywords=num_keywords
        )
        seg_plot = self.plot_segments(segments)
        emo_plot = self.plot_emotions(segments, keys=['valence', 'arousal', 'dominance'])
        text = [seg['text'] for seg in segments]
        text = [f'Segment {i+1}: {seg["text"]}' for i,seg in enumerate(segments)]
        return seg_plot, emo_plot, '\n\n'.join(text)
        

    def generate_video(
        self,
        style,
        width, 
        height,
        seed,
        strength,
        num_inference_steps,
        durations,
        prompts, 
        zoom_directions, 
        zoom_factors, 
        rotate_directions,
        rotate_factors,
        move_directions,
        move_factors,
        negative_prompt,
        generation_fps,
        music_path
    ):
        durations = list(map(float, durations.strip().split(', ')))
        prompts = prompts.strip().split('\n')
        zoom_directions = None if zoom_directions == '' else zoom_directions.strip().split(', ')
        rotate_directions = None if rotate_directions == '' else rotate_directions.strip().split(', ')
        move_directions = None if move_directions == '' else move_directions.strip().split(', ')
        zoom_factors = None if zoom_factors == '' else list(map(float, zoom_factors.strip().split(', ')))
        rotate_factors = None if rotate_factors == '' else list(map(float, rotate_factors.strip().split(', ')))
        move_factors = None if move_factors == '' else list(map(float, move_factors.strip().split(', ')))
    
        self.frames = None
        gc.collect()
    
        # Progress bar handling for Gradio
        progress_bar = gr.Progress().tqdm
        self.frames = gen.generate(
            durations, 
            style=style,
            generation_fps=generation_fps,
            final_fps=generation_fps,
            width=width, 
            height=height,
            strength=strength,
            prompts=prompts, 
            rotate_directions=rotate_directions,
            zoom_directions=zoom_directions,
            move_directions=move_directions,
            zoom_factors=zoom_factors,
            rotate_factors=rotate_factors,
            move_factors=move_factors,
            guidance_scale=7.,
            num_inference_steps=num_inference_steps,
            progress_bar=progress_bar,
            negative_prompt=negative_prompt,
            seed=seed
        )
        indices = [
            int(np.ceil(sum(durations[:i])*generation_fps)) for i in range(len(durations))
        ]
        selected_frames = [self.frames[idx] for idx in indices]
        self.gen.save_video(self.frames, music_path, 'temp.mp4', fps=generation_fps)
        return selected_frames, 'temp.mp4'


    def upsample_video(self, generation_fps, final_fps, interpolation_type, music_path):
        gc.collect()
        temp_frames = self.gen.upsample(
            self.frames, 
            scale=final_fps // generation_fps, 
            linear_interpolation=interpolation_type == 'linear',
            progress_bar=gr.Progress().tqdm
        )
        self.gen.save_video(temp_frames, music_path, 'temp_up.mp4', fps=final_fps)
        del temp_frames
        gc.collect()
        return 'temp_up.mp4'

    def plot_segments(self, segments):
        segments.sort(key=lambda x: x['start'])
        fig, ax = plt.subplots(1, 1, figsize=(15, 5), sharex=True)
    
        colors = plt.cm.rainbow(np.linspace(0, 1, len(segments)))
        for idx, seg in enumerate(segments):
            rect = plt.Rectangle(
                xy=(seg['start'], 0), 
                width=seg['end'] - seg['start'], 
                height=1, facecolor=colors[idx], alpha=0.5, edgecolor='black'
            )
            ax.add_patch(rect)
        ax.set_title('Segments')
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.set_xlabel('Time')
        ax.set_xlim(min(seg['start'] for seg in segments), max(seg['end'] for seg in segments))
        plt.tight_layout()
        return fig
    
    def plot_emotions(self, segments, keys):
        fig, ax = plt.subplots(1, 1, figsize=(15, 5))
        ax.set_title('Emotion Dimenions')
        for key in keys:
            arr = []
            for seg in segments:
                arr += [np.mean(seg[key])]
            ax.plot(arr, label=key)
        plt.legend()
        return fig

    def setup_video_interface(self):
        with gr.Blocks() as demo:
            with gr.Column():
                gr.Markdown("# Music Video Generation")
                with gr.Row():
                    style = gr.Textbox(lines=1, placeholder="Enter style", label="Style")
                    width = gr.Slider(minimum=128, step=128, maximum=1024, value=512, label="Video Width")
                    height = gr.Slider(minimum=128, step=128, maximum=1024, value=512, label="Video Height")
                    seed = gr.Slider(minimum=0, step=1, maximum=10000, value=7777, label="Random Seed")
                    strength = gr.Slider(minimum=0.1, step=0.01, maximum=0.9, value=0.4, label="Dynamicity")
                    num_inference_steps = gr.Slider(
                        minimum=1, step=1, maximum=10, value=4, label="Number of refinement steps"
                    )
            
                durations = gr.Textbox(
                    lines=1, placeholder="Enter durations separated by commas", label="Durations"
                )
                prompts = gr.Textbox(
                    lines=5, placeholder="Enter prompts separated by enter", label="Prompts"
                )
                zoom_directions = gr.Textbox(
                    lines=1, placeholder="Enter zoom directions separated by commas", label="Zoom Directions"
                )
                zoom_factors = gr.Textbox(
                    lines=1, placeholder="Enter zoom factors separated by commas", label="Zoom Factors"
                )
                rotation_directions = gr.Textbox(
                    lines=1, placeholder="Enter rotation directions separated by commas", label="Rotation Directions"
                )
                rotation_factors = gr.Textbox(
                    lines=1, placeholder="Enter rotation factors separated by commas", label="Rotation Factors"
                )
                move_directions = gr.Textbox(
                    lines=1, placeholder="Enter move directions separated by commas", label="Move Directions"
                )
                move_factors = gr.Textbox(
                    lines=1, placeholder="Enter move factors separated by commas", label="Move Factors"
                )
                negative_prompt = gr.Textbox(
                    lines=1, placeholder="Enter negative prompt", label="Negative Prompt"
                )
                generation_fps = gr.Slider(minimum=1, step=1, maximum=5, value=2, label="Generation FPS")
                music_file = gr.Audio(label="Upload Music File", type='filepath', sources='upload')
        
                generate_button = gr.Button("Generate")
                video_gallery = gr.Gallery(
                    label="Segment Init Frames", 
                    type='pil',
                    columns=10,
                    # height="auto"
                )
                generated_video = gr.Video(label="Generated Video")
        
                gr.Markdown("# Video Upsampling")
                with gr.Row():
                    interpolation_type = gr.Dropdown(["linear", "spherical"], label="Interpolation Type")
                    final_fps = gr.Slider(minimum=10, step=10, maximum=30, value=10, label="Final FPS")
                upsample_button = gr.Button("Upsample")
                upsample_video = gr.Video(label="Upsampled Video")
                
            generate_button.click(
                self.generate_video,
                inputs=[
                    style, 
                    width, 
                    height, 
                    seed,
                    strength, 
                    num_inference_steps, 
                    durations,
                    prompts, 
                    zoom_directions, 
                    zoom_factors, 
                    rotation_directions, 
                    rotation_factors, 
                    move_directions, 
                    move_factors, 
                    negative_prompt, 
                    generation_fps, 
                    music_file
                ],
                outputs=[video_gallery, generated_video]
            )
            upsample_button.click(
                self.upsample_video,
                inputs=[generation_fps, final_fps, interpolation_type, music_file],
                outputs=[upsample_video]
            )
        return demo

    def setup_music_interface(self):
        with gr.Blocks() as demo:
            with gr.Column():
                gr.Markdown("# Music Analysis")
                music_file = gr.Audio(label="Upload Music File", type='filepath', sources='upload')
                with gr.Row():
                    segment_threshold = gr.Slider(
                        minimum=0.01, step=0.01, maximum=0.99, value=0.1, label="Segmentation Threshold"
                    )
                    instrument_threshold = gr.Slider(
                        minimum=0.01, step=0.01, maximum=0.99, value=0.3, label="Instrumentation Threshold"
                    )
                    genre_threshold = gr.Slider(
                        minimum=0.01, step=0.01, maximum=0.99, value=0.3, label="Genre Threshold"
                    )

                analyse_button = gr.Button("Analyse")
                segment_plot = gr.Plot(label="Segments", format="png")
                emotion_plot = gr.Plot(label="Segment Emotion", format="png")
                durations = gr.Text(label='Segments Durations')
            
            analyse_button.click(
                self.analyse_music,
                inputs=[music_file, segment_threshold, instrument_threshold, genre_threshold],
                outputs=[segment_plot, emotion_plot, durations]
            )
        return demo

    def setup_speech_interface(self):
        with gr.Blocks() as demo:
            with gr.Column():
                gr.Markdown("# Voice Over Analysis")
                speech_file = gr.Audio(label="Upload Voice-over File", type='filepath', sources='upload')
                preprocess_button = gr.Button("Preprocess")
                
                with gr.Row():
                    max_dist = gr.Slider(
                        minimum=1., step=0.5, maximum=10, value=5, label="Segments Max Distance"
                    )
                    extract_kw = gr.Checkbox(label='extract_keywords', info='Extract Keywords')
                    num_keywords = gr.Slider(
                        minimum=1, step=1, maximum=10, value=5, label="Number of Keywords"
                    )

                analyse_button = gr.Button("Analyse")
                segment_plot = gr.Plot(label="Segments", format="png")
                emotion_plot = gr.Plot(label="Segment Emotion", format="png")
                text = gr.Text(label='Segments Transcription')

                preprocess_button.click(
                    self.preprocess_speech,
                    inputs=[speech_file],
                    outputs=[segment_plot, emotion_plot, text]
                )
                analyse_button.click(
                    self.analyse_speech,
                    inputs=[max_dist, extract_kw, num_keywords],
                    outputs=[segment_plot, emotion_plot, text]
                )
        return demo

    def run(self, *args, **kwargs):
        self.interface.launch(*args, **kwargs)


if __name__ == '__main__':
    mir = MusicInformationRetreiver(weights_path='mir_weights/')
    sir = SpeechInformationRetreiver(
        model_name='stt_en_fastconformer_hybrid_large_streaming_1040ms',
        lookahead_size=1040,
        decoder_type='rnnt',
        num_keywords=10,
        device='cuda'
    )
    gen = VideoGenerator()
    
    demo = Demo(mir, sir, gen)
    demo.run(share=True)