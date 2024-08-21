import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import gradio as gr
import gc

from src.music.mir import MusicInformationRetreiver
from src.video.gen import VideoGenerator

gen = VideoGenerator()
frames = None

def gen_fn(
    style,
    width, 
    height,
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

    global gen, frames
    
    durations = list(map(float, durations.strip().split(', ')))
    prompts = prompts.strip().split('\n')
    zoom_directions = None if zoom_directions == '' else zoom_directions.strip().split(', ')
    rotate_directions = None if rotate_directions == '' else rotate_directions.strip().split(', ')
    move_directions = None if move_directions == '' else move_directions.strip().split(', ')
    zoom_factors = None if zoom_factors == '' else list(map(float, zoom_factors.strip().split(', ')))
    rotate_factors = None if rotate_factors == '' else list(map(float, rotate_factors.strip().split(', ')))
    move_factors = None if move_factors == '' else list(map(float, move_factors.strip().split(', ')))

    frames = None
    gc.collect()

    # Progress bar handling for Gradio
    progress_bar = gr.Progress().tqdm
    
    frames = gen.generate(
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
        guidance_scale=10.,
        num_inference_steps=num_inference_steps,
        progress_bar=progress_bar,
        negative_prompt=negative_prompt
    )
    gen.save_video(frames, music_path, 'temp.mp4', fps=generation_fps)
    return 'temp.mp4'


def upsample_fn(generation_fps, final_fps, interpolation_type, music_path):
    global gen, frames
    gc.collect()
    temp_frames = gen.upsample(
        frames, 
        scale=final_fps // generation_fps, 
        linear_interpolation=interpolation_type == 'linear',
        progress_bar=gr.Progress().tqdm
    )
    gen.save_video(temp_frames, music_path, 'temp_up.mp4', fps=final_fps)
    del temp_frames
    gc.collect()
    return 'temp_up.mp4'


with gr.Blocks() as demo:
    with gr.Column():
        gr.Markdown("# Music Video Generation")
        with gr.Row():
            style = gr.Textbox(lines=1, placeholder="Enter style", label="Style")
            width = gr.Slider(minimum=128, step=128, maximum=1024, value=512, label="Video width")
            height = gr.Slider(minimum=128, step=128, maximum=1024, value=512, label="Video height")
            strength = gr.Slider(minimum=0.1, step=0.01, maximum=0.9, value=0.45, label="Dynamicity")
            num_inference_steps = gr.Slider(minimum=1, step=1, maximum=10, value=4, label="Number of refinement steps")
    
        durations = gr.Textbox(lines=1, placeholder="Enter durations separated by commas", label="Durations")
        prompts = gr.Textbox(lines=5, placeholder="Enter prompts separated by enter", label="Prompts")
        zoom_directions = gr.Textbox(lines=1, placeholder="Enter zoom directions separated by commas", label="Zoom Directions")
        zoom_factors = gr.Textbox(lines=1, placeholder="Enter zoom factors separated by commas", label="Zoom Factors")
        rotation_directions = gr.Textbox(lines=1, placeholder="Enter rotation directions separated by commas", label="Rotation Directions")
        rotation_factors = gr.Textbox(lines=1, placeholder="Enter rotation factors separated by commas", label="Rotation Factors")
        move_directions = gr.Textbox(lines=1, placeholder="Enter move directions separated by commas", label="Move Directions")
        move_factors = gr.Textbox(lines=1, placeholder="Enter move factors separated by commas", label="Move Factors")
        negative_prompt = gr.Textbox(lines=1, placeholder="Enter negative prompt", label="Negative Prompt")
        generation_fps = gr.Slider(minimum=1, step=1, maximum=5, value=2, label="Generation FPS")
        music_file = gr.Audio(label="Upload Music File", type='filepath', sources='upload')

        generate_button = gr.Button("Generate")
        generated_video = gr.Video(label="Generated Video")

        gr.Markdown("# Video Upsampling")
        with gr.Row():
            interpolation_type = gr.Dropdown(["linear", "spherical"], label="Interpolation Type")
            final_fps = gr.Slider(minimum=10, step=10, maximum=30, value=10, label="Final FPS")
        upsample_button = gr.Button("Upsample")
        upsample_video = gr.Video(label="Upsampled Video")
        
    generate_button.click(
        gen_fn,
        inputs=[
            style, 
            width, 
            height, 
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
        outputs=[generated_video]
    )
    upsample_button.click(
        upsample_fn,
        inputs=[generation_fps, final_fps, interpolation_type, music_file],
        outputs=[upsample_video]
    )

demo.launch(share=True)