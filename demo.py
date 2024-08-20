import gradio as gr

from src.music.mir import MusicInformationRetreiver
from src.video.gen import VideoGenerator

mir = MusicInformationRetreiver(weights_path='mir_weights/')
gen = VideoGenerator()


def get_segment_durations(music_path):
    segments = mir(
        music_path, 
        segment_threshold=0.1, 
        genre_threshold=0.4, 
        inst_threshold=0.4, 
        add_segment_info=False
    )
    return [seg['duration'] for seg in segments]


def fn(
    prompts, 
    zoom_directions, 
    zoom_factors, 
    rotate_directions,
    rotate_factors,
    move_directions,
    move_factors,
    fps, 
    music_path
):
    prompts = prompts.strip().split('\n')
    zoom_directions = None if zoom_directions == '' else zoom_directions.strip().split(',')
    rotate_directions = None if rotate_directions == '' else rotate_directions.strip().split(',')
    move_directions = None if move_directions == '' else move_directions.strip().split(',')
    zoom_factors = None if zoom_factors == '' else list(map(float, zoom_factors.strip().split(',')))
    rotate_factors = None if rotate_factors == '' else list(map(float, rotate_factors.strip().split(',')))
    move_factors = None if move_factors == '' else list(map(float, move_factors.strip().split(',')))
    
    durations = get_segment_durations(music_path)
    frames = gen.generate(
        durations, 
        style='realistic',
        generation_fps=2,
        final_fps=fps,
        width=512, 
        height=512,
        strength=0.4,
        prompts=prompts, 
        rotate_directions=rotate_directions,
        zoom_directions=zoom_directions,
        move_directions=move_directions,
        zoom_factors=zoom_factors,
        rotate_factors=rotate_factors,
        move_factors=move_factors,
        guidance_scale=8.,
        num_inference_steps=4,
        progress_bar=gr.Progress().tqdm,
        negative_prompt='blurry, fuzzy, low quality, chaotic, poor details, dark, sad, text, letters, alphabet'
    )
    gen.save_video(frames, music_path, 'temp.mp4', fps=fps)
    return 'temp.mp4'



demo = gr.Interface(
    fn=fn,
    inputs=[
        gr.Textbox(lines=5, placeholder="Enter prompts separated by enter"),
        gr.Textbox(lines=5, placeholder="Enter zoom directions separated by commas"),
        gr.Textbox(lines=5, placeholder="Enter zoom factors separated by commas"),
        gr.Textbox(lines=5, placeholder="Enter rotation directions separated by commas"),
        gr.Textbox(lines=5, placeholder="Enter rotation factors separated by commas"),
        gr.Textbox(lines=5, placeholder="Enter move directions separated by commas"),
        gr.Textbox(lines=5, placeholder="Enter move factors separated by commas"),
        gr.Slider(minimum=10, step=10, maximum=30, value=20, label="Frames per second"),
        gr.Audio(label="Upload Music File", type='filepath', sources='upload')
    ],
    outputs=[
        gr.Video(label="Generated Video")
    ],
    title="Music Video Generation App",
    description="Generate a music video using Stable Diffusion based on prompts and music input."
)

demo.launch(share=True)
    