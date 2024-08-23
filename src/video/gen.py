import numpy as np
import torch
from PIL import Image
import random
import pickle
from compel import Compel
from diffusers import AutoPipelineForImage2Image, AutoPipelineForText2Image
from diffusers.utils import load_image, export_to_video
from tqdm.auto import tqdm
import librosa
from torchvision.io import write_video
from torchvision.transforms.functional import pil_to_tensor

from .utils import *


class VideoGenerator:
    def __init__(self, model_name="SimianLuo/LCM_Dreamshaper_v7"):
        self.i2i_pipe = AutoPipelineForImage2Image.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            safety_checker=None
        ).to("cuda")
        
        self.t2i_pipe = AutoPipelineForText2Image.from_pretrained(
            model_name,
            vae=self.i2i_pipe.vae,
            tokenizer=self.i2i_pipe.tokenizer,
            text_encoder=self.i2i_pipe.text_encoder,
            unet=self.i2i_pipe.unet,
            scheduler=self.i2i_pipe.scheduler,
            torch_dtype=torch.float16,
            safety_checker=None
        ).to('cuda')
        self.i2i_pipe.enable_xformers_memory_efficient_attention()
        self.i2i_pipe.set_progress_bar_config(disable=True)
        self.t2i_pipe.enable_xformers_memory_efficient_attention()
        self.t2i_pipe.set_progress_bar_config(disable=True)

        self.compel = Compel(
            tokenizer=self.i2i_pipe.tokenizer, 
            text_encoder=self.i2i_pipe.text_encoder
        )

    def _text_to_image(
        self,
        prompt, 
        negative_prompt=None, 
        num_inference_steps=4, 
        guidance_scale=10., 
        width=512, 
        height=512,
        seed=7
    ):
        torch.cuda.empty_cache()
        return self.t2i_pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=torch.Generator("cuda").manual_seed(seed),
            width=width,
            height=height
        ).images[0]

    def _generate_segment_frames(
        self, 
        prompt,
        style,
        num_frames, 
        rotate_direction, 
        zoom_direction,
        move_direction,
        zoom_factor,
        rotate_factor,
        move_factor,
        width=512, 
        height=512,
        init_frame=None, 
        negative_prompt=None,
        num_inference_steps=4,
        guidance_scale=8.,
        strength=0.4,
        seed=7
    ):
        torch.cuda.empty_cache()
        prompt = f'a {style} photo of {prompt} with high quality, high details, 4k.'
        if init_frame is None:
            init_frame = self._text_to_image(
                prompt=prompt, 
                negative_prompt=negative_prompt, 
                num_inference_steps=num_inference_steps, 
                guidance_scale=guidance_scale,
                width=width,
                height=height,
                seed=seed
            )
            init_strength = strength
        init_frame = init_frame.resize((width, height))
        
        frames = [init_frame]
        for _ in range(num_frames):
            image = frames[-1]
            image = zoom(
                image, 
                factor=zoom_factor, 
                direction=zoom_direction, 
                fill_blanks=True
            )
            image = move(
                image, 
                factor=move_factor, 
                direction=move_direction, 
                fill_blanks=True
            )
            image = rotate(
                image, 
                factor=rotate_factor, 
                direction=rotate_direction, 
                fill_blanks=True
            )
    
            frames += self.i2i_pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=image,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                strength=strength,
                generator=torch.Generator("cuda").manual_seed(seed)
            ).images
        return frames[1:]


    def _encode_image(self, image):
        x = self.t2i_pipe.image_processor.preprocess(image).to(torch.float16).to('cuda')
        return self.t2i_pipe.vae.encode(x).latent_dist.sample().cpu().detach()[0]
    
    def _decode_image(self, z):
        image = self.t2i_pipe.vae.decode(
            z.unsqueeze(0).to('cuda'), 
            return_dict=False
        )[0].cpu().detach()
        return self.t2i_pipe.image_processor.postprocess(image, output_type='pil', do_denormalize=[True])[0]
    
    def _interpolate(self, image1, image2, steps=1, linear=True, weights=None):
        if weights is None:
            weights = torch.linspace(0.0, 1.0, 2 + steps).to(torch.float16)
        else:
            weights = (weights - weights[0]) / (weights[-1] - weights[0])
            
        x = self._encode_image(image1)
        y = self._encode_image(image2)
        res = []
        for w in weights[1:-1]:
            if linear:
                z = torch.lerp(x, y, w)
            else:
                z = slerp(x, y, w)
            res += [self._decode_image(z)]
        return res
    
    def upsample(
        self, 
        frames, 
        scale=1, 
        linear_interpolation=True, 
        audio_reactivity=False, 
        audio_path=None, 
        smooth=0.,
        progress_bar=None
    ):
        if audio_reactivity:
            weights = self.load_audio_weights(audio_path, (len(frames)-1) * scale + 1, smooth=smooth)
        else:
            weights = None
            
        progress_bar = tqdm if progress_bar is None else progress_bar
        res = []
        for i, f in enumerate(progress_bar(frames[:-1])):
            res += [f]
            res += self._interpolate(
                image1=f, 
                image2=frames[i+1], 
                steps=scale - 1, 
                linear=linear_interpolation,
                weights=None if not audio_reactivity else weights[i*scale: (i+1)*scale]
            )
        res += [frames[-1]]
        return res
    
    def save_video(self, frames, audio_path, save_path, fps=20):
        audio, sr = librosa.load(audio_path)
        idx = int(len(frames) / fps * sr) + 1
        audio = torch.tensor(audio[:idx]).unsqueeze(0)
        
        write_video(
            save_path,
            torch.stack([pil_to_tensor(frame) for frame in frames]).permute(0, 2, 3, 1),
            fps=fps,
            audio_array=audio,
            audio_fps=sr,
            audio_codec="aac",
            options={"crf": "10", "pix_fmt": "yuv420p"},
        )

    def load_audio_weights(self, audio_path, n_frames, smooth=0.):
        audio, sr = librosa.load(audio_path)
        spec_raw = librosa.feature.melspectrogram(y=audio, sr=sr)
        spec_max = np.amax(spec_raw, axis=0)
        spec_norm = (spec_max - np.min(spec_max)) / np.ptp(spec_max)
    
        # Resize cumsum of spec norm to our desired number of interpolation frames
        x_norm = np.linspace(0, spec_norm.shape[-1], spec_norm.shape[-1])
        y_norm = np.cumsum(spec_norm)
        y_norm /= y_norm[-1]
        x_resize = np.linspace(0, y_norm.shape[-1], n_frames)
    
        T = np.interp(x_resize, x_norm, y_norm)
        return T * (1 - smooth) + np.linspace(0.0, 1.0, T.shape[0]) * smooth
        

    def generate(
        self, 
        durations, 
        style='realistic',
        generation_fps=2,
        final_fps=20,
        linear_interpolation=True,
        width=512, 
        height=512,
        strength=0.45,
        prompts=None, 
        rotate_directions=None,
        zoom_directions=None,
        move_directions=None,
        zoom_factors=None,
        rotate_factors=None,
        move_factors=None,
        guidance_scale=10.,
        num_inference_steps=4,
        seed=7,
        negative_prompt='blurry, fuzzy, low quality, chaotic, poor details, dark, sad',
        save_path=None,
        progress_bar=None
    ):
        frames = []
        progress_bar = tqdm if progress_bar is None else progress_bar
        for i, dur in enumerate(progress_bar(durations)):
            n = int(np.ceil(generation_fps * dur))
            prev = None if i == 0 else frames[-1]
            seg_frames = self._generate_segment_frames(
                prompt='' if prompts is None else prompts[i],
                style=style,
                num_frames=n, 
                rotate_direction=random.choice(['pos', 'neg']) if rotate_directions is None else rotate_directions[i], 
                zoom_direction=random.choice(['in', 'out']) if zoom_directions is None else zoom_directions[i],
                move_direction=random.choice(['up', 'down', 'left', 'right']) if move_directions is None else move_directions[i],
                zoom_factor=random.choice([0., 0.006]) if zoom_factors is None else zoom_factors[i],
                rotate_factor=random.choice([0., 0.003]) if rotate_factors is None else rotate_factors[i],
                move_factor=random.choice([0., 0.006]) if move_factors is None else move_factors[i],
                width=width, 
                height=height,
                init_frame=prev, 
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                strength=strength,
                seed=seed
            )
            frames += seg_frames
        
        if final_fps > generation_fps:
            assert final_fps % generation_fps == 0
            frames = self.upsample(
                frames, 
                scale=final_fps // generation_fps, 
                linear_interpolation=linear_interpolation, 
                progress_bar=progress_bar
            )
        if isinstance(save_path, str):
            pickle.dump(frames, open(save_path, 'wb'))
        return frames
        