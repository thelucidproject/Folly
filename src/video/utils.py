import numpy as np
import cv2
from PIL import Image
import torch

def slerp(v0, v1, t, DOT_THRESHOLD=0.999):
    """ helper function to spherically interpolate two arrays v1 v2 """
    device = v0.device
    v0 = v0.cpu().numpy()
    v1 = v1.cpu().numpy()
    t = t.item()

    dot = np.sum(v0 * v1 / (np.linalg.norm(v0) * np.linalg.norm(v1)))
    if np.abs(dot) > DOT_THRESHOLD:
        v2 = (1 - t) * v0 + t * v1
    else:
        theta_0 = np.arccos(dot)
        sin_theta_0 = np.sin(theta_0)
        theta_t = theta_0 * t
        sin_theta_t = np.sin(theta_t)
        s0 = np.sin(theta_0 - theta_t) / sin_theta_0
        s1 = sin_theta_t / sin_theta_0
        v2 = s0 * v0 + s1 * v1
    return torch.from_numpy(v2).to(device)



def get_mask(image):
    mask = np.array(image).sum(axis=2)
    mask[mask > 0] = 255
    return 255 - mask.astype('uint8')

def zoom(image, factor=0.05, direction='in', fill_blanks=False):
    if direction == 'out':
        mask_width = int(factor * image.width / 2)
        mask_height = int(factor * image.height / 2)
        if mask_width == 0 and mask_height == 0:
            return image
        resized_image = image.resize((image.height - 2 * mask_height, image.width - 2 * mask_width))
        blank = np.zeros_like(image)
        blank[mask_width:-mask_width, mask_width:-mask_width] = resized_image
        resized_image = Image.fromarray(blank)
        
    elif direction == 'in':
        original_width, original_height = image.size
        new_width = int(original_width * (1 + factor))
        new_height = int(original_height * (1 + factor))
        
        resized_image = image.resize((new_width, new_height), Image.LANCZOS)
        
        left = (new_width - original_width) / 2
        top = (new_height - original_height) / 2
        right = left + original_width
        bottom = top + original_height
        resized_image = resized_image.crop((left, top, right, bottom))
        
    if fill_blanks:
        resized_image = inpaint(resized_image)
    return resized_image.convert('RGB')
    
def rotate(image, factor=0.001, direction='pos', fill_blanks=False):
    angle = 360 * factor
    if direction == 'neg':
        angle = -angle
    rotated_image = image.rotate(angle, expand=False)
    if fill_blanks:
        rotated_image = inpaint(rotated_image)
    return rotated_image.convert('RGB')

def move(image, factor, direction, fill_blanks=False):
    temp = np.array(image)
    if direction == 'up':
        width = int(factor * image.height)
        temp = np.pad(temp, ((width, 0), (0, 0), (0, 0)))
        temp = temp[:image.height]
    elif direction == 'down':
        width = int(factor * image.height)
        temp = np.pad(temp, ((0, width), (0, 0), (0, 0)))
        temp = temp[-image.height:]
    elif direction == 'left':
        width = int(factor * image.width)
        temp = np.pad(temp, ((0, 0), (width, 0), (0, 0)))
        temp = temp[:, :image.width]
    elif direction == 'right':
        width = int(factor * image.width)
        temp = np.pad(temp, ((0, 0), (0, width), (0, 0)))
        temp = temp[:, -image.width:]

    res = Image.fromarray(temp)
    if fill_blanks:
        res = inpaint(res)
    return res.convert('RGB')

def inpaint(image):
    mask = get_mask(image)
    img = np.array(image)
    img = cv2.inpaint(img, mask, 3, cv2.INPAINT_NS)
    return Image.fromarray(img)