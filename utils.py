from PIL import Image
import torch
import os

def convert_into_int(image: torch.Tensor) -> torch.Tensor:
    return torch.clamp(image * 255.0, 0, 255).to(torch.uint8)

def save_image(image: torch.Tensor, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    image = convert_into_int(image.squeeze(0).cpu()).permute(1, 2, 0)
    
    image = Image.fromarray(image.numpy().astype("uint8")).convert("RGB")
    image.save(path)

def de_normalize(image: torch.Tensor, mean:list, std:list) -> torch.Tensor:
    mean = torch.tensor(mean, device=image.device, dtype=image.dtype)
    std = torch.tensor(std, device=image.device, dtype=image.dtype)
    image = image * std[:, None, None] + mean[:, None, None]
    return torch.clamp(image, 0, 1)