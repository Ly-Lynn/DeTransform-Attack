import argparse
import os
from models import Model
from attack import WhiteAttack
from utils import save_image
from PIL import Image
from torchvision import transforms
import torch
import random
import numpy as np
import time
import cv2
from tqdm import tqdm

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def parse_args():
    parser = argparse.ArgumentParser(description="DeTransform Attack")
    parser.add_argument("--model", type=str, default="efficientnet_v2_m", help="Model name")
    parser.add_argument("--data_dir", type=str, default="data/", help="Directory containing images")
    parser.add_argument("--output_dir", type=str, default="output/", help="Directory to save adversarial images")
    parser.add_argument("--attack_type", type=str, default="FGSM", choices=["FGSM", "PGD", "DDN"], help="Attack algorithm")
    parser.add_argument("--epsilon", type=float, default=0.1, help="Epsilon value for attack")
    parser.add_argument("--num_steps", type=int, default=10, help="Number of steps for PGD attack")
    parser.add_argument("--alpha", type=float, default=0.01, help="Alpha value for PGD attack")
    parser.add_argument("--seed", type=int, default=22520766, help="Random seed for reproducibility")
    return parser.parse_args()
    
if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    print(f"Parameters:\nModel: {args.model}\n Attack Type: {args.attack_type}\n Epsilon: {args.epsilon}\n Alpha: {args.alpha}\nNum Steps: {args.num_steps}")
    if args.attack_type == "FGSM":
        params = {
            "epsilon": args.epsilon,
        }
    elif args.attack_type == "PGD":
        params = {
            "epsilon": args.epsilon,
            "alpha": args.alpha,
            "num_steps": args.num_steps,
        }
    elif args.attack_type == "DDN":
        params = {
            "epsilon": args.epsilon,
            "num_steps": args.num_steps,
        }
    txt_output = os.path.join(args.output_dir, "results.txt")

    MODEL = Model(args.model)
    model = MODEL.get_model()
    class_names = MODEL.get_class_name()
    attack = WhiteAttack(model, args.attack_type)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    os.makedirs(args.output_dir, exist_ok=True)
    success_rate = 0
    flag = False
    image_files = [f for f in os.listdir(args.data_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    for image_file in tqdm(image_files, desc="Processing images"):
        image_path = os.path.join(args.data_dir, image_file)
        image = Image.open(image_path).convert("RGB")
        x = transform(image).unsqueeze(0).to(MODEL.device)
        y = model(x)
        predicted = torch.max(y, 1)[1].item()
        score = torch.max(y, 1)[0].item()

        adv_image = attack.attack(x=x, y=y, **params)
        adv_inference = model(adv_image)
        adv_predicted = torch.max(adv_inference, 1)[1].item()
        adv_score = torch.max(adv_inference, 1)[0].item()
        if predicted != adv_predicted:
            success_rate += 1
            flag = True
        else: flag = False
        adv_path = os.path.join(args.output_dir, args.attack_type, f"{flag}_{image_file}")
        os.makedirs(os.path.dirname(adv_path), exist_ok=True)
        save_image(adv_image, adv_path)

        with open(txt_output, "a") as f:
            f.write(f"Original {image_path}: {score} - {predicted} - {class_names[predicted]}\n")
            f.write(f"Adversarial {adv_path}: {adv_score} - {adv_predicted} - {class_names[adv_predicted]}\n")
            f.write("-" * 50 + "\n")

    success_rate /= len(image_files)
    print(f"Success Rate: {success_rate:.2f}")
    with open(txt_output, "a") as f:
        f.write(f"Success Rate: {success_rate:.2f}\n")