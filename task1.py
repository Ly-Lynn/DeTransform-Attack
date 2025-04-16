import argparse
import os
from models import Model
from attack import WhiteAttack
from utils import save_image, de_normalize
from PIL import Image
from torchvision import transforms
import torch
import random
import numpy as np
import time
import cv2
from tqdm import tqdm
from logger import Logger

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
    print(f"Parameters:\nModel: {args.model}\nAttack Type: {args.attack_type}\nEpsilon: {args.epsilon}\nAlpha: {args.alpha}\nNum Steps: {args.num_steps}")
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
    
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    MODEL = Model(args.model)
    model = MODEL.get_model()
    class_names = MODEL.get_class_name()
    output_dir = os.path.join(args.output_dir, args.model, args.attack_type)
    attack = WhiteAttack(model, args.attack_type)
    pkl_logger = Logger(os.path.join(args.output_dir, args.model), types="pkl")
    txt_logger = Logger(os.path.join(args.output_dir, args.model), types="txt")   

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    os.makedirs(output_dir,  exist_ok=True)
    success_rate = 0
    case1, case2, case3, case4 = 0, 0, 0, 0
    flag = False
    image_files = [f for f in os.listdir(args.data_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    for image_file in tqdm(image_files, desc="Processing images"):
        image_path = os.path.join(args.data_dir, image_file)
        image = Image.open(image_path).convert("RGB")
        x = transform(image).unsqueeze(0).to(MODEL.device)
        y, predicted, score = MODEL.inference(x)

        adv_image = attack.attack(x=x, y=y, **params)
        adv_res, adv_predicted, adv_score = MODEL.inference(adv_image)

        if predicted != adv_predicted:
            success_rate += 1
            flag = True
        else:
            flag = False

        adv_path = os.path.join(output_dir, f"{flag}_{image_file}")
        os.makedirs(os.path.dirname(adv_path), exist_ok=True)
        
        adv_image = de_normalize(adv_image, mean, std)
        save_image(adv_image, adv_path)

        before_data = (predicted, class_names[predicted], score, y)
        after_data = (adv_predicted, class_names[adv_predicted], adv_score, adv_res)
        pkl_data = {
            "img_path": image_path,
            "adv_img": adv_path,
            "adv": adv_image,
            "before_atk": before_data[:3],
            "after_atk": after_data[:3],
            "softmax_before": before_data[-1],
            "softmax_after": after_data[-1],
        }

        pkl_logger(pkl_data, f"{args.attack_type}.pkl")

        # ------------- Re-evaluate attacks -------------
        _adv_img = Image.open(adv_path).convert("RGB")
        _adv_img = transform(_adv_img).unsqueeze(0).to(MODEL.device)
        re_res, re_predicted, re_score = MODEL.inference(_adv_img)
        """
        1/ adv class == re predicted
        1/ adv class != re predicted but re predicted == original class
        3/ adv class != re predicted and original class but adv class == original class (atk failed)
        4/ adv class != re predicted and original class
        """            
        if re_predicted == adv_predicted:
            case1 += 1
            txt_logger(f"\nCase 1: {flag} success {re_predicted}=={adv_predicted} ({re_score:.4f}-{adv_score:.4f})",
                       f"{args.attack_type}.txt")
        elif re_predicted != adv_predicted:
            if re_predicted == predicted:
                case2 += 1
                txt_logger(f"\nCase 2: {flag} failed {re_predicted} != {adv_predicted} ({re_score:.4f}-{adv_score:.4f}) but {re_predicted}=={predicted} ({re_score:.4f}-{score:.4f})",
                       f"{args.attack_type}.txt")
            elif re_predicted != predicted:
                if adv_predicted == predicted:
                    case3 += 1
                    txt_logger(f"\nCase 3: {flag} failed {re_predicted}!={adv_predicted} ({re_score:.4f}-{adv_score:.4f}) and {re_predicted}!={predicted} ({re_score:.4f}-{score:.4f})",
                       f"{args.attack_type}.txt")
                else:
                    case4 += 1
                    txt_logger(f"\nCase 4: {flag} failed {re_predicted}!={adv_predicted} ({re_score:.4f}-{adv_score:.4f}) and {re_predicted}!={predicted} ({re_score:.4f}-{score:.4f})",
                       f"{args.attack_type}.txt")
            
        
    success_rate /= len(image_files)
    print(f"Total images: {len(image_files)}")
    print(f"1st phase - Attack Success Rate: {success_rate:.2f}")
    print(f"2nd phase - Re-evaluate Rate: {case1} - {case2} - {case3} - {case4}")


    