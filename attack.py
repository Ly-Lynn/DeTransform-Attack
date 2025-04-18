import torch
import torch.nn.functional as F
from torch.nn.parallel import DataParallel
from utils import (save_image, 
                   convert_into_int)
class WhiteAttack:
    def __init__(self,
                model,
                alg="FGSM",):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model

        alg_func = {
            "FGSM": self.fgsm,
            "PGD": self.pgd,
            "DDN": self.ddn,
        }    
        if alg in alg_func:
            self.attack_call = alg_func[alg]
        else:
            raise ValueError(f"Unknown attack algorithm: {alg}. Supported algorithms are: {list(alg_func.keys())}")
    def convert_xy(self, x, y):
        x = x.clone().detach().to(self.device)
        y = y.clone().detach().to(self.device)
        return x, y
    def fgsm(self, x, y, epsilon=0.1):
        x, y = self.convert_xy(x, y)
        x.requires_grad = True
        output = self.model(x)
        self.model.zero_grad()
        loss = F.cross_entropy(output, y)
        loss.backward()
        x_grad = x.grad.data
        x_adv = x + epsilon * x_grad.sign()
        return x_adv.detach()
    def pgd(self, x, y, epsilon=0.1, alpha=0.01, num_steps=10):
        x, y = self.convert_xy(x, y)
        x_adv = x.clone().detach()
        for _ in range(num_steps):
            x_adv.requires_grad = True
            output = self.model(x_adv)
            self.model.zero_grad()
            loss = F.cross_entropy(output, y)
            loss.backward()
            x_grad = x_adv.grad.data
            x_adv = x_adv + alpha * x_grad.sign()
            x_adv = torch.clamp(x_adv, 0, 1)
            x_adv = torch.clamp(x_adv, x - epsilon, x + epsilon)
        return x_adv.detach()
    def ddn(self, x, y, epsilon=0.1, num_steps=10):
        x, y = self.convert_xy(x, y)
        x_adv = x.clone().detach()
        for _ in range(num_steps):
            x_adv.requires_grad = True
            output = self.model(x_adv)
            self.model.zero_grad()
            loss = F.cross_entropy(output, y)
            loss.backward()
            x_grad = x_adv.grad.data
            x_adv = x_adv + epsilon * x_grad.sign()
            x_adv = torch.clamp(x_adv, 0, 1)
        return x_adv.detach()
    def attack(self, **kwargs):
        return self.attack_call(**kwargs) if self.attack_call else None
    

if __name__ == "__main__":
    from models import Model
    from torchvision import transforms
    from PIL import Image

    MODEL = Model("efficientnet_v2_m")
    model = MODEL.get_model()
    class_name = MODEL.get_class_name()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img = Image.open(r"D:\codePJ\RESEARCH\DeTransform-Attack\data\DOG.jpg").convert("RGB")
    x = transform(img).unsqueeze(0).to(MODEL.device)  
    save_image(x, "test/test_org.jpg")
    y = model(x)

    predicted = torch.max(y, 1)[1].item()
    print(f"Original class: {predicted} - {class_name[predicted]}")
    
    attack = WhiteAttack(alg="FGSM", model=model)

    x_adv = attack.attack(x=x, y=y, epsilon=0.01)
    y_adv = model(x_adv)
    predicted_adv = torch.max(y_adv, 1)[1].item()
    print(f"{torch.max(y_adv, 1)}")
    print(f"Adversarial class: {predicted_adv} - {class_name[predicted_adv]}")
    save_image(x_adv, "test/test.jpg")