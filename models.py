import torchvision.models as models
import torch

from torchvision import transforms
from PIL import Image

class Model:
    def __init__(self, name = "efficientnet_v2_m"):
        self.models_list = {
            "efficientnet_v2_m": [models.efficientnet_v2_m(weights=models.EfficientNet_V2_M_Weights.IMAGENET1K_V1), 
            models.EfficientNet_V2_M_Weights.IMAGENET1K_V1],
            "vgg16": [models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1), models.VGG16_Weights.IMAGENET1K_V1],
            "resnet50": [models.resnet50(weights=
                                         models.ResNet50_Weights.IMAGENET1K_V1), 
                         models.ResNet50_Weights.IMAGENET1K_V1],
        }
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.models_list[name][0].to(self.device)
        self.class_name = self.models_list[name][1].meta["categories"] 
        if torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model)
        self.model.eval()
    def get_model(self):
        return self.model
    def get_class_name(self):
        return self.class_name

if __name__ == "__main__":
    MODEL = Model("efficientnet_v2_m")
    model = MODEL.get_model()
    print(model)
    image_path = r"DOG.jpg"
    image = Image.open(image_path).convert("RGB")
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_tensor = preprocess(image).unsqueeze(0).to(device)  

    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = torch.max(output, 1)
        predicted_class = predicted.item()
    
    print(f"Predicted class: {predicted_class} - {MODEL.get_class_name()[predicted_class]}")