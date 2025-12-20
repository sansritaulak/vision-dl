# day5/inference.py
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import json
from timm import create_model

# Load your best model (change path if needed)
MODEL_PATH = "artifacts/randaug_final.pth"   # ← UPDATE IF DIFFERENT
model = create_model("resnet18", pretrained=False, num_classes=10)
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

# Temperature from Day 4 (change if yours is different)
TEMPERATURE = 1.042

classes = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize([0.4914,0.4822,0.4465], [0.2470,0.2430,0.2610])
])

def predict(image_path: str):
    img = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        logits = model(tensor)
        logits = logits / TEMPERATURE       
        probs = F.softmax(logits, dim=1).squeeze(0)
    
    conf, pred = torch.max(probs, 0)
    return {
        "prediction": classes[pred.item()],
        "confidence": float(conf.item()),
        "all_probabilities": {c: float(p) for c, p in zip(classes, probs)}
    }

if __name__ == "__main__":
    import sys
    result = predict(sys.argv[1])
    print(json.dumps(result, indent=2))