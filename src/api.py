from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import torch
import torch.nn.functional as F
import timm
import torch.nn as nn
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import cv2
from pathlib import Path
import io

app = FastAPI(
    title="Brain Tumor MRI Diagnostic API",
    description="PFA 2024/2025 — Nadia Zemani — Pr. Mohamed LAZAAR",
    version="1.0.0"
)

# ── Config ────────────────────────────────────────────────────────────────────
CLASSES    = ['glioma', 'meningioma', 'notumor', 'pituitary']
IMG_SIZE   = 300
DEVICE     = torch.device('cpu')
MODEL_PATH = Path('models/model_complet_nadia_94pct.pth')

CLASS_INFO = {
    'glioma'     : {'fr': 'Gliome',               'severity': 'ÉLEVÉE',  'urgent': True},
    'meningioma' : {'fr': 'Méningiome',            'severity': 'MODÉRÉE', 'urgent': False},
    'pituitary'  : {'fr': 'Tumeur hypophysaire',   'severity': 'MODÉRÉE', 'urgent': False},
    'notumor'    : {'fr': 'Aucune tumeur détectée','severity': 'FAIBLE',  'urgent': False},
}

# ── Modèle ────────────────────────────────────────────────────────────────────
class BrainTumorClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone   = timm.create_model('efficientnet_b3', pretrained=False, num_classes=0)
        in_features     = self.backbone.num_features
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Dropout(0.4),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2),
            nn.Linear(512, 4),
        )
    def forward(self, x):
        return self.classifier(self.backbone(x))

# ── Chargement du modèle au démarrage ────────────────────────────────────────
print("⏳ Chargement du modèle...")
model = BrainTumorClassifier().to(DEVICE)

if MODEL_PATH.exists():
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    TRAIN_MEAN = checkpoint.get('train_mean', [0.485, 0.456, 0.406])
    TRAIN_STD  = checkpoint.get('train_std',  [0.229, 0.224, 0.225])
    print(f"✅ Modèle chargé — best_val_acc: {checkpoint.get('best_val_acc', 'N/A')}")
else:
    TRAIN_MEAN = [0.485, 0.456, 0.406]
    TRAIN_STD  = [0.229, 0.224, 0.225]
    print("⚠️  Modèle non trouvé — utilisation des poids aléatoires")

model.eval()

transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=TRAIN_MEAN, std=TRAIN_STD),
    ToTensorV2(),
])

# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "message" : "Brain Tumor MRI Diagnostic API",
        "version" : "1.0.0",
        "auteur"  : "Nadia Zemani",
        "status"  : "running"
    }

@app.get("/health")
def health():
    return {"status": "healthy", "model": "EfficientNet-B3", "classes": CLASSES}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Lire l'image
        contents = await file.read()
        nparr    = np.frombuffer(contents, np.uint8)
        img      = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img      = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Prétraitement
        augmented   = transform(image=img)
        input_tensor = augmented['image'].unsqueeze(0).to(DEVICE)

        # Prédiction
        with torch.no_grad():
            logits = model(input_tensor)
            probs  = F.softmax(logits, dim=1)[0].cpu().numpy()

        pred_idx   = int(np.argmax(probs))
        pred_class = CLASSES[pred_idx]
        confidence = float(probs[pred_idx])
        info       = CLASS_INFO[pred_class]

        # Incertitude
        entropy     = -np.sum(probs * np.log(probs + 1e-10))
        uncertainty = float(entropy / np.log(len(CLASSES)))

        return JSONResponse({
            "status"        : "success",
            "prediction"    : pred_class,
            "prediction_fr" : info['fr'],
            "confidence"    : round(confidence, 4),
            "uncertainty"   : round(uncertainty, 4),
            "severity"      : info['severity'],
            "urgent"        : info['urgent'],
            "probabilities" : {
                cls: round(float(p), 4)
                for cls, p in zip(CLASSES, probs)
            },
            "model"         : "EfficientNet-B3",
            "author"        : "Nadia Zemani — PFA 2024/2025"
        })

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )
