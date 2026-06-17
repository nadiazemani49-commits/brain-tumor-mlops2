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
import time
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response
import time
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

app = FastAPI(
    title="Brain Tumor MRI Diagnostic API",
    description="PFA — Nadia Zemani",
    version="1.0.0"
)

# ── Métriques Prometheus ──────────────────────────────────────────────────────
PREDICT_COUNTER    = Counter("predictions_total", "Total prédictions", ["predicted_class"])
PREDICT_LATENCY    = Histogram("prediction_latency_seconds", "Latence prédictions",
                               buckets=[.1, .25, .5, 1.0, 2.5, 5.0])
CONFIDENCE_GAUGE   = Gauge("prediction_confidence_last", "Confiance dernière prédiction")
UNCERTAINTY_GAUGE  = Gauge("prediction_uncertainty_last", "Incertitude dernière prédiction")
ERROR_COUNTER      = Counter("prediction_errors_total", "Total erreurs")


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
import os
CI_MODE = os.getenv("CI", "false").lower() == "true"
print("⏳ Chargement du modèle...")
model = BrainTumorClassifier().to(DEVICE)

if MODEL_PATH.exists() and not CI_MODE:
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    TRAIN_MEAN = checkpoint.get('train_mean', [0.485, 0.456, 0.406])
    TRAIN_STD  = checkpoint.get('train_std',  [0.229, 0.224, 0.225])
    print(f"✅ Modèle chargé — best_val_acc: {checkpoint.get('best_val_acc', 'N/A')}")
else:
    TRAIN_MEAN = [0.485, 0.456, 0.406]
    TRAIN_STD  = [0.229, 0.224, 0.225]
    if CI_MODE:
        print("⚠️  CI MODE — modèle non chargé (weights random)")
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
    start_time = time.time()
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

        # ── Métriques Prometheus ──────────────────────────────────────
        PREDICT_COUNTER.labels(predicted_class=pred_class).inc()
        PREDICT_LATENCY.observe(time.time() - start_time)
        CONFIDENCE_GAUGE.set(confidence)
        UNCERTAINTY_GAUGE.set(uncertainty)

        # Log pour monitoring
        try:
            log_prediction(
                filename=file.filename,
                prediction=pred_class,
                confidence=confidence,
                probabilities={cls: round(float(p), 4) for cls, p in zip(CLASSES, probs)},
                uncertainty=uncertainty,
            )
        except Exception as log_err:
            print(f"LOG ERROR: {log_err}", flush=True)
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

# ── Import monitoring ─────────────────────────────────────────────────────────
# ── Import monitoring ─────────────────────────────────────────────────────────
import sys, os
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent))
try:
    from monitoring import log_prediction, get_monitoring_stats, generate_drift_report
    MONITORING_ENABLED = True
    print("✅ Monitoring chargé")
except Exception as e:
    MONITORING_ENABLED = False
    print(f"⚠️  Monitoring non disponible: {e}")
    def log_prediction(*args, **kwargs): pass
    def get_monitoring_stats(): return {}
    def generate_drift_report(): return None

@app.get("/metrics")
def metrics():
    """Endpoint Prometheus — métriques temps réel."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/monitoring")
def monitoring():
    """Statistiques de monitoring en temps réel."""
    if not MONITORING_ENABLED:
        return {"error": "Monitoring non disponible"}
    return get_monitoring_stats()

@app.get("/monitoring/report")
def monitoring_report():
    """Génère et retourne le rapport HTML de drift."""
    if not MONITORING_ENABLED:
        return {"error": "Monitoring non disponible"}
    path = generate_drift_report()
    if path:
        return {"status": "ok", "report_path": path}
    return {"status": "not_enough_data", "message": "Minimum 50 prédictions nécessaires"}

@app.get("/model/info")
def model_info():
    """Retourne la version du modèle en production depuis MLflow Registry."""
    try:
        import mlflow
        from mlflow.tracking import MlflowClient
        mlflow.set_tracking_uri("http://brain-tumor-mlflow:5000")
        client = MlflowClient()
        versions = client.get_latest_versions("BrainTumorEfficientNetB3", stages=["Production"])
        if versions:
            v = versions[0]
            return {
                "model"      : v.name,
                "version"    : v.version,
                "stage"      : v.current_stage,
                "description": v.description,
                "run_id"     : v.run_id,
            }
        return {"model": "BrainTumorEfficientNetB3", "version": "1", "stage": "Production"}
    except Exception as e:
        return {"model": "EfficientNet-B3", "version": "1", "stage": "Production", "note": str(e)}
