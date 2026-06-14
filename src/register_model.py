"""
register_model.py — Log les métriques + enregistre dans MLflow Registry
Nadia Zemani — PFA 2024/2025
"""
import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient
import os

MLFLOW_URI = "http://localhost:5000"
MODEL_NAME = "BrainTumorEfficientNetB3"

mlflow.set_tracking_uri(MLFLOW_URI)
client = MlflowClient()

# ── 1. Créer l'experiment ─────────────────────────────────────────────────────
exp_name = "Brain-Tumor-MRI-Classification"
exp = mlflow.get_experiment_by_name(exp_name)
if exp is None:
    exp_id = mlflow.create_experiment(exp_name)
else:
    exp_id = exp.experiment_id
print(f"✅ Experiment ID : {exp_id}")

# ── 2. Logger les métriques réelles de ton entraînement Kaggle ────────────────
with mlflow.start_run(experiment_id=exp_id, run_name="EfficientNet-B3-final") as run:
    run_id = run.info.run_id

    # Paramètres du modèle
    mlflow.log_params({
        "model_arch"    : "EfficientNet-B3",
        "dataset"       : "Brain Tumor MRI Dataset",
        "num_classes"   : 4,
        "classes"       : "glioma,meningioma,notumor,pituitary",
        "img_size"      : 300,
        "epochs"        : 40,
        "best_epoch"    : 34,
        "batch_size"    : 32,
        "optimizer"     : "AdamW",
        "augmentation"  : "Albumentations",
        "device"        : "GPU P100 Kaggle",
        "training_time" : "61 minutes",
        "num_images"    : 7023,
    })

    # Métriques finales réelles
    mlflow.log_metrics({
        "val_acc"       : 0.9437,
        "f1_macro"      : 0.944,
        "auc_roc"       : 0.972,
        "val_loss"      : 0.1823,
        # Métriques par classe
        "f1_glioma"     : 0.97,
        "f1_meningioma" : 0.89,
        "f1_notumor"    : 0.98,
        "f1_pituitary"  : 0.95,
    })

    # Tag du run
    mlflow.set_tags({
        "author"    : "Nadia Zemani",
        "project"   : "PFA 2024/2025 ENSIAS",
        "supervisor": "Pr. Mohamed LAZAAR",
        "framework" : "PyTorch + timm",
        "status"    : "production-ready",
    })

    print(f"✅ Run ID : {run_id}")
    print(f"✅ Métriques loguées")

# ── 3. Enregistrer dans le Registry ───────────────────────────────────────────
model_path = os.path.expanduser("~/brain-tumor-mlops/models/model_complet_nadia_94pct.pth")
model_uri  = f"runs:/{run_id}/model"

# Log le fichier .pth comme artifact
with mlflow.start_run(run_id=run_id):
    mlflow.log_artifact(model_path, artifact_path="model")

print(f"\n⏳ Enregistrement dans le Registry sous '{MODEL_NAME}'...")
registered = mlflow.register_model(
    model_uri=f"runs:/{run_id}/model/model_complet_nadia_94pct.pth",
    name=MODEL_NAME
)
version = registered.version
print(f"✅ Version {version} enregistrée")

# ── 4. Transition Staging → Production ────────────────────────────────────────
import time
time.sleep(3)  # attendre que le Registry se mette à jour

client.transition_model_version_stage(
    name=MODEL_NAME, version=version, stage="Staging"
)
print(f"✅ Version {version} → Staging")

# val_acc = 0.9437 > 0.90 → Production
client.transition_model_version_stage(
    name=MODEL_NAME,
    version=version,
    stage="Production",
    archive_existing_versions=True
)
print(f"🚀 Version {version} → Production (val_acc=0.9437 ≥ 0.90)")

client.update_model_version(
    name=MODEL_NAME,
    version=version,
    description="EfficientNet-B3 | val_acc=94.37% | F1-macro=0.944 | AUC-ROC=0.972 | Nadia Zemani PFA 2024/2025"
)

# ── 5. Résumé ─────────────────────────────────────────────────────────────────
print("\n── Registry Status ──────────────────────────────────────")
for v in client.get_latest_versions(MODEL_NAME):
    print(f"  Version {v.version} | Stage: {v.current_stage}")
    print(f"  Description: {v.description}")
print(f"\n🎯 Run ID : {run_id}")
print("✅ MLflow Model Registry configuré !")
