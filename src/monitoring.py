import pandas as pd
import json
from pathlib import Path
from datetime import datetime

# Chemin dans le conteneur Docker
LOGS_DIR = Path('/app/logs')
LOGS_DIR.mkdir(parents=True, exist_ok=True)
PREDICTIONS_LOG = LOGS_DIR / 'predictions.jsonl'

def log_prediction(filename, prediction, confidence, probabilities, uncertainty):
    record = {
        'timestamp'  : datetime.now().isoformat(),
        'filename'   : str(filename),
        'prediction' : prediction,
        'confidence' : float(confidence),
        'uncertainty': float(uncertainty),
        **{f'prob_{cls}': float(prob) for cls, prob in probabilities.items()}
    }
    with open(PREDICTIONS_LOG, 'a') as f:
        f.write(json.dumps(record) + '\n')

def load_predictions_log():
    if not PREDICTIONS_LOG.exists():
        return pd.DataFrame()
    records = []
    with open(PREDICTIONS_LOG, 'r') as f:
        for line in f:
            try:
                records.append(json.loads(line.strip()))
            except:
                pass
    return pd.DataFrame(records)

def generate_drift_report(output_path=None):
    try:
        from evidently.report import Report
        from evidently.metric_preset import DataDriftPreset, DataQualityPreset
    except:
        return None

    df = load_predictions_log()
    if len(df) < 50:
        return None

    feature_cols = ['confidence', 'uncertainty',
                   'prob_glioma', 'prob_meningioma',
                   'prob_notumor', 'prob_pituitary']
    cols = [c for c in feature_cols if c in df.columns]
    mid  = len(df) // 2

    report = Report(metrics=[DataDriftPreset(), DataQualityPreset()])
    report.run(
        reference_data=df.iloc[:mid][cols],
        current_data  =df.iloc[mid:][cols]
    )

    if output_path is None:
        output_path = str(LOGS_DIR / 'drift_report.html')
    report.save_html(output_path)
    return output_path

def get_monitoring_stats():
    df = load_predictions_log()
    if df.empty:
        return {
            'total_predictions' : 0,
            'avg_confidence'    : 0.0,
            'avg_uncertainty'   : 0.0,
            'class_distribution': {},
            'drift_detected'    : False,
            'last_prediction'   : None,
        }
    avg_conf = float(df['confidence'].mean())
    return {
        'total_predictions' : len(df),
        'avg_confidence'    : round(avg_conf, 4),
        'avg_uncertainty'   : round(float(df['uncertainty'].mean()), 4),
        'class_distribution': df['prediction'].value_counts().to_dict(),
        'drift_detected'    : avg_conf < 0.70,
        'last_prediction'   : str(df['timestamp'].iloc[-1]),
    }
