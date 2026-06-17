import pandas as pd
import json
from pathlib import Path
from datetime import datetime

_BASE = Path('/app/logs') if Path('/app').exists() else Path(__file__).parent.parent / 'logs'
LOGS_DIR = _BASE
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
        from evidently import Dataset, DataDefinition, Report
        from evidently.presets import DataDriftPreset
    except Exception as e:
        print(f"Evidently import error: {e}")
        return None

    df = load_predictions_log()
    if len(df) < 50:
        return None

    feature_cols = ['confidence', 'uncertainty',
                    'prob_glioma', 'prob_meningioma',
                    'prob_notumor', 'prob_pituitary']
    cols = [c for c in feature_cols if c in df.columns]
    mid  = len(df) // 2

    if output_path is None:
        output_path = str(LOGS_DIR / 'drift_report.html')

    try:
        ref = Dataset.from_pandas(df.iloc[:mid][cols], data_definition=DataDefinition())
        cur = Dataset.from_pandas(df.iloc[mid:][cols], data_definition=DataDefinition())
        report = Report([DataDriftPreset()])
        result = report.run(ref, cur)
        result.save_html(output_path)
        return output_path
    except Exception as e:
        print(f"Drift report error: {e}")
        return None

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
    return {
        'total_predictions' : len(df),
        'avg_confidence'    : round(float(df['confidence'].mean()), 4),
        'avg_uncertainty'   : round(float(df['uncertainty'].mean()), 4),
        'class_distribution': df['prediction'].value_counts().to_dict(),
        'drift_detected'    : float(df['confidence'].mean()) < 0.70,
        'last_prediction'   : str(df['timestamp'].iloc[-1]),
    }
