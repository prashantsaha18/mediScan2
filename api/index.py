"""
MediScan — Vercel Serverless Entry Point
All paths are relative so this works on any machine / Vercel deployment.
"""

from flask import Flask, request, jsonify
import joblib, json, os, numpy as np
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
# api/index.py lives in  <root>/api/
# model files live in    <root>/model/
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, 'model')

app = Flask(__name__)

# ── Allow CORS for Vercel static ──────────────────────────────────────────────
@app.after_request
def add_cors(response):
    response.headers['Access-Control-Allow-Origin']  = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    response.headers['Access-Control-Allow-Methods'] = 'POST, GET, OPTIONS'
    return response

# ── Load models (once per cold start) ─────────────────────────────────────────
print("Loading models …")
MODELS = {
    name: joblib.load(os.path.join(MODEL_DIR, f'{name}_model.pkl'))
    for name in ['overall', 'diabetes', 'heart', 'hypertension', 'kidney']
}
FEATURES        = joblib.load(os.path.join(MODEL_DIR, 'features.pkl'))
IMPORTANCE_MAP  = json.load(open(os.path.join(MODEL_DIR, 'importance_map.json')))
THRESHOLDS      = json.load(open(os.path.join(MODEL_DIR, 'thresholds.json')))
RECOMMENDATIONS = json.load(open(os.path.join(MODEL_DIR, 'recommendations.json')))
print("Models loaded.")

# ── Helpers ───────────────────────────────────────────────────────────────────

def build_input(data):
    return pd.DataFrame([{f: float(data.get(f, 0)) for f in FEATURES}])

def predict_all(df_input):
    return {
        name: {
            'probability': round(float(model.predict_proba(df_input)[0][1]), 4),
            'prediction':  int(model.predict(df_input)[0])
        }
        for name, model in MODELS.items()
    }

def classify_risk(prob):
    if prob < 0.30: return 'Low'
    if prob < 0.55: return 'Moderate'
    if prob < 0.75: return 'High'
    return 'Critical'

def analyze_risk_factors(data):
    flags = []
    for key, cfg in THRESHOLDS.items():
        val    = float(data.get(key, 0))
        status = None
        if cfg.get('high') and val >= cfg['high']:
            status = 'critical' if val >= cfg['high'] * 1.25 else 'high'
        elif cfg.get('borderline') and val >= cfg['borderline']:
            status = 'borderline'
        elif cfg.get('low') and val < cfg['low']:
            status = 'high'

        if status:
            ref   = cfg.get('high') or cfg.get('borderline') or cfg.get('low')
            delta = abs(val - ref) / ref * 100 if ref else 0
            flags.append({
                'key': key, 'label': cfg['label'],
                'value': val, 'unit': cfg['unit'],
                'status': status, 'delta_pct': round(delta, 1),
            })

    if int(data.get('smoking', 0)):
        flags.append({'key':'smoking','label':'Smoking','value':'Active','unit':'','status':'critical','delta_pct':100})
    if int(data.get('alcohol_consumption', 0)) >= 3:
        flags.append({'key':'alcohol','label':'Alcohol Consumption','value':'Heavy','unit':'','status':'high','delta_pct':50})
    if int(data.get('exercise_days', 7)) < 2:
        flags.append({'key':'exercise','label':'Physical Activity','value':int(data.get('exercise_days',0)),'unit':'days/wk','status':'borderline','delta_pct':30})
    if int(data.get('family_history', 0)):
        flags.append({'key':'family_history','label':'Family History','value':'Positive','unit':'','status':'borderline','delta_pct':20})

    order = {'critical':0,'high':1,'borderline':2}
    flags.sort(key=lambda x: (order.get(x['status'],3), -x['delta_pct']))
    return flags

def build_recommendations(data, predictions, risk_flags):
    recs, seen = [], set()

    def add(key):
        if key in RECOMMENDATIONS and key not in seen:
            seen.add(key)
            recs.append({'key': key, **RECOMMENDATIONS[key]})

    flag_key_map = {
        'glucose':'glucose_high','hba1c':'hba1c_high',
        'systolic_bp':'systolic_bp_high','diastolic_bp':'diastolic_bp_high',
        'bmi':'bmi_high','cholesterol':'cholesterol_high','ldl':'ldl_high',
        'hdl':'hdl_low','triglycerides':'triglycerides_high',
        'heart_rate':'heart_rate_high','creatinine':'creatinine_high',
        'waist_cm':'waist_cm_high','sleep_hours':'sleep_hours_low',
        'stress_level':'stress_level_high','smoking':'smoking_active',
        'alcohol':'alcohol_heavy','exercise':'exercise_low',
        'family_history':'family_history',
    }
    for flag in risk_flags:
        rec_key = flag_key_map.get(flag['key'])
        if rec_key: add(rec_key)

    disease_map = {'diabetes':'diabetes_risk','heart':'heart_risk',
                   'hypertension':'hypertension_risk','kidney':'kidney_risk'}
    for model_name, rec_key in disease_map.items():
        if predictions.get(model_name,{}).get('prediction') == 1:
            add(rec_key)

    if predictions.get('overall',{}).get('probability', 0) >= 0.65:
        add('see_doctor')

    if not recs:
        add('healthy')

    urgency_order = {'critical':0,'high':1,'medium':2,'low':3}
    recs.sort(key=lambda x: urgency_order.get(x.get('urgency','medium'), 2))
    return recs

def top_contributing_factors(model_name='overall', top_n=5):
    imp = IMPORTANCE_MAP.get(model_name, {})
    sorted_imp = sorted(imp.items(), key=lambda x: -x[1])
    return [{'feature': f, 'importance': round(v, 4)} for f, v in sorted_imp[:top_n]]

# ── Routes ────────────────────────────────────────────────────────────────────

@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    if request.method == 'OPTIONS':
        return '', 204

    try:
        data         = request.get_json()
        df_input     = build_input(data)
        predictions  = predict_all(df_input)
        risk_flags   = analyze_risk_factors(data)
        recs         = build_recommendations(data, predictions, risk_flags)
        top_factors  = top_contributing_factors()

        overall_prob = predictions['overall']['probability']

        return jsonify({
            'prediction':         predictions['overall']['prediction'],
            'probability':        overall_prob,
            'risk_level':         classify_risk(overall_prob),
            'diseases': {
                name: {
                    'detected':    predictions[name]['prediction'] == 1,
                    'probability': predictions[name]['probability'],
                    'risk_level':  classify_risk(predictions[name]['probability'])
                }
                for name in ['diabetes', 'heart', 'hypertension', 'kidney']
            },
            'diseases_detected':  [n for n in ['diabetes','heart','hypertension','kidney']
                                    if predictions[n]['prediction'] == 1],
            'risk_factors':       risk_flags,
            'recommendations':    recs,
            'top_contributors':   top_factors,
        })

    except Exception as e:
        import traceback
        return jsonify({'error': str(e), 'trace': traceback.format_exc()}), 400

@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'models': list(MODELS.keys())})

# ── Vercel exposes the `app` object ───────────────────────────────────────────
# Do NOT call app.run() here
