"""
MediScan — Training Script
Run this LOCALLY before deploying to Vercel.
Generates all .pkl and .json files inside the model/ folder.

Usage:
    python model/train_model.py
"""

import numpy as np
import pandas as pd
import json, os, joblib
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

np.random.seed(42)
N = 8000

# Paths — always relative to this file
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Generate synthetic dataset ────────────────────────────────────────────────
age           = np.random.randint(18, 85, N)
gender        = np.random.randint(0, 2, N)
bmi           = np.random.normal(27, 6, N).clip(15, 55)
systolic_bp   = np.random.normal(120, 20, N).clip(80, 200)
diastolic_bp  = np.random.normal(80, 12, N).clip(50, 130)
glucose       = np.random.normal(100, 30, N).clip(60, 300)
cholesterol   = np.random.normal(200, 40, N).clip(100, 400)
hdl           = np.random.normal(55, 15, N).clip(20, 100)
ldl           = np.random.normal(130, 35, N).clip(50, 300)
triglycerides = np.random.normal(150, 60, N).clip(50, 500)
heart_rate    = np.random.normal(75, 12, N).clip(45, 130)
smoking       = np.random.randint(0, 2, N)
exercise      = np.random.randint(0, 8, N)
family_hist   = np.random.randint(0, 2, N)
alcohol       = np.random.randint(0, 4, N)
stress        = np.random.randint(1, 11, N)
sleep         = np.random.normal(7, 1.5, N).clip(3, 12)
waist         = np.random.normal(88, 14, N).clip(55, 150)
creatinine    = np.random.normal(1.0, 0.3, N).clip(0.4, 4.0)
hba1c         = np.random.normal(5.5, 1.2, N).clip(4.0, 14.0)

def sigmoid(x): return 1 / (1 + np.exp(-x))

diabetes = (np.random.random(N) < sigmoid(0.6*(
    (glucose>126)*3.0 + (hba1c>6.5)*3.5 + (bmi>30)*1.5 +
    (age>45)*1.0 + family_hist*1.2 + (exercise<2)*0.8 +
    (alcohol>2)*0.5 + np.random.normal(0,.5,N) - 4))).astype(int)

heart = (np.random.random(N) < sigmoid(0.5*(
    (systolic_bp>140)*2.0 + (ldl>160)*1.5 + (hdl<40)*1.5 +
    (cholesterol>240)*1.0 + (triglycerides>200)*0.8 + smoking*2.0 +
    (age>55)*1.5 + (gender==1)*0.5 + family_hist*1.5 +
    (heart_rate>100)*0.8 + (exercise<2)*0.7 + (stress>7)*0.6 +
    np.random.normal(0,.5,N) - 5))).astype(int)

hypertension = (np.random.random(N) < sigmoid(0.5*(
    (systolic_bp>140)*3.5 + (diastolic_bp>90)*3.0 + (bmi>30)*1.2 +
    (age>50)*1.0 + (alcohol>2)*1.0 + smoking*0.8 + (stress>7)*0.9 +
    (sleep<6)*0.7 + (exercise<2)*0.6 +
    np.random.normal(0,.5,N) - 5))).astype(int)

kidney = (np.random.random(N) < sigmoid(0.6*(
    (creatinine>1.5)*3.0 + (systolic_bp>140)*1.5 + diabetes*2.0 +
    (age>60)*1.0 + np.random.normal(0,.5,N) - 3))).astype(int)

overall = (np.random.random(N) < sigmoid(0.7*(
    diabetes*2 + heart*2 + hypertension*1.5 + kidney*1 +
    (bmi>30)*0.5 + smoking*0.5 + family_hist*0.5 +
    np.random.normal(0,.3,N) - 2))).astype(int)

FEATURES = [
    'age','gender','bmi','systolic_bp','diastolic_bp',
    'glucose','cholesterol','hdl','ldl','triglycerides',
    'heart_rate','smoking','exercise_days','family_history',
    'alcohol_consumption','stress_level','sleep_hours',
    'waist_cm','creatinine','hba1c'
]

df = pd.DataFrame({
    'age':age,'gender':gender,'bmi':np.round(bmi,1),
    'systolic_bp':systolic_bp.astype(int),'diastolic_bp':diastolic_bp.astype(int),
    'glucose':np.round(glucose,1),'cholesterol':np.round(cholesterol,1),
    'hdl':np.round(hdl,1),'ldl':np.round(ldl,1),
    'triglycerides':np.round(triglycerides,1),'heart_rate':heart_rate.astype(int),
    'smoking':smoking,'exercise_days':exercise,'family_history':family_hist,
    'alcohol_consumption':alcohol,'stress_level':stress,
    'sleep_hours':np.round(sleep,1),'waist_cm':np.round(waist,1),
    'creatinine':np.round(creatinine,2),'hba1c':np.round(hba1c,1),
    'overall':overall,'diabetes':diabetes,'heart':heart,
    'hypertension':hypertension,'kidney':kidney,
})

print(f"Dataset: {N} samples")

# ── Train ─────────────────────────────────────────────────────────────────────
def train(X, y, label):
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2,
                                               stratify=y, random_state=42)
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('model',  GradientBoostingClassifier(
            n_estimators=200, max_depth=4,
            learning_rate=0.1, subsample=0.8,
            min_samples_leaf=20, random_state=42
        ))
    ])
    pipe.fit(X_tr, y_tr)
    prob = pipe.predict_proba(X_te)[:,1]
    pred = pipe.predict(X_te)
    print(f"  [{label}]  Acc={accuracy_score(y_te,pred):.3f}  AUC={roc_auc_score(y_te,prob):.3f}")
    return pipe

X = df[FEATURES]
print("\nTraining models …")
models = {
    'overall':      train(X, df['overall'],      'Overall'),
    'diabetes':     train(X, df['diabetes'],     'Diabetes'),
    'heart':        train(X, df['heart'],        'Heart'),
    'hypertension': train(X, df['hypertension'], 'Hypertension'),
    'kidney':       train(X, df['kidney'],       'Kidney'),
}

# ── Feature importances ───────────────────────────────────────────────────────
importance_map = {
    name: {f: round(float(v),4) for f,v in
           zip(FEATURES, pipe.named_steps['model'].feature_importances_)}
    for name, pipe in models.items()
}

# ── Clinical thresholds ───────────────────────────────────────────────────────
THRESHOLDS = {
    'glucose':       {'low':None,'borderline':100,'high':126,  'unit':'mg/dL', 'label':'Blood Glucose'},
    'hba1c':         {'low':None,'borderline':5.7,'high':6.5,  'unit':'%',     'label':'HbA1c'},
    'systolic_bp':   {'low':None,'borderline':130,'high':140,  'unit':'mmHg',  'label':'Systolic BP'},
    'diastolic_bp':  {'low':None,'borderline':85, 'high':90,   'unit':'mmHg',  'label':'Diastolic BP'},
    'bmi':           {'low':None,'borderline':25, 'high':30,   'unit':'kg/m²', 'label':'BMI'},
    'cholesterol':   {'low':None,'borderline':200,'high':240,  'unit':'mg/dL', 'label':'Total Cholesterol'},
    'ldl':           {'low':None,'borderline':130,'high':160,  'unit':'mg/dL', 'label':'LDL Cholesterol'},
    'triglycerides': {'low':None,'borderline':150,'high':200,  'unit':'mg/dL', 'label':'Triglycerides'},
    'hdl':           {'low':40,  'borderline':None,'high':None,'unit':'mg/dL', 'label':'HDL (Good Cholesterol)'},
    'heart_rate':    {'low':None,'borderline':90, 'high':100,  'unit':'bpm',   'label':'Heart Rate'},
    'creatinine':    {'low':None,'borderline':1.2,'high':1.5,  'unit':'mg/dL', 'label':'Creatinine'},
    'waist_cm':      {'low':None,'borderline':88, 'high':102,  'unit':'cm',    'label':'Waist Circumference'},
    'sleep_hours':   {'low':6,   'borderline':None,'high':None,'unit':'hrs',   'label':'Sleep Hours'},
    'stress_level':  {'low':None,'borderline':6,  'high':8,    'unit':'/10',   'label':'Stress Level'},
}

# ── Recommendations ───────────────────────────────────────────────────────────
RECOMMENDATIONS = {
    'glucose_high':      {'title':'Manage Blood Sugar','detail':'Your glucose is elevated. Reduce refined carbs, sugary beverages, and white rice. Eat more fiber-rich foods and check levels regularly.','urgency':'high'},
    'hba1c_high':        {'title':'HbA1c Requires Attention','detail':'A raised HbA1c suggests poor long-term glucose control. Consult a doctor for a diabetes management plan.','urgency':'high'},
    'systolic_bp_high':  {'title':'Lower Blood Pressure','detail':'High systolic BP strains the heart. Reduce sodium (<2g/day), increase potassium, limit caffeine, and manage stress.','urgency':'high'},
    'diastolic_bp_high': {'title':'Diastolic BP Elevated','detail':'Elevated diastolic pressure increases stroke risk. Follow the DASH diet, exercise regularly, and reduce alcohol intake.','urgency':'medium'},
    'bmi_high':          {'title':'Weight Management Needed','detail':'Excess weight raises risk for diabetes and heart disease. Aim for a 500 kcal daily deficit through diet and 150 min exercise per week.','urgency':'medium'},
    'cholesterol_high':  {'title':'Reduce Total Cholesterol','detail':'Cut saturated fats and trans fats. Increase soluble fiber (oats, beans) and omega-3 fatty acids (fish, flaxseeds).','urgency':'medium'},
    'ldl_high':          {'title':'LDL "Bad" Cholesterol High','detail':'Elevated LDL accelerates plaque build-up in arteries. Avoid fried foods, increase plant sterols, and consider a statin if advised by your doctor.','urgency':'high'},
    'hdl_low':           {'title':'Raise HDL "Good" Cholesterol','detail':'Low HDL means less cholesterol removal from arteries. Exercise aerobically, quit smoking, eat healthy fats (olive oil, avocado, nuts).','urgency':'medium'},
    'triglycerides_high':{'title':'High Triglycerides','detail':'Excess sugar and alcohol raises triglycerides. Limit sugar, refined carbs, and alcohol. Eat more omega-3s and increase physical activity.','urgency':'medium'},
    'heart_rate_high':   {'title':'Elevated Resting Heart Rate','detail':'A resting heart rate above 100 bpm can signal stress or cardiovascular strain. Practice deep breathing, stay hydrated, and reduce caffeine.','urgency':'medium'},
    'creatinine_high':   {'title':'Kidney Function Concern','detail':'Raised creatinine may indicate reduced kidney function. Stay well hydrated, avoid NSAIDs, control blood pressure, and get a kidney function panel.','urgency':'high'},
    'waist_cm_high':     {'title':'Excess Abdominal Fat','detail':'Central obesity increases heart and diabetes risk. Reduce refined carbs, alcohol, and add core-strengthening exercises.','urgency':'medium'},
    'sleep_hours_low':   {'title':'Improve Sleep Quality','detail':'Sleeping under 6 hours raises cortisol, blood pressure, and insulin resistance. Aim for 7–9 hours with a consistent sleep schedule.','urgency':'medium'},
    'stress_level_high': {'title':'Reduce Chronic Stress','detail':'Chronic stress elevates cortisol and blood pressure. Try mindfulness, yoga, 20-min daily walks, journaling, or speaking with a therapist.','urgency':'medium'},
    'smoking_active':    {'title':'Quit Smoking — #1 Priority','detail':'Smoking doubles heart disease risk and directly damages lungs and blood vessels. Use NRT patches or medication to quit.','urgency':'critical'},
    'alcohol_heavy':     {'title':'Reduce Alcohol Intake','detail':'Heavy drinking raises blood pressure, triglycerides, and liver risk. Limit to ≤1 drink/day (women) or ≤2/day (men).','urgency':'high'},
    'exercise_low':      {'title':'Increase Physical Activity','detail':'Less than 2 active days/week significantly raises disease risk. Start with 30-minute walks 5x/week, add resistance training twice a week.','urgency':'medium'},
    'family_history':    {'title':'Monitor Due to Family History','detail':'Genetic predisposition raises your baseline risk. Get annual check-ups including lipid panel, glucose, and blood pressure.','urgency':'medium'},
    'diabetes_risk':     {'title':'Diabetes Risk Detected','detail':'Your profile matches pre-diabetic or diabetic patterns. Request an HbA1c and fasting glucose test. Early lifestyle changes can reverse pre-diabetes.','urgency':'high'},
    'heart_risk':        {'title':'Cardiovascular Risk Elevated','detail':'Multiple cardiac risk markers are raised. Consider a lipid panel, ECG, and stress test. Prioritize cholesterol control and blood pressure.','urgency':'high'},
    'hypertension_risk': {'title':'Hypertension Risk','detail':'Your blood pressure pattern suggests hypertension. Monitor BP twice daily for 2 weeks and consult a doctor. The DASH diet can reduce SBP by up to 11 mmHg.','urgency':'high'},
    'kidney_risk':       {'title':'Kidney Health Concern','detail':'Markers suggest kidney stress. Drink 2–3 L of water daily, avoid excessive protein supplements, and control BP and blood sugar.','urgency':'high'},
    'see_doctor':        {'title':'Consult a Healthcare Professional','detail':'Your overall risk profile is elevated. A comprehensive health screening with blood work, physical exam, and specialist referral is strongly recommended.','urgency':'critical'},
    'healthy':           {'title':'Maintain Your Healthy Lifestyle','detail':'Your parameters are within healthy ranges. Continue regular exercise, a balanced diet, adequate sleep, and annual health check-ups.','urgency':'low'},
}

# ── Save everything ───────────────────────────────────────────────────────────
print(f"\nSaving to {MODEL_DIR} …")
for name, pipe in models.items():
    path = os.path.join(MODEL_DIR, f'{name}_model.pkl')
    joblib.dump(pipe, path)
    print(f"  Saved {name}_model.pkl  ({os.path.getsize(path)//1024} KB)")

joblib.dump(FEATURES, os.path.join(MODEL_DIR, 'features.pkl'))
json.dump(importance_map, open(os.path.join(MODEL_DIR,'importance_map.json'),'w'), indent=2)
json.dump(THRESHOLDS,     open(os.path.join(MODEL_DIR,'thresholds.json'),'w'),     indent=2)
json.dump(RECOMMENDATIONS,open(os.path.join(MODEL_DIR,'recommendations.json'),'w'),indent=2)

print("\n✅ All models and config saved. Ready to deploy!")
