# MediScan 

## Project Structure
```
mediscan/
├── api/
│   └── index.py              ← Flask serverless entry point (Vercel reads this)
├── model/
│   ├── train_model.py        ← run locally to generate .pkl files
│   ├── overall_model.pkl     ← generated after training
│   ├── diabetes_model.pkl
│   ├── heart_model.pkl
│   ├── hypertension_model.pkl
│   ├── kidney_model.pkl
│   ├── features.pkl
│   ├── thresholds.json
│   ├── recommendations.json
│   └── importance_map.json
├── static/
│   ├── index.html
│   ├── css/style.css
│   └── js/app.js
├── vercel.json               ← Vercel routing config
└── requirements.txt
```

---

## Step 1 — Run Locally First

```bash
# Install dependencies
pip install -r requirements.txt

# Train models (generates all .pkl and .json files)
python model/train_model.py

# Test locally
cd api && flask --app index run --port 5000
# Open http://127.0.0.1:5000
```

---

 for medical advice.
