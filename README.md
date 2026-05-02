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

## Step 2 — Deploy to Vercel

### Option A — Vercel CLI (recommended)
```bash
# Install Vercel CLI
npm install -g vercel

# Login
vercel login

# Deploy from project root
vercel

# Follow the prompts:
#   Set up and deploy? → Y
#   Which scope? → your account
#   Link to existing project? → N
#   Project name → mediscan (or any name)
#   Directory → ./  (current directory)
#   Override settings? → N

# For production deployment:
vercel --prod
```

### Option B — GitHub + Vercel Dashboard
1. Push this folder to a GitHub repo
2. Go to https://vercel.com/new
3. Import your GitHub repo
4. Leave all settings as default (Vercel auto-detects `vercel.json`)
5. Click **Deploy**

---

## How It Works on Vercel

| File | Purpose |
|------|---------|
| `vercel.json` | Routes `/predict` and `/health` to Flask, `/static/*` to static files, `/` to `index.html` |
| `api/index.py` | Flask app exposed as `app` — Vercel runs this as a serverless function |
| `requirements.txt` | Vercel installs these automatically at build time |
| `model/*.pkl` | Included in deployment, loaded at cold start |

---

## Important Notes

- **Do NOT call `app.run()`** in `api/index.py` — Vercel manages the server itself
- **Model files** must exist before deploying — run `train_model.py` first
- **Cold starts** — first request after inactivity may take 3–5s while models load
- **Free tier limits** — 100GB bandwidth/month, 10s function timeout (sufficient for this app)

---

## Troubleshooting

| Error | Fix |
|-------|-----|
| `FileNotFoundError: model/*.pkl` | Run `python model/train_model.py` first |
| `ModuleNotFoundError` | Check `requirements.txt` has all packages |
| `504 Gateway Timeout` | Model load exceeded 10s — upgrade to Vercel Pro for 60s timeout |
| CORS errors | Already handled in `api/index.py` via `@app.after_request` |

---

> ⚠️ Educational purposes only. Not a substitute for medical advice.
