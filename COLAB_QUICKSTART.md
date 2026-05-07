# 🚀 COLAB QUICK START - 5 MINUTES

## ✨ What This Does

Runs the **entire prediction engine on Google Colab's free GPU** and connects it to your local backend.

---

## 📋 Steps

### 1. Upload Project to Google Drive (2 minutes)

1. Open Google Drive: https://drive.google.com
2. Create folder: `Deriv_Predictor`
3. Upload the entire `/home/tim/Downloads/2026/der/` folder

Your structure should be:
```
Google Drive/
└── Deriv_Predictor/
    ├── core/
    ├── backend/
    ├── docs/
    ├── colab/
    └── requirements.txt
```

---

### 2. Open Colab Notebook (1 minute)

1. In Google Drive, navigate to: `Deriv_Predictor/colab/`
2. Double-click: `Deriv_Prediction_API.ipynb`
3. It will open in Google Colab

**OR** create new notebook:
1. Go to: https://colab.research.google.com
2. File → Upload notebook
3. Upload: `colab/Deriv_Prediction_API.ipynb`

---

### 3. Enable GPU (30 seconds)

1. In Colab: Runtime → Change runtime type
2. Hardware accelerator: **GPU**
3. Click **Save**

---

### 4. Get ngrok Token (1 minute)

1. Go to: https://ngrok.com
2. Sign up (free)
3. Go to: https://dashboard.ngrok.com/get-started/your-authtoken
4. Copy your token (looks like: `2abc...xyz`)

---

### 5. Run Notebook (1 minute)

1. In Colab: Runtime → **Run all**
2. When prompted, paste your ngrok token
3. Wait ~30 seconds for setup
4. **Copy the public URL** (looks like: `https://xxxx.ngrok.io`)

You'll see:
```
🎉 SERVER RUNNING!
📡 Public URL: https://xxxx.ngrok.io
```

---

### 6. Update Local Backend (30 seconds)

Edit `/home/tim/Downloads/2026/der/backend/main.py`:

```python
# Line ~27 - Set your Colab URL
COLAB_URL = "https://xxxx.ngrok.io"  # Paste your URL here
```

---

### 7. Start Local Backend

```bash
cd /home/tim/Downloads/2026/der
python -m uvicorn backend.main:app --reload
```

You'll see:
```
✅ Colab API connected: https://xxxx.ngrok.io
✅ Connected to Deriv: R_100
```

---

## 🎉 Done!

Your system is now:
1. ✅ Receiving live ticks from Deriv (R_100)
2. ✅ Sending them to Colab for prediction
3. ✅ Getting predictions back (GPU-accelerated)
4. ✅ Auto-retraining every 2,000 ticks

---

## 📊 Test It

```bash
# Check status
curl http://localhost:8000/api/status

# Get prediction
curl -X POST http://localhost:8000/api/predict
```

---

## 🔍 Monitor

### In Colab

Run the monitoring cell (cell #9) to see:
- Tick count
- Training status
- Prediction accuracy
- GPU usage

### In Local Terminal

Watch the logs:
```
📊 Buffered 50 ticks | Predictions: 1
🎯 Prediction: BUY (confidence: 65%)
```

---

## ⚠️ Important

- **Keep Colab notebook running**: Server stops if you close it
- **Free tier limit**: 12 hours max session
- **ngrok URL changes**: Every 2 hours (free tier)

**Solution**: Use Colab Pro ($10/month) for 24-hour sessions

---

## 🆘 Troubleshooting

### "Colab API not reachable"

1. Make sure Colab notebook is running
2. Check ngrok URL is correct
3. Run cell #8 again to restart server

### "Session disconnected"

1. In Colab: Runtime → Run all
2. Models auto-load from Google Drive

### "No predictions"

1. Wait for 500 ticks (~5.5 minutes) for initial training
2. Check Colab logs for errors

---

## 🎯 What Happens Now

1. **Deriv sends ticks** → Your local backend
2. **Backend buffers** → Every 50 ticks
3. **Sends to Colab** → GPU processes
4. **Colab returns prediction** → Backend receives
5. **Backend logs** → You see results

---

## 📈 Performance

With free Colab GPU (T4):
- **Initial training**: ~3 seconds (500 ticks)
- **Retraining**: ~5 seconds (2000 ticks)
- **Prediction**: ~0.05 seconds
- **Speedup**: 10-20x faster than CPU

---

## 🚀 Next Steps

1. ✅ **Build frontend**: Display predictions
2. ✅ **Add trade execution**: Connect to Deriv trading API
3. ✅ **Monitor performance**: Track accuracy
4. ✅ **Tune parameters**: Optimize in `core/config.py`

---

**🎉 You're now running a GPU-accelerated prediction system!**

All heavy lifting (LSTM training) happens on Google's free GPU, while your local machine just handles tick ingestion and display.
