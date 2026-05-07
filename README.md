# 🎯 DERIV HYBRID PREDICTOR

**Production-Grade Adaptive Market Signal Predictor**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Status: Production Ready](https://img.shields.io/badge/status-production%20ready-green.svg)]()

---

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt && pip install -r backend/requirements.txt

# 2. Test Deriv connection (your API key pre-configured)
python run_backend.py

# 3. Start backend API
python -m uvicorn backend.main:app --reload
```

**Access**: http://localhost:8000/docs

---

## ✨ What's Included

### 🧠 Core Prediction Engine
- ✅ Hybrid Architecture (LSTM + Decision Tree + Ensemble)
- ✅ Live Retraining (every 2,000 ticks ~33 minutes)
- ✅ RL Execution Filter (learns when to trade)
- ✅ Risk Management (auto-shutdown protection)
- ✅ Google Colab Ready (GPU-accelerated)

### 🔌 Deriv Integration (NEW!)
- ✅ WebSocket Client (auto-reconnect, error handling)
- ✅ Pre-configured (API key: V35FbErHFzWjhj5)
- ✅ Optimized for R_100 (~1.5 ticks/second)
- ✅ FastAPI Backend (REST + WebSocket)

### 📊 Your Configuration
- **Symbol**: R_100 (Volatility 100 Index)
- **Tick Rate**: ~1.5/second
- **Buffer**: 15,000 ticks (~4 hours)
- **Retrain**: Every 2,000 ticks (~33 minutes)

---

## 📁 Project Structure

```
der/
├── 📄 GETTING_STARTED.md          ← START HERE!
├── 📄 PROJECT_SUMMARY.md          ← Complete overview
├── 📄 run_backend.py              ← Test Deriv connection
│
├── 🧠 core/                       ← Prediction Engine
│   ├── config.py                  ← Optimized for R_100
│   ├── data_engine.py
│   ├── feature_engine.py
│   ├── risk_engine.py
│   └── models/                    ← LSTM, Tree, Ensemble, RL
│
├── 🔌 backend/                    ← Deriv Integration (NEW!)
│   ├── main.py                    ← FastAPI backend
│   └── deriv_websocket.py         ← WebSocket client
│
└── 📚 docs/                       ← Documentation
    ├── DERIV_INTEGRATION.md       ← Your API guide
    ├── COLAB_INTEGRATION.md       ← Remote GPU setup
    └── ARCHITECTURE.md            ← Technical details
```

---

## 🎯 Quick Test

Test your Deriv connection:

```bash
python run_backend.py
```

Expected output:
```
✅ Connected to R_100
📊 Tick #1: Quote: 1234.56
⚡ Average rate: 1.52 ticks/second
✅ TEST SUCCESSFUL!
```

---

## 📚 Documentation

### Getting Started
- **[GETTING_STARTED.md](GETTING_STARTED.md)** - 5-minute overview
- **[docs/DERIV_INTEGRATION.md](docs/DERIV_INTEGRATION.md)** - Test your API
- **[docs/QUICKSTART.md](docs/QUICKSTART.md)** - Usage examples

### Integration
- **[RESPONSIBILITY_MATRIX.md](RESPONSIBILITY_MATRIX.md)** - What to build
- **[docs/COLAB_INTEGRATION.md](docs/COLAB_INTEGRATION.md)** - Remote GPU
- **[PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)** - Complete summary

### Technical
- **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** - System design
- **[docs/INSTALLATION.md](docs/INSTALLATION.md)** - Setup guide

---

## 🧠 Architecture

```
Deriv API (R_100) → WebSocket → Backend API → Core Engine → Predictions
   ~1.5 ticks/sec                FastAPI        (Colab)
```

**Components**:
1. **Data Engine**: 15k tick buffer, sequence generation
2. **Feature Engine**: 13 features, drift detection
3. **LSTM + Tree**: Hybrid prediction
4. **Ensemble**: Probability fusion, adaptive threshold
5. **RL Agent**: Execution optimizer (skip vs execute)
6. **Risk Engine**: Drawdown protection, auto-shutdown

---

## ⚙️ Configuration

Your system is pre-configured for:

| Setting | Value | Details |
|---------|-------|---------|
| API Key | V35FbErHFzWjhj5 | Pre-configured |
| Symbol | R_100 | ~1.5 ticks/sec |
| Buffer | 15,000 ticks | ~4 hours |
| Retrain | 2,000 ticks | ~33 minutes |
| Min Train | 500 ticks | ~5.5 minutes |

Edit `core/config.py` to customize.

---

## 📊 Performance Expectations

- **Accuracy**: 51-54% (on executed trades)
- **Trade Frequency**: 40-60% of signals
- **Edge**: Comes from filtering, not raw accuracy
- **Risk**: Auto-shutdown at 10% drawdown

**Reality Check**: Synthetic indices are near-random. Edge is thin.

---

## 🎓 Next Steps

1. ✅ **Test connection**: `python run_backend.py`
2. ✅ **Start backend**: `uvicorn backend.main:app --reload`
3. ✅ **Read docs**: Start with `GETTING_STARTED.md`
4. ✅ **Set up Colab**: See `docs/COLAB_INTEGRATION.md`
5. ✅ **Build frontend**: Create dashboard

---

## 🆘 Troubleshooting

**"Connection refused"**
```bash
python run_backend.py  # Test connection
```

**"No module named 'websockets'"**
```bash
pip install websockets
```

**"No ticks received"**
- Check API key: V35FbErHFzWjhj5
- Try different symbol (R_50, R_75)
- Check firewall

**See**: `docs/DERIV_INTEGRATION.md` for more help

---

## 📄 License

MIT License - Use responsibly. Trading involves risk.

---

**🚀 Ready? Start with: `python run_backend.py`**

---

**Last Updated**: 2026-02-16  
**Version**: 1.0.0  
**Status**: ✅ Production Ready  
**API Key**: V35FbErHFzWjhj5
