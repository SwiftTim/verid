# 🏗️ Architecture Documentation

## System Overview

The Deriv Hybrid Predictor is a production-grade machine learning system designed for single-tick direction prediction with live adaptive learning.

---

## 🎯 Core Design Principles

### 1. **Separation of Concerns**

```
┌─────────────────────────────────────────────────────────┐
│                    YOUR RESPONSIBILITY                   │
├─────────────────────────────────────────────────────────┤
│  • Frontend (Dashboard, Visualization)                  │
│  • Backend (API, Tick Ingestion, Execution)             │
│  • Database (Trade Logging, Performance Tracking)       │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│                  CORE ENGINE (Colab)                     │
├─────────────────────────────────────────────────────────┤
│  • Prediction Models (LSTM + Tree)                      │
│  • Feature Engineering                                   │
│  • Ensemble Logic                                        │
│  • Reinforcement Learning                                │
│  • Risk Management                                       │
└─────────────────────────────────────────────────────────┘
```

### 2. **Stateless Execution**

- Models saved to Google Drive
- Q-table persisted via pickle
- No long-running loops in Colab
- Periodic checkpointing

### 3. **Adaptive Learning**

- Live retraining every 1,000 ticks
- Adaptive confidence thresholding
- RL learns execution timing
- Drift detection and response

---

## 📊 Data Flow

```
Tick Stream (Deriv API)
    ↓
Streaming Buffer (10k ticks)
    ↓
Feature Engineering (13 features)
    ↓
Sequence Creation (20-tick windows)
    ↓
┌─────────────────┬─────────────────┐
│   LSTM Engine   │   Tree Engine   │
│   (Temporal)    │  (Conditional)  │
└────────┬────────┴────────┬────────┘
         │                 │
         └────────┬────────┘
                  ↓
         Ensemble Fusion
                  ↓
         Adaptive Threshold
                  ↓
         RL Execution Filter
                  ↓
         Risk Approval
                  ↓
    Final Decision (BUY/SELL/SKIP)
```

---

## 🧠 Engine Components

### 1️⃣ **Data Engine**

**File**: `core/data_engine.py`

**Purpose**:
- Maintain rolling buffer of ticks
- Create sequences for LSTM
- Time-based train/test splits (NO shuffle)
- Prevent data leakage

**Key Classes**:
- `StreamingBuffer`: Fixed-size deque (10k ticks)
- `DataEngine`: Sequence generation and splitting

**Critical Design**:
```python
# Time-based split (maintains temporal order)
split_idx = int(len(X) * 0.8)
X_train = X[:split_idx]  # Earlier data
X_test = X[split_idx:]   # Later data
```

---

### 2️⃣ **Feature Engine**

**File**: `core/feature_engine.py`

**Purpose**:
- Generate short-memory features (3-10 tick windows)
- Detect feature drift
- Normalize features adaptively

**Features** (13 total):
1. `tick_diff` - Price change
2. `direction` - Binary direction (target)
3-4. `rolling_mean_3/5` - Short-term averages
5-6. `rolling_std_5/10` - Volatility proxies
7-8. `momentum_3/5` - Price momentum
9. `volatility` - 10-tick std
10. `streak` - Consecutive direction count
11. `mean_reversion` - Distance from mean
12. `acceleration` - 2nd derivative
13. `up_down_ratio` - Recent direction bias

**Why Short Memory?**
- Synthetic indices drift quickly
- Long memory = overfitting
- Fast adaptation > deep patterns

---

### 3️⃣ **LSTM Engine**

**File**: `core/models/lstm_engine.py`

**Architecture**:
```
Input (20 ticks × 13 features)
    ↓
LSTM (32 units)
    ↓
Dropout (0.2)
    ↓
Dense (1, sigmoid)
    ↓
Output (probability)
```

**Design Philosophy**:
- **Shallow** (1 LSTM layer) → Fast training, less overfitting
- **Small** (32 units) → Adapts quickly
- **Short epochs** (5) → For live retraining

**Training**:
- Batch size: 64
- Optimizer: Adam
- Loss: Binary crossentropy
- Early stopping (patience=2)

---

### 4️⃣ **Tree Engine**

**File**: `core/models/tree_engine.py`

**Configuration**:
- Max depth: 5
- Min samples split: 50
- Min samples leaf: 20
- Class weight: Balanced

**Why Decision Tree?**:
- Captures conditional logic (if-then rules)
- Fast training (no GPU needed)
- Interpretable (can export rules)
- Complements LSTM temporal patterns

**Flattening**:
```python
# LSTM uses: (samples, 20, 13)
# Tree needs: (samples, 260)  # 20 × 13 flattened
X_flat = X.reshape(X.shape[0], -1)
```

---

### 5️⃣ **Ensemble Engine**

**File**: `core/models/ensemble_engine.py`

**Fusion Logic**:
```python
combined_prob = 0.5 * lstm_prob + 0.5 * tree_prob
```

**Adaptive Thresholding**:
```python
if accuracy < 52%:
    threshold += 0.01  # Trade less
elif accuracy > 60%:
    threshold -= 0.01  # Trade more
```

**Decision Rules**:
- `prob > threshold` → BUY
- `prob < (1 - threshold)` → SELL
- Otherwise → SKIP

**Key Insight**:
> Edge comes from FILTERING, not just prediction accuracy

---

### 6️⃣ **Q-Learning Agent**

**File**: `core/models/q_agent.py`

**Purpose**:
- Learn **WHEN** to trust predictions
- Filter low-quality signals

**State Space** (discretized):
1. Ensemble confidence (0-1)
2. Recent win rate (0-1)
3. Volatility regime (low/med/high)
4. Streak length (-10 to +10)

**Action Space**:
- 0: SKIP (don't trade)
- 1: EXECUTE (trust prediction)

**Reward**:
- +1: Correct prediction
- -1: Wrong prediction
- 0: SKIP (neutral)

**Update Rule**:
```python
Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]
```

**Exploration**:
- Epsilon-greedy (ε = 0.1)
- Decays over time (0.995 per retrain)
- Min epsilon: 0.05

---

### 7️⃣ **Risk Engine**

**File**: `core/risk_engine.py`

**Purpose**:
- Prevent catastrophic losses
- Auto-shutdown on performance decay

**Monitoring**:
1. **Drawdown**: Max 10%
2. **Consecutive losses**: Max 10
3. **Rolling accuracy (200)**: Min 49%
4. **Rolling accuracy (50)**: Min 47% (warning)

**Auto-Shutdown Triggers**:
```python
if drawdown > 10%:
    STOP TRADING
if consecutive_losses >= 10:
    STOP TRADING
if accuracy_200 < 49%:
    STOP TRADING
```

**Philosophy**:
> Survival > Optimization

---

## 🔄 Live Retraining Strategy

### Trigger Conditions

1. **Every 1,000 ticks** (primary)
2. **Accuracy drops below 48%** (emergency)

### Retraining Process

```python
1. Get latest 80% of buffer
2. Retrain LSTM (3-5 epochs only)
3. Refit Decision Tree
4. Decay RL epsilon
5. Update adaptive threshold
6. Save checkpoint
```

### Why This Works

- **Incremental**: Doesn't forget old patterns
- **Fast**: 3-5 epochs = seconds
- **Adaptive**: Responds to drift
- **Stable**: Doesn't overreact to noise

---

## 📈 Performance Expectations

### Reality Check

**Synthetic indices are near pseudo-random.**

Expected performance:
- **Best case**: 51-54% accuracy
- **Realistic**: 49-52% accuracy
- **Random baseline**: 50%

### Where Edge Comes From

1. ✅ **Trade filtering** (skip 40-60% of signals)
2. ✅ **Risk control** (drawdown limits)
3. ✅ **Adaptive thresholding** (adjust to regime)
4. ✅ **Execution discipline** (RL timing)
5. ❌ **NOT model complexity**

### Key Metrics

- **Accuracy**: 51-54% (on executed trades)
- **Trade frequency**: 40-60% of signals
- **Sharpe ratio**: 0.5-1.0 (if edge exists)
- **Max drawdown**: <10%

---

## 🚀 Google Colab Deployment

### File Structure

```
/content/drive/MyDrive/deriv_predictor/
├── core/                 # Engine code
├── models/               # Saved models
│   ├── lstm_model.h5
│   ├── tree_model.pkl
│   └── q_table.pkl
├── logs/                 # Performance logs
└── checkpoints/          # Periodic saves
```

### GPU Usage

- **LSTM training**: Uses GPU (if available)
- **Tree training**: CPU only
- **RL updates**: CPU only
- **Inference**: CPU (fast enough)

### Persistence

```python
# Save every 1000 ticks
if tick_count % 1000 == 0:
    engine.save_models('/content/drive/MyDrive/deriv_predictor/models')
```

---

## 🔌 Integration with Backend

### API Contract

**Endpoint**: `/predict`

**Request**:
```json
{
  "ticks": [
    {"timestamp": 1700000000, "quote": 1000.5, "symbol": "R_100"},
    ...
  ]
}
```

**Response**:
```json
{
  "prediction": {
    "decision": "BUY",
    "confidence": 0.65,
    "lstm_prob": 0.62,
    "tree_prob": 0.68,
    "combined_prob": 0.65,
    "rl_action": "EXECUTE",
    "risk_approved": true
  },
  "status": {
    "accuracy_200": 0.52,
    "trade_frequency": 0.45,
    "threshold": 0.56
  }
}
```

---

## 🛠️ Maintenance

### Daily

- Monitor accuracy metrics
- Check risk status
- Review trade frequency

### Weekly

- Analyze feature importance
- Review RL Q-table convergence
- Check for drift

### Monthly

- Full backtest on new data
- Hyperparameter tuning
- Model architecture review

---

## ⚠️ Critical Warnings

1. **No Overfitting**: Keep models shallow
2. **No Shuffle**: Always time-based splits
3. **No Hallucination**: Edge is thin (51-54%)
4. **No Overtrading**: Filter aggressively
5. **No Ignoring Risk**: Respect drawdown limits

---

## 📚 References

- Q-Learning: Sutton & Barto (2018)
- LSTM: Hochreiter & Schmidhuber (1997)
- Ensemble Methods: Dietterich (2000)
- Risk Management: Tharp (2008)

---

**Last Updated**: 2026-02-16
