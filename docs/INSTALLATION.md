# 📦 Installation Guide

## Quick Install (Google Colab - Recommended)

```python
# In Google Colab notebook
!pip install tensorflow scikit-learn pandas numpy matplotlib seaborn
```

That's it! Colab has most dependencies pre-installed.

---

## Local Installation

### Prerequisites

- Python 3.10 or higher
- pip (Python package manager)

### Step 1: Install Dependencies

```bash
cd /home/tim/Downloads/2026/der
pip install -r requirements.txt
```

### Step 2: Verify Installation

```bash
python test_structure.py
```

If successful, you should see:
```
🎉 STRUCTURE TEST PASSED!
```

### Step 3: Run Full Test (with TensorFlow)

```bash
python test_installation.py
```

---

## Dependencies Explained

### Core Dependencies (Required)

```
tensorflow>=2.13.0      # For LSTM neural network
scikit-learn>=1.3.0     # For Decision Tree
numpy>=1.24.0           # Numerical computing
pandas>=2.0.0           # Data manipulation
```

### Optional Dependencies

```
matplotlib>=3.7.0       # Visualization
seaborn>=0.12.0         # Statistical plots
tqdm>=4.65.0            # Progress bars
```

---

## Troubleshooting

### Issue: "No module named 'tensorflow'"

**Solution**:
```bash
pip install tensorflow
```

### Issue: "No module named 'sklearn'"

**Solution**:
```bash
pip install scikit-learn
```

### Issue: TensorFlow GPU not working

**Check GPU availability**:
```python
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
```

**Install CUDA** (if needed):
- Follow: https://www.tensorflow.org/install/gpu

### Issue: Import errors

**Solution**: Ensure you're in the correct directory:
```bash
cd /home/tim/Downloads/2026/der
python -c "from core import HybridEngine; print('✅ Success')"
```

---

## Platform-Specific Notes

### Google Colab

✅ **Recommended for this project**

Advantages:
- Free GPU access
- Pre-installed dependencies
- Easy persistence via Google Drive
- No local setup required

Setup:
```python
from google.colab import drive
drive.mount('/content/drive')

# Upload project to Drive
# Then install additional dependencies
!pip install -q tensorflow scikit-learn
```

### Linux

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install python3-pip python3-dev

# Install dependencies
pip3 install -r requirements.txt
```

### macOS

```bash
# Install Python 3.10+
brew install python@3.10

# Install dependencies
pip3 install -r requirements.txt
```

### Windows

```bash
# Install Python from python.org
# Then in Command Prompt:
pip install -r requirements.txt
```

---

## Virtual Environment (Recommended)

### Create Virtual Environment

```bash
# Linux/macOS
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

### Install in Virtual Environment

```bash
pip install -r requirements.txt
```

---

## Minimal Installation (No TensorFlow)

If you only want to test the structure without LSTM:

```bash
pip install scikit-learn pandas numpy
```

This installs:
- ✅ Decision Tree engine
- ✅ Feature engineering
- ✅ Risk management
- ✅ Q-Learning agent
- ❌ LSTM engine (requires TensorFlow)

---

## Development Installation

For development with additional tools:

```bash
pip install -r requirements.txt
pip install jupyter ipython black flake8 pytest
```

---

## Verification Checklist

After installation, verify:

- [ ] Python 3.10+ installed: `python --version`
- [ ] Dependencies installed: `pip list | grep tensorflow`
- [ ] Structure test passes: `python test_structure.py`
- [ ] Full test passes: `python test_installation.py`
- [ ] Can import engine: `python -c "from core import HybridEngine"`

---

## Next Steps

Once installed:

1. ✅ Read `docs/QUICKSTART.md` for usage
2. ✅ Read `docs/ARCHITECTURE.md` for technical details
3. ✅ Check `colab/TRAINING_GUIDE.md` for Colab setup
4. ✅ Review `docs/COLAB_INTEGRATION.md` for API integration

---

## Support

If you encounter issues:

1. Check this installation guide
2. Review error messages carefully
3. Ensure Python 3.10+ is installed
4. Try in a fresh virtual environment
5. Use Google Colab (easiest option)

---

**Recommended Path**: Use Google Colab for the core engine, keep your backend/frontend local.

See `docs/COLAB_INTEGRATION.md` for the complete setup.
