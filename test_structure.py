"""
Simple Test: Verify Core Structure (No TensorFlow Required)
Tests basic imports and structure without training
"""

import sys

print("=" * 60)
print("🧪 DERIV HYBRID PREDICTOR - STRUCTURE TEST")
print("=" * 60)

# Test 1: Check file structure
print("\n1️⃣ Checking file structure...")
import os

required_files = [
    'core/__init__.py',
    'core/config.py',
    'core/data_engine.py',
    'core/feature_engine.py',
    'core/risk_engine.py',
    'core/core_engine.py',
    'core/models/__init__.py',
    'core/models/lstm_engine.py',
    'core/models/tree_engine.py',
    'core/models/ensemble_engine.py',
    'core/models/q_agent.py',
    'requirements.txt',
    'README.md'
]

missing = []
for file in required_files:
    if not os.path.exists(file):
        missing.append(file)

if missing:
    print(f"   ❌ Missing files: {missing}")
    sys.exit(1)
else:
    print(f"   ✅ All {len(required_files)} core files present")

# Test 2: Import config
print("\n2️⃣ Testing configuration...")
try:
    from core import config
    print(f"   ✅ Config loaded")
    print(f"      - Buffer size: {config.BUFFER_SIZE}")
    print(f"      - Sequence length: {config.SEQUENCE_LENGTH}")
    print(f"      - LSTM units: {config.LSTM_CONFIG['units_layer1']}")
except Exception as e:
    print(f"   ❌ Config failed: {e}")
    sys.exit(1)

# Test 3: Import data engine
print("\n3️⃣ Testing data engine...")
try:
    from core.data_engine import DataEngine, StreamingBuffer
    
    buffer = StreamingBuffer(max_size=100)
    buffer.add_tick({'timestamp': 1700000000, 'quote': 1000.0, 'symbol': 'R_100'})
    
    print(f"   ✅ Data engine working")
    print(f"      - Buffer size: {buffer.size()}")
except Exception as e:
    print(f"   ❌ Data engine failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Import feature engine
print("\n4️⃣ Testing feature engine...")
try:
    from core.feature_engine import FeatureEngine
    import pandas as pd
    import numpy as np
    
    # Create test data
    df = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=100, freq='s'),
        'quote': 1000 + np.random.randn(100).cumsum()
    })
    
    fe = FeatureEngine()
    df_features = fe.transform(df)
    
    print(f"   ✅ Feature engine working")
    print(f"      - Generated {len(fe.feature_names)} features")
    print(f"      - Output shape: {df_features.shape}")
except Exception as e:
    print(f"   ❌ Feature engine failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Import risk engine
print("\n5️⃣ Testing risk engine...")
try:
    from core.risk_engine import RiskEngine
    
    risk = RiskEngine()
    risk.update(1)  # Win
    risk.update(0)  # Loss
    
    stats = risk.get_statistics()
    
    print(f"   ✅ Risk engine working")
    print(f"      - Active: {stats['is_active']}")
    print(f"      - Win rate: {stats['win_rate_all']:.2%}")
except Exception as e:
    print(f"   ❌ Risk engine failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Import ensemble engine
print("\n6️⃣ Testing ensemble engine...")
try:
    from core.models.ensemble_engine import EnsembleEngine
    
    ensemble = EnsembleEngine()
    
    # Test probability fusion
    combined = ensemble.combine_probabilities(0.6, 0.7)
    decision, confidence = ensemble.make_decision(combined)
    
    print(f"   ✅ Ensemble engine working")
    print(f"      - Combined prob: {combined:.2%}")
    print(f"      - Decision: {decision}")
except Exception as e:
    print(f"   ❌ Ensemble engine failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 7: Import Q-Agent
print("\n7️⃣ Testing Q-Learning agent...")
try:
    from core.models.q_agent import QAgent
    
    agent = QAgent()
    
    # Test state discretization
    state = agent.get_state(
        confidence=0.6,
        win_rate=0.52,
        volatility=0.5,
        streak=2
    )
    
    action = agent.act(state, explore=False)
    
    print(f"   ✅ Q-Agent working")
    print(f"      - State index: {state}")
    print(f"      - Action: {'EXECUTE' if action == 1 else 'SKIP'}")
except Exception as e:
    print(f"   ❌ Q-Agent failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 8: Import tree engine
print("\n8️⃣ Testing tree engine...")
try:
    from core.models.tree_engine import TreeEngine
    import numpy as np
    
    tree = TreeEngine()
    tree.build()
    
    # Test with dummy data
    X = np.random.randn(100, 10)
    y = np.random.randint(0, 2, 100)
    
    tree.train(X, y)
    
    print(f"   ✅ Tree engine working")
    print(f"      - Tree depth: {tree.model.get_depth()}")
except Exception as e:
    print(f"   ❌ Tree engine failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 9: Check TensorFlow availability
print("\n9️⃣ Checking TensorFlow...")
try:
    import tensorflow as tf
    print(f"   ✅ TensorFlow {tf.__version__} installed")
    print(f"      - GPU available: {len(tf.config.list_physical_devices('GPU')) > 0}")
except ImportError:
    print("   ⚠️ TensorFlow not installed")
    print("      Install with: pip install tensorflow")
    print("      (Required for LSTM engine)")

# Final summary
print("\n" + "=" * 60)
print("🎉 STRUCTURE TEST PASSED!")
print("=" * 60)
print("\n✅ Core structure verified")
print("✅ All non-LSTM components working")
print("\n📚 Next steps:")
print("   1. Install TensorFlow: pip install -r requirements.txt")
print("   2. Run full test: python test_installation.py")
print("   3. Read docs/QUICKSTART.md for usage")
print("\n🚀 Ready for development!")
print("=" * 60)
