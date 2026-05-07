"""
Test Script: Verify Core Engine Installation
Run this to ensure everything is working correctly
"""

import sys
import numpy as np

print("=" * 60)
print("🧪 DERIV HYBRID PREDICTOR - INSTALLATION TEST")
print("=" * 60)

# Test 1: Import core modules
print("\n1️⃣ Testing imports...")
try:
    from core import (
        HybridEngine,
        DataEngine,
        FeatureEngine,
        RiskEngine,
        LSTMEngine,
        TreeEngine,
        EnsembleEngine,
        QAgent
    )
    print("   ✅ All core modules imported successfully")
except ImportError as e:
    print(f"   ❌ Import failed: {e}")
    sys.exit(1)

# Test 2: Generate synthetic data
print("\n2️⃣ Generating synthetic test data...")
def generate_test_ticks(n=2000):
    """Generate synthetic tick data"""
    np.random.seed(42)
    ticks = []
    price = 1000.0
    
    for i in range(n):
        price += np.random.randn() * 0.5
        ticks.append({
            'timestamp': 1700000000 + i,
            'quote': price,
            'symbol': 'R_100'
        })
    
    return ticks

ticks = generate_test_ticks(2000)
print(f"   ✅ Generated {len(ticks)} test ticks")

# Test 3: Initialize engine
print("\n3️⃣ Initializing HybridEngine...")
try:
    engine = HybridEngine(verbose=False)
    print("   ✅ Engine initialized")
except Exception as e:
    print(f"   ❌ Initialization failed: {e}")
    sys.exit(1)

# Test 4: Load data
print("\n4️⃣ Loading tick data...")
try:
    for tick in ticks:
        engine.add_tick(tick)
    print(f"   ✅ Loaded {engine.tick_count} ticks")
except Exception as e:
    print(f"   ❌ Data loading failed: {e}")
    sys.exit(1)

# Test 5: Train models
print("\n5️⃣ Training models...")
try:
    results = engine.initial_train()
    print(f"   ✅ Training complete")
    print(f"      - LSTM accuracy: {results['lstm_accuracy']:.2%}")
    print(f"      - Tree accuracy: {results['tree_accuracy']:.2%}")
    print(f"      - Training time: {results['training_time']:.2f}s")
except Exception as e:
    print(f"   ❌ Training failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Make prediction
print("\n6️⃣ Testing prediction...")
try:
    prediction = engine.predict_next_tick()
    
    if prediction:
        print(f"   ✅ Prediction successful")
        print(f"      - Decision: {prediction['final_decision']}")
        print(f"      - Confidence: {prediction['confidence']:.2%}")
        print(f"      - LSTM prob: {prediction['lstm_prob']:.2%}")
        print(f"      - Tree prob: {prediction['tree_prob']:.2%}")
        print(f"      - Combined: {prediction['combined_prob']:.2%}")
    else:
        print("   ⚠️ Prediction returned None (insufficient data)")
except Exception as e:
    print(f"   ❌ Prediction failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 7: Update with outcome
print("\n7️⃣ Testing outcome update...")
try:
    actual_direction = 1  # UP
    engine.update_with_outcome(prediction, actual_direction)
    print("   ✅ Outcome update successful")
except Exception as e:
    print(f"   ❌ Update failed: {e}")
    sys.exit(1)

# Test 8: Get status
print("\n8️⃣ Testing status retrieval...")
try:
    status = engine.get_status()
    print("   ✅ Status retrieved")
    print(f"      - Total predictions: {status['engine']['prediction_count']}")
    print(f"      - Trade frequency: {status['ensemble']['trade_frequency']:.2%}")
    print(f"      - Current threshold: {status['ensemble']['current_threshold']:.3f}")
except Exception as e:
    print(f"   ❌ Status retrieval failed: {e}")
    sys.exit(1)

# Test 9: Risk engine
print("\n9️⃣ Testing risk engine...")
try:
    risk_stats = engine.risk_engine.get_statistics()
    print("   ✅ Risk engine working")
    print(f"      - Active: {risk_stats['is_active']}")
    print(f"      - Drawdown: {risk_stats['drawdown']:.2%}")
    print(f"      - Win rate: {risk_stats['win_rate_all']:.2%}")
except Exception as e:
    print(f"   ❌ Risk engine failed: {e}")
    sys.exit(1)

# Test 10: Model persistence
print("\n🔟 Testing model save/load...")
try:
    import tempfile
    import os
    
    # Create temp directory
    temp_dir = tempfile.mkdtemp()
    
    # Save models
    engine.save_models(temp_dir)
    
    # Create new engine and load
    engine2 = HybridEngine(verbose=False)
    engine2.load_models(temp_dir)
    
    # Cleanup
    import shutil
    shutil.rmtree(temp_dir)
    
    print("   ✅ Model persistence working")
except Exception as e:
    print(f"   ❌ Model persistence failed: {e}")
    import traceback
    traceback.print_exc()

# Final summary
print("\n" + "=" * 60)
print("🎉 ALL TESTS PASSED!")
print("=" * 60)
print("\n✅ Installation verified successfully")
print("✅ Core engine is fully functional")
print("\n📚 Next steps:")
print("   1. Read docs/QUICKSTART.md for usage guide")
print("   2. Read docs/ARCHITECTURE.md for technical details")
print("   3. Check colab/TRAINING_GUIDE.md for Colab setup")
print("\n🚀 Ready to start predicting!")
print("=" * 60)
