"""
Core Package Initializer
"""

from .data_engine import DataEngine, StreamingBuffer
from .feature_engine import FeatureEngine
from .risk_engine import RiskEngine
from .core_engine import HybridEngine
from .models import LSTMEngine, TreeEngine, EnsembleEngine, RegimeFilter

# Alias for backward compatibility
QAgent = RegimeFilter

__version__ = '2.0.0'

__all__ = [
    'DataEngine',
    'StreamingBuffer',
    'FeatureEngine',
    'RiskEngine',
    'HybridEngine',
    'LSTMEngine',
    'TreeEngine',
    'EnsembleEngine',
    'RegimeFilter',
    'QAgent',  # backward compat alias
]
