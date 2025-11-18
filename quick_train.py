#!/usr/bin/env python3
"""Quick model training with minimal data"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'backend'))

print("🔥 QUICK MODEL TRAINING")
print("=" * 60)

# Create models dir
models_dir = Path(__file__).parent / 'backend' / 'ml' / 'models'
models_dir.mkdir(parents=True, exist_ok=True)

# Train Credit Scorer
print("\n1. Credit Scorer...")
try:
    from ml.credit_scorer import CreditScorer, generate_synthetic_credit_data
    
    X, y = generate_synthetic_credit_data(n_samples=1000)  # Small dataset
    print(f"   Generated {len(X)} samples")
    
    scorer = CreditScorer()
    scorer.train(X, y)
    scorer.save_models()
    print("   ✅ Saved")
except Exception as e:
    print(f"   ❌ Failed: {e}")

print("\n" + "=" * 60)
print("✅ TRAINING COMPLETE")
print(f"Models in: {models_dir}")
