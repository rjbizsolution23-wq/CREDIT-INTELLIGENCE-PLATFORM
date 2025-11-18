"""
Credit Scoring ML Model
XGBoost + LightGBM + CatBoost Ensemble with SHAP Explainability
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple
import joblib
from pathlib import Path

# ML Models
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score, classification_report
from imblearn.over_sampling import SMOTE

# Explainability
import shap


class CreditScorer:
    """
    Elite credit scoring system with ensemble ML + SHAP explainability
    """
    
    def __init__(self, model_path: str = None):
        self.model_path = model_path or Path(__file__).parent / "models"
        self.model_path = Path(self.model_path)
        self.model_path.mkdir(parents=True, exist_ok=True)
        
        self.ensemble = None
        self.xgb_model = None
        self.lgbm_model = None
        self.catboost_model = None
        self.explainer = None
        
        self.feature_names = [
            # Payment History (35% of FICO)
            'late_payments_12mo',
            'late_payments_24mo',
            'late_payments_36mo',
            'delinquencies_total',
            'payment_history_score',
            
            # Credit Utilization (30% of FICO)
            'credit_utilization_ratio',
            'total_balance',
            'total_credit_limit',
            'avg_utilization_per_account',
            'revolving_balance',
            
            # Credit Age (15% of FICO)
            'oldest_account_age_months',
            'avg_account_age_months',
            'newest_account_age_months',
            'length_of_credit_history',
            
            # Credit Mix (10% of FICO)
            'total_accounts',
            'revolving_accounts',
            'installment_accounts',
            'mortgage_accounts',
            'auto_loan_accounts',
            'credit_mix_diversity',
            
            # New Credit (10% of FICO)
            'hard_inquiries_6mo',
            'hard_inquiries_12mo',
            'hard_inquiries_24mo',
            'new_accounts_6mo',
            'new_accounts_12mo',
            
            # Derogatory Items
            'collections_count',
            'chargeoffs_count',
            'bankruptcies_count',
            'judgments_count',
            'tax_liens_count',
            'public_records_total',
            
            # Additional Factors
            'total_debt_to_income',
            'monthly_payment_to_income',
            'available_credit',
            'debt_consolidation_flag'
        ]
    
    def train(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2) -> Dict[str, Any]:
        """
        Train ensemble model with hyperparameter tuning
        
        Args:
            X: Feature matrix
            y: Target variable (0 = bad credit, 1 = good credit)
            test_size: Test set proportion
        
        Returns:
            Training metrics
        """
        print("🚀 Training Credit Scoring Ensemble...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=42
        )
        
        # Handle class imbalance with SMOTE (if needed)
        class_counts = y_train.value_counts()
        minority_ratio = class_counts.min() / class_counts.max()
        
        if minority_ratio < 0.5:
            print(f"⚖️  Applying SMOTE for class balance (ratio: {minority_ratio:.2f})...")
            smote = SMOTE(sampling_strategy=0.8, random_state=42)
            X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
        else:
            print(f"✅ Classes already balanced (ratio: {minority_ratio:.2f}), skipping SMOTE")
            X_train_balanced, y_train_balanced = X_train, y_train
        
        # 1. XGBoost
        print("🌳 Training XGBoost...")
        self.xgb_model = XGBClassifier(
            n_estimators=1000,
            max_depth=7,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            gamma=0.1,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            eval_metric='auc'
        )
        self.xgb_model.fit(
            X_train_balanced, y_train_balanced,
            eval_set=[(X_test, y_test)],
            verbose=False
        )
        
        # 2. LightGBM
        print("⚡ Training LightGBM...")
        self.lgbm_model = LGBMClassifier(
            n_estimators=1000,
            max_depth=7,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=20,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        self.lgbm_model.fit(
            X_train_balanced, y_train_balanced,
            eval_set=[(X_test, y_test)],
            callbacks=[]
        )
        
        # 3. CatBoost
        print("🐱 Training CatBoost...")
        self.catboost_model = CatBoostClassifier(
            iterations=1000,
            depth=6,
            learning_rate=0.05,
            l2_leaf_reg=3,
            random_seed=42,
            verbose=False
        )
        self.catboost_model.fit(
            X_train_balanced, y_train_balanced,
            eval_set=(X_test, y_test),
            verbose=False
        )
        
        # 4. Create Voting Ensemble
        print("🎯 Building Ensemble...")
        self.ensemble = VotingClassifier(
            estimators=[
                ('xgb', self.xgb_model),
                ('lgbm', self.lgbm_model),
                ('catboost', self.catboost_model)
            ],
            voting='soft',
            weights=[1.2, 1.0, 1.1]  # XGBoost slightly higher weight
        )
        self.ensemble.fit(X_train_balanced, y_train_balanced)
        
        # Evaluate
        print("📊 Evaluating Models...")
        metrics = {}
        
        for name, model in [
            ('xgb', self.xgb_model),
            ('lgbm', self.lgbm_model),
            ('catboost', self.catboost_model),
            ('ensemble', self.ensemble)
        ]:
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_pred_proba)
            metrics[name] = {'auc': auc}
            print(f"  {name.upper()}: AUC = {auc:.4f}")
        
        # Initialize SHAP explainer (use XGBoost for speed)
        print("🔍 Initializing SHAP Explainer...")
        self.explainer = shap.TreeExplainer(self.xgb_model)
        
        # Save models
        print("💾 Saving models...")
        self.save_models()
        
        print("✅ Training complete!")
        return metrics
    
    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict credit scores
        
        Returns:
            (scores, probabilities)
        """
        if self.ensemble is None:
            self.load_models()
        
        # Get probability (0-1)
        proba = self.ensemble.predict_proba(X)[:, 1]
        
        # Convert to credit score (300-850)
        scores = self._proba_to_credit_score(proba)
        
        return scores, proba
    
    def explain(self, X: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate SHAP explanations
        
        Returns:
            SHAP values and feature importance
        """
        if self.explainer is None:
            self.explainer = shap.TreeExplainer(self.xgb_model)
        
        # Calculate SHAP values
        shap_values = self.explainer.shap_values(X)
        
        # Get feature importance
        feature_importance = {}
        for i, feature in enumerate(self.feature_names[:X.shape[1]]):
            feature_importance[feature] = float(np.mean(np.abs(shap_values[:, i])))
        
        # Sort by importance
        feature_importance = dict(
            sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        )
        
        return {
            'shap_values': shap_values.tolist() if len(shap_values.shape) > 1 else shap_values[0].tolist(),
            'feature_importance': feature_importance,
            'expected_value': float(self.explainer.expected_value)
        }
    
    def generate_recommendations(self, X: pd.DataFrame, shap_values: np.ndarray) -> List[str]:
        """
        Generate actionable recommendations based on SHAP values
        """
        recommendations = []
        
        # Get feature impacts
        feature_impacts = {}
        for i, feature in enumerate(self.feature_names[:X.shape[1]]):
            feature_impacts[feature] = float(shap_values[i] if len(shap_values.shape) == 1 else shap_values[0][i])
        
        # Sort by negative impact (hurting factors)
        negative_impacts = {k: v for k, v in feature_impacts.items() if v < 0}
        negative_impacts = dict(sorted(negative_impacts.items(), key=lambda x: x[1]))
        
        # Generate recommendations
        if 'credit_utilization_ratio' in negative_impacts:
            util = X['credit_utilization_ratio'].iloc[0] * 100
            if util > 30:
                target_reduction = (util - 25) / 100 * X['total_credit_limit'].iloc[0]
                recommendations.append(
                    f"💰 Reduce credit utilization from {util:.1f}% to below 30% "
                    f"(pay down ${target_reduction:,.0f})"
                )
        
        if 'late_payments_12mo' in negative_impacts:
            recommendations.append(
                "📅 Set up autopay to avoid future late payments (35% of credit score)"
            )
        
        if 'hard_inquiries_6mo' in negative_impacts:
            inquiries = X['hard_inquiries_6mo'].iloc[0]
            if inquiries > 3:
                recommendations.append(
                    f"🚫 Avoid new credit applications for 6 months "
                    f"({int(inquiries)} recent inquiries detected)"
                )
        
        if 'collections_count' in negative_impacts or 'chargeoffs_count' in negative_impacts:
            recommendations.append(
                "📝 Consider dispute process for negative accounts (FCRA rights)"
            )
        
        if 'credit_mix_diversity' in negative_impacts:
            recommendations.append(
                "🎯 Improve credit mix by adding different account types "
                "(installment loan, secured card)"
            )
        
        return recommendations[:5]  # Top 5 recommendations
    
    def predict_with_explanation(self, credit_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict credit score with full SHAP explanation and recommendations
        
        Args:
            credit_data: Dictionary with credit features
        
        Returns:
            Complete analysis with score, confidence, risk level, SHAP values, recommendations
        """
        # Prepare features
        feature_values = []
        for feature in self.feature_names:
            feature_values.append(credit_data.get(feature, 0))
        
        X = pd.DataFrame([feature_values], columns=self.feature_names)
        
        # Get prediction
        scores, proba = self.predict(X)
        credit_score = int(scores[0])
        confidence = float(proba[0])
        
        # Determine risk level
        if credit_score >= 740:
            risk_level = 'LOW'
        elif credit_score >= 670:
            risk_level = 'MEDIUM'
        elif credit_score >= 580:
            risk_level = 'HIGH'
        else:
            risk_level = 'CRITICAL'
        
        # Generate SHAP explanations
        explanation = self.explain(X)
        
        # Generate recommendations
        shap_values = np.array(explanation['shap_values'])
        recommendations = self.generate_recommendations(X, shap_values)
        
        # Get top positive and negative factors
        feature_importance = explanation['feature_importance']
        sorted_features = sorted(feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)
        
        factors_helping = []
        factors_hurting = []
        
        for feature, impact in sorted_features[:10]:
            impact_value = shap_values[self.feature_names.index(feature)] if feature in self.feature_names else 0
            feature_display = feature.replace('_', ' ').title()
            
            if impact_value > 0:
                factors_helping.append({
                    'factor': feature_display,
                    'impact': f"+{int(impact_value * 100)} points"
                })
            elif impact_value < 0:
                factors_hurting.append({
                    'factor': feature_display,
                    'impact': f"{int(impact_value * 100)} points"
                })
        
        return {
            'credit_score': credit_score,
            'confidence': confidence,
            'risk_level': risk_level,
            'feature_importance': feature_importance,
            'shap_values': explanation.get('shap_values', []),
            'recommendations': recommendations,
            'factors_helping': factors_helping[:5],
            'factors_hurting': factors_hurting[:5]
        }
    
    def _proba_to_credit_score(self, proba: np.ndarray) -> np.ndarray:
        """
        Convert probability to FICO-style credit score (300-850)
        
        Args:
            proba: Probability of good credit (0-1)
        
        Returns:
            Credit scores (300-850)
        """
        # Sigmoid transformation for realistic distribution
        # proba 0.0 → score 300
        # proba 0.5 → score 575
        # proba 1.0 → score 850
        
        scores = 300 + (proba * 550)
        return np.clip(scores, 300, 850).astype(int)
    
    def save_models(self):
        """Save trained models"""
        joblib.dump(self.xgb_model, self.model_path / 'xgb_model.pkl')
        joblib.dump(self.lgbm_model, self.model_path / 'lgbm_model.pkl')
        joblib.dump(self.catboost_model, self.model_path / 'catboost_model.pkl')
        joblib.dump(self.ensemble, self.model_path / 'ensemble_model.pkl')
        print(f"✅ Models saved to {self.model_path}")
    
    def load_models(self):
        """Load trained models"""
        try:
            self.xgb_model = joblib.load(self.model_path / 'xgb_model.pkl')
            self.lgbm_model = joblib.load(self.model_path / 'lgbm_model.pkl')
            self.catboost_model = joblib.load(self.model_path / 'catboost_model.pkl')
            self.ensemble = joblib.load(self.model_path / 'ensemble_model.pkl')
            self.explainer = shap.TreeExplainer(self.xgb_model)
            print("✅ Models loaded successfully")
        except FileNotFoundError:
            print("⚠️  No trained models found. Please train first.")


# Synthetic data generator for initial training
def generate_synthetic_credit_data(n_samples: int = 10000) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Generate synthetic credit data for initial model training
    """
    np.random.seed(42)
    
    data = {
        # Payment history
        'late_payments_12mo': np.random.poisson(0.5, n_samples),
        'late_payments_24mo': np.random.poisson(1.0, n_samples),
        'late_payments_36mo': np.random.poisson(1.5, n_samples),
        'delinquencies_total': np.random.poisson(2.0, n_samples),
        'payment_history_score': np.random.normal(85, 15, n_samples).clip(0, 100),
        
        # Credit utilization
        'credit_utilization_ratio': np.random.beta(2, 5, n_samples),
        'total_balance': np.random.lognormal(9, 1, n_samples),
        'total_credit_limit': np.random.lognormal(10.5, 0.8, n_samples),
        'avg_utilization_per_account': np.random.beta(2, 5, n_samples),
        'revolving_balance': np.random.lognormal(8.5, 1.2, n_samples),
        
        # Credit age
        'oldest_account_age_months': np.random.exponential(120, n_samples).clip(6, 600),
        'avg_account_age_months': np.random.exponential(60, n_samples).clip(3, 400),
        'newest_account_age_months': np.random.exponential(24, n_samples).clip(1, 200),
        'length_of_credit_history': np.random.exponential(120, n_samples).clip(6, 600),
        
        # Credit mix
        'total_accounts': np.random.poisson(12, n_samples).clip(1, 50),
        'revolving_accounts': np.random.poisson(6, n_samples).clip(0, 30),
        'installment_accounts': np.random.poisson(4, n_samples).clip(0, 20),
        'mortgage_accounts': np.random.binomial(2, 0.3, n_samples),
        'auto_loan_accounts': np.random.binomial(2, 0.4, n_samples),
        'credit_mix_diversity': np.random.uniform(0, 1, n_samples),
        
        # New credit
        'hard_inquiries_6mo': np.random.poisson(1.5, n_samples).clip(0, 10),
        'hard_inquiries_12mo': np.random.poisson(2.5, n_samples).clip(0, 15),
        'hard_inquiries_24mo': np.random.poisson(4, n_samples).clip(0, 20),
        'new_accounts_6mo': np.random.poisson(0.5, n_samples).clip(0, 5),
        'new_accounts_12mo': np.random.poisson(1, n_samples).clip(0, 8),
        
        # Derogatory items
        'collections_count': np.random.poisson(0.3, n_samples),
        'chargeoffs_count': np.random.poisson(0.2, n_samples),
        'bankruptcies_count': np.random.binomial(1, 0.05, n_samples),
        'judgments_count': np.random.poisson(0.1, n_samples),
        'tax_liens_count': np.random.poisson(0.05, n_samples),
        'public_records_total': np.random.poisson(0.5, n_samples),
        
        # Additional
        'total_debt_to_income': np.random.beta(2, 3, n_samples),
        'monthly_payment_to_income': np.random.beta(2, 5, n_samples),
        'available_credit': np.random.lognormal(9.5, 1, n_samples),
        'debt_consolidation_flag': np.random.binomial(1, 0.15, n_samples),
    }
    
    X = pd.DataFrame(data)
    
    # Generate target based on features (good credit = 1, bad credit = 0)
    credit_score = (
        X['payment_history_score'] * 0.35 +
        (1 - X['credit_utilization_ratio']) * 100 * 0.30 +
        (X['oldest_account_age_months'] / 6) * 0.15 +
        X['credit_mix_diversity'] * 100 * 0.10 +
        (1 - X['hard_inquiries_6mo'] / 10) * 100 * 0.10 -
        X['late_payments_12mo'] * 10 -
        X['collections_count'] * 15 -
        X['bankruptcies_count'] * 50
    )
    
    # Binary classification (good vs bad credit)
    y = (credit_score > 60).astype(int)
    
    return X, y


if __name__ == "__main__":
    # Demo: Train model on synthetic data
    print("🎯 Credit Scoring ML Model - Demo\n")
    
    print("1️⃣  Generating synthetic credit data...")
    X, y = generate_synthetic_credit_data(n_samples=50000)
    print(f"   Generated {len(X)} samples")
    print(f"   Good credit: {y.sum()} ({y.mean()*100:.1f}%)")
    print(f"   Bad credit: {(~y.astype(bool)).sum()} ({(1-y.mean())*100:.1f}%)\n")
    
    print("2️⃣  Training ensemble model...\n")
    scorer = CreditScorer()
    metrics = scorer.train(X, y)
    
    print("\n3️⃣  Testing prediction...")
    # Test on single sample
    test_sample = X.iloc[[0]]
    scores, proba = scorer.predict(test_sample)
    print(f"   Predicted credit score: {scores[0]}")
    print(f"   Confidence: {proba[0]:.4f}\n")
    
    print("4️⃣  Generating SHAP explanation...")
    explanation = scorer.explain(test_sample)
    print("   Top 5 features:")
    for i, (feature, importance) in enumerate(list(explanation['feature_importance'].items())[:5], 1):
        print(f"   {i}. {feature}: {importance:.4f}")
    
    print("\n✅ Demo complete!")
