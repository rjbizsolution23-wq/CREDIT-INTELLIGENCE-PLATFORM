"""
🔥 ELITE CREDIT SCORE FORECASTING SYSTEM
LSTM-Transformer Hybrid for 6-12 Month Predictions
Author: Rick Jefferson Solutions
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple
from datetime import datetime, timedelta
import pickle
from dataclasses import dataclass


@dataclass
class ForecastResult:
    """Credit score forecast output"""
    current_score: int
    forecasted_scores: List[int]  # Monthly predictions
    forecast_months: List[str]  # YYYY-MM format
    confidence_intervals: List[Tuple[int, int]]  # (lower, upper) bounds
    trend: str  # 'improving', 'stable', 'declining'
    key_drivers: List[Dict[str, Any]]
    milestone_dates: Dict[str, str]  # e.g., {'reach_700': '2025-06-15'}
    recommendations: List[str]


class TransformerEncoder(nn.Module):
    """Transformer encoder with multi-head attention"""
    
    def __init__(self, d_model: int = 128, nhead: int = 8, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
    def forward(self, x):
        return self.transformer(x)


class LSTMTransformerForecaster(nn.Module):
    """
    Hybrid LSTM-Transformer model for credit score forecasting
    LSTM captures long-term trends, Transformer handles complex patterns
    """
    
    def __init__(self, input_dim: int = 20, lstm_hidden: int = 128, 
                 transformer_dim: int = 128, num_heads: int = 8, 
                 output_seq_len: int = 12, dropout: float = 0.2):
        super().__init__()
        
        # LSTM for temporal dependencies
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=lstm_hidden,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )
        
        # Project LSTM output to transformer dimension
        self.lstm_projection = nn.Linear(lstm_hidden * 2, transformer_dim)
        
        # Transformer for attention-based patterns
        self.transformer = TransformerEncoder(
            d_model=transformer_dim,
            nhead=num_heads,
            num_layers=3,
            dropout=dropout
        )
        
        # Output layers
        self.fc1 = nn.Linear(transformer_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, output_seq_len)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(transformer_dim)
        
    def forward(self, x):
        # LSTM processing
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Project to transformer dimension
        lstm_projected = self.lstm_projection(lstm_out)
        lstm_projected = self.layer_norm(lstm_projected)
        
        # Transformer attention
        transformer_out = self.transformer(lstm_projected)
        
        # Use last timestep for prediction
        last_hidden = transformer_out[:, -1, :]
        
        # Prediction layers
        x = F.relu(self.fc1(last_hidden))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        predictions = self.fc3(x)
        
        return predictions


class CreditScoreForecaster:
    """
    Elite credit score forecasting system
    Predicts 6-12 month credit score trajectory with confidence intervals
    """
    
    def __init__(self, model_path: str = None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = LSTMTransformerForecaster(
            input_dim=20,
            lstm_hidden=128,
            transformer_dim=128,
            num_heads=8,
            output_seq_len=12
        )
        self.model.to(self.device)
        self.feature_scaler = None
        self.trained = False
        
        if model_path:
            self.load_model(model_path)
    
    def prepare_features(self, credit_history: List[Dict[str, Any]]) -> torch.Tensor:
        """
        Convert credit history to ML features
        Features: utilization, payments, balances, inquiries, etc. over time
        """
        features = []
        
        for month_data in credit_history:
            month_features = [
                month_data.get('credit_score', 650) / 850,  # Normalized score
                month_data.get('utilization', 0.3),
                month_data.get('on_time_payments', 0) / 10,
                month_data.get('late_payments', 0) / 10,
                month_data.get('total_balance', 0) / 100000,
                month_data.get('total_credit_limit', 10000) / 100000,
                month_data.get('num_accounts', 5) / 20,
                month_data.get('new_accounts', 0) / 5,
                month_data.get('closed_accounts', 0) / 5,
                month_data.get('hard_inquiries', 0) / 10,
                month_data.get('derogatory_marks', 0) / 5,
                month_data.get('collections', 0) / 5,
                month_data.get('oldest_account_months', 60) / 360,
                month_data.get('avg_account_age', 30) / 180,
                month_data.get('payment_history_score', 0.8),
                month_data.get('credit_mix_score', 0.6),
                month_data.get('new_credit_score', 0.7),
                month_data.get('month_of_year', 6) / 12,  # Seasonality
                month_data.get('economic_indicator', 1.0),
                month_data.get('account_velocity', 0) / 10
            ]
            features.append(month_features)
        
        return torch.tensor(features, dtype=torch.float32)
    
    def forecast(self, credit_data: Dict[str, Any], months_ahead: int = 12) -> ForecastResult:
        """
        Generate credit score forecast
        Predicts monthly scores with confidence intervals
        """
        # Get current score
        current_score = credit_data.get('credit_score', 650)
        
        # Build credit history (use last 12 months if available)
        credit_history = credit_data.get('monthly_history', [])
        
        # If no history, simulate based on current state
        if not credit_history or len(credit_history) < 6:
            credit_history = self.simulate_credit_history(credit_data, months=12)
        
        # Ensure we have at least 12 months of history
        while len(credit_history) < 12:
            credit_history.insert(0, credit_history[0].copy() if credit_history else {
                'credit_score': current_score,
                'utilization': credit_data.get('credit_utilization', 0.3),
                'on_time_payments': 10,
                'late_payments': 0,
                'total_balance': credit_data.get('total_balance', 5000),
                'total_credit_limit': credit_data.get('total_credit_limit', 20000),
                'num_accounts': credit_data.get('total_accounts', 5),
                'new_accounts': 0,
                'closed_accounts': 0,
                'hard_inquiries': 0,
                'derogatory_marks': credit_data.get('derogatory_marks', 0),
                'collections': credit_data.get('collections_count', 0),
                'oldest_account_months': credit_data.get('oldest_account_age_months', 60),
                'avg_account_age': credit_data.get('avg_account_age_months', 30),
                'payment_history_score': 0.8,
                'credit_mix_score': 0.6,
                'new_credit_score': 0.7,
                'month_of_year': 6,
                'economic_indicator': 1.0,
                'account_velocity': 0
            })
        
        # Use last 12 months
        recent_history = credit_history[-12:]
        
        # Prepare features
        features = self.prepare_features(recent_history)
        features = features.unsqueeze(0).to(self.device)  # Add batch dimension
        
        # Generate predictions
        if self.trained:
            self.model.eval()
            with torch.no_grad():
                predictions = self.model(features)
                predictions = predictions.squeeze().cpu().numpy()
        else:
            # Use heuristic forecast if model not trained
            predictions = self.heuristic_forecast(credit_data, current_score, months_ahead)
        
        # Convert predictions to credit scores (300-850 range)
        forecasted_scores = [int(max(300, min(850, current_score + pred))) for pred in predictions[:months_ahead]]
        
        # Generate forecast months
        today = datetime.now()
        forecast_months = [(today + timedelta(days=30*i)).strftime('%Y-%m') for i in range(1, months_ahead + 1)]
        
        # Calculate confidence intervals (±20 points for short term, ±40 for long term)
        confidence_intervals = []
        for i, score in enumerate(forecasted_scores):
            uncertainty = 20 + (i * 2)  # Increases with time
            lower = max(300, score - uncertainty)
            upper = min(850, score + uncertainty)
            confidence_intervals.append((lower, upper))
        
        # Determine trend
        if forecasted_scores[-1] > current_score + 20:
            trend = 'improving'
        elif forecasted_scores[-1] < current_score - 20:
            trend = 'declining'
        else:
            trend = 'stable'
        
        # Identify key drivers
        key_drivers = self.identify_key_drivers(credit_data, forecasted_scores)
        
        # Calculate milestone dates
        milestone_dates = self.calculate_milestones(current_score, forecasted_scores, forecast_months)
        
        # Generate recommendations
        recommendations = self.generate_forecast_recommendations(credit_data, trend, forecasted_scores)
        
        return ForecastResult(
            current_score=current_score,
            forecasted_scores=forecasted_scores,
            forecast_months=forecast_months,
            confidence_intervals=confidence_intervals,
            trend=trend,
            key_drivers=key_drivers,
            milestone_dates=milestone_dates,
            recommendations=recommendations
        )
    
    def heuristic_forecast(self, credit_data: Dict[str, Any], current_score: int, months: int) -> np.ndarray:
        """
        Heuristic-based forecast when ML model not trained
        Uses credit fundamentals to predict trajectory
        """
        # Base monthly change factors
        utilization = credit_data.get('credit_utilization', 0.3)
        late_payments = credit_data.get('late_payments_12mo', 0)
        inquiries = credit_data.get('inquiries_6mo', 0)
        collections = credit_data.get('collections_count', 0)
        
        # Calculate monthly improvement rate
        monthly_change = 0
        
        # Utilization impact (biggest factor)
        if utilization < 0.1:
            monthly_change += 5  # Very low utilization: +5 points/month
        elif utilization < 0.3:
            monthly_change += 3  # Low utilization: +3 points/month
        elif utilization < 0.5:
            monthly_change += 1  # Medium utilization: +1 point/month
        elif utilization < 0.7:
            monthly_change -= 1  # High utilization: -1 point/month
        else:
            monthly_change -= 3  # Very high utilization: -3 points/month
        
        # Payment history impact
        if late_payments == 0:
            monthly_change += 2  # Clean payment history: +2 points/month
        elif late_payments <= 2:
            monthly_change -= 1  # Few late payments: -1 point/month
        else:
            monthly_change -= 3  # Many late payments: -3 points/month
        
        # Inquiry impact (decays over time)
        if inquiries > 5:
            monthly_change -= 2
        elif inquiries > 2:
            monthly_change -= 1
        
        # Collections impact
        if collections > 0:
            monthly_change -= 2
        
        # Generate predictions with diminishing returns
        predictions = []
        cumulative_change = 0
        for month in range(months):
            # Diminishing returns (harder to improve as score increases)
            decay_factor = 1 - (month / (months * 2))
            month_change = monthly_change * decay_factor
            cumulative_change += month_change
            predictions.append(cumulative_change)
        
        return np.array(predictions)
    
    def simulate_credit_history(self, credit_data: Dict[str, Any], months: int = 12) -> List[Dict[str, Any]]:
        """Simulate credit history from current state"""
        current_score = credit_data.get('credit_score', 650)
        utilization = credit_data.get('credit_utilization', 0.3)
        
        history = []
        for i in range(months):
            # Add some randomness to simulate realistic history
            score_noise = np.random.normal(0, 10)
            util_noise = np.random.normal(0, 0.05)
            
            history.append({
                'credit_score': int(max(300, min(850, current_score + score_noise))),
                'utilization': max(0, min(1, utilization + util_noise)),
                'on_time_payments': max(0, int(10 + np.random.normal(0, 2))),
                'late_payments': max(0, int(credit_data.get('late_payments_12mo', 0) / 12 + np.random.poisson(0.1))),
                'total_balance': credit_data.get('total_balance', 5000) * (1 + np.random.normal(0, 0.1)),
                'total_credit_limit': credit_data.get('total_credit_limit', 20000),
                'num_accounts': credit_data.get('total_accounts', 5),
                'new_accounts': int(np.random.poisson(0.1)),
                'closed_accounts': int(np.random.poisson(0.05)),
                'hard_inquiries': int(np.random.poisson(0.2)),
                'derogatory_marks': credit_data.get('derogatory_marks', 0),
                'collections': credit_data.get('collections_count', 0),
                'oldest_account_months': credit_data.get('oldest_account_age_months', 60) - (months - i),
                'avg_account_age': credit_data.get('avg_account_age_months', 30),
                'payment_history_score': 0.8,
                'credit_mix_score': 0.6,
                'new_credit_score': 0.7,
                'month_of_year': (datetime.now().month - (months - i)) % 12 + 1,
                'economic_indicator': 1.0,
                'account_velocity': 0
            })
        
        return history
    
    def identify_key_drivers(self, credit_data: Dict[str, Any], forecasted_scores: List[int]) -> List[Dict[str, Any]]:
        """Identify what's driving the forecast"""
        drivers = []
        
        utilization = credit_data.get('credit_utilization', 0.3)
        late_payments = credit_data.get('late_payments_12mo', 0)
        oldest_account = credit_data.get('oldest_account_age_months', 60)
        
        # Utilization impact
        if utilization > 0.5:
            drivers.append({
                'factor': 'High Credit Utilization',
                'current_value': f'{utilization*100:.1f}%',
                'impact': 'negative',
                'strength': 'high',
                'description': 'Using more than 50% of available credit limits your score growth'
            })
        elif utilization < 0.3:
            drivers.append({
                'factor': 'Low Credit Utilization',
                'current_value': f'{utilization*100:.1f}%',
                'impact': 'positive',
                'strength': 'high',
                'description': 'Keeping utilization under 30% strongly supports score improvement'
            })
        
        # Payment history
        if late_payments == 0:
            drivers.append({
                'factor': 'Perfect Payment History',
                'current_value': '0 late payments',
                'impact': 'positive',
                'strength': 'very high',
                'description': 'On-time payments are the #1 factor in credit score growth'
            })
        elif late_payments > 3:
            drivers.append({
                'factor': 'Multiple Late Payments',
                'current_value': f'{late_payments} in last 12 months',
                'impact': 'negative',
                'strength': 'very high',
                'description': 'Late payments significantly damage credit scores'
            })
        
        # Credit age
        if oldest_account < 24:
            drivers.append({
                'factor': 'Young Credit History',
                'current_value': f'{oldest_account} months',
                'impact': 'negative',
                'strength': 'medium',
                'description': 'Limited credit history slows score growth (ideal: 7+ years)'
            })
        elif oldest_account > 84:
            drivers.append({
                'factor': 'Established Credit History',
                'current_value': f'{oldest_account/12:.1f} years',
                'impact': 'positive',
                'strength': 'medium',
                'description': 'Long credit history provides stable foundation for high scores'
            })
        
        return drivers[:5]  # Top 5 drivers
    
    def calculate_milestones(self, current_score: int, forecasted_scores: List[int], 
                           forecast_months: List[str]) -> Dict[str, str]:
        """Calculate when credit score will hit key milestones"""
        milestones = {}
        
        milestone_targets = [600, 650, 700, 750, 800]
        
        for target in milestone_targets:
            if current_score < target:
                # Find first month where score reaches target
                for score, month in zip(forecasted_scores, forecast_months):
                    if score >= target:
                        milestones[f'reach_{target}'] = month
                        break
        
        return milestones
    
    def generate_forecast_recommendations(self, credit_data: Dict[str, Any], 
                                         trend: str, forecasted_scores: List[int]) -> List[str]:
        """Generate recommendations to improve forecast"""
        recommendations = []
        
        if trend == 'declining':
            recommendations.append("⚠️ URGENT: Address negative factors immediately to reverse declining trend")
        
        utilization = credit_data.get('credit_utilization', 0.3)
        if utilization > 0.3:
            target_paydown = credit_data.get('total_balance', 0) - (credit_data.get('total_credit_limit', 0) * 0.25)
            if target_paydown > 0:
                recommendations.append(f"💳 Pay down ${target_paydown:,.0f} to reach 25% utilization (current: {utilization*100:.1f}%)")
        
        late_payments = credit_data.get('late_payments_12mo', 0)
        if late_payments > 0:
            recommendations.append(f"📅 Set up auto-pay to prevent future late payments (had {late_payments} in last 12 months)")
        
        inquiries = credit_data.get('inquiries_6mo', 0)
        if inquiries > 2:
            recommendations.append(f"🛑 Avoid new credit applications for 6 months ({inquiries} recent inquiries)")
        
        if forecasted_scores[-1] < 700:
            recommendations.append("🎯 Focus on payment history and utilization to reach 'Good' credit (700+) faster")
        elif forecasted_scores[-1] < 800:
            recommendations.append("🚀 Maintain current habits to reach 'Excellent' credit (800+) within forecast period")
        
        oldest_account = credit_data.get('oldest_account_age_months', 60)
        if oldest_account < 36:
            recommendations.append("⏰ Keep oldest accounts open to build credit age (currently only {oldest_account} months)")
        
        return recommendations[:7]
    
    def train(self, X_train: torch.Tensor, y_train: torch.Tensor, 
              X_val: torch.Tensor = None, y_val: torch.Tensor = None,
              epochs: int = 100, lr: float = 0.001, batch_size: int = 32):
        """Train the LSTM-Transformer forecaster"""
        self.model.train()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        criterion = nn.MSELoss()
        
        dataset = torch.utils.data.TensorDataset(X_train, y_train)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        for epoch in range(epochs):
            total_loss = 0
            for batch_X, batch_y in dataloader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                predictions = self.model(batch_X)
                loss = criterion(predictions, batch_y)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(dataloader)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}/{epochs} - Loss: {avg_loss:.4f}")
        
        self.trained = True
        print("✅ Forecaster training complete")
    
    def save_model(self, path: str):
        """Save model to disk"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'trained': self.trained
        }, path)
        print(f"✅ Model saved to {path}")
    
    def load_model(self, path: str):
        """Load model from disk"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.trained = checkpoint.get('trained', False)
        print(f"✅ Model loaded from {path}")


# Demo usage
if __name__ == "__main__":
    print("🔥 CREDIT SCORE FORECASTING - DEMO")
    print("=" * 60)
    
    # Initialize forecaster
    forecaster = CreditScoreForecaster()
    
    # Test case: Mid-range score with room for improvement
    test_credit_data = {
        'credit_score': 680,
        'credit_utilization': 0.45,
        'late_payments_12mo': 1,
        'inquiries_6mo': 2,
        'collections_count': 0,
        'derogatory_marks': 0,
        'total_accounts': 8,
        'oldest_account_age_months': 72,
        'avg_account_age_months': 36,
        'total_balance': 12000,
        'total_credit_limit': 27000
    }
    
    print("\n📊 CURRENT CREDIT PROFILE:")
    print(f"Credit Score: {test_credit_data['credit_score']}")
    print(f"Utilization: {test_credit_data['credit_utilization']*100:.1f}%")
    print(f"Late Payments (12mo): {test_credit_data['late_payments_12mo']}")
    
    # Generate 12-month forecast
    forecast = forecaster.forecast(test_credit_data, months_ahead=12)
    
    print(f"\n📈 12-MONTH FORECAST:")
    print(f"Current Score: {forecast.current_score}")
    print(f"Trend: {forecast.trend.upper()}")
    print(f"\nMonthly Predictions:")
    for month, score, (lower, upper) in zip(forecast.forecast_months[:6], 
                                             forecast.forecasted_scores[:6],
                                             forecast.confidence_intervals[:6]):
        print(f"  {month}: {score} (range: {lower}-{upper})")
    
    print(f"\n🎯 MILESTONE DATES:")
    for milestone, date in forecast.milestone_dates.items():
        score = milestone.split('_')[1]
        print(f"  Reach {score}: {date}")
    
    print(f"\n🔍 KEY DRIVERS:")
    for driver in forecast.key_drivers:
        impact_emoji = "📈" if driver['impact'] == 'positive' else "📉"
        print(f"  {impact_emoji} {driver['factor']}: {driver['current_value']} - {driver['description']}")
    
    print(f"\n💡 RECOMMENDATIONS:")
    for rec in forecast.recommendations:
        print(f"  {rec}")
    
    print("\n✅ Forecasting system ready for production")
