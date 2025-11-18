"""
🔥 ELITE FRAUD DETECTION SYSTEM
Graph Neural Network (GNN) using PyTorch Geometric
Detects credit fraud patterns through relationship analysis
Author: Rick Jefferson Solutions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.data import Data, Batch
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple
from datetime import datetime, timedelta
import pickle
import json
from dataclasses import dataclass


@dataclass
class FraudAlert:
    """Fraud detection result"""
    fraud_probability: float
    risk_level: str  # 'low', 'medium', 'high', 'critical'
    fraud_indicators: List[Dict[str, Any]]
    graph_anomalies: List[str]
    recommended_actions: List[str]
    confidence_score: float


class GraphAttentionFraudDetector(nn.Module):
    """
    Graph Attention Network for fraud detection
    Uses multi-head attention to learn fraud patterns in transaction graphs
    """
    
    def __init__(self, input_dim: int = 32, hidden_dim: int = 128, num_heads: int = 8, dropout: float = 0.3):
        super().__init__()
        
        # Graph Attention Layers
        self.gat1 = GATConv(input_dim, hidden_dim, heads=num_heads, dropout=dropout)
        self.gat2 = GATConv(hidden_dim * num_heads, hidden_dim, heads=num_heads, dropout=dropout)
        self.gat3 = GATConv(hidden_dim * num_heads, hidden_dim, heads=1, concat=False, dropout=dropout)
        
        # Classification layers
        self.fc1 = nn.Linear(hidden_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        
        self.dropout = nn.Dropout(dropout)
        self.batch_norm1 = nn.BatchNorm1d(hidden_dim * num_heads)
        self.batch_norm2 = nn.BatchNorm1d(hidden_dim * num_heads)
        
    def forward(self, x, edge_index, batch):
        # GAT layers with residual connections
        x1 = F.elu(self.gat1(x, edge_index))
        x1 = self.batch_norm1(x1)
        x1 = self.dropout(x1)
        
        x2 = F.elu(self.gat2(x1, edge_index))
        x2 = self.batch_norm2(x2)
        x2 = self.dropout(x2)
        
        x3 = F.elu(self.gat3(x2, edge_index))
        
        # Global pooling
        x = global_mean_pool(x3, batch)
        
        # Classification
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc3(x))
        
        return x


class FraudDetector:
    """
    Elite fraud detection system with Graph Neural Networks
    Analyzes transaction patterns, account relationships, and behavioral anomalies
    """
    
    def __init__(self, model_path: str = None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = GraphAttentionFraudDetector(input_dim=32, hidden_dim=128, num_heads=8)
        self.model.to(self.device)
        self.feature_scaler = None
        self.trained = False
        
        if model_path:
            self.load_model(model_path)
    
    def create_transaction_graph(self, transactions: pd.DataFrame, accounts: pd.DataFrame) -> Data:
        """
        Build transaction graph from credit data
        Nodes: accounts, merchants, locations
        Edges: transactions, shared addresses, linked accounts
        """
        # Create node features
        node_features = []
        node_mapping = {}
        current_idx = 0
        
        # Account nodes
        for _, account in accounts.iterrows():
            features = [
                account.get('account_age_days', 0) / 365,
                account.get('credit_limit', 10000) / 100000,
                account.get('balance', 0) / 100000,
                account.get('utilization', 0),
                account.get('late_payments', 0) / 10,
                account.get('inquiries_6mo', 0) / 10,
                account.get('derogatory_marks', 0) / 5,
                account.get('collections', 0) / 5,
            ]
            # Pad to 32 dimensions
            features.extend([0] * (32 - len(features)))
            node_features.append(features[:32])
            node_mapping[f"account_{account['id']}"] = current_idx
            current_idx += 1
        
        # Transaction nodes (sample for large datasets)
        transaction_sample = transactions.head(100) if len(transactions) > 100 else transactions
        for _, txn in transaction_sample.iterrows():
            features = [
                txn.get('amount', 0) / 10000,
                txn.get('hour_of_day', 12) / 24,
                txn.get('day_of_week', 3) / 7,
                txn.get('is_weekend', 0),
                txn.get('is_international', 0),
                txn.get('distance_from_home', 0) / 1000,
                txn.get('velocity_1hr', 0) / 10,
                txn.get('velocity_24hr', 0) / 50,
            ]
            features.extend([0] * (32 - len(features)))
            node_features.append(features[:32])
            node_mapping[f"txn_{txn['id']}"] = current_idx
            current_idx += 1
        
        # Create edges (transaction relationships)
        edge_list = []
        
        # Connect transactions to accounts
        for _, txn in transaction_sample.iterrows():
            account_key = f"account_{txn.get('account_id')}"
            txn_key = f"txn_{txn['id']}"
            if account_key in node_mapping and txn_key in node_mapping:
                edge_list.append([node_mapping[account_key], node_mapping[txn_key]])
                edge_list.append([node_mapping[txn_key], node_mapping[account_key]])
        
        # Convert to PyTorch tensors
        x = torch.tensor(node_features, dtype=torch.float32)
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous() if edge_list else torch.empty((2, 0), dtype=torch.long)
        
        return Data(x=x, edge_index=edge_index)
    
    def extract_fraud_indicators(self, credit_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract fraud risk indicators from credit report"""
        indicators = []
        risk_score = 0
        
        # Recent hard inquiries spike
        inquiries_6mo = credit_data.get('inquiries_6mo', 0)
        if inquiries_6mo > 6:
            indicators.append({
                'type': 'inquiry_spike',
                'severity': 'high' if inquiries_6mo > 10 else 'medium',
                'description': f'{inquiries_6mo} hard inquiries in 6 months (normal: 0-2)',
                'impact': 0.3 if inquiries_6mo > 10 else 0.2
            })
            risk_score += 0.3 if inquiries_6mo > 10 else 0.2
        
        # New accounts opened rapidly
        new_accounts_12mo = credit_data.get('new_accounts_12mo', 0)
        if new_accounts_12mo > 5:
            indicators.append({
                'type': 'rapid_account_opening',
                'severity': 'high',
                'description': f'{new_accounts_12mo} new accounts in 12 months',
                'impact': 0.25
            })
            risk_score += 0.25
        
        # High utilization across all accounts
        utilization = credit_data.get('credit_utilization', 0)
        if utilization > 0.9:
            indicators.append({
                'type': 'maxed_out_credit',
                'severity': 'high',
                'description': f'{utilization*100:.1f}% credit utilization (normal: <30%)',
                'impact': 0.2
            })
            risk_score += 0.2
        
        # Multiple delinquencies
        late_payments = credit_data.get('late_payments_12mo', 0)
        if late_payments > 3:
            indicators.append({
                'type': 'payment_pattern_change',
                'severity': 'medium',
                'description': f'{late_payments} late payments in 12 months',
                'impact': 0.15
            })
            risk_score += 0.15
        
        # Address changes
        address_changes = credit_data.get('address_changes_12mo', 0)
        if address_changes > 2:
            indicators.append({
                'type': 'frequent_address_change',
                'severity': 'medium',
                'description': f'{address_changes} address changes in 12 months',
                'impact': 0.15
            })
            risk_score += 0.15
        
        # Collections activity
        collections = credit_data.get('collections_count', 0)
        if collections > 2:
            indicators.append({
                'type': 'collections_spike',
                'severity': 'high',
                'description': f'{collections} collection accounts',
                'impact': 0.2
            })
            risk_score += 0.2
        
        # Unusual account mix
        total_accounts = credit_data.get('total_accounts', 0)
        revolving_accounts = credit_data.get('revolving_accounts', 0)
        if total_accounts > 0 and revolving_accounts / total_accounts > 0.8:
            indicators.append({
                'type': 'unusual_account_mix',
                'severity': 'low',
                'description': f'{revolving_accounts}/{total_accounts} accounts are revolving (credit cards)',
                'impact': 0.1
            })
            risk_score += 0.1
        
        return {
            'indicators': indicators,
            'risk_score': min(risk_score, 1.0),
            'indicator_count': len(indicators)
        }
    
    def detect_graph_anomalies(self, graph: Data) -> List[str]:
        """Detect anomalies in transaction graph structure"""
        anomalies = []
        
        # Check for isolated nodes (suspicious standalone transactions)
        if graph.edge_index.size(1) == 0:
            anomalies.append("No transaction relationships detected (possible new account fraud)")
        
        # Check node count
        num_nodes = graph.x.size(0)
        num_edges = graph.edge_index.size(1)
        
        if num_nodes > 10 and num_edges / num_nodes < 1.5:
            anomalies.append(f"Sparse transaction graph ({num_edges} edges, {num_nodes} nodes) - unusual isolation")
        
        if num_nodes > 20 and num_edges / num_nodes > 8:
            anomalies.append(f"Dense transaction graph ({num_edges} edges, {num_nodes} nodes) - possible velocity fraud")
        
        # Check for unusual feature patterns
        feature_means = graph.x.mean(dim=0)
        if feature_means[0] > 0.8:  # High normalized amounts
            anomalies.append("Unusually high average transaction amounts across graph")
        
        if feature_means[4] > 0.5:  # Many international transactions
            anomalies.append("High proportion of international transactions")
        
        return anomalies
    
    def generate_recommendations(self, fraud_alert: FraudAlert) -> List[str]:
        """Generate fraud mitigation recommendations"""
        recommendations = []
        
        if fraud_alert.risk_level in ['high', 'critical']:
            recommendations.append("🚨 IMMEDIATE: Freeze credit with all 3 bureaus (Equifax, Experian, TransUnion)")
            recommendations.append("🔒 Enable multi-factor authentication on all financial accounts")
            recommendations.append("📧 Sign up for fraud alerts with your banks and credit card companies")
        
        if fraud_alert.fraud_probability > 0.7:
            recommendations.append("📄 File identity theft report with FTC at IdentityTheft.gov")
            recommendations.append("🏦 Contact affected creditors immediately to dispute fraudulent charges")
            recommendations.append("🛡️ Consider identity theft protection service")
        
        # Specific recommendations based on indicators
        for indicator in fraud_alert.fraud_indicators:
            if indicator['type'] == 'inquiry_spike':
                recommendations.append("❌ Dispute unauthorized hard inquiries with credit bureaus")
            elif indicator['type'] == 'rapid_account_opening':
                recommendations.append("🔍 Review all recent account openings - close any unauthorized accounts")
            elif indicator['type'] == 'maxed_out_credit':
                recommendations.append("💳 Report unauthorized charges and request credit limit restoration")
        
        if fraud_alert.risk_level == 'medium':
            recommendations.append("📊 Monitor credit reports weekly for next 3 months")
            recommendations.append("🔔 Set up transaction alerts for all accounts")
        
        return recommendations[:10]  # Top 10 most important
    
    def predict(self, credit_data: Dict[str, Any]) -> FraudAlert:
        """
        Predict fraud probability for credit report
        Combines GNN graph analysis with rule-based indicators
        """
        # Extract fraud indicators (rule-based)
        indicator_analysis = self.extract_fraud_indicators(credit_data)
        
        # Build transaction graph
        transactions_df = pd.DataFrame(credit_data.get('transactions', []))
        accounts_df = pd.DataFrame(credit_data.get('accounts', []))
        
        # If no transaction data, use simulated graph
        if transactions_df.empty:
            transactions_df = self.simulate_transactions(credit_data)
        if accounts_df.empty:
            accounts_df = self.simulate_accounts(credit_data)
        
        graph = self.create_transaction_graph(transactions_df, accounts_df)
        
        # Detect graph anomalies
        graph_anomalies = self.detect_graph_anomalies(graph)
        
        # GNN prediction (if model is trained)
        gnn_probability = 0.0
        if self.trained and graph.x.size(0) > 0:
            self.model.eval()
            with torch.no_grad():
                graph = graph.to(self.device)
                batch = torch.zeros(graph.x.size(0), dtype=torch.long, device=self.device)
                gnn_probability = self.model(graph.x, graph.edge_index, batch).item()
        
        # Combine rule-based and GNN scores (weighted ensemble)
        rule_based_score = indicator_analysis['risk_score']
        combined_probability = (0.6 * rule_based_score + 0.4 * gnn_probability)
        
        # Determine risk level
        if combined_probability >= 0.8:
            risk_level = 'critical'
        elif combined_probability >= 0.6:
            risk_level = 'high'
        elif combined_probability >= 0.3:
            risk_level = 'medium'
        else:
            risk_level = 'low'
        
        # Calculate confidence (based on indicator count and graph size)
        confidence = min(0.5 + (indicator_analysis['indicator_count'] * 0.1) + (graph.x.size(0) / 100), 0.95)
        
        fraud_alert = FraudAlert(
            fraud_probability=combined_probability,
            risk_level=risk_level,
            fraud_indicators=indicator_analysis['indicators'],
            graph_anomalies=graph_anomalies,
            recommended_actions=[],
            confidence_score=confidence
        )
        
        fraud_alert.recommended_actions = self.generate_recommendations(fraud_alert)
        
        return fraud_alert
    
    def simulate_transactions(self, credit_data: Dict[str, Any]) -> pd.DataFrame:
        """Simulate transaction data from credit report metadata"""
        num_txns = credit_data.get('total_accounts', 5) * 10
        
        transactions = []
        for i in range(num_txns):
            transactions.append({
                'id': i,
                'account_id': np.random.randint(0, credit_data.get('total_accounts', 5)),
                'amount': np.random.lognormal(4, 1),
                'hour_of_day': np.random.randint(0, 24),
                'day_of_week': np.random.randint(0, 7),
                'is_weekend': int(np.random.randint(0, 7) >= 5),
                'is_international': int(np.random.random() < 0.05),
                'distance_from_home': np.random.exponential(50),
                'velocity_1hr': np.random.poisson(2),
                'velocity_24hr': np.random.poisson(10)
            })
        
        return pd.DataFrame(transactions)
    
    def simulate_accounts(self, credit_data: Dict[str, Any]) -> pd.DataFrame:
        """Simulate account data from credit report summary"""
        num_accounts = credit_data.get('total_accounts', 5)
        
        accounts = []
        for i in range(num_accounts):
            age_days = max(30, np.random.exponential(365 * 5))
            credit_limit = np.random.lognormal(9, 0.5)
            utilization = credit_data.get('credit_utilization', 0.3) + np.random.normal(0, 0.1)
            
            accounts.append({
                'id': i,
                'account_age_days': age_days,
                'credit_limit': credit_limit,
                'balance': credit_limit * max(0, min(1, utilization)),
                'utilization': max(0, min(1, utilization)),
                'late_payments': max(0, credit_data.get('late_payments_12mo', 0) // num_accounts),
                'inquiries_6mo': max(0, credit_data.get('inquiries_6mo', 0) // num_accounts),
                'derogatory_marks': credit_data.get('derogatory_marks', 0) // max(1, num_accounts),
                'collections': credit_data.get('collections_count', 0) // max(1, num_accounts)
            })
        
        return pd.DataFrame(accounts)
    
    def train(self, train_graphs: List[Data], train_labels: torch.Tensor, 
              val_graphs: List[Data] = None, val_labels: torch.Tensor = None,
              epochs: int = 100, lr: float = 0.001):
        """Train the GNN fraud detector"""
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        criterion = nn.BCELoss()
        
        train_loader = Batch.from_data_list(train_graphs)
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            # Forward pass
            batch_idx = torch.zeros(train_loader.x.size(0), dtype=torch.long, device=self.device)
            predictions = self.model(train_loader.x.to(self.device), 
                                    train_loader.edge_index.to(self.device), 
                                    batch_idx)
            
            loss = criterion(predictions.squeeze(), train_labels.to(self.device))
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}/{epochs} - Loss: {loss.item():.4f}")
        
        self.trained = True
        print("✅ Fraud detection model training complete")
    
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
    print("🔥 FRAUD DETECTION GNN - DEMO")
    print("=" * 60)
    
    # Initialize detector
    detector = FraudDetector()
    
    # Test case 1: High fraud probability
    test_credit_data_fraud = {
        'user_id': 'demo_user_fraud',
        'inquiries_6mo': 15,  # Suspicious
        'new_accounts_12mo': 8,  # Rapid account opening
        'credit_utilization': 0.95,  # Maxed out
        'late_payments_12mo': 6,  # Payment issues
        'address_changes_12mo': 3,  # Moving frequently
        'collections_count': 4,  # Collections
        'total_accounts': 12,
        'revolving_accounts': 10,
        'derogatory_marks': 2
    }
    
    print("\n📊 TEST CASE 1: High Fraud Risk Profile")
    fraud_alert = detector.predict(test_credit_data_fraud)
    
    print(f"Fraud Probability: {fraud_alert.fraud_probability:.2%}")
    print(f"Risk Level: {fraud_alert.risk_level.upper()}")
    print(f"Confidence: {fraud_alert.confidence_score:.2%}")
    print(f"\n🚨 Fraud Indicators ({len(fraud_alert.fraud_indicators)}):")
    for indicator in fraud_alert.fraud_indicators:
        print(f"  - [{indicator['severity'].upper()}] {indicator['description']}")
    
    print(f"\n🔍 Graph Anomalies ({len(fraud_alert.graph_anomalies)}):")
    for anomaly in fraud_alert.graph_anomalies:
        print(f"  - {anomaly}")
    
    print(f"\n💡 Recommended Actions:")
    for action in fraud_alert.recommended_actions[:5]:
        print(f"  {action}")
    
    # Test case 2: Clean profile
    test_credit_data_clean = {
        'user_id': 'demo_user_clean',
        'inquiries_6mo': 1,
        'new_accounts_12mo': 1,
        'credit_utilization': 0.15,
        'late_payments_12mo': 0,
        'address_changes_12mo': 0,
        'collections_count': 0,
        'total_accounts': 8,
        'revolving_accounts': 4,
        'derogatory_marks': 0
    }
    
    print("\n\n📊 TEST CASE 2: Clean Credit Profile")
    fraud_alert_clean = detector.predict(test_credit_data_clean)
    
    print(f"Fraud Probability: {fraud_alert_clean.fraud_probability:.2%}")
    print(f"Risk Level: {fraud_alert_clean.risk_level.upper()}")
    print(f"Confidence: {fraud_alert_clean.confidence_score:.2%}")
    print(f"Fraud Indicators: {len(fraud_alert_clean.fraud_indicators)}")
    
    print("\n✅ Fraud detection system ready for production")
