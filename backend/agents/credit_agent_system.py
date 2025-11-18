"""
🔥 ELITE MULTI-AGENT AI SYSTEM
AutoGen + LangGraph orchestration for credit intelligence
Author: Rick Jefferson Solutions
"""

import os
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass, asdict
import asyncio

# AutoGen imports
try:
    import autogen
    from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager
    AUTOGEN_AVAILABLE = True
except ImportError:
    AUTOGEN_AVAILABLE = False
    print("⚠️  AutoGen not available, using fallback agent system")

# LangGraph imports
try:
    from langgraph.graph import StateGraph, END
    from langchain.schema import HumanMessage, SystemMessage
    from langchain_openai import ChatOpenAI
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    print("⚠️  LangGraph not available, using fallback agent system")

from ml.credit_scorer import CreditScorer
from ml.fraud_detector import FraudDetector
from ml.forecaster import CreditScoreForecaster
from services.openrouter_service import OpenRouterService


@dataclass
class AgentExecutionState:
    """Track agent execution state"""
    execution_id: str
    user_id: str
    report_id: str
    status: str  # 'queued', 'running', 'completed', 'failed'
    current_agent: Optional[str]
    results: Dict[str, Any]
    errors: List[str]
    started_at: datetime
    completed_at: Optional[datetime]
    total_cost: float  # USD


class CreditAgentSystem:
    """
    Elite multi-agent system for credit intelligence
    Orchestrates: Credit Scorer, Fraud Detector, Forecaster, Dispute Generator
    """
    
    def __init__(self):
        # Initialize ML models
        self.credit_scorer = CreditScorer()
        self.fraud_detector = FraudDetector()
        self.forecaster = CreditScoreForecaster()
        self.openrouter = OpenRouterService()
        
        # Agent configuration
        self.agent_config = {
            "openai_api_key": os.getenv("OPENROUTER_API_KEY"),
            "model": "google/gemini-2.0-flash-exp:free",  # Free tier
            "temperature": 0.7,
            "max_tokens": 2000
        }
        
        self.execution_states: Dict[str, AgentExecutionState] = {}
    
    async def execute_full_analysis(
        self,
        user_id: str,
        report_id: str,
        credit_data: Dict[str, Any]
    ) -> AgentExecutionState:
        """
        Execute complete multi-agent credit analysis
        Runs all agents in parallel where possible
        """
        # Create execution state
        execution_id = f"exec-{report_id}-{int(datetime.utcnow().timestamp())}"
        state = AgentExecutionState(
            execution_id=execution_id,
            user_id=user_id,
            report_id=report_id,
            status='running',
            current_agent='orchestrator',
            results={},
            errors=[],
            started_at=datetime.utcnow(),
            completed_at=None,
            total_cost=0.0
        )
        
        self.execution_states[execution_id] = state
        
        try:
            # Phase 1: Parallel execution of ML models
            print(f"🚀 Starting multi-agent analysis for {report_id}")
            
            tasks = [
                self._run_credit_scoring_agent(credit_data, state),
                self._run_fraud_detection_agent(credit_data, state),
                self._run_forecasting_agent(credit_data, state)
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Collect results
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    state.errors.append(f"Agent {i} failed: {str(result)}")
                else:
                    state.results.update(result)
            
            # Phase 2: Generate comprehensive insights using OpenRouter
            state.current_agent = 'insight_generator'
            insights = await self._generate_comprehensive_insights(state.results, credit_data)
            state.results['comprehensive_insights'] = insights
            
            # Phase 3: Generate dispute letters if needed
            if state.results.get('fraud_check', {}).get('risk_level') in ['high', 'critical']:
                state.current_agent = 'dispute_generator'
                disputes = await self._generate_dispute_letters(credit_data, state.results)
                state.results['dispute_letters'] = disputes
            
            # Phase 4: Generate action plan
            state.current_agent = 'action_planner'
            action_plan = await self._generate_action_plan(state.results, credit_data)
            state.results['action_plan'] = action_plan
            
            # Mark as completed
            state.status = 'completed'
            state.completed_at = datetime.utcnow()
            state.current_agent = None
            
            print(f"✅ Multi-agent analysis completed for {report_id}")
            
        except Exception as e:
            state.status = 'failed'
            state.errors.append(f"Execution failed: {str(e)}")
            state.completed_at = datetime.utcnow()
            print(f"❌ Multi-agent analysis failed: {e}")
        
        return state
    
    async def _run_credit_scoring_agent(
        self,
        credit_data: Dict[str, Any],
        state: AgentExecutionState
    ) -> Dict[str, Any]:
        """Agent 1: Credit Scoring + SHAP Explainability"""
        try:
            print("  🤖 Credit Scoring Agent: Running ensemble ML models...")
            
            result = self.credit_scorer.predict_with_explanation(credit_data)
            
            return {
                'credit_score': {
                    'predicted_score': result['credit_score'],
                    'confidence': result['confidence'],
                    'risk_level': result['risk_level'],
                    'feature_importance': result['feature_importance'],
                    'recommendations': result['recommendations'],
                    'shap_analysis': result.get('shap_values', {}),
                    'timestamp': datetime.utcnow().isoformat()
                }
            }
        except Exception as e:
            state.errors.append(f"Credit scoring failed: {e}")
            return {}
    
    async def _run_fraud_detection_agent(
        self,
        credit_data: Dict[str, Any],
        state: AgentExecutionState
    ) -> Dict[str, Any]:
        """Agent 2: Fraud Detection with GNN"""
        try:
            print("  🤖 Fraud Detection Agent: Analyzing transaction graphs...")
            
            fraud_alert = self.fraud_detector.predict(credit_data)
            
            return {
                'fraud_check': {
                    'fraud_probability': fraud_alert.fraud_probability,
                    'risk_level': fraud_alert.risk_level,
                    'fraud_indicators': fraud_alert.fraud_indicators,
                    'graph_anomalies': fraud_alert.graph_anomalies,
                    'recommended_actions': fraud_alert.recommended_actions,
                    'confidence': fraud_alert.confidence_score,
                    'timestamp': datetime.utcnow().isoformat()
                }
            }
        except Exception as e:
            state.errors.append(f"Fraud detection failed: {e}")
            return {}
    
    async def _run_forecasting_agent(
        self,
        credit_data: Dict[str, Any],
        state: AgentExecutionState
    ) -> Dict[str, Any]:
        """Agent 3: Credit Score Forecasting with LSTM-Transformer"""
        try:
            print("  🤖 Forecasting Agent: Predicting 12-month trajectory...")
            
            forecast = self.forecaster.forecast(credit_data, months_ahead=12)
            
            return {
                'forecast': {
                    'current_score': forecast.current_score,
                    'forecasted_scores': forecast.forecasted_scores,
                    'forecast_months': forecast.forecast_months,
                    'confidence_intervals': [
                        {'lower': lower, 'upper': upper}
                        for lower, upper in forecast.confidence_intervals
                    ],
                    'trend': forecast.trend,
                    'key_drivers': forecast.key_drivers,
                    'milestone_dates': forecast.milestone_dates,
                    'recommendations': forecast.recommendations,
                    'timestamp': datetime.utcnow().isoformat()
                }
            }
        except Exception as e:
            state.errors.append(f"Forecasting failed: {e}")
            return {}
    
    async def _generate_comprehensive_insights(
        self,
        agent_results: Dict[str, Any],
        credit_data: Dict[str, Any]
    ) -> str:
        """Generate comprehensive insights combining all agent outputs"""
        try:
            print("  🤖 Insight Generator: Synthesizing all agent results...")
            
            # Build comprehensive prompt
            prompt = f"""Analyze this complete credit intelligence report and provide executive summary:

CREDIT SCORE ANALYSIS:
- Predicted Score: {agent_results.get('credit_score', {}).get('predicted_score', 'N/A')}
- Risk Level: {agent_results.get('credit_score', {}).get('risk_level', 'N/A')}
- Top Factors: {agent_results.get('credit_score', {}).get('recommendations', [])}

FRAUD RISK ASSESSMENT:
- Fraud Probability: {agent_results.get('fraud_check', {}).get('fraud_probability', 0):.1%}
- Risk Level: {agent_results.get('fraud_check', {}).get('risk_level', 'N/A')}
- Indicators: {len(agent_results.get('fraud_check', {}).get('fraud_indicators', []))}

12-MONTH FORECAST:
- Current: {agent_results.get('forecast', {}).get('current_score', 'N/A')}
- Predicted (12mo): {agent_results.get('forecast', {}).get('forecasted_scores', [0])[-1] if agent_results.get('forecast', {}).get('forecasted_scores') else 'N/A'}
- Trend: {agent_results.get('forecast', {}).get('trend', 'N/A')}

Provide:
1. Executive Summary (3 sentences)
2. Top 3 Priority Actions
3. Expected Outcome Timeline
4. Risk Mitigation Strategy

Be direct and actionable."""

            insights = await self.openrouter.generate_credit_insights(
                {'summary': prompt},
                'comprehensive'
            )
            
            return insights
            
        except Exception as e:
            return f"Insight generation failed: {e}"
    
    async def _generate_dispute_letters(
        self,
        credit_data: Dict[str, Any],
        agent_results: Dict[str, Any]
    ) -> List[Dict[str, str]]:
        """Generate FCRA-compliant dispute letters for fraud indicators"""
        try:
            print("  🤖 Dispute Generator: Creating FCRA-compliant letters...")
            
            letters = []
            fraud_indicators = agent_results.get('fraud_check', {}).get('fraud_indicators', [])
            
            # Generate letter for each high-severity indicator
            for indicator in fraud_indicators[:3]:  # Top 3
                if indicator.get('severity') in ['high', 'critical']:
                    dispute_details = {
                        'creditor_name': 'Credit Bureau',
                        'account_number': '****XXXX',
                        'dispute_type': indicator.get('type', 'fraud'),
                        'description': indicator.get('description', ''),
                        'reason': 'Identity theft - unauthorized account activity'
                    }
                    
                    letter = await self.openrouter.generate_dispute_letter(dispute_details)
                    
                    letters.append({
                        'indicator_type': indicator.get('type'),
                        'severity': indicator.get('severity'),
                        'letter': letter,
                        'generated_at': datetime.utcnow().isoformat()
                    })
            
            return letters
            
        except Exception as e:
            return [{'error': f"Dispute generation failed: {e}"}]
    
    async def _generate_action_plan(
        self,
        agent_results: Dict[str, Any],
        credit_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate 90-day action plan based on all insights"""
        try:
            print("  🤖 Action Planner: Creating 90-day roadmap...")
            
            # Determine primary goal based on results
            current_score = agent_results.get('credit_score', {}).get('predicted_score', 650)
            fraud_risk = agent_results.get('fraud_check', {}).get('risk_level', 'low')
            
            if fraud_risk in ['high', 'critical']:
                goal = "fraud_mitigation"
            elif current_score < 640:
                goal = "score_recovery"
            elif current_score < 720:
                goal = "improve_score"
            else:
                goal = "maintain_excellent"
            
            # Generate detailed action plan
            advice = await self.openrouter.generate_financial_advice(credit_data, goal)
            
            # Structure the action plan
            action_plan = {
                'primary_goal': goal,
                'current_score': current_score,
                'target_score': current_score + 50 if current_score < 750 else 800,
                'timeline': '90 days',
                'detailed_plan': advice,
                'weekly_tasks': self._extract_weekly_tasks(agent_results),
                'monthly_milestones': self._extract_milestones(agent_results),
                'estimated_cost': 0.0,  # Free with current setup
                'generated_at': datetime.utcnow().isoformat()
            }
            
            return action_plan
            
        except Exception as e:
            return {'error': f"Action plan generation failed: {e}"}
    
    def _extract_weekly_tasks(self, agent_results: Dict[str, Any]) -> List[Dict[str, str]]:
        """Extract weekly tasks from recommendations"""
        tasks = []
        
        # From credit scorer recommendations
        credit_recs = agent_results.get('credit_score', {}).get('recommendations', [])
        for i, rec in enumerate(credit_recs[:4], 1):
            tasks.append({
                'week': i,
                'task': rec,
                'priority': 'high' if i <= 2 else 'medium',
                'category': 'credit_improvement'
            })
        
        # From fraud recommendations
        fraud_actions = agent_results.get('fraud_check', {}).get('recommended_actions', [])
        for action in fraud_actions[:2]:
            tasks.append({
                'week': 1,
                'task': action,
                'priority': 'urgent',
                'category': 'fraud_prevention'
            })
        
        return tasks[:8]  # Max 8 tasks
    
    def _extract_milestones(self, agent_results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract milestones from forecast"""
        forecast = agent_results.get('forecast', {})
        milestones = forecast.get('milestone_dates', {})
        
        return {
            'month_1': 'Complete all urgent fraud actions',
            'month_2': 'Reduce utilization to target levels',
            'month_3': 'See measurable score improvement',
            **milestones
        }
    
    def get_execution_status(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """Get current status of an execution"""
        state = self.execution_states.get(execution_id)
        if not state:
            return None
        
        return {
            'execution_id': state.execution_id,
            'status': state.status,
            'current_agent': state.current_agent,
            'progress': self._calculate_progress(state),
            'results_count': len(state.results),
            'errors_count': len(state.errors),
            'elapsed_time': (
                (state.completed_at or datetime.utcnow()) - state.started_at
            ).total_seconds(),
            'total_cost': state.total_cost
        }
    
    def _calculate_progress(self, state: AgentExecutionState) -> float:
        """Calculate execution progress percentage"""
        if state.status == 'completed':
            return 100.0
        elif state.status == 'failed':
            return 0.0
        
        # Estimate based on completed agents
        total_agents = 4  # credit_scorer, fraud_detector, forecaster, insight_generator
        completed = len(state.results)
        return min((completed / total_agents) * 100, 99.0)
    
    async def cleanup_old_executions(self, hours: int = 24):
        """Clean up execution states older than N hours"""
        cutoff = datetime.utcnow().timestamp() - (hours * 3600)
        
        to_remove = [
            eid for eid, state in self.execution_states.items()
            if state.started_at.timestamp() < cutoff
        ]
        
        for eid in to_remove:
            del self.execution_states[eid]
        
        print(f"🧹 Cleaned up {len(to_remove)} old execution states")


# Singleton instance
_agent_system: Optional[CreditAgentSystem] = None


def get_agent_system() -> CreditAgentSystem:
    """Get singleton agent system instance"""
    global _agent_system
    if _agent_system is None:
        _agent_system = CreditAgentSystem()
    return _agent_system


# Demo usage
if __name__ == "__main__":
    import asyncio
    
    async def demo():
        print("🔥 MULTI-AGENT SYSTEM - DEMO")
        print("=" * 60)
        
        # Initialize agent system
        agent_system = CreditAgentSystem()
        
        # Test credit data
        test_credit_data = {
            'credit_score': 680,
            'late_payments_12mo': 2,
            'credit_utilization': 0.45,
            'inquiries_6mo': 8,
            'new_accounts_12mo': 4,
            'collections_count': 1,
            'derogatory_marks': 2,
            'total_accounts': 10,
            'oldest_account_age_months': 72,
            'avg_account_age_months': 36,
            'total_balance': 22000,
            'total_credit_limit': 49000,
            'monthly_income': 6500,
            'address_changes_12mo': 1,
            'revolving_accounts': 7
        }
        
        # Execute full analysis
        print("\n🚀 Starting multi-agent analysis...")
        state = await agent_system.execute_full_analysis(
            user_id='demo_user',
            report_id='report-demo-001',
            credit_data=test_credit_data
        )
        
        print(f"\n📊 EXECUTION RESULTS:")
        print(f"Status: {state.status}")
        print(f"Execution ID: {state.execution_id}")
        print(f"Time Elapsed: {(state.completed_at - state.started_at).total_seconds():.2f}s")
        print(f"Total Cost: ${state.total_cost:.4f}")
        print(f"Errors: {len(state.errors)}")
        
        print(f"\n📈 RESULTS SUMMARY:")
        for key, value in state.results.items():
            if isinstance(value, dict):
                print(f"  {key}: {len(value)} fields")
            elif isinstance(value, list):
                print(f"  {key}: {len(value)} items")
            else:
                print(f"  {key}: {str(value)[:100]}...")
        
        if state.errors:
            print(f"\n⚠️  ERRORS:")
            for error in state.errors:
                print(f"  - {error}")
        
        print("\n✅ Multi-agent system demo complete")
    
    asyncio.run(demo())
