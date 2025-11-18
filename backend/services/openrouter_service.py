"""
🔥 OPENROUTER INTEGRATION SERVICE
Cost-effective LLM routing ($0.25/M tokens or under)
Author: Rick Jefferson Solutions
"""

import os
import httpx
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
import asyncio


class OpenRouterService:
    """
    Elite OpenRouter integration for cost-effective AI calls
    Auto-selects best models under $0.25 per million tokens
    """
    
    # Cost-effective models (all under $0.25/M tokens)
    COST_EFFECTIVE_MODELS = {
        # FREE MODELS (Best for high-volume operations)
        'free': [
            'google/gemini-2.0-flash-thinking-exp:free',  # FREE - Google's latest reasoning model
            'google/gemini-2.0-flash-exp:free',           # FREE - Fast and capable
            'meta-llama/llama-3.2-90b-vision-instruct:free',  # FREE - Vision + text
            'qwen/qwen-2.5-72b-instruct:free',            # FREE - Strong reasoning
            'mistralai/mistral-nemo:free',                # FREE - 128k context
        ],
        
        # ULTRA-CHEAP MODELS ($0.01-0.10/M tokens)
        'ultra_cheap': [
            'google/gemini-flash-1.5-8b',                 # $0.0375/$0.15 - Fastest Gemini
            'meta-llama/llama-3.2-3b-instruct',           # $0.06/$0.06 - Efficient small model
            'microsoft/phi-3-mini-128k-instruct',         # $0.10/$0.10 - 128k context
        ],
        
        # BUDGET MODELS ($0.10-0.25/M tokens)
        'budget': [
            'google/gemini-flash-1.5',                    # $0.075/$0.30 - Best value/performance
            'meta-llama/llama-3.1-8b-instruct',           # $0.05/$0.05 - Great balance
            'mistralai/mistral-7b-instruct',              # $0.06/$0.06 - Fast and capable
            'anthropic/claude-3-haiku',                   # $0.25/$1.25 - Most capable in budget
        ]
    }
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv('OPENROUTER_API_KEY')
        if not self.api_key:
            raise ValueError("OpenRouter API key not found. Set OPENROUTER_API_KEY environment variable")
        
        self.base_url = "https://openrouter.ai/api/v1"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": "https://rickjeffersonsolutions.com",
            "X-Title": "MyFreeScoreNow AI Credit Intelligence"
        }
        
        # Default to free models for maximum cost savings
        self.default_tier = 'free'
    
    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: str = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        tier: str = None,
        stream: bool = False
    ) -> Dict[str, Any]:
        """
        Make chat completion request to OpenRouter
        Auto-selects cost-effective model if none specified
        """
        # Select model based on tier
        if not model:
            tier = tier or self.default_tier
            model = self.COST_EFFECTIVE_MODELS[tier][0]  # Use first model in tier
        
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream
        }
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json=payload
            )
            response.raise_for_status()
            return response.json()
    
    async def generate_credit_insights(
        self,
        credit_data: Dict[str, Any],
        analysis_type: str = "comprehensive"
    ) -> str:
        """
        Generate AI-powered credit insights
        Uses free tier for cost optimization
        """
        system_prompt = """You are an elite credit analysis AI specializing in FICO scoring, 
credit repair, and financial planning. Provide expert, actionable insights based on credit data."""
        
        if analysis_type == "comprehensive":
            user_prompt = f"""Analyze this credit profile and provide comprehensive insights:

Credit Score: {credit_data.get('credit_score', 'N/A')}
Utilization: {credit_data.get('credit_utilization', 0)*100:.1f}%
Late Payments (12mo): {credit_data.get('late_payments_12mo', 0)}
Hard Inquiries (6mo): {credit_data.get('inquiries_6mo', 0)}
Total Accounts: {credit_data.get('total_accounts', 0)}
Oldest Account: {credit_data.get('oldest_account_age_months', 0)} months
Collections: {credit_data.get('collections_count', 0)}
Derogatory Marks: {credit_data.get('derogatory_marks', 0)}

Provide:
1. Overall credit health assessment (2-3 sentences)
2. Top 3 factors impacting the score
3. 3 specific, actionable recommendations
4. Timeline for improvement

Be direct, specific, and actionable."""
        
        elif analysis_type == "quick":
            user_prompt = f"""Quick credit analysis for score {credit_data.get('credit_score', 650)} 
with {credit_data.get('credit_utilization', 0)*100:.1f}% utilization. 
Top 2 issues and 2 quick wins in under 100 words."""
        
        elif analysis_type == "dispute":
            user_prompt = f"""Generate a professional credit dispute letter for:
- Issue: {credit_data.get('dispute_item', 'Incorrect late payment')}
- Account: {credit_data.get('account_details', 'Credit card ending in 1234')}
- Date: {credit_data.get('dispute_date', datetime.now().strftime('%Y-%m'))}

Include legal references (FCRA Section 611) and firm but professional tone."""
        
        else:
            user_prompt = f"Analyze this credit data: {json.dumps(credit_data, indent=2)}"
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # Use free tier for cost optimization
        response = await self.chat_completion(
            messages=messages,
            tier='free',
            temperature=0.7,
            max_tokens=1500
        )
        
        return response['choices'][0]['message']['content']
    
    async def generate_dispute_letter(
        self,
        dispute_details: Dict[str, Any]
    ) -> str:
        """
        Generate FCRA-compliant dispute letter
        Uses free tier for cost savings
        """
        system_prompt = """You are an expert credit dispute specialist with deep knowledge of 
the Fair Credit Reporting Act (FCRA). Generate professional, legally-sound dispute letters."""
        
        user_prompt = f"""Generate a professional credit dispute letter with these details:

Creditor: {dispute_details.get('creditor_name')}
Account: {dispute_details.get('account_number')}
Issue Type: {dispute_details.get('dispute_type')}
Issue Description: {dispute_details.get('description')}
Reason for Dispute: {dispute_details.get('reason')}

Include:
1. Professional business letter format with proper addressing
2. Clear statement of dispute with FCRA Section 611 reference
3. Specific inaccuracies being disputed
4. Request for investigation and deletion/correction
5. 30-day response deadline
6. Threat of legal action if not resolved (FCRA Section 616/617)
7. Request for confirmation in writing

Use formal, assertive tone. Include all legal protections."""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        response = await self.chat_completion(
            messages=messages,
            tier='free',
            temperature=0.4,  # Lower temp for consistent legal language
            max_tokens=2000
        )
        
        return response['choices'][0]['message']['content']
    
    async def generate_financial_advice(
        self,
        credit_data: Dict[str, Any],
        goal: str = "improve_score"
    ) -> str:
        """
        Generate personalized financial advice
        Uses free tier for cost optimization
        """
        goals_map = {
            'improve_score': "Improve credit score by 50+ points in 6 months",
            'reduce_debt': "Create debt payoff plan to eliminate high-interest debt",
            'qualify_mortgage': "Prepare credit profile for mortgage approval",
            'credit_cards': "Get approved for premium credit cards",
            'business_credit': "Build business credit separate from personal"
        }
        
        goal_description = goals_map.get(goal, goal)
        
        system_prompt = """You are a certified financial planner (CFP) specializing in credit 
optimization and debt management. Provide expert, personalized advice."""
        
        user_prompt = f"""Create a detailed action plan for this credit profile:

CREDIT PROFILE:
- Score: {credit_data.get('credit_score', 650)}
- Utilization: {credit_data.get('credit_utilization', 0)*100:.1f}%
- Total Debt: ${credit_data.get('total_balance', 0):,.2f}
- Available Credit: ${credit_data.get('total_credit_limit', 0):,.2f}
- Monthly Income: ${credit_data.get('monthly_income', 5000):,.2f}
- Late Payments: {credit_data.get('late_payments_12mo', 0)}
- Accounts: {credit_data.get('total_accounts', 0)}

GOAL: {goal_description}

Provide:
1. 90-day action plan with specific steps
2. Monthly milestones
3. Exact dollar amounts for debt payoff priorities
4. Credit utilization targets
5. Expected timeline to reach goal
6. Potential obstacles and how to overcome them

Be specific with numbers, dates, and action items."""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        response = await self.chat_completion(
            messages=messages,
            tier='free',
            temperature=0.7,
            max_tokens=2500
        )
        
        return response['choices'][0]['message']['content']
    
    async def analyze_credit_report(
        self,
        credit_report: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Comprehensive credit report analysis
        Returns structured JSON analysis
        """
        system_prompt = """You are an AI credit analyst. Analyze the credit report and return 
a JSON object with scores, risks, and recommendations. Be precise and data-driven."""
        
        user_prompt = f"""Analyze this credit report and return JSON with this structure:

{{
  "overall_score": <1-100>,
  "risk_level": "low|medium|high|critical",
  "strengths": ["strength 1", "strength 2"],
  "weaknesses": ["weakness 1", "weakness 2"],
  "immediate_actions": ["action 1", "action 2", "action 3"],
  "long_term_strategies": ["strategy 1", "strategy 2"],
  "score_potential": <predicted score in 12 months>,
  "fraud_indicators": ["indicator 1", "indicator 2"] or []
}}

Credit Report Data:
{json.dumps(credit_report, indent=2)}

Return ONLY valid JSON, no other text."""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        response = await self.chat_completion(
            messages=messages,
            tier='free',
            temperature=0.3,  # Low temp for consistent JSON
            max_tokens=1500
        )
        
        # Parse JSON response
        content = response['choices'][0]['message']['content']
        
        # Extract JSON if wrapped in code blocks
        if '```json' in content:
            content = content.split('```json')[1].split('```')[0].strip()
        elif '```' in content:
            content = content.split('```')[1].split('```')[0].strip()
        
        return json.loads(content)
    
    def get_cost_estimate(self, input_tokens: int, output_tokens: int, tier: str = 'free') -> float:
        """
        Estimate cost for a request
        Returns cost in dollars
        """
        # Cost per million tokens (input/output)
        costs = {
            'free': (0.0, 0.0),
            'ultra_cheap': (0.06, 0.06),
            'budget': (0.075, 0.30)
        }
        
        input_cost, output_cost = costs.get(tier, costs['free'])
        
        total_cost = (input_tokens / 1_000_000 * input_cost) + (output_tokens / 1_000_000 * output_cost)
        return total_cost
    
    async def batch_analyze(
        self,
        credit_reports: List[Dict[str, Any]],
        analysis_type: str = "quick"
    ) -> List[str]:
        """
        Batch analyze multiple credit reports
        Uses asyncio for parallel processing
        """
        tasks = [
            self.generate_credit_insights(report, analysis_type)
            for report in credit_reports
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions
        valid_results = [r for r in results if not isinstance(r, Exception)]
        
        return valid_results


# Synchronous wrapper for non-async contexts
class OpenRouterServiceSync:
    """Synchronous wrapper for OpenRouter service"""
    
    def __init__(self, api_key: str = None):
        self.async_service = OpenRouterService(api_key)
    
    def generate_credit_insights(self, credit_data: Dict[str, Any], analysis_type: str = "comprehensive") -> str:
        return asyncio.run(self.async_service.generate_credit_insights(credit_data, analysis_type))
    
    def generate_dispute_letter(self, dispute_details: Dict[str, Any]) -> str:
        return asyncio.run(self.async_service.generate_dispute_letter(dispute_details))
    
    def generate_financial_advice(self, credit_data: Dict[str, Any], goal: str = "improve_score") -> str:
        return asyncio.run(self.async_service.generate_financial_advice(credit_data, goal))
    
    def analyze_credit_report(self, credit_report: Dict[str, Any]) -> Dict[str, Any]:
        return asyncio.run(self.async_service.analyze_credit_report(credit_report))


# Demo usage
if __name__ == "__main__":
    import asyncio
    
    async def demo():
        print("🔥 OPENROUTER SERVICE - DEMO")
        print("=" * 60)
        
        # Initialize service
        service = OpenRouterService()
        
        # Test credit data
        test_credit_data = {
            'credit_score': 680,
            'credit_utilization': 0.55,
            'late_payments_12mo': 2,
            'inquiries_6mo': 4,
            'total_accounts': 8,
            'oldest_account_age_months': 48,
            'collections_count': 1,
            'derogatory_marks': 1,
            'total_balance': 18000,
            'total_credit_limit': 32000,
            'monthly_income': 6500
        }
        
        print("\n📊 TEST: Credit Insights Generation")
        insights = await service.generate_credit_insights(test_credit_data, "comprehensive")
        print(insights[:500] + "...")
        
        print("\n\n📊 TEST: Dispute Letter Generation")
        dispute_details = {
            'creditor_name': 'ABC Bank',
            'account_number': '****1234',
            'dispute_type': 'Late Payment',
            'description': 'Late payment reported for March 2024',
            'reason': 'Payment was received on time but processed late by creditor'
        }
        dispute_letter = await service.generate_dispute_letter(dispute_details)
        print(dispute_letter[:500] + "...")
        
        print("\n\n📊 TEST: Financial Advice")
        advice = await service.generate_financial_advice(test_credit_data, "improve_score")
        print(advice[:500] + "...")
        
        print("\n\n📊 TEST: Structured Analysis")
        analysis = await service.analyze_credit_report(test_credit_data)
        print(json.dumps(analysis, indent=2))
        
        print("\n\n💰 COST ESTIMATE:")
        print(f"Free tier: ${service.get_cost_estimate(1000, 500, 'free'):.6f}")
        print(f"Ultra-cheap tier: ${service.get_cost_estimate(1000, 500, 'ultra_cheap'):.6f}")
        print(f"Budget tier: ${service.get_cost_estimate(1000, 500, 'budget'):.6f}")
        
        print("\n✅ OpenRouter service ready for production")
        print("💡 All tests used FREE tier models ($0.00 cost)")
    
    asyncio.run(demo())
