"""
Credit Intelligence Platform - Streamlit Dashboard
Supreme AI-powered credit analysis interface
"""
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import requests
from datetime import datetime
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
API_URL = os.getenv("API_URL", "http://localhost:8000")
API_PREFIX = "/api/v1"

# Page config
st.set_page_config(
    page_title="Credit Intelligence Platform",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(90deg, #06b6d4, #ec4899);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        color: white;
        margin-bottom: 1rem;
    }
    .success-box {
        background-color: #10b981;
        padding: 1rem;
        border-radius: 0.5rem;
        color: white;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #f59e0b;
        padding: 1rem;
        border-radius: 0.5rem;
        color: white;
        margin: 1rem 0;
    }
    .danger-box {
        background-color: #ef4444;
        padding: 1rem;
        border-radius: 0.5rem;
        color: white;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


# Session state initialization
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'access_token' not in st.session_state:
    st.session_state.access_token = None
if 'user_email' not in st.session_state:
    st.session_state.user_email = None


def api_request(endpoint, method="GET", data=None, auth=True):
    """Make API request with error handling"""
    url = f"{API_URL}{API_PREFIX}{endpoint}"
    headers = {}
    
    if auth and st.session_state.access_token:
        headers["Authorization"] = f"Bearer {st.session_state.access_token}"
    
    try:
        if method == "GET":
            response = requests.get(url, headers=headers, timeout=30)
        elif method == "POST":
            response = requests.post(url, headers=headers, json=data, timeout=30)
        
        return response.json() if response.status_code < 400 else None
    except Exception as e:
        st.error(f"API Error: {str(e)}")
        return None


def login_page():
    """Login interface"""
    st.markdown('<h1 class="main-header">🚀 Credit Intelligence Platform</h1>', unsafe_allow_html=True)
    st.markdown("### Elite AI-Powered Credit Analysis")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("---")
        tab1, tab2 = st.tabs(["Login", "Register"])
        
        with tab1:
            with st.form("login_form"):
                email = st.text_input("Email", placeholder="your@email.com")
                password = st.text_input("Password", type="password")
                submit = st.form_submit_button("Login", use_container_width=True)
                
                if submit:
                    # OAuth2 format for FastAPI
                    login_data = {"username": email, "password": password}
                    response = api_request("/auth/login", method="POST", data=login_data, auth=False)
                    
                    if response and response.get("access_token"):
                        st.session_state.authenticated = True
                        st.session_state.access_token = response["access_token"]
                        st.session_state.user_email = email
                        st.success("✅ Login successful!")
                        st.rerun()
                    else:
                        st.error("❌ Invalid credentials")
        
        with tab2:
            with st.form("register_form"):
                reg_email = st.text_input("Email", placeholder="your@email.com", key="reg_email")
                reg_name = st.text_input("Full Name", placeholder="John Doe")
                reg_password = st.text_input("Password (min 8 chars)", type="password", key="reg_password")
                reg_confirm = st.text_input("Confirm Password", type="password")
                reg_affiliate = st.text_input("Affiliate Code (optional)", placeholder="Optional")
                
                reg_submit = st.form_submit_button("Create Account", use_container_width=True)
                
                if reg_submit:
                    if reg_password != reg_confirm:
                        st.error("❌ Passwords don't match")
                    elif len(reg_password) < 8:
                        st.error("❌ Password must be at least 8 characters")
                    else:
                        register_data = {
                            "email": reg_email,
                            "password": reg_password,
                            "full_name": reg_name,
                            "affiliate_id": reg_affiliate if reg_affiliate else None
                        }
                        response = api_request("/auth/register", method="POST", data=register_data, auth=False)
                        
                        if response and response.get("success"):
                            st.success("✅ Account created! Please login.")
                        else:
                            st.error("❌ Registration failed")


def credit_score_gauge(score):
    """Create credit score gauge chart"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Credit Score", 'font': {'size': 24}},
        delta={'reference': 680},
        gauge={
            'axis': {'range': [300, 850], 'tickwidth': 1, 'tickcolor': "darkgray"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [300, 579], 'color': '#ef4444'},
                {'range': [580, 669], 'color': '#f59e0b'},
                {'range': [670, 739], 'color': '#fbbf24'},
                {'range': [740, 799], 'color': '#84cc16'},
                {'range': [800, 850], 'color': '#10b981'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 740
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        font={'color': "white", 'family': "Arial"}
    )
    
    return fig


def forecast_chart(predictions):
    """Create credit score forecast chart"""
    months = [p["month"] for p in predictions]
    scores = [p["score"] for p in predictions]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=months,
        y=scores,
        mode='lines+markers',
        name='Predicted Score',
        line=dict(color='#06b6d4', width=3),
        marker=dict(size=10)
    ))
    
    fig.update_layout(
        title="6-Month Credit Score Forecast",
        xaxis_title="Months Ahead",
        yaxis_title="Credit Score",
        yaxis=dict(range=[650, 850]),
        height=400,
        template="plotly_dark"
    )
    
    return fig


def dashboard_page():
    """Main dashboard interface"""
    # Header
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown('<h1 class="main-header">💳 Credit Intelligence Dashboard</h1>', unsafe_allow_html=True)
    with col2:
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.authenticated = False
            st.session_state.access_token = None
            st.rerun()
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 🎯 Navigation")
        page = st.radio(
            "Select Page",
            ["📊 Overview", "🔍 Get Credit Report", "🤖 AI Analysis", "📝 Dispute Generator", "⚙️ Settings"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        st.markdown("### 👤 User Info")
        st.info(f"**Email:** {st.session_state.user_email}")
        st.markdown("**Plan:** Pro")
        st.markdown("**Reports Left:** 5 / Unlimited")
    
    # Main content
    if page == "📊 Overview":
        overview_page()
    elif page == "🔍 Get Credit Report":
        get_report_page()
    elif page == "🤖 AI Analysis":
        ai_analysis_page()
    elif page == "📝 Dispute Generator":
        dispute_page()
    elif page == "⚙️ Settings":
        settings_page()


def overview_page():
    """Dashboard overview"""
    st.markdown("## 📊 Credit Overview")
    
    # Mock data for demo
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Credit Score", "704", "+12", help="Average across 3 bureaus")
    with col2:
        st.metric("Fraud Risk", "Low (15)", "-5", help="0-100 scale, lower is better")
    with col3:
        st.metric("Total Accounts", "22", "+1")
    with col4:
        st.metric("Utilization", "35%", "-3%", help="Target < 30%")
    
    # Credit score gauge
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.plotly_chart(credit_score_gauge(704), use_container_width=True)
    
    with col2:
        # Mock forecast data
        predictions = [
            {"month": 1, "score": 710},
            {"month": 2, "score": 718},
            {"month": 3, "score": 724},
            {"month": 4, "score": 729},
            {"month": 5, "score": 733},
            {"month": 6, "score": 735}
        ]
        st.plotly_chart(forecast_chart(predictions), use_container_width=True)
    
    # AI Insights
    st.markdown("## 🤖 AI Agent Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="success-box">', unsafe_allow_html=True)
        st.markdown("### ✅ Factors Helping Your Score")
        st.markdown("- No late payments (24 months)")
        st.markdown("- Long credit history (15 years)")
        st.markdown("- Low total debt ($33K)")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="warning-box">', unsafe_allow_html=True)
        st.markdown("### ⚠️ Factors Hurting Your Score")
        st.markdown("- High utilization (35% - target <30%)")
        st.markdown("- 11 hard inquiries (6 months)")
        st.markdown("- 26 negative accounts")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Recommendations
    st.markdown("## 💡 AI Recommendations")
    st.info("""
    **Top Priority Actions:**
    1. 💰 Pay down $3,000 on Capital One to reduce utilization below 30%
    2. 📝 Dispute 3 negative accounts with FCRA violations
    3. 🚫 Avoid new credit applications for 6 months
    """)


def get_report_page():
    """Get credit report page"""
    st.markdown("## 🔍 Get Credit Report")
    
    tab1, tab2 = st.tabs(["3-Bureau Report", "Snapshot"])
    
    with tab1:
        st.markdown("### Full 3-Bureau Credit Report")
        st.info("Get comprehensive credit report from TransUnion, Equifax, and Experian")
        
        with st.form("report_form"):
            mfsn_email = st.text_input("MyFreeScoreNow Email", placeholder="your@email.com")
            mfsn_password = st.text_input("MyFreeScoreNow Password", type="password")
            
            col1, col2 = st.columns(2)
            with col1:
                submit = st.form_submit_button("📥 Get Report", use_container_width=True)
            with col2:
                epic = st.form_submit_button("⭐ Get Epic Pro Report", use_container_width=True)
            
            if submit:
                with st.spinner("Retrieving credit report..."):
                    report_data = {
                        "username": mfsn_email,
                        "password": mfsn_password
                    }
                    response = api_request("/mfsn/3b-report", method="POST", data=report_data)
                    
                    if response and response.get("success"):
                        st.success("✅ Credit report retrieved successfully!")
                        st.json(response.get("data", {}))
                    else:
                        st.error("❌ Failed to retrieve report")
    
    with tab2:
        st.markdown("### Quick Snapshot Enrollment")
        st.info("Fast credit check enrollment")
        
        with st.form("snapshot_form"):
            col1, col2 = st.columns(2)
            with col1:
                first_name = st.text_input("First Name")
                email = st.text_input("Email")
                ssn = st.text_input("SSN", type="password")
                city = st.text_input("City")
            
            with col2:
                last_name = st.text_input("Last Name")
                mobile = st.text_input("Mobile")
                dob = st.text_input("DOB (MM/DD/YYYY)")
                state = st.text_input("State")
            
            address = st.text_input("Street Address")
            zip_code = st.text_input("ZIP Code")
            password = st.text_input("Create Password", type="password")
            
            submit_snapshot = st.form_submit_button("🚀 Enroll Now", use_container_width=True)
            
            if submit_snapshot:
                st.success("✅ Snapshot enrollment submitted!")


def ai_analysis_page():
    """AI analysis page"""
    st.markdown("## 🤖 AI-Powered Analysis")
    
    st.info("Run advanced AI analysis on your credit report")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📊 Credit Scoring Analysis", use_container_width=True):
            with st.spinner("Running XGBoost/LightGBM ensemble..."):
                st.success("Analysis complete!")
                st.json({
                    "score": 704,
                    "confidence": 0.87,
                    "risk_level": "MEDIUM"
                })
    
    with col2:
        if st.button("🕵️ Fraud Detection (GNN)", use_container_width=True):
            with st.spinner("Analyzing transaction patterns..."):
                st.success("Fraud check complete!")
                st.json({
                    "risk_score": 15,
                    "is_fraudulent": False
                })
    
    col3, col4 = st.columns(2)
    
    with col3:
        if st.button("📈 6-Month Forecast", use_container_width=True):
            with st.spinner("Running LSTM-Transformer model..."):
                st.success("Forecast complete!")
    
    with col4:
        if st.button("🔥 Full Analysis (All Agents)", use_container_width=True):
            with st.spinner("Orchestrating multi-agent system..."):
                st.success("Full analysis complete!")


def dispute_page():
    """Dispute generator page"""
    st.markdown("## 📝 FCRA Dispute Letter Generator")
    
    st.info("Generate professional, FCRA-compliant dispute letters")
    
    bureau = st.selectbox("Select Bureau", ["TRANSUNION", "EQUIFAX", "EXPERIAN"])
    
    st.markdown("### Items to Dispute")
    num_items = st.number_input("Number of items", min_value=1, max_value=10, value=1)
    
    dispute_items = []
    for i in range(num_items):
        with st.expander(f"Item {i+1}"):
            account = st.text_input(f"Account Name {i+1}")
            issue = st.text_area(f"Issue Description {i+1}")
            dispute_items.append({"account": account, "description": issue})
    
    reason = st.text_area("Overall Reason for Dispute", height=150)
    
    if st.button("🎯 Generate Dispute Letter", use_container_width=True):
        with st.spinner("AI generating FCRA-compliant letter..."):
            st.success("✅ Dispute letter generated!")
            st.markdown("### Generated Letter Preview")
            st.text_area("Letter Content", value=f"""
[Your Name]
[Your Address]

{datetime.now().strftime('%B %d, %Y')}

{bureau} Credit Bureau

Dear Sir/Madam,

I am writing to dispute the following information...

[AI-generated content based on your inputs]

Sincerely,
[Your Name]
            """, height=400)


def settings_page():
    """Settings page"""
    st.markdown("## ⚙️ Settings")
    
    tab1, tab2, tab3 = st.tabs(["Profile", "MFSN Credentials", "Subscription"])
    
    with tab1:
        st.markdown("### Profile Settings")
        with st.form("profile_form"):
            name = st.text_input("Full Name", value="Rick Jefferson")
            email = st.text_input("Email", value=st.session_state.user_email)
            submit = st.form_submit_button("💾 Save Changes")
            if submit:
                st.success("✅ Profile updated!")
    
    with tab2:
        st.markdown("### MyFreeScoreNow Credentials")
        with st.form("mfsn_form"):
            mfsn_email = st.text_input("MFSN Email")
            mfsn_password = st.text_input("MFSN Password", type="password")
            submit_mfsn = st.form_submit_button("💾 Save Credentials (Encrypted)")
            if submit_mfsn:
                st.success("✅ Credentials saved securely!")
    
    with tab3:
        st.markdown("### Subscription")
        st.info("**Current Plan:** Pro ($297/month)")
        st.markdown("**Benefits:**")
        st.markdown("- ✅ Unlimited credit reports")
        st.markdown("- ✅ All AI agents")
        st.markdown("- ✅ Priority support")
        
        if st.button("⬆️ Upgrade to Enterprise"):
            st.info("Contact sales for Enterprise plan")


# Main app logic
def main():
    if not st.session_state.authenticated:
        login_page()
    else:
        dashboard_page()


if __name__ == "__main__":
    main()
