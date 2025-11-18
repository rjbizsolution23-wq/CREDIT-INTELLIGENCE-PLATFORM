-- Credit Intelligence Platform Database Schema
-- PostgreSQL 15+

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ============= USERS & AUTHENTICATION =============

CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    full_name VARCHAR(255) NOT NULL,
    role VARCHAR(50) DEFAULT 'user' CHECK (role IN ('user', 'admin', 'super_admin')),
    affiliate_id VARCHAR(100),
    is_active BOOLEAN DEFAULT TRUE,
    email_verified BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_users_affiliate_id ON users(affiliate_id);


-- ============= MFSN CREDENTIALS =============

CREATE TABLE mfsn_credentials (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    mfsn_email VARCHAR(255) NOT NULL,
    mfsn_password_encrypted TEXT NOT NULL,
    tracking_token TEXT,
    customer_token TEXT,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(user_id, mfsn_email)
);

CREATE INDEX idx_mfsn_credentials_user_id ON mfsn_credentials(user_id);


-- ============= CREDIT REPORTS =============

CREATE TABLE credit_reports (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    report_type VARCHAR(50) NOT NULL CHECK (report_type IN ('3B', 'EPIC_PRO', 'SNAPSHOT_CREDIT', 'SNAPSHOT_FUNDING')),
    report_data JSONB NOT NULL,
    
    -- Extracted key metrics for quick access
    transunion_score INTEGER,
    equifax_score INTEGER,
    experian_score INTEGER,
    average_score INTEGER,
    total_accounts INTEGER,
    total_balance DECIMAL(12,2),
    total_inquiries INTEGER,
    
    retrieved_at TIMESTAMP DEFAULT NOW(),
    expires_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_credit_reports_user_id ON credit_reports(user_id);
CREATE INDEX idx_credit_reports_type ON credit_reports(report_type);
CREATE INDEX idx_credit_reports_retrieved ON credit_reports(retrieved_at DESC);


-- ============= AI ANALYSIS RESULTS =============

CREATE TABLE ai_analysis (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    report_id UUID REFERENCES credit_reports(id) ON DELETE CASCADE,
    analysis_type VARCHAR(50) NOT NULL CHECK (analysis_type IN ('CREDIT_SCORE', 'FRAUD_DETECTION', 'FORECAST', 'FULL_ANALYSIS')),
    
    -- Analysis results
    score DECIMAL(10,2),
    confidence DECIMAL(5,4),
    risk_level VARCHAR(20),
    
    -- Detailed results (JSONB for flexibility)
    shap_values JSONB,
    recommendations JSONB,
    factors_helping JSONB,
    factors_hurting JSONB,
    
    -- Metadata
    model_version VARCHAR(50),
    execution_time DECIMAL(8,3),
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_ai_analysis_user_id ON ai_analysis(user_id);
CREATE INDEX idx_ai_analysis_report_id ON ai_analysis(report_id);
CREATE INDEX idx_ai_analysis_type ON ai_analysis(analysis_type);
CREATE INDEX idx_ai_analysis_created ON ai_analysis(created_at DESC);


-- ============= FRAUD ALERTS =============

CREATE TABLE fraud_alerts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    report_id UUID REFERENCES credit_reports(id) ON DELETE CASCADE,
    
    alert_type VARCHAR(100) NOT NULL,
    risk_score INTEGER CHECK (risk_score >= 0 AND risk_score <= 100),
    severity VARCHAR(20) CHECK (severity IN ('LOW', 'MEDIUM', 'HIGH', 'CRITICAL')),
    
    flagged_items JSONB,
    anomalies JSONB,
    
    status VARCHAR(50) DEFAULT 'ACTIVE' CHECK (status IN ('ACTIVE', 'INVESTIGATING', 'RESOLVED', 'FALSE_POSITIVE')),
    resolved_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_fraud_alerts_user_id ON fraud_alerts(user_id);
CREATE INDEX idx_fraud_alerts_status ON fraud_alerts(status);
CREATE INDEX idx_fraud_alerts_severity ON fraud_alerts(severity);


-- ============= CREDIT FORECASTS =============

CREATE TABLE credit_forecasts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    report_id UUID REFERENCES credit_reports(id) ON DELETE CASCADE,
    
    current_score INTEGER,
    predicted_scores JSONB NOT NULL,
    confidence_intervals JSONB,
    scenario VARCHAR(50) CHECK (scenario IN ('OPTIMISTIC', 'REALISTIC', 'PESSIMISTIC')),
    
    months_ahead INTEGER,
    model_version VARCHAR(50),
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_credit_forecasts_user_id ON credit_forecasts(user_id);
CREATE INDEX idx_credit_forecasts_report_id ON credit_forecasts(report_id);


-- ============= DISPUTE LETTERS =============

CREATE TABLE dispute_letters (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    report_id UUID REFERENCES credit_reports(id),
    
    bureau VARCHAR(50) NOT NULL CHECK (bureau IN ('TRANSUNION', 'EQUIFAX', 'EXPERIAN', 'ALL')),
    dispute_items JSONB NOT NULL,
    reason TEXT NOT NULL,
    letter_content TEXT NOT NULL,
    
    status VARCHAR(50) DEFAULT 'DRAFT' CHECK (status IN ('DRAFT', 'SENT', 'IN_PROGRESS', 'RESOLVED', 'DENIED')),
    sent_at TIMESTAMP,
    resolved_at TIMESTAMP,
    
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_dispute_letters_user_id ON dispute_letters(user_id);
CREATE INDEX idx_dispute_letters_bureau ON dispute_letters(bureau);
CREATE INDEX idx_dispute_letters_status ON dispute_letters(status);


-- ============= AGENT EXECUTIONS =============

CREATE TABLE agent_executions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    report_id UUID REFERENCES credit_reports(id),
    
    execution_id VARCHAR(255) UNIQUE NOT NULL,
    tasks JSONB NOT NULL,
    priority VARCHAR(20) DEFAULT 'normal' CHECK (priority IN ('low', 'normal', 'high')),
    
    status VARCHAR(50) DEFAULT 'QUEUED' CHECK (status IN ('QUEUED', 'RUNNING', 'COMPLETED', 'FAILED')),
    results JSONB,
    error_message TEXT,
    
    execution_time DECIMAL(8,3),
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_agent_executions_user_id ON agent_executions(user_id);
CREATE INDEX idx_agent_executions_execution_id ON agent_executions(execution_id);
CREATE INDEX idx_agent_executions_status ON agent_executions(status);


-- ============= SUBSCRIPTIONS =============

CREATE TABLE subscriptions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    
    stripe_customer_id VARCHAR(255),
    stripe_subscription_id VARCHAR(255),
    
    tier VARCHAR(50) NOT NULL CHECK (tier IN ('STARTER', 'PRO', 'ENTERPRISE')),
    status VARCHAR(50) DEFAULT 'ACTIVE' CHECK (status IN ('ACTIVE', 'PAST_DUE', 'CANCELED', 'UNPAID')),
    
    price_monthly DECIMAL(10,2),
    
    current_period_start TIMESTAMP,
    current_period_end TIMESTAMP,
    cancel_at TIMESTAMP,
    canceled_at TIMESTAMP,
    
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_subscriptions_user_id ON subscriptions(user_id);
CREATE INDEX idx_subscriptions_stripe_customer ON subscriptions(stripe_customer_id);
CREATE INDEX idx_subscriptions_status ON subscriptions(status);


-- ============= AFFILIATE CONVERSIONS =============

CREATE TABLE affiliate_conversions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    
    affiliate_id VARCHAR(100) NOT NULL,
    pid VARCHAR(50) NOT NULL,
    
    conversion_type VARCHAR(50) CHECK (conversion_type IN ('MFSN_ENROLLMENT', 'SUBSCRIPTION', 'UPGRADE')),
    commission_amount DECIMAL(10,2),
    
    status VARCHAR(50) DEFAULT 'PENDING' CHECK (status IN ('PENDING', 'APPROVED', 'PAID')),
    
    conversion_date TIMESTAMP DEFAULT NOW(),
    paid_at TIMESTAMP
);

CREATE INDEX idx_affiliate_conversions_affiliate_id ON affiliate_conversions(affiliate_id);
CREATE INDEX idx_affiliate_conversions_status ON affiliate_conversions(status);
CREATE INDEX idx_affiliate_conversions_date ON affiliate_conversions(conversion_date DESC);


-- ============= AUDIT LOGS (FCRA Compliance) =============

CREATE TABLE audit_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    
    action VARCHAR(100) NOT NULL,
    resource VARCHAR(100),
    resource_id UUID,
    
    ip_address INET,
    user_agent TEXT,
    
    request_data JSONB,
    response_status INTEGER,
    
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_audit_logs_user_id ON audit_logs(user_id);
CREATE INDEX idx_audit_logs_action ON audit_logs(action);
CREATE INDEX idx_audit_logs_created ON audit_logs(created_at DESC);


-- ============= RAG SEARCH INDEX METADATA =============

CREATE TABLE vector_index_metadata (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    report_id UUID REFERENCES credit_reports(id) ON DELETE CASCADE,
    
    pinecone_vector_id VARCHAR(255) NOT NULL,
    content_type VARCHAR(50) CHECK (content_type IN ('TRADELINE', 'INQUIRY', 'PUBLIC_RECORD', 'NARRATIVE')),
    content_summary TEXT,
    
    bureau VARCHAR(50),
    account_id VARCHAR(255),
    
    indexed_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_vector_metadata_user_id ON vector_index_metadata(user_id);
CREATE INDEX idx_vector_metadata_report_id ON vector_index_metadata(report_id);
CREATE INDEX idx_vector_metadata_pinecone_id ON vector_index_metadata(pinecone_vector_id);


-- ============= FUNCTIONS & TRIGGERS =============

-- Update updated_at timestamp automatically
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_mfsn_credentials_updated_at BEFORE UPDATE ON mfsn_credentials
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_dispute_letters_updated_at BEFORE UPDATE ON dispute_letters
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_subscriptions_updated_at BEFORE UPDATE ON subscriptions
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
