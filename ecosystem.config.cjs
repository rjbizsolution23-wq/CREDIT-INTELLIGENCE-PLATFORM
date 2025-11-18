/**
 * PM2 Configuration for Credit Intelligence Platform
 * For local development/sandbox environment
 */

module.exports = {
  apps: [
    {
      name: 'credit-intel-backend',
      script: 'uvicorn',
      args: 'api.main:app --host 0.0.0.0 --port 8000 --reload',
      cwd: './backend',
      interpreter: 'python3',
      env: {
        PYTHONPATH: './backend',
        ENVIRONMENT: 'development',
        DEBUG: 'True',
        PORT: 8000
      },
      instances: 1,
      exec_mode: 'fork',
      watch: false,
      max_memory_restart: '500M',
      error_file: './logs/backend-error.log',
      out_file: './logs/backend-out.log',
      log_date_format: 'YYYY-MM-DD HH:mm:ss Z'
    },
    {
      name: 'credit-intel-frontend',
      script: 'streamlit',
      args: 'run app.py --server.port 8501 --server.address 0.0.0.0',
      cwd: './frontend',
      interpreter: 'python3',
      env: {
        API_URL: 'http://localhost:8000',
        PORT: 8501
      },
      instances: 1,
      exec_mode: 'fork',
      watch: false,
      max_memory_restart: '500M',
      error_file: './logs/frontend-error.log',
      out_file: './logs/frontend-out.log',
      log_date_format: 'YYYY-MM-DD HH:mm:ss Z'
    }
  ]
};
