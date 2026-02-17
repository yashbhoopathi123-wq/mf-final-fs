# 📊 Mutual Fund Performance Analyzer

A Streamlit web application for analyzing mutual fund performance with AI-powered recommendations.

## Features

- **3-Year to 15-Year CAGR Analysis** - Track fund performance over different time periods
- **Alpha & Beta Calculations** - Compare performance against Nifty 50
- **Risk Metrics** - Standard deviation and Sharpe ratio analysis
- **SIP Calculator** - Simulate returns on Systematic Investment Plans
- **Cost Efficiency Analysis** - Understand fund expense ratios
- **Fund Manager Tenure** - Track manager experience
- **AI Recommendations** - Get personalized fund suggestions

## Installation

1. Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/mutualfund-analyzer.git
cd mutualfund-analyzer
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running Locally

```bash
streamlit run mutualfund.py
```

The app will open at `http://localhost:8501`

## Deployment

### Deploy on Streamlit Cloud

1. Push your repository to GitHub
2. Go to [Streamlit Cloud](https://streamlit.io/cloud)
3. Sign in with your GitHub account
4. Click "New app" and select this repository
5. Configure settings and deploy

## Requirements

- Python 3.8+
- See `requirements.txt` for package dependencies

## License

MIT
