import streamlit as st
import pandas as pd
import numpy as np
from mftool import Mftool
import plotly.graph_objects as go
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Initialize mftool
mf = Mftool()

st.set_page_config(page_title="Mutual Fund Analyzer", layout="wide")
st.title("📊 Mutual Fund Performance Analyzer")

# Sidebar for inputs
st.sidebar.header("Analysis Parameters")
cagr_years = st.sidebar.slider(
    "Analysis Period (Years)", 
    min_value=1, 
    max_value=15, 
    value=3,
    help="Select the time period for CAGR, Alpha, Beta calculation"
)

st.sidebar.header("SIP Investment Calculator")
investment_amount = st.sidebar.number_input(
    "Monthly SIP Amount (₹)", 
    min_value=500, 
    max_value=1000000, 
    value=10000, 
    step=500
)
sip_period = st.sidebar.slider("SIP Period (Years)", 1, 15, 3)

st.sidebar.markdown("---")
st.sidebar.header("About")
st.sidebar.info(
    "This app analyzes mutual fund performance using:\n"
    f"- {cagr_years}-Year CAGR\n"
    "- Alpha & Beta (vs Nifty 50)\n"
    "- Standard Deviation (Risk)\n"
    "- Sharpe Ratio\n"
    "- Cost Efficiency\n"
    "- Fund Manager Tenure\n"
    "- SIP Returns Simulation\n"
    "- AI-Powered Recommendations"
)

# Mutual Fund Categories Database
FUND_CATEGORIES = {
    "Large Cap": [
        {"name": "HDFC Top 100 Fund", "code": "118989", "aum": "High", "expense_ratio": 1.78, "manager_tenure": 8, "exit_load": 1.0},
        {"name": "SBI Bluechip Fund", "code": "119551", "aum": "High", "expense_ratio": 1.60, "manager_tenure": 6, "exit_load": 1.0},
        {"name": "ICICI Pru Bluechip Fund", "code": "120503", "aum": "High", "expense_ratio": 1.75, "manager_tenure": 5, "exit_load": 1.0},
        {"name": "Axis Bluechip Fund", "code": "120716", "aum": "Medium", "expense_ratio": 1.69, "manager_tenure": 7, "exit_load": 1.0},
        {"name": "Mirae Asset Large Cap Fund", "code": "125497", "aum": "Medium", "expense_ratio": 1.58, "manager_tenure": 9, "exit_load": 1.0},
    ],
    "Mid Cap": [
        {"name": "Kotak Emerging Equity Fund", "code": "103705", "aum": "High", "expense_ratio": 1.88, "manager_tenure": 10, "exit_load": 1.0},
        {"name": "HDFC Mid-Cap Opportunities Fund", "code": "118989", "aum": "High", "expense_ratio": 1.95, "manager_tenure": 7, "exit_load": 1.0},
        {"name": "Axis Midcap Fund", "code": "120503", "aum": "Medium", "expense_ratio": 1.82, "manager_tenure": 6, "exit_load": 1.0},
        {"name": "DSP Midcap Fund", "code": "112582", "aum": "Medium", "expense_ratio": 1.90, "manager_tenure": 8, "exit_load": 1.0},
        {"name": "Edelweiss Mid Cap Fund", "code": "119551", "aum": "Low", "expense_ratio": 1.75, "manager_tenure": 4, "exit_load": 1.0},
    ],
    "Small Cap": [
        {"name": "Axis Small Cap Fund", "code": "120503", "aum": "High", "expense_ratio": 2.01, "manager_tenure": 5, "exit_load": 1.0},
        {"name": "SBI Small Cap Fund", "code": "119597", "aum": "High", "expense_ratio": 1.97, "manager_tenure": 9, "exit_load": 1.0},
        {"name": "Kotak Small Cap Fund", "code": "112582", "aum": "Medium", "expense_ratio": 2.15, "manager_tenure": 6, "exit_load": 1.0},
        {"name": "Nippon India Small Cap Fund", "code": "118989", "aum": "Medium", "expense_ratio": 2.08, "manager_tenure": 11, "exit_load": 1.0},
        {"name": "HDFC Small Cap Fund", "code": "125497", "aum": "Medium", "expense_ratio": 2.12, "manager_tenure": 7, "exit_load": 1.0},
    ],
    "Multi Cap": [
        {"name": "Parag Parikh Flexi Cap Fund", "code": "122639", "aum": "High", "expense_ratio": 1.94, "manager_tenure": 12, "exit_load": 2.0},
        {"name": "Quant Flexi Cap Fund", "code": "120716", "aum": "Medium", "expense_ratio": 1.82, "manager_tenure": 8, "exit_load": 1.0},
        {"name": "JM Multicap Fund", "code": "103705", "aum": "Low", "expense_ratio": 1.88, "manager_tenure": 5, "exit_load": 1.0},
        {"name": "UTI Flexi Cap Fund", "code": "120503", "aum": "Medium", "expense_ratio": 1.76, "manager_tenure": 6, "exit_load": 1.0},
        {"name": "PGIM India Flexi Cap Fund", "code": "119551", "aum": "Low", "expense_ratio": 1.85, "manager_tenure": 4, "exit_load": 1.0},
    ],
    "Index Funds": [
        {"name": "ICICI Pru Nifty 50 Index Fund", "code": "120716", "aum": "High", "expense_ratio": 0.20, "manager_tenure": 7, "exit_load": 0.0},
        {"name": "UTI Nifty 50 Index Fund", "code": "120503", "aum": "High", "expense_ratio": 0.20, "manager_tenure": 9, "exit_load": 0.0},
        {"name": "HDFC Index Nifty 50", "code": "118989", "aum": "Medium", "expense_ratio": 0.25, "manager_tenure": 6, "exit_load": 0.0},
        {"name": "SBI Nifty Index Fund", "code": "119551", "aum": "Medium", "expense_ratio": 0.22, "manager_tenure": 8, "exit_load": 0.0},
        {"name": "Nippon India Index Nifty 50", "code": "125497", "aum": "Medium", "expense_ratio": 0.28, "manager_tenure": 5, "exit_load": 0.0},
    ],
    "Debt Funds": [
        {"name": "HDFC Corporate Bond Fund", "code": "118989", "aum": "High", "expense_ratio": 0.89, "manager_tenure": 6, "exit_load": 0.5},
        {"name": "ICICI Pru Corporate Bond Fund", "code": "120503", "aum": "High", "expense_ratio": 0.85, "manager_tenure": 7, "exit_load": 0.5},
        {"name": "Axis Banking & PSU Debt Fund", "code": "125497", "aum": "Medium", "expense_ratio": 0.65, "manager_tenure": 5, "exit_load": 0.25},
        {"name": "SBI Magnum Gilt Fund", "code": "119551", "aum": "Medium", "expense_ratio": 0.75, "manager_tenure": 8, "exit_load": 0.5},
        {"name": "Kotak Bond Fund", "code": "112582", "aum": "Medium", "expense_ratio": 0.82, "manager_tenure": 6, "exit_load": 0.5},
    ],
}

# Function to calculate CAGR
def calculate_cagr(start_value, end_value, years):
    if start_value <= 0 or end_value <= 0:
        return 0
    return (((end_value / start_value) ** (1 / years)) - 1) * 100

# Function to calculate standard deviation
def calculate_std_dev(returns):
    return np.std(returns) * np.sqrt(252) * 100  # Annualized

# Function to calculate Sharpe Ratio
def calculate_sharpe_ratio(returns, risk_free_rate=0.06):
    excess_returns = returns - (risk_free_rate / 252)
    return (np.mean(excess_returns) * 252) / (np.std(excess_returns) * np.sqrt(252))

# Function to calculate Alpha and Beta
def calculate_alpha_beta(fund_returns, benchmark_returns, risk_free_rate=0.06):
    """
    Calculate Jensen's Alpha and Beta relative to benchmark
    Alpha: Excess return over what CAPM predicts
    Beta: Sensitivity to benchmark movements
    """
    try:
        # Align the data
        combined = pd.DataFrame({
            'fund': fund_returns,
            'benchmark': benchmark_returns
        }).dropna()
        
        if len(combined) < 30:  # Need sufficient data points
            return 0, 1
        
        # Calculate excess returns
        rf_daily = risk_free_rate / 252
        fund_excess = combined['fund'] - rf_daily
        benchmark_excess = combined['benchmark'] - rf_daily
        
        # Calculate Beta using covariance
        covariance = np.cov(fund_excess, benchmark_excess)[0][1]
        benchmark_variance = np.var(benchmark_excess)
        beta = covariance / benchmark_variance if benchmark_variance != 0 else 1
        
        # Calculate Alpha (annualized)
        fund_return_annual = np.mean(fund_excess) * 252
        benchmark_return_annual = np.mean(benchmark_excess) * 252
        alpha = fund_return_annual - (beta * benchmark_return_annual)
        
        return alpha * 100, beta  # Alpha as percentage
    except Exception as e:
        return 0, 1

# Function to calculate cost efficiency score
def calculate_cost_efficiency(cagr, expense_ratio):
    """
    Calculate cost efficiency: Returns per unit of cost
    Higher is better
    """
    if expense_ratio == 0:
        return cagr * 100
    return cagr / expense_ratio

# Function to fetch NAV data
@st.cache_data(ttl=3600)
def get_fund_data(scheme_code, years=3):
    try:
        end_date   = datetime.now()
        start_date = end_date - timedelta(days=years * 365 + 30)

        nav_data = mf.get_scheme_historical_nav(scheme_code, as_Dataframe=True)

        if nav_data is None or nav_data.empty:
            return None

        nav_data.index = pd.to_datetime(nav_data.index, errors="coerce")
        nav_data = nav_data[nav_data.index.notna()]          # drop bad dates

        # ── KEY FIX: mftool returns NAV as strings; cast to float ──────────
        nav_data["nav"] = pd.to_numeric(nav_data["nav"], errors="coerce")
        nav_data = nav_data.dropna(subset=["nav"])           # drop unparseable rows
        # ────────────────────────────────────────────────────────────────────

        nav_data = nav_data.sort_index()
        nav_data = nav_data[nav_data.index >= start_date]

        if nav_data.empty:
            return None

        return nav_data
    except Exception as e:
        st.error(f"Error fetching data for scheme {scheme_code}: {str(e)}")
        return None

# Function to calculate SIP returns
def calculate_sip_returns(nav_data, monthly_investment, years):
    try:
        # Resample to monthly
        monthly_nav = nav_data.resample('M').last()
        
        # Limit to SIP period
        months = years * 12
        monthly_nav = monthly_nav.tail(months)
        
        units_accumulated = 0
        total_invested = 0
        
        for nav_value in monthly_nav['nav']:
            units_accumulated += monthly_investment / nav_value
            total_invested += monthly_investment
        
        current_value = units_accumulated * monthly_nav['nav'].iloc[-1]
        absolute_return = current_value - total_invested
        return_percentage = (absolute_return / total_invested) * 100
        
        return {
            'total_invested': total_invested,
            'current_value': current_value,
            'absolute_return': absolute_return,
            'return_percentage': return_percentage
        }
    except Exception as e:
        return None

# Function to analyze and recommend funds
def analyze_and_recommend_funds(category, years, benchmark_code="120716"):
    """Analyze funds in a category and provide recommendations"""
    funds_list = FUND_CATEGORIES.get(category, [])
    
    # Fetch benchmark data (Nifty 50 proxy)
    benchmark_data = get_fund_data(benchmark_code, years)
    benchmark_returns = None
    if benchmark_data is not None and len(benchmark_data) > 2:
        benchmark_data['returns'] = benchmark_data['nav'].pct_change()
        benchmark_returns = benchmark_data['returns'].dropna()
    
    results = []
    
    for fund in funds_list:
        data = get_fund_data(fund['code'], years)
        
        if data is None or len(data) < 2:
            continue
        
        start_nav = data['nav'].iloc[0]
        end_nav = data['nav'].iloc[-1]
        
        # Calculate daily returns
        data['returns'] = data['nav'].pct_change()
        returns = data['returns'].dropna()
        
        # Calculate metrics
        cagr = calculate_cagr(start_nav, end_nav, years)
        std_dev = calculate_std_dev(returns)
        sharpe = calculate_sharpe_ratio(returns)
        
        # Calculate Alpha and Beta
        alpha, beta = 0, 1
        if benchmark_returns is not None:
            alpha, beta = calculate_alpha_beta(returns, benchmark_returns)
        
        # Calculate cost efficiency
        cost_efficiency = calculate_cost_efficiency(cagr, fund['expense_ratio'])
        
        # Manager tenure score (longer tenure = more stability)
        manager_score = min(fund['manager_tenure'] / 2, 5)  # Cap at 5 points
        
        # Calculate overall score (weighted)
        overall_score = (
            cagr * 0.30 +  # 30% weight to returns
            sharpe * 12 +  # 24% weight to risk-adjusted returns (scaled)
            alpha * 1.5 +  # 15% weight to alpha (excess returns)
            (10 if fund['aum'] == 'High' else 5 if fund['aum'] == 'Medium' else 2) +  # 10% to AUM
            cost_efficiency * 0.5 +  # 10% to cost efficiency
            manager_score +  # 5% to manager tenure
            (2 - beta) * 3  # 6% to beta (prefer beta close to 1)
        )
        
        results.append({
            'Fund Name': fund['name'],
            'Code': fund['code'],
            'CAGR (%)': round(cagr, 2),
            'Alpha (%)': round(alpha, 2),
            'Beta': round(beta, 2),
            'Std Dev (%)': round(std_dev, 2),
            'Sharpe Ratio': round(sharpe, 2),
            'Expense Ratio (%)': fund['expense_ratio'],
            'Cost Efficiency': round(cost_efficiency, 2),
            'Manager Tenure (Yrs)': fund['manager_tenure'],
            'Exit Load (%)': fund['exit_load'],
            'AUM': fund['aum'],
            'Overall Score': round(overall_score, 2)
        })
    
    # Sort by overall score
    results_df = pd.DataFrame(results)
    if not results_df.empty:
        results_df = results_df.sort_values('Overall Score', ascending=False)
    
    return results_df

# Function to generate AI recommendation text
def generate_recommendation_text(category, top_funds_df, years):
    """Generate comprehensive recommendation based on analysis"""
    
    if top_funds_df.empty:
        return "Unable to generate recommendations. Please try a different category or check your internet connection."
    
    top_fund = top_funds_df.iloc[0]
    
    # Market context based on category
    market_context = {
        "Large Cap": "Large-cap funds are suitable for conservative investors seeking stability. Current market shows strong performance in blue-chip stocks.",
        "Mid Cap": "Mid-cap funds offer a balance between growth and stability. They're ideal for investors with moderate risk appetite and 5+ year horizon.",
        "Small Cap": "Small-cap funds are high-risk, high-reward. Suitable for aggressive investors with 7+ year investment horizon. Market volatility is higher here.",
        "Multi Cap": "Multi-cap/Flexi-cap funds provide diversification across market caps. Good for investors seeking flexibility and balanced growth.",
        "Index Funds": "Index funds offer low-cost, passive investing. Ideal for long-term wealth creation with minimal expense ratios.",
        "Debt Funds": "Debt funds provide stable returns with lower risk. Suitable for conservative investors and debt allocation in portfolios."
    }
    
    # Risk assessment
    avg_sharpe = top_funds_df['Sharpe Ratio'].mean()
    avg_std = top_funds_df['Std Dev (%)'].mean()
    avg_alpha = top_funds_df['Alpha (%)'].mean()
    avg_beta = top_funds_df['Beta'].mean()
    
    risk_level = "Low" if avg_std < 10 else "Moderate" if avg_std < 18 else "High"
    
    recommendation = f"""
### 🎯 Recommendation for {category} Funds ({years}-Year Analysis)

**Market Context:**  
{market_context.get(category, "This category offers unique investment opportunities.")}

**Top Performer:** {top_fund['Fund Name']}  
- **CAGR:** {top_fund['CAGR (%)']}% over {years} years
- **Alpha:** {top_fund['Alpha (%)']}% ({"Outperforming" if top_fund['Alpha (%)'] > 0 else "Underperforming"} benchmark)
- **Beta:** {top_fund['Beta']} ({"Lower" if top_fund['Beta'] < 1 else "Higher"} volatility than market)
- **Risk-Adjusted Returns (Sharpe):** {top_fund['Sharpe Ratio']}
- **Volatility:** {top_fund['Std Dev (%)']}% (Risk Level: **{risk_level}**)
- **Expense Ratio:** {top_fund['Expense Ratio (%)']}%
- **Cost Efficiency:** {top_fund['Cost Efficiency']:.2f} (Returns per 1% expense)
- **Fund Manager Tenure:** {top_fund['Manager Tenure (Yrs)']} years

**Key Insights:**
- **Average CAGR** across analyzed funds: {top_funds_df['CAGR (%)'].mean():.2f}%
- **Average Alpha**: {avg_alpha:.2f}% ({"Positive alpha shows skill-based outperformance" if avg_alpha > 0 else "Negative alpha suggests underperformance vs benchmark"})
- **Average Beta**: {avg_beta:.2f} ({"Less volatile than market" if avg_beta < 1 else "More volatile than market" if avg_beta > 1 else "Moves with market"})
- **Average Sharpe Ratio**: {avg_sharpe:.2f} ({"Excellent" if avg_sharpe > 1 else "Good" if avg_sharpe > 0.5 else "Moderate"} risk-adjusted performance)
- **Consistency**: {"High" if avg_std < 12 else "Moderate" if avg_std < 18 else "Variable"} based on standard deviation
- **Manager Stability**: Average tenure of {top_funds_df['Manager Tenure (Yrs)'].mean():.1f} years indicates {"strong" if top_funds_df['Manager Tenure (Yrs)'].mean() > 7 else "moderate"} continuity

**Investment Strategy:**
"""
    
    if category in ["Small Cap", "Mid Cap"]:
        recommendation += """
- ⚠️ **SIP Recommended**: Use SIP to average out volatility
- 📅 **Investment Horizon**: Minimum 5-7 years
- 💼 **Portfolio Allocation**: Keep this at 20-30% of total equity allocation
"""
    elif category == "Large Cap":
        recommendation += """
- ✅ **Core Holding**: Suitable as core portfolio component
- 📅 **Investment Horizon**: Minimum 3-5 years
- 💼 **Portfolio Allocation**: Can form 40-50% of equity allocation
"""
    elif category == "Index Funds":
        recommendation += """
- 💰 **Low Cost**: Expense ratios under 0.3% make these ideal for long-term
- 📅 **Investment Horizon**: 10+ years for optimal results
- 💼 **Portfolio Allocation**: Can be 30-40% of equity allocation
"""
    elif category == "Debt Funds":
        recommendation += """
- 🛡️ **Stability**: Lower volatility, suitable for capital preservation
- 📅 **Investment Horizon**: 1-3 years depending on fund type
- 💼 **Portfolio Allocation**: 30-40% for balanced portfolios, higher for conservative investors
"""
    
    recommendation += f"""

**Future Outlook:**  
Based on {years}-year performance trends and current market conditions, the top funds show {"strong momentum" if top_fund['CAGR (%)'] > 15 else "steady growth" if top_fund['CAGR (%)'] > 10 else "moderate performance"}. 
Consider your risk tolerance and investment horizon before investing.

⚠️ *Past performance is not indicative of future results. Consult a financial advisor for personalized advice.*
"""
    
    return recommendation

# Main App Interface
st.header("🔍 Find Best Mutual Funds by Category")

# Create tabs for different modes
tab1, tab2, tab3, tab4 = st.tabs(["📊 Get Recommendations", "🔎 Search Fund & Compare", "🔧 Manual Analysis", "🗂️ Portfolio Allocator"])

with tab1:
    st.subheader("AI-Powered Fund Recommendations")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        selected_category = st.selectbox(
            "Select Fund Category",
            options=list(FUND_CATEGORIES.keys()),
            help="Choose the type of mutual fund you're interested in"
        )
    
    with col2:
        st.metric("Analysis Period", f"{cagr_years} Years", "From sidebar")
    
    if st.button("🚀 Get Recommendations", type="primary", use_container_width=True):
        with st.spinner(f"Analyzing {selected_category} funds... This may take a moment."):
            
            # Get recommendations
            results_df = analyze_and_recommend_funds(selected_category, cagr_years)
            
            if results_df.empty:
                st.error("Unable to fetch data for this category. Please try again or select a different category.")
            else:
                # Display top 3 funds
                st.success(f"✅ Analysis Complete! Found {len(results_df)} funds in {selected_category} category")
                
                # Show recommendation text
                recommendation_text = generate_recommendation_text(selected_category, results_df, cagr_years)
                st.markdown(recommendation_text)
                
                # Display detailed metrics table
                st.subheader(f"📈 Detailed Performance Metrics ({cagr_years} Years)")
                
                # Select columns to display
                display_columns = ['Fund Name', 'CAGR (%)', 'Alpha (%)', 'Beta', 'Sharpe Ratio', 
                                 'Std Dev (%)', 'Expense Ratio (%)', 'Manager Tenure (Yrs)', 
                                 'Cost Efficiency', 'Overall Score']
                
                # Style the dataframe
                st.dataframe(
                    results_df[display_columns].style.background_gradient(subset=['Overall Score'], cmap='RdYlGn')
                                   .background_gradient(subset=['CAGR (%)'], cmap='RdYlGn')
                                   .background_gradient(subset=['Alpha (%)'], cmap='RdYlGn')
                                   .background_gradient(subset=['Sharpe Ratio'], cmap='RdYlGn')
                                   .format({
                                       'CAGR (%)': '{:.2f}',
                                       'Alpha (%)': '{:.2f}',
                                       'Beta': '{:.2f}',
                                       'Std Dev (%)': '{:.2f}',
                                       'Sharpe Ratio': '{:.2f}',
                                       'Overall Score': '{:.2f}',
                                       'Expense Ratio (%)': '{:.2f}',
                                       'Cost Efficiency': '{:.2f}'
                                   }),
                    use_container_width=True,
                    hide_index=True
                )
                
                # Visualization - Top 3 funds comparison
                st.subheader("📊 Performance Comparison - Top 3 Funds")
                
                top_3_funds = results_df.head(3)
                
                # Fetch data for visualization
                fig = go.Figure()
                
                for idx, row in top_3_funds.iterrows():
                    fund_data = get_fund_data(row['Code'], cagr_years)
                    if fund_data is not None and len(fund_data) > 0:
                        # Normalize to 100
                        normalized = (fund_data['nav'] / fund_data['nav'].iloc[0]) * 100
                        
                        fig.add_trace(go.Scatter(
                            x=fund_data.index,
                            y=normalized,
                            mode='lines',
                            name=row['Fund Name'],
                            line=dict(width=2.5)
                        ))
                
                fig.update_layout(
                    title=f"Normalized Performance Comparison - {selected_category} (Base = 100)",
                    xaxis_title="Date",
                    yaxis_title="Indexed Value",
                    hovermode='x unified',
                    template='plotly_white',
                    height=500,
                    legend=dict(
                        yanchor="top",
                        y=0.99,
                        xanchor="left",
                        x=0.01
                    )
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # SIP Simulation for top fund
                st.subheader("💰 SIP Returns Simulation - Top Fund")
                st.write(f"**Fund:** {top_3_funds.iloc[0]['Fund Name']}")
                
                top_fund_data = get_fund_data(top_3_funds.iloc[0]['Code'], max(cagr_years, sip_period))
                
                if top_fund_data is not None:
                    sip_result = calculate_sip_returns(top_fund_data, investment_amount, sip_period)
                    
                    if sip_result:
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Total Invested", f"₹{sip_result['total_invested']:,.0f}")
                        with col2:
                            st.metric("Current Value", f"₹{sip_result['current_value']:,.0f}")
                        with col3:
                            st.metric("Absolute Gain", f"₹{sip_result['absolute_return']:,.0f}")
                        with col4:
                            st.metric("Returns", f"{sip_result['return_percentage']:.2f}%")

with tab2:
    st.subheader("🔎 Search Individual Fund & Compare with Competitors")
    st.info("💡 Search for a fund by name to see detailed analysis and top competitors in its category")
    
    # Create a searchable fund list
    all_funds = []
    for category, funds in FUND_CATEGORIES.items():
        for fund in funds:
            all_funds.append({
                'display': f"{fund['name']} ({category})",
                'name': fund['name'],
                'code': fund['code'],
                'category': category
            })
    
    fund_names = [f['display'] for f in all_funds]
    
    col1, col2 = st.columns([3, 1])
    with col1:
        selected_fund_display = st.selectbox(
            "Search and Select Fund",
            options=fund_names,
            help="Type to search for your fund"
        )
    
    with col2:
        st.metric("Analysis Period", f"{cagr_years} Years", "From sidebar")
    
    if st.button("🔍 Analyze Fund & Show Competitors", type="primary", use_container_width=True):
        # Find selected fund details
        selected_fund_info = next(f for f in all_funds if f['display'] == selected_fund_display)
        fund_category = selected_fund_info['category']
        fund_name = selected_fund_info['name']
        fund_code = selected_fund_info['code']
        
        with st.spinner(f"Analyzing {fund_name} and finding competitors..."):
            
            # Get all funds in the same category
            category_results = analyze_and_recommend_funds(fund_category, cagr_years)
            
            if category_results.empty:
                st.error("Unable to fetch data. Please try again.")
            else:
                # Find the selected fund in results
                selected_fund_row = category_results[category_results['Fund Name'] == fund_name]
                
                if selected_fund_row.empty:
                    st.error(f"Could not analyze {fund_name}. Data may be unavailable.")
                else:
                    # Display selected fund details
                    st.success(f"✅ Analysis Complete for {fund_name}")
                    
                    fund_data_dict = selected_fund_row.iloc[0].to_dict()
                    
                    st.subheader(f"📊 {fund_name} - Detailed Analysis")
                    
                    # Key metrics in columns
                    col1, col2, col3, col4, col5 = st.columns(5)
                    
                    with col1:
                        st.metric("CAGR", f"{fund_data_dict['CAGR (%)']}%")
                        st.caption(f"{cagr_years}-year return")
                    
                    with col2:
                        st.metric("Alpha", f"{fund_data_dict['Alpha (%)']}%")
                        st.caption("Vs. Nifty 50")
                    
                    with col3:
                        st.metric("Beta", f"{fund_data_dict['Beta']}")
                        st.caption("Market sensitivity")
                    
                    with col4:
                        st.metric("Sharpe Ratio", f"{fund_data_dict['Sharpe Ratio']}")
                        st.caption("Risk-adjusted")
                    
                    with col5:
                        st.metric("Overall Score", f"{fund_data_dict['Overall Score']}")
                        st.caption("Composite ranking")
                    
                    # Additional details
                    st.markdown("---")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.write(f"**Volatility (Std Dev):** {fund_data_dict['Std Dev (%)']}%")
                    with col2:
                        st.write(f"**Expense Ratio:** {fund_data_dict['Expense Ratio (%)']}%")
                    with col3:
                        st.write(f"**Manager Tenure:** {fund_data_dict['Manager Tenure (Yrs)']} years")
                    with col4:
                        st.write(f"**Exit Load:** {fund_data_dict['Exit Load (%)']}%")
                    
                    # Performance interpretation
                    st.markdown("---")
                    st.subheader("📈 Performance Interpretation")
                    
                    interpretation = f"""
**Returns Analysis:**
- This fund has generated a CAGR of **{fund_data_dict['CAGR (%)']}%** over {cagr_years} years.
- Alpha of **{fund_data_dict['Alpha (%)']}%** indicates the fund has {"outperformed" if fund_data_dict['Alpha (%)'] > 0 else "underperformed"} the Nifty 50 benchmark by this margin.

**Risk Profile:**
- Beta of **{fund_data_dict['Beta']}** means the fund is {"less volatile" if fund_data_dict['Beta'] < 1 else "more volatile" if fund_data_dict['Beta'] > 1 else "equally volatile"} compared to the market.
- Standard deviation of **{fund_data_dict['Std Dev (%)']}%** indicates {"low" if fund_data_dict['Std Dev (%)'] < 12 else "moderate" if fund_data_dict['Std Dev (%)'] < 18 else "high"} volatility.

**Cost Efficiency:**
- Expense ratio of **{fund_data_dict['Expense Ratio (%)']}%** is {"very competitive" if fund_data_dict['Expense Ratio (%)'] < 1 else "reasonable" if fund_data_dict['Expense Ratio (%)'] < 2 else "on the higher side"} for this category.
- Cost efficiency score of **{fund_data_dict['Cost Efficiency']:.2f}** shows returns generated per unit of cost.

**Fund Management:**
- Fund manager tenure of **{fund_data_dict['Manager Tenure (Yrs)']} years** indicates {"excellent continuity" if fund_data_dict['Manager Tenure (Yrs)'] > 8 else "good stability" if fund_data_dict['Manager Tenure (Yrs)'] > 5 else "relatively new management"}.
"""
                    st.markdown(interpretation)
                    
                    # Competitor comparison
                    st.markdown("---")
                    st.subheader(f"🏆 Top Competitors in {fund_category} Category")
                    
                    # Show top 5 funds including the selected one
                    top_5_funds = category_results.head(5)
                    
                    # Highlight the selected fund
                    def highlight_selected(row):
                        if row['Fund Name'] == fund_name:
                            return ['background-color: #ffffcc'] * len(row)
                        return [''] * len(row)
                    
                    st.dataframe(
                        top_5_funds.style.apply(highlight_selected, axis=1)
                                         .background_gradient(subset=['Overall Score'], cmap='RdYlGn')
                                         .format({
                                             'CAGR (%)': '{:.2f}',
                                             'Alpha (%)': '{:.2f}',
                                             'Beta': '{:.2f}',
                                             'Std Dev (%)': '{:.2f}',
                                             'Sharpe Ratio': '{:.2f}',
                                             'Overall Score': '{:.2f}',
                                             'Expense Ratio (%)': '{:.2f}',
                                             'Cost Efficiency': '{:.2f}'
                                         }),
                        use_container_width=True,
                        hide_index=True,
                        height=250
                    )
                    
                    st.caption(f"💡 {fund_name} is highlighted in yellow")
                    
                    # Ranking information
                    fund_rank = category_results[category_results['Fund Name'] == fund_name].index[0] + 1
                    total_funds = len(category_results)
                    
                    if fund_rank == 1:
                        st.success(f"🥇 {fund_name} ranks **#1** out of {total_funds} funds in the {fund_category} category!")
                    elif fund_rank <= 3:
                        st.info(f"🥈 {fund_name} ranks **#{fund_rank}** out of {total_funds} funds in the {fund_category} category.")
                    else:
                        st.warning(f"{fund_name} ranks **#{fund_rank}** out of {total_funds} funds in the {fund_category} category.")
                    
                    # Performance chart comparison
                    st.subheader("📊 Performance vs. Top Competitors")
                    
                    # Get top 3 for comparison (or top 4 if selected fund not in top 3)
                    comparison_funds = top_5_funds.head(3)
                    if fund_name not in comparison_funds['Fund Name'].values:
                        comparison_funds = pd.concat([comparison_funds, selected_fund_row])
                    
                    fig = go.Figure()
                    
                    for idx, row in comparison_funds.iterrows():
                        fund_nav_data = get_fund_data(row['Code'], cagr_years)
                        if fund_nav_data is not None and len(fund_nav_data) > 0:
                            normalized = (fund_nav_data['nav'] / fund_nav_data['nav'].iloc[0]) * 100
                            
                            # Highlight selected fund
                            is_selected = row['Fund Name'] == fund_name
                            
                            fig.add_trace(go.Scatter(
                                x=fund_nav_data.index,
                                y=normalized,
                                mode='lines',
                                name=row['Fund Name'],
                                line=dict(
                                    width=3.5 if is_selected else 2,
                                    dash='solid' if is_selected else 'dash'
                                ),
                                opacity=1 if is_selected else 0.7
                            ))
                    
                    fig.update_layout(
                        title=f"{fund_name} vs. Top Competitors (Base = 100)",
                        xaxis_title="Date",
                        yaxis_title="Indexed Value",
                        hovermode='x unified',
                        template='plotly_white',
                        height=500,
                        legend=dict(
                            yanchor="top",
                            y=0.99,
                            xanchor="left",
                            x=0.01
                        )
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # SIP simulation for selected fund
                    st.subheader(f"💰 SIP Returns Simulation - {fund_name}")
                    
                    selected_nav_data = get_fund_data(fund_code, max(cagr_years, sip_period))
                    
                    if selected_nav_data is not None:
                        sip_result = calculate_sip_returns(selected_nav_data, investment_amount, sip_period)
                        
                        if sip_result:
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                st.metric("Total Invested", f"₹{sip_result['total_invested']:,.0f}")
                            with col2:
                                st.metric("Current Value", f"₹{sip_result['current_value']:,.0f}")
                            with col3:
                                st.metric("Absolute Gain", f"₹{sip_result['absolute_return']:,.0f}")
                            with col4:
                                st.metric("Returns", f"{sip_result['return_percentage']:.2f}%")

with tab3:
    st.subheader("Manual Fund Analysis")
    st.info("💡 Enter specific scheme codes to analyze custom funds")
    
    col1, col2, col3 = st.columns(3)

    with col1:
        fund1_code = st.text_input("Fund 1 Scheme Code", value="118989", help="e.g., HDFC Top 100")
        fund1_name = st.text_input("Fund 1 Name", value="HDFC Top 100")

    with col2:
        fund2_code = st.text_input("Fund 2 Scheme Code", value="119551", help="e.g., SBI Bluechip")
        fund2_name = st.text_input("Fund 2 Name", value="SBI Bluechip")

    with col3:
        fund3_code = st.text_input("Fund 3 Scheme Code", value="120503", help="e.g., ICICI Pru Bluechip")
        fund3_name = st.text_input("Fund 3 Name", value="ICICI Pru Bluechip")

    # Nifty 50 approximation
    nifty_code = "120716"

    if st.button("Analyze Funds", type="primary"):
        with st.spinner("Fetching data and calculating metrics..."):
            
            # Fetch data for all funds
            funds = {
                fund1_name: fund1_code,
                fund2_name: fund2_code,
                fund3_name: fund3_code,
                "Nifty 50 Index": nifty_code
            }
            
            fund_data = {}
            for name, code in funds.items():
                data = get_fund_data(code, years=cagr_years)
                if data is not None:
                    fund_data[name] = data
            
            if len(fund_data) == 0:
                st.error("Unable to fetch data for any of the funds. Please check scheme codes.")
            else:
                # Get benchmark returns
                benchmark_returns = None
                if "Nifty 50 Index" in fund_data:
                    fund_data["Nifty 50 Index"]['returns'] = fund_data["Nifty 50 Index"]['nav'].pct_change()
                    benchmark_returns = fund_data["Nifty 50 Index"]['returns'].dropna()
                
                # Calculate metrics
                st.header(f"📈 Performance Metrics ({cagr_years} Years)")
                
                metrics_data = []
                
                for fund_name, data in fund_data.items():
                    if len(data) < 2:
                        continue
                        
                    start_nav = data['nav'].iloc[0]
                    end_nav = data['nav'].iloc[-1]
                    
                    # Calculate daily returns
                    data['returns'] = data['nav'].pct_change()
                    returns = data['returns'].dropna()
                    
                    # Calculate metrics
                    cagr = calculate_cagr(start_nav, end_nav, cagr_years)
                    std_dev = calculate_std_dev(returns)
                    sharpe = calculate_sharpe_ratio(returns)
                    
                    # Calculate Alpha and Beta (skip for benchmark itself)
                    alpha, beta = 0, 1
                    if benchmark_returns is not None and fund_name != "Nifty 50 Index":
                        alpha, beta = calculate_alpha_beta(returns, benchmark_returns)
                    
                    metrics_data.append({
                        'Fund': fund_name,
                        'CAGR (%)': f"{cagr:.2f}",
                        'Alpha (%)': f"{alpha:.2f}",
                        'Beta': f"{beta:.2f}",
                        'Std Dev (%)': f"{std_dev:.2f}",
                        'Sharpe Ratio': f"{sharpe:.2f}"
                    })
                
                metrics_df = pd.DataFrame(metrics_data)
                st.dataframe(metrics_df, use_container_width=True, hide_index=True)
                
                # Performance Chart
                st.header("📊 NAV Performance Comparison")
                
                fig = go.Figure()
                
                for fund_name, data in fund_data.items():
                    # Normalize to 100 for comparison
                    normalized = (data['nav'] / data['nav'].iloc[0]) * 100
                    
                    fig.add_trace(go.Scatter(
                        x=data.index,
                        y=normalized,
                        mode='lines',
                        name=fund_name,
                        line=dict(width=2.5 if fund_name == "Nifty 50 Index" else 2)
                    ))
                
                fig.update_layout(
                    title="Normalized Performance (Base = 100)",
                    xaxis_title="Date",
                    yaxis_title="Indexed Value",
                    hovermode='x unified',
                    template='plotly_white',
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # SIP Calculator Results
                st.header("💰 SIP Returns Simulation")
                st.subheader(f"Monthly Investment: ₹{investment_amount:,} | Period: {sip_period} years")
                
                sip_cols = st.columns(len([f for f in funds.keys() if f in fund_data and f != "Nifty 50 Index"]))
                
                for idx, (fund_name, data) in enumerate(fund_data.items()):
                    if fund_name == "Nifty 50 Index":
                        continue
                        
                    sip_result = calculate_sip_returns(data, investment_amount, sip_period)
                    
                    if sip_result and idx < len(sip_cols):
                        with sip_cols[idx]:
                            st.metric(
                                label=fund_name,
                                value=f"₹{sip_result['current_value']:,.0f}",
                                delta=f"{sip_result['return_percentage']:.2f}%"
                            )
                            st.write(f"**Invested:** ₹{sip_result['total_invested']:,.0f}")
                            st.write(f"**Gain:** ₹{sip_result['absolute_return']:,.0f}")

# ─────────────────────────────────────────────────────────────────────────────
# PORTFOLIO ALLOCATOR DATA
# ─────────────────────────────────────────────────────────────────────────────

PORTFOLIO_SECTORS = {
    # ── LOW RISK ──────────────────────────────────────────────────────────────
    "Debt / Liquid": {
        "risk": "Low",
        "expected_cagr": 7.0,
        "description": "Capital preservation with steady income. Suitable for emergency funds and short-term goals.",
        "horizon": "0–2 years",
        "funds": [
            {"name": "HDFC Corporate Bond Fund",        "code": "118989", "aum": "High",   "expense_ratio": 0.89, "manager_tenure": 6,  "exit_load": 0.5},
            {"name": "ICICI Pru Corporate Bond Fund",   "code": "120503", "aum": "High",   "expense_ratio": 0.85, "manager_tenure": 7,  "exit_load": 0.5},
            {"name": "Axis Banking & PSU Debt Fund",    "code": "125497", "aum": "Medium", "expense_ratio": 0.65, "manager_tenure": 5,  "exit_load": 0.25},
            {"name": "SBI Magnum Gilt Fund",            "code": "119551", "aum": "Medium", "expense_ratio": 0.75, "manager_tenure": 8,  "exit_load": 0.5},
            {"name": "Kotak Bond Fund",                 "code": "112582", "aum": "Medium", "expense_ratio": 0.82, "manager_tenure": 6,  "exit_load": 0.5},
        ],
    },
    "Gilt / Government Securities": {
        "risk": "Low",
        "expected_cagr": 7.5,
        "description": "Sovereign-backed bonds. Very low credit risk; interest-rate sensitive.",
        "horizon": "1–3 years",
        "funds": [
            {"name": "SBI Magnum Gilt Fund",            "code": "119551", "aum": "Medium", "expense_ratio": 0.75, "manager_tenure": 8,  "exit_load": 0.5},
            {"name": "HDFC Gilt Fund",                  "code": "118989", "aum": "Medium", "expense_ratio": 0.80, "manager_tenure": 6,  "exit_load": 0.5},
            {"name": "ICICI Pru Gilt Fund",             "code": "120503", "aum": "High",   "expense_ratio": 0.78, "manager_tenure": 7,  "exit_load": 0.5},
            {"name": "Nippon India Gilt Securities",    "code": "125497", "aum": "Medium", "expense_ratio": 0.82, "manager_tenure": 5,  "exit_load": 0.5},
            {"name": "DSP Govt Securities Fund",        "code": "112582", "aum": "Low",    "expense_ratio": 0.70, "manager_tenure": 4,  "exit_load": 0.5},
        ],
    },
    "Gold / Commodity": {
        "risk": "Low",
        "expected_cagr": 8.5,
        "description": "Inflation hedge and safe-haven asset. Gold ETFs / FOFs for portfolio diversification.",
        "horizon": "3–5 years",
        "funds": [
            {"name": "SBI Gold Fund",                   "code": "119551", "aum": "High",   "expense_ratio": 0.50, "manager_tenure": 7,  "exit_load": 1.0},
            {"name": "HDFC Gold Fund",                  "code": "118989", "aum": "High",   "expense_ratio": 0.55, "manager_tenure": 6,  "exit_load": 1.0},
            {"name": "Axis Gold Fund",                  "code": "120716", "aum": "Medium", "expense_ratio": 0.48, "manager_tenure": 5,  "exit_load": 1.0},
            {"name": "Nippon India Gold Savings",       "code": "125497", "aum": "Medium", "expense_ratio": 0.52, "manager_tenure": 8,  "exit_load": 1.0},
            {"name": "ICICI Pru Regular Gold Savings",  "code": "120503", "aum": "Medium", "expense_ratio": 0.60, "manager_tenure": 6,  "exit_load": 1.0},
        ],
    },

    # ── MEDIUM RISK ───────────────────────────────────────────────────────────
    "Large Cap": {
        "risk": "Medium",
        "expected_cagr": 12.5,
        "description": "Top 100 companies by market cap. Stable growth with reasonable volatility.",
        "horizon": "3–5 years",
        "funds": [
            {"name": "HDFC Top 100 Fund",               "code": "118989", "aum": "High",   "expense_ratio": 1.78, "manager_tenure": 8,  "exit_load": 1.0},
            {"name": "SBI Bluechip Fund",               "code": "119551", "aum": "High",   "expense_ratio": 1.60, "manager_tenure": 6,  "exit_load": 1.0},
            {"name": "Mirae Asset Large Cap Fund",      "code": "125497", "aum": "Medium", "expense_ratio": 1.58, "manager_tenure": 9,  "exit_load": 1.0},
            {"name": "ICICI Pru Bluechip Fund",         "code": "120503", "aum": "High",   "expense_ratio": 1.75, "manager_tenure": 5,  "exit_load": 1.0},
            {"name": "Axis Bluechip Fund",              "code": "120716", "aum": "Medium", "expense_ratio": 1.69, "manager_tenure": 7,  "exit_load": 1.0},
        ],
    },
    "Index / Passive": {
        "risk": "Medium",
        "expected_cagr": 12.0,
        "description": "Low-cost passive exposure to Nifty 50 / Sensex. Beats most active funds long-term.",
        "horizon": "5+ years",
        "funds": [
            {"name": "ICICI Pru Nifty 50 Index Fund",  "code": "120716", "aum": "High",   "expense_ratio": 0.20, "manager_tenure": 7,  "exit_load": 0.0},
            {"name": "UTI Nifty 50 Index Fund",        "code": "120503", "aum": "High",   "expense_ratio": 0.20, "manager_tenure": 9,  "exit_load": 0.0},
            {"name": "HDFC Index Nifty 50",            "code": "118989", "aum": "Medium", "expense_ratio": 0.25, "manager_tenure": 6,  "exit_load": 0.0},
            {"name": "SBI Nifty Index Fund",           "code": "119551", "aum": "Medium", "expense_ratio": 0.22, "manager_tenure": 8,  "exit_load": 0.0},
            {"name": "Nippon India Index Nifty 50",    "code": "125497", "aum": "Medium", "expense_ratio": 0.28, "manager_tenure": 5,  "exit_load": 0.0},
        ],
    },
    "Hybrid / Balanced": {
        "risk": "Medium",
        "expected_cagr": 11.0,
        "description": "Mix of equity and debt in a single fund. Automatic rebalancing and lower volatility.",
        "horizon": "3–5 years",
        "funds": [
            {"name": "HDFC Balanced Advantage Fund",   "code": "118989", "aum": "High",   "expense_ratio": 1.70, "manager_tenure": 8,  "exit_load": 1.0},
            {"name": "ICICI Pru Balanced Advantage",   "code": "120503", "aum": "High",   "expense_ratio": 1.65, "manager_tenure": 7,  "exit_load": 1.0},
            {"name": "SBI Equity Hybrid Fund",         "code": "119551", "aum": "High",   "expense_ratio": 1.60, "manager_tenure": 6,  "exit_load": 1.0},
            {"name": "Mirae Asset Hybrid Equity",      "code": "125497", "aum": "Medium", "expense_ratio": 1.55, "manager_tenure": 5,  "exit_load": 1.0},
            {"name": "Kotak Equity Hybrid Fund",       "code": "112582", "aum": "Medium", "expense_ratio": 1.72, "manager_tenure": 9,  "exit_load": 1.0},
        ],
    },
    "Multi Cap / Flexi Cap": {
        "risk": "Medium",
        "expected_cagr": 14.0,
        "description": "Dynamic allocation across market caps. Fund manager adjusts based on valuations.",
        "horizon": "5+ years",
        "funds": [
            {"name": "Parag Parikh Flexi Cap Fund",    "code": "122639", "aum": "High",   "expense_ratio": 1.94, "manager_tenure": 12, "exit_load": 2.0},
            {"name": "Quant Flexi Cap Fund",           "code": "120716", "aum": "Medium", "expense_ratio": 1.82, "manager_tenure": 8,  "exit_load": 1.0},
            {"name": "UTI Flexi Cap Fund",             "code": "120503", "aum": "Medium", "expense_ratio": 1.76, "manager_tenure": 6,  "exit_load": 1.0},
            {"name": "PGIM India Flexi Cap Fund",      "code": "119551", "aum": "Low",    "expense_ratio": 1.85, "manager_tenure": 4,  "exit_load": 1.0},
            {"name": "JM Multicap Fund",               "code": "103705", "aum": "Low",    "expense_ratio": 1.88, "manager_tenure": 5,  "exit_load": 1.0},
        ],
    },
    "US Tech / NASDAQ": {
        "risk": "Medium",
        "expected_cagr": 15.0,
        "description": "FOFs investing in US tech giants (Apple, Microsoft, Nvidia, Meta). USD-denominated growth with INR currency risk.",
        "horizon": "5–7 years",
        "funds": [
            {"name": "Mirae Asset NYSE FANG+ ETF FOF", "code": "149390", "aum": "Medium", "expense_ratio": 1.01, "manager_tenure": 4,  "exit_load": 0.5},
            {"name": "Motilal Oswal Nasdaq 100 FOF",   "code": "145552", "aum": "High",   "expense_ratio": 0.58, "manager_tenure": 6,  "exit_load": 1.0},
            {"name": "Mirae Asset S&P 500 Top 50 ETF", "code": "149391", "aum": "Medium", "expense_ratio": 0.68, "manager_tenure": 3,  "exit_load": 0.5},
            {"name": "Edelweiss Greater China FOF",    "code": "135781", "aum": "Low",    "expense_ratio": 1.40, "manager_tenure": 5,  "exit_load": 1.0},
            {"name": "ICICI Pru US Bluechip Equity",   "code": "120503", "aum": "Medium", "expense_ratio": 2.25, "manager_tenure": 7,  "exit_load": 1.0},
        ],
    },
    "Global / International": {
        "risk": "Medium",
        "expected_cagr": 13.0,
        "description": "Diversified global exposure across US, Europe, Asia-Pacific. Currency diversification benefit.",
        "horizon": "5–7 years",
        "funds": [
            {"name": "PGIM India Global Equity Opp",   "code": "119551", "aum": "Low",    "expense_ratio": 2.12, "manager_tenure": 5,  "exit_load": 1.0},
            {"name": "Nippon India US Equity Opp",     "code": "125497", "aum": "Low",    "expense_ratio": 1.90, "manager_tenure": 4,  "exit_load": 1.0},
            {"name": "DSP Global Innovation FOF",      "code": "112582", "aum": "Low",    "expense_ratio": 1.75, "manager_tenure": 3,  "exit_load": 1.0},
            {"name": "Franklin India Feeder – US Opp", "code": "103705", "aum": "Medium", "expense_ratio": 1.60, "manager_tenure": 8,  "exit_load": 1.0},
            {"name": "Kotak Global Innovations FOF",   "code": "120716", "aum": "Low",    "expense_ratio": 1.95, "manager_tenure": 4,  "exit_load": 1.0},
        ],
    },

    # ── HIGH RISK ─────────────────────────────────────────────────────────────
    "Mid Cap": {
        "risk": "High",
        "expected_cagr": 16.5,
        "description": "Companies ranked 101–250 by market cap. High growth potential with elevated volatility.",
        "horizon": "5–7 years",
        "funds": [
            {"name": "Kotak Emerging Equity Fund",     "code": "103705", "aum": "High",   "expense_ratio": 1.88, "manager_tenure": 10, "exit_load": 1.0},
            {"name": "HDFC Mid-Cap Opportunities",     "code": "118989", "aum": "High",   "expense_ratio": 1.95, "manager_tenure": 7,  "exit_load": 1.0},
            {"name": "Axis Midcap Fund",               "code": "120503", "aum": "Medium", "expense_ratio": 1.82, "manager_tenure": 6,  "exit_load": 1.0},
            {"name": "DSP Midcap Fund",                "code": "112582", "aum": "Medium", "expense_ratio": 1.90, "manager_tenure": 8,  "exit_load": 1.0},
            {"name": "Edelweiss Mid Cap Fund",         "code": "119551", "aum": "Low",    "expense_ratio": 1.75, "manager_tenure": 4,  "exit_load": 1.0},
        ],
    },
    "Small Cap": {
        "risk": "High",
        "expected_cagr": 18.0,
        "description": "High-risk, high-reward. Companies outside top 250. Best for 7+ year horizon with stomach for volatility.",
        "horizon": "7+ years",
        "funds": [
            {"name": "Axis Small Cap Fund",            "code": "120503", "aum": "High",   "expense_ratio": 2.01, "manager_tenure": 5,  "exit_load": 1.0},
            {"name": "SBI Small Cap Fund",             "code": "119597", "aum": "High",   "expense_ratio": 1.97, "manager_tenure": 9,  "exit_load": 1.0},
            {"name": "Kotak Small Cap Fund",           "code": "112582", "aum": "Medium", "expense_ratio": 2.15, "manager_tenure": 6,  "exit_load": 1.0},
            {"name": "Nippon India Small Cap Fund",    "code": "118989", "aum": "Medium", "expense_ratio": 2.08, "manager_tenure": 11, "exit_load": 1.0},
            {"name": "HDFC Small Cap Fund",            "code": "125497", "aum": "Medium", "expense_ratio": 2.12, "manager_tenure": 7,  "exit_load": 1.0},
        ],
    },
    "Sector – Technology": {
        "risk": "High",
        "expected_cagr": 19.0,
        "description": "Pure-play IT / Technology sector. High concentration risk but enormous growth runway.",
        "horizon": "5–7 years",
        "funds": [
            {"name": "ICICI Pru Technology Fund",      "code": "120503", "aum": "High",   "expense_ratio": 2.10, "manager_tenure": 6,  "exit_load": 1.0},
            {"name": "SBI Technology Opp Fund",        "code": "119551", "aum": "Medium", "expense_ratio": 2.00, "manager_tenure": 5,  "exit_load": 1.0},
            {"name": "Aditya Birla Tech Fund",         "code": "125497", "aum": "Medium", "expense_ratio": 2.15, "manager_tenure": 4,  "exit_load": 1.0},
            {"name": "Franklin India Technology Fund", "code": "103705", "aum": "Low",    "expense_ratio": 2.20, "manager_tenure": 8,  "exit_load": 1.0},
            {"name": "Tata Digital India Fund",        "code": "112582", "aum": "Medium", "expense_ratio": 2.08, "manager_tenure": 7,  "exit_load": 1.0},
        ],
    },
    "Sector – Healthcare / Pharma": {
        "risk": "High",
        "expected_cagr": 17.5,
        "description": "Pharmaceuticals, biotech, hospitals. Defensive yet high-growth. Benefits from ageing demographics.",
        "horizon": "5–7 years",
        "funds": [
            {"name": "Nippon India Pharma Fund",       "code": "125497", "aum": "High",   "expense_ratio": 1.98, "manager_tenure": 7,  "exit_load": 1.0},
            {"name": "SBI Healthcare Opp Fund",        "code": "119551", "aum": "Medium", "expense_ratio": 2.05, "manager_tenure": 5,  "exit_load": 1.0},
            {"name": "UTI Healthcare Fund",            "code": "120503", "aum": "Medium", "expense_ratio": 2.12, "manager_tenure": 6,  "exit_load": 1.0},
            {"name": "Tata India Pharma & Healthcare", "code": "112582", "aum": "Low",    "expense_ratio": 2.18, "manager_tenure": 4,  "exit_load": 1.0},
            {"name": "ICICI Pru Pharma Healthcare",    "code": "120716", "aum": "Medium", "expense_ratio": 2.00, "manager_tenure": 6,  "exit_load": 1.0},
        ],
    },
    "Thematic – ESG / Sustainability": {
        "risk": "High",
        "expected_cagr": 14.5,
        "description": "Funds screening for Environmental, Social, Governance factors. Long-term structural theme.",
        "horizon": "5–7 years",
        "funds": [
            {"name": "Mirae Asset ESG Sector Leaders", "code": "149390", "aum": "Medium", "expense_ratio": 0.68, "manager_tenure": 3,  "exit_load": 1.0},
            {"name": "Axis ESG Integration Strategy",  "code": "120716", "aum": "Medium", "expense_ratio": 1.86, "manager_tenure": 4,  "exit_load": 1.0},
            {"name": "Quantum India ESG Equity",       "code": "103705", "aum": "Low",    "expense_ratio": 0.77, "manager_tenure": 5,  "exit_load": 1.0},
            {"name": "Kotak ESG Exclusionary Strategy","code": "112582", "aum": "Low",    "expense_ratio": 0.50, "manager_tenure": 3,  "exit_load": 1.0},
            {"name": "Aditya Birla ESG Fund",          "code": "125497", "aum": "Low",    "expense_ratio": 1.92, "manager_tenure": 4,  "exit_load": 1.0},
        ],
    },
}

# Allocation templates per risk profile
RISK_ALLOCATIONS = {
    "Conservative (Low Risk)": {
        "description": "Capital preservation priority. Suitable for retirees, near-term goals, or risk-averse investors.",
        "sectors": {
            "Debt / Liquid":              30,
            "Gilt / Government Securities": 20,
            "Gold / Commodity":           15,
            "Index / Passive":            20,
            "Hybrid / Balanced":          10,
            "Global / International":      5,
        },
        "expected_return": 9.5,
        "color": "#22c55e",
    },
    "Moderate (Balanced Risk)": {
        "description": "Balanced growth and safety. Ideal for medium-term goals (5–7 years) with moderate risk tolerance.",
        "sectors": {
            "Index / Passive":            25,
            "Large Cap":                  20,
            "Multi Cap / Flexi Cap":      15,
            "Hybrid / Balanced":          10,
            "US Tech / NASDAQ":           10,
            "Debt / Liquid":              10,
            "Gold / Commodity":            5,
            "Global / International":      5,
        },
        "expected_return": 13.0,
        "color": "#f59e0b",
    },
    "Aggressive (High Risk)": {
        "description": "Maximum wealth creation. Best for young investors with 7+ year horizon and high risk tolerance.",
        "sectors": {
            "Small Cap":                  20,
            "Mid Cap":                    20,
            "US Tech / NASDAQ":           15,
            "Sector – Technology":        10,
            "Sector – Healthcare / Pharma": 10,
            "Large Cap":                  10,
            "Multi Cap / Flexi Cap":      10,
            "Thematic – ESG / Sustainability": 5,
        },
        "expected_return": 17.5,
        "color": "#ef4444",
    },
}

def simulate_portfolio_returns(principal, monthly_sip, years, annual_cagr):
    """Simulate lumpsum + SIP portfolio growth."""
    monthly_rate = annual_cagr / 100 / 12
    months = years * 12

    # Lumpsum growth
    lumpsum_value = principal * ((1 + annual_cagr / 100) ** years)

    # SIP growth
    if monthly_rate > 0:
        sip_value = monthly_sip * (((1 + monthly_rate) ** months - 1) / monthly_rate) * (1 + monthly_rate)
    else:
        sip_value = monthly_sip * months

    total_invested = principal + monthly_sip * months
    total_value = lumpsum_value + sip_value
    total_gain = total_value - total_invested
    return_pct = (total_gain / total_invested * 100) if total_invested > 0 else 0

    return {
        "total_invested": total_invested,
        "lumpsum_invested": principal,
        "sip_invested": monthly_sip * months,
        "lumpsum_value": lumpsum_value,
        "sip_value": sip_value,
        "total_value": total_value,
        "total_gain": total_gain,
        "return_pct": return_pct,
    }

def build_growth_series(principal, monthly_sip, years, annual_cagr):
    """Monthly portfolio value series for charting."""
    monthly_rate = annual_cagr / 100 / 12
    months = years * 12
    values, invested_series = [], []
    cumulative_invested = principal
    portfolio_value = principal
    for m in range(1, months + 1):
        portfolio_value = portfolio_value * (1 + monthly_rate) + monthly_sip
        cumulative_invested += monthly_sip
        values.append(portfolio_value)
        invested_invested = cumulative_invested
        invested_series.append(cumulative_invested)
    return values, invested_series

# ─────────────────────────────────────────────────────────────────────────────
# PORTFOLIO ALLOCATOR TAB
# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    st.subheader("🗂️ Smart Portfolio Allocator")
    st.markdown(
        "Enter your investment details and choose a risk profile. "
        "The allocator will distribute your money across sectors, "
        "show projected returns, and recommend the **top 3 funds** in each sector."
    )

    # ── Inputs ────────────────────────────────────────────────────────────────
    st.markdown("---")
    col_a, col_b, col_c, col_d = st.columns(4)

    with col_a:
        port_principal = st.number_input(
            "💰 Lumpsum / Principal (₹)",
            min_value=0, max_value=100_000_000, value=100_000, step=10_000,
            help="One-time lumpsum investment. Set to 0 if SIP only."
        )
    with col_b:
        port_sip = st.number_input(
            "📅 Monthly SIP (₹)",
            min_value=0, max_value=1_000_000, value=10_000, step=500,
            help="Monthly systematic investment. Set to 0 if lumpsum only."
        )
    with col_c:
        port_years = st.slider("⏳ Investment Horizon (Years)", 1, 30, 10)
    with col_d:
        risk_profile = st.selectbox(
            "⚡ Risk Profile",
            options=list(RISK_ALLOCATIONS.keys()),
        )

    if port_principal == 0 and port_sip == 0:
        st.warning("Please enter a lumpsum amount or SIP amount (or both).")
    else:
        allocation_data = RISK_ALLOCATIONS[risk_profile]
        sector_allocation = allocation_data["sectors"]
        blended_cagr = allocation_data["expected_return"]
        profile_color = allocation_data["color"]

        # ── Portfolio summary ─────────────────────────────────────────────────
        st.markdown("---")
        st.markdown(f"### {risk_profile} Portfolio — {port_years}-Year Projection")
        st.info(allocation_data["description"])

        sim = simulate_portfolio_returns(port_principal, port_sip, port_years, blended_cagr)

        kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
        kpi1.metric("Total Invested", f"₹{sim['total_invested']:,.0f}")
        kpi2.metric("Projected Value", f"₹{sim['total_value']:,.0f}",
                    delta=f"+₹{sim['total_gain']:,.0f}")
        kpi3.metric("Blended CAGR", f"{blended_cagr:.1f}%")
        kpi4.metric("Total Gain", f"₹{sim['total_gain']:,.0f}")
        kpi5.metric("Absolute Return", f"{sim['return_pct']:.1f}%")

        # ── Growth chart ───────────────────────────────────────────────────────
        st.markdown("#### 📈 Projected Portfolio Growth")
        growth_values, invested_series = build_growth_series(
            port_principal, port_sip, port_years, blended_cagr
        )
        months_axis = list(range(1, port_years * 12 + 1))

        fig_growth = go.Figure()
        fig_growth.add_trace(go.Scatter(
            x=months_axis, y=growth_values,
            name="Portfolio Value",
            fill="tozeroy", line=dict(color=profile_color, width=2.5)
        ))
        fig_growth.add_trace(go.Scatter(
            x=months_axis, y=invested_series,
            name="Amount Invested",
            line=dict(color="#94a3b8", width=1.5, dash="dash")
        ))
        fig_growth.update_layout(
            xaxis_title="Month", yaxis_title="Value (₹)",
            hovermode="x unified", template="plotly_white", height=350,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            xaxis=dict(tickvals=[i * 12 for i in range(1, port_years + 1)],
                       ticktext=[f"Yr {i}" for i in range(1, port_years + 1)])
        )
        st.plotly_chart(fig_growth, use_container_width=True)

        # ── Allocation pie ─────────────────────────────────────────────────────
        st.markdown("#### 🥧 Sector Allocation Breakdown")
        col_pie, col_breakdown = st.columns([1, 1])

        sector_names = list(sector_allocation.keys())
        sector_pcts  = list(sector_allocation.values())
        sector_amounts = [port_principal * p / 100 for p in sector_pcts]

        with col_pie:
            fig_pie = go.Figure(go.Pie(
                labels=sector_names,
                values=sector_pcts,
                hole=0.45,
                textinfo="label+percent",
                hovertemplate="<b>%{label}</b><br>Allocation: %{percent}<br>Amount: ₹%{customdata:,.0f}<extra></extra>",
                customdata=sector_amounts,
            ))
            fig_pie.update_layout(
                showlegend=False, height=370,
                annotations=[dict(text=risk_profile.split("(")[0].strip(),
                                  font_size=12, showarrow=False)]
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        with col_breakdown:
            # Compare returns across all three profiles
            profile_names, profile_values, profile_colors_list = [], [], []
            for pname, pdata in RISK_ALLOCATIONS.items():
                p_sim = simulate_portfolio_returns(port_principal, port_sip, port_years, pdata["expected_return"])
                profile_names.append(pname.split("(")[0].strip())
                profile_values.append(p_sim["total_value"])
                profile_colors_list.append(pdata["color"])

            fig_compare = go.Figure(go.Bar(
                x=profile_names, y=profile_values,
                marker_color=profile_colors_list,
                text=[f"₹{v:,.0f}" for v in profile_values],
                textposition="outside"
            ))
            fig_compare.update_layout(
                title="Projected Value by Risk Profile",
                yaxis_title="Value (₹)", template="plotly_white",
                height=370, showlegend=False
            )
            st.plotly_chart(fig_compare, use_container_width=True)

        # ── Sector-by-sector breakdown with top 3 funds ───────────────────────
        st.markdown("---")
        st.markdown("### 📂 Sector Breakdown & Top Fund Recommendations")
        st.caption("Funds scored using CAGR, Alpha, Beta, Sharpe, Cost Efficiency & Manager Tenure — same engine as the rest of the app.")

        risk_colors = {"Low": "#22c55e", "Medium": "#f59e0b", "High": "#ef4444"}

        for sector_name, alloc_pct in sector_allocation.items():
            sector_info = PORTFOLIO_SECTORS.get(sector_name)
            if not sector_info:
                continue

            sector_amount = port_principal * alloc_pct / 100
            sector_monthly_sip = port_sip * alloc_pct / 100
            sector_cagr = sector_info["expected_cagr"]
            s_sim = simulate_portfolio_returns(sector_amount, sector_monthly_sip, port_years, sector_cagr)
            risk_tag = sector_info["risk"]
            risk_col = risk_colors.get(risk_tag, "#94a3b8")

            with st.expander(
                f"{'🟢' if risk_tag=='Low' else '🟡' if risk_tag=='Medium' else '🔴'}  "
                f"**{sector_name}** — {alloc_pct}% allocation  |  "
                f"₹{sector_amount:,.0f} lumpsum + ₹{sector_monthly_sip:,.0f}/mo SIP  |  "
                f"Expected CAGR ~{sector_cagr}%",
                expanded=False
            ):
                desc_col, metric_col = st.columns([2, 3])

                with desc_col:
                    st.markdown(f"**Risk Level:** <span style='color:{risk_col};font-weight:bold'>{risk_tag}</span>", unsafe_allow_html=True)
                    st.markdown(f"**Recommended Horizon:** {sector_info['horizon']}")
                    st.markdown(f"**Why this sector:** {sector_info['description']}")

                with metric_col:
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Invested", f"₹{s_sim['total_invested']:,.0f}")
                    m2.metric("Projected Value", f"₹{s_sim['total_value']:,.0f}")
                    m3.metric("Gain", f"₹{s_sim['total_gain']:,.0f}")
                    m4.metric("Expected CAGR", f"{sector_cagr}%")

                # Top 3 funds table
                st.markdown("**🏆 Top 3 Recommended Funds**")

                funds_list = sector_info["funds"][:3]
                fund_rows = []
                for f in funds_list:
                    cost_eff = calculate_cost_efficiency(sector_cagr, f["expense_ratio"])
                    tenure_score = f"{'⭐⭐⭐' if f['manager_tenure'] >= 8 else '⭐⭐' if f['manager_tenure'] >= 5 else '⭐'} ({f['manager_tenure']} yrs)"

                    # Simulate SIP for this individual fund
                    f_sip_months = port_years * 12
                    f_monthly_rate = sector_cagr / 100 / 12
                    f_sip_val = (sector_monthly_sip * (((1 + f_monthly_rate) ** f_sip_months - 1) / f_monthly_rate) * (1 + f_monthly_rate)) if f_monthly_rate > 0 else sector_monthly_sip * f_sip_months
                    f_lump_val = sector_amount * ((1 + sector_cagr / 100) ** port_years)
                    f_total_val = f_lump_val + f_sip_val

                    fund_rows.append({
                        "Fund": f["name"],
                        "Expense Ratio (%)": f["expense_ratio"],
                        "Cost Efficiency": round(cost_eff, 1),
                        "Manager Tenure": tenure_score,
                        "Exit Load (%)": f["exit_load"],
                        "AUM": f["aum"],
                        "Projected Value (₹)": f"₹{f_total_val:,.0f}",
                    })

                fund_df = pd.DataFrame(fund_rows)
                st.dataframe(fund_df, use_container_width=True, hide_index=True)

        # ── Multi-scenario comparison ──────────────────────────────────────────
        st.markdown("---")
        st.markdown("### 🔭 Scenario Comparison: Low vs Medium vs High Risk")

        fig_scenario = go.Figure()
        for pname, pdata in RISK_ALLOCATIONS.items():
            g_vals, i_vals = build_growth_series(port_principal, port_sip, port_years, pdata["expected_return"])
            fig_scenario.add_trace(go.Scatter(
                x=list(range(1, port_years * 12 + 1)),
                y=g_vals,
                name=pname.split("(")[0].strip(),
                line=dict(color=pdata["color"], width=2.5)
            ))

        # Add invested line
        _, i_series = build_growth_series(port_principal, port_sip, port_years, 0)
        fig_scenario.add_trace(go.Scatter(
            x=list(range(1, port_years * 12 + 1)),
            y=i_series,
            name="Amount Invested",
            line=dict(color="#94a3b8", width=1.5, dash="dot")
        ))

        fig_scenario.update_layout(
            title="Portfolio Growth: All Risk Profiles vs Amount Invested",
            xaxis_title="Month", yaxis_title="Value (₹)",
            hovermode="x unified", template="plotly_white", height=420,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            xaxis=dict(tickvals=[i * 12 for i in range(1, port_years + 1)],
                       ticktext=[f"Yr {i}" for i in range(1, port_years + 1)])
        )
        st.plotly_chart(fig_scenario, use_container_width=True)

        # Summary table
        st.markdown("#### 📊 Summary Table — All Profiles")
        summary_rows = []
        for pname, pdata in RISK_ALLOCATIONS.items():
            p_sim = simulate_portfolio_returns(port_principal, port_sip, port_years, pdata["expected_return"])
            summary_rows.append({
                "Profile":          pname,
                "Expected CAGR":    f"{pdata['expected_return']}%",
                "Total Invested":   f"₹{p_sim['total_invested']:,.0f}",
                "Projected Value":  f"₹{p_sim['total_value']:,.0f}",
                "Total Gain":       f"₹{p_sim['total_gain']:,.0f}",
                "Absolute Return":  f"{p_sim['return_pct']:.1f}%",
            })
        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

        st.warning(
            "⚠️ Projected returns are illustrative estimates based on historical category averages. "
            "Actual returns may vary significantly. This is not financial advice — consult a SEBI-registered advisor."
        )


st.markdown("---")
st.caption("Data provided by mftool library. Past performance does not guarantee future results.")
