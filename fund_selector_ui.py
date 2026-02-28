"""
Enhanced Fund Selector UI
Allows users to select specific funds and input custom amounts per fund
"""

import streamlit as st
import pandas as pd
from datetime import datetime

def render_fund_selector_interface(sector_name, sector_allocation_pct, total_portfolio_value, available_funds):
    """
    Complete UI for selecting funds within a sector
    
    Args:
        sector_name: Name of sector (e.g., "Large Cap")
        sector_allocation_pct: Percentage allocated to this sector
        total_portfolio_value: Total investment (lumpsum + SIP over tenure)
        available_funds: List of fund dictionaries for this sector
    
    Returns:
        List of holding dictionaries with user selections
    """
    
    st.markdown(f"#### 📂 {sector_name} ({sector_allocation_pct:.1f}% allocation)")
    
    sector_total_amount = total_portfolio_value * sector_allocation_pct / 100
    st.caption(f"💰 Sector budget: ₹{sector_total_amount:,.0f}")
    
    # Number of funds to select
    num_funds = st.number_input(
        f"How many {sector_name} funds do you want?",
        min_value=1,
        max_value=min(5, len(available_funds)),
        value=min(2, len(available_funds)),
        key=f"num_funds_{sector_name.replace(' ', '_').replace('/', '_')}"
    )
    
    selected_holdings = []
    
    # Create fund selectors
    for i in range(num_funds):
        with st.expander(f"💼 Fund Selection #{i+1} in {sector_name}", expanded=(i==0)):
            
            # Fund selection dropdown
            fund_options = [f"{f['name']} (Manager: {f.get('manager', 'N/A')})" for f in available_funds]
            
            selected_display = st.selectbox(
                f"Select Fund #{i+1}",
                options=fund_options,
                key=f"fund_select_{sector_name.replace(' ', '_').replace('/', '_')}_{i}"
            )
            
            # Get actual fund object
            selected_index = fund_options.index(selected_display)
            selected_fund = available_funds[selected_index]
            
            # Show fund details
            col_info1, col_info2, col_info3 = st.columns(3)
            col_info1.caption(f"📊 Expense: {selected_fund['expense_ratio']}%")
            col_info2.caption(f"👨‍💼 Tenure: {selected_fund['manager_tenure']} yrs")
            col_info3.caption(f"🏢 AUM: {selected_fund['aum']}")
            
            st.markdown("---")
            
            # Allocation within sector
            col_alloc, col_lump, col_sip = st.columns(3)
            
            with col_alloc:
                fund_pct_of_sector = st.number_input(
                    "% of this sector",
                    min_value=0,
                    max_value=100,
                    value=100 // num_funds if num_funds > 0 else 100,
                    step=5,
                    key=f"fund_pct_{sector_name.replace(' ', '_').replace('/', '_')}_{i}",
                    help=f"What % of {sector_name} allocation goes to this fund?"
                )
            
            # Calculate suggested amounts
            fund_total_amount = sector_total_amount * fund_pct_of_sector / 100
            suggested_lumpsum = int(fund_total_amount * 0.3)  # 30% lumpsum
            suggested_sip = int((fund_total_amount * 0.7) / 12)  # 70% via SIP
            
            with col_lump:
                lumpsum_amount = st.number_input(
                    "Lumpsum Amount (₹)",
                    min_value=0,
                    max_value=int(fund_total_amount),
                    value=suggested_lumpsum,
                    step=1000,
                    key=f"lumpsum_{sector_name.replace(' ', '_').replace('/', '_')}_{i}",
                    help="One-time investment in this fund"
                )
            
            with col_sip:
                monthly_sip = st.number_input(
                    "Monthly SIP (₹)",
                    min_value=0,
                    max_value=100000,
                    value=suggested_sip,
                    step=500,
                    key=f"sip_{sector_name.replace(' ', '_').replace('/', '_')}_{i}",
                    help="Monthly SIP for this fund"
                )
            
            # Summary for this fund
            total_for_fund = lumpsum_amount + (monthly_sip * 12)  # 1 year total
            st.info(f"📊 **Total first year:** ₹{total_for_fund:,} | Lumpsum: ₹{lumpsum_amount:,} + SIP: ₹{monthly_sip:,}/mo")
            
            # Store the holding
            selected_holdings.append({
                'sector': sector_name,
                'fund_name': selected_fund['name'],
                'fund_code': selected_fund['code'],
                'fund_manager': selected_fund.get('manager', 'N/A'),
                'expense_ratio': selected_fund['expense_ratio'],
                'manager_tenure': selected_fund['manager_tenure'],
                'allocation_pct_of_sector': fund_pct_of_sector,
                'lumpsum_amount': lumpsum_amount,
                'monthly_sip': monthly_sip,
                'start_date': datetime.now().strftime('%Y-%m-%d'),
                'entry_nav': 0,  # Will be fetched on first update
                'aum': selected_fund['aum']
            })
    
    return selected_holdings


def render_portfolio_summary(all_holdings):
    """Display summary of all selected holdings"""
    
    st.markdown("### 📊 Your Complete Portfolio Summary")
    
    # Calculate totals
    total_lumpsum = sum(h['lumpsum_amount'] for h in all_holdings)
    total_monthly_sip = sum(h['monthly_sip'] for h in all_holdings)
    total_funds = len(all_holdings)
    
    # Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("🏦 Total Funds", total_funds)
    col2.metric("💰 Total Lumpsum", f"₹{total_lumpsum:,}")
    col3.metric("📅 Total Monthly SIP", f"₹{total_monthly_sip:,}")
    
    # Detailed table
    st.markdown("#### 📋 Holdings Breakdown")
    
    holdings_display = []
    for h in all_holdings:
        holdings_display.append({
            'Sector': h['sector'],
            'Fund Name': h['fund_name'],
            'Manager': h['fund_manager'],
            'Lumpsum': f"₹{h['lumpsum_amount']:,}",
            'Monthly SIP': f"₹{h['monthly_sip']:,}",
            'Expense %': f"{h['expense_ratio']}%",
            '1st Year Total': f"₹{h['lumpsum_amount'] + (h['monthly_sip'] * 12):,}"
        })
    
    holdings_df = pd.DataFrame(holdings_display)
    st.dataframe(holdings_df, use_container_width=True, hide_index=True)
    
    # Sector-wise summary
    st.markdown("#### 🗂️ Sector-wise Investment")
    
    sector_summary = {}
    for h in all_holdings:
        sector = h['sector']
        if sector not in sector_summary:
            sector_summary[sector] = {'lumpsum': 0, 'sip': 0, 'funds': 0}
        sector_summary[sector]['lumpsum'] += h['lumpsum_amount']
        sector_summary[sector]['sip'] += h['monthly_sip']
        sector_summary[sector]['funds'] += 1
    
    sector_display = []
    for sector, data in sector_summary.items():
        sector_display.append({
            'Sector': sector,
            'Funds': data['funds'],
            'Lumpsum': f"₹{data['lumpsum']:,}",
            'Monthly SIP': f"₹{data['sip']:,}",
            'Total 1st Year': f"₹{data['lumpsum'] + (data['sip'] * 12):,}"
        })
    
    sector_df = pd.DataFrame(sector_display)
    st.dataframe(sector_df, use_container_width=True, hide_index=True)
    
    return {
        'total_lumpsum': total_lumpsum,
        'total_sip': total_monthly_sip,
        'total_funds': total_funds
    }


# Example integration code (to be added to main app):
"""
# In Portfolio Allocator tab, after showing recommended allocation:

st.markdown("---")
st.markdown("## 🎯 Step 2: Select Your Funds & Enter Amounts")

st.info("💡 Choose specific funds for each sector and enter your investment amounts")

enable_custom_selection = st.checkbox(
    "✅ I want to customize my fund selection",
    value=True,
    help="Enable this to choose specific funds and amounts"
)

if enable_custom_selection:
    all_holdings = []
    
    # Calculate total portfolio value
    total_investment = port_principal + (port_sip * port_years * 12)
    
    # For each sector in allocation
    for sector_name, alloc_pct in sector_allocation.items():
        if alloc_pct > 0 and sector_name in PORTFOLIO_SECTORS:
            with st.expander(f"🔹 {sector_name} ({alloc_pct:.1f}%)", expanded=True):
                sector_funds = PORTFOLIO_SECTORS[sector_name]['funds']
                
                holdings = render_fund_selector_interface(
                    sector_name=sector_name,
                    sector_allocation_pct=alloc_pct,
                    total_portfolio_value=total_investment,
                    available_funds=sector_funds
                )
                
                all_holdings.extend(holdings)
    
    # Show summary
    if all_holdings:
        st.markdown("---")
        summary = render_portfolio_summary(all_holdings)
        
        # Save button
        st.markdown("---")
        st.markdown("### 💾 Save Your Portfolio")
        
        col_save1, col_save2 = st.columns([3, 1])
        
        with col_save1:
            portfolio_name = st.text_input(
                "Portfolio Name",
                value=f"My Portfolio {datetime.now().strftime('%Y-%m-%d')}",
                key='custom_portfolio_name'
            )
        
        with col_save2:
            st.write("")
            st.write("")
            
            if st.button("💾 Save Portfolio", type="primary", use_container_width=True):
                # Save to database with user authentication
                portfolio_data = {
                    'name': portfolio_name,
                    'created_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'investment_params': {
                        'principal': port_principal,
                        'monthly_sip': port_sip,
                        'years': port_years,
                        'risk_profile': risk_profile
                    },
                    'sector_allocations': sector_allocation,
                    'holdings': all_holdings,
                    'snapshots': [],
                    'alerts': []
                }
                
                result = save_portfolio_to_db(st.session_state.user_id, portfolio_data)
                
                if result['success']:
                    st.success(f"✅ {result['message']} Portfolio saved with {len(all_holdings)} funds!")
                    st.balloons()
                else:
                    st.error(result['message'])

else:
    st.warning("Enable custom selection to choose specific funds and amounts")
"""
