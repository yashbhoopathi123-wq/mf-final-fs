"""
ENHANCED PORTFOLIO TRACKER - Fund Selection & Daily Updates
This module adds fund selection, investment tracking, and daily rebalancing alerts
"""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import json
import hashlib

# ═════════════════════════════════════════════════════════════════════════════
# ENHANCED PORTFOLIO STORAGE WITH FUND SELECTION
# ═════════════════════════════════════════════════════════════════════════════

def init_portfolio_storage():
    """Initialize session state for enhanced portfolio tracking"""
    if 'saved_portfolios' not in st.session_state:
        st.session_state.saved_portfolios = {}
    if 'active_portfolio_id' not in st.session_state:
        st.session_state.active_portfolio_id = None
    if 'portfolio_holdings' not in st.session_state:
        st.session_state.portfolio_holdings = {}
    if 'last_daily_update' not in st.session_state:
        st.session_state.last_daily_update = {}

def save_enhanced_portfolio(portfolio_name, sector_allocations, selected_funds_data):
    """
    Save portfolio with specific fund selections and amounts
    
    Args:
        portfolio_name: User-given name
        sector_allocations: Dict of {sector: total_allocation_%}
        selected_funds_data: List of dicts with fund details
            [{
                'sector': 'Large Cap',
                'fund_name': 'HDFC Top 100',
                'fund_code': '118989',
                'allocation_%': 15,
                'lumpsum_amount': 50000,
                'monthly_sip': 5000,
                'start_date': '2026-02-19',
                'current_nav': 523.45
            }, ...]
    
    Returns:
        portfolio_id: Unique ID
    """
    portfolio_id = hashlib.md5(
        f"{portfolio_name}{datetime.now()}".encode()
    ).hexdigest()[:8]
    
    portfolio = {
        'id': portfolio_id,
        'name': portfolio_name,
        'created_date': datetime.now().strftime('%Y-%m-%d %H:%M'),
        'last_reviewed': datetime.now().strftime('%Y-%m-%d'),
        'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M'),
        
        # Sector level allocation
        'sector_allocations': sector_allocations,
        
        # Individual fund holdings with amounts
        'holdings': selected_funds_data,
        
        # Performance tracking
        'snapshots': [{
            'date': datetime.now().strftime('%Y-%m-%d'),
            'total_value': sum(h['lumpsum_amount'] for h in selected_funds_data),
            'note': 'Portfolio created'
        }],
        
        # Alerts and recommendations
        'alerts': [],
        'rebalancing_suggestions': [],
        
        # User notes
        'manual_notes': ''
    }
    
    st.session_state.saved_portfolios[portfolio_id] = portfolio
    st.session_state.active_portfolio_id = portfolio_id
    
    return portfolio_id

def update_portfolio_daily(portfolio_id, get_fund_data_func):
    """
    Update portfolio with latest NAV data and generate alerts
    
    Args:
        portfolio_id: Portfolio to update
        get_fund_data_func: Function to fetch latest NAV data
    
    Returns:
        Dict with updates and alerts
    """
    if portfolio_id not in st.session_state.saved_portfolios:
        return None
    
    portfolio = st.session_state.saved_portfolios[portfolio_id]
    
    # Check if already updated today
    last_update = portfolio.get('last_updated', '')
    today = datetime.now().strftime('%Y-%m-%d')
    
    if last_update.startswith(today):
        return {
            'status': 'already_updated',
            'message': 'Portfolio already updated today',
            'alerts': portfolio.get('alerts', [])
        }
    
    # Update each holding
    alerts = []
    updated_holdings = []
    total_current_value = 0
    
    for holding in portfolio['holdings']:
        fund_code = holding['fund_code']
        
        # Fetch latest NAV
        nav_data = get_fund_data_func(fund_code, years=0.5)  # Last 6 months
        
        if nav_data is not None and len(nav_data) > 0:
            current_nav = nav_data['nav'].iloc[-1]
            start_nav = holding.get('current_nav', current_nav)
            
            # Calculate current value
            # For lumpsum: units = lumpsum_amount / start_nav
            lumpsum_units = holding['lumpsum_amount'] / start_nav if start_nav > 0 else 0
            lumpsum_value = lumpsum_units * current_nav
            
            # For SIP: Calculate accumulated units (simplified - assumes monthly)
            months_elapsed = max(1, (datetime.now() - datetime.strptime(
                holding.get('start_date', datetime.now().strftime('%Y-%m-%d')), 
                '%Y-%m-%d'
            )).days // 30)
            
            sip_units = 0
            if nav_data is not None:
                # Simplified: Use average NAV for SIP accumulation
                avg_nav = nav_data['nav'].mean()
                sip_units = (holding['monthly_sip'] * months_elapsed) / avg_nav if avg_nav > 0 else 0
            
            sip_value = sip_units * current_nav
            
            total_value = lumpsum_value + sip_value
            total_invested = holding['lumpsum_amount'] + (holding['monthly_sip'] * months_elapsed)
            
            gain_loss = total_value - total_invested
            gain_loss_pct = (gain_loss / total_invested * 100) if total_invested > 0 else 0
            
            # Generate alerts
            if gain_loss_pct > 30:
                alerts.append({
                    'type': 'profit_booking',
                    'severity': 'medium',
                    'fund': holding['fund_name'],
                    'sector': holding['sector'],
                    'gain': gain_loss_pct,
                    'message': f"🎯 {holding['fund_name']}: Up {gain_loss_pct:.1f}%! Consider booking partial profits.",
                    'current_value': total_value,
                    'invested': total_invested
                })
            
            elif gain_loss_pct < -20:
                alerts.append({
                    'type': 'review_needed',
                    'severity': 'high',
                    'fund': holding['fund_name'],
                    'sector': holding['sector'],
                    'loss': gain_loss_pct,
                    'message': f"⚠️ {holding['fund_name']}: Down {gain_loss_pct:.1f}%. Review fundamentals or average down via SIP.",
                    'current_value': total_value,
                    'invested': total_invested
                })
            
            elif -5 < gain_loss_pct < 5:
                alerts.append({
                    'type': 'stable',
                    'severity': 'info',
                    'fund': holding['fund_name'],
                    'sector': holding['sector'],
                    'change': gain_loss_pct,
                    'message': f"✅ {holding['fund_name']}: Stable ({gain_loss_pct:+.1f}%). Stay invested.",
                    'current_value': total_value,
                    'invested': total_invested
                })
            
            # Update holding
            updated_holding = holding.copy()
            updated_holding.update({
                'current_nav': current_nav,
                'current_value': total_value,
                'total_invested': total_invested,
                'gain_loss': gain_loss,
                'gain_loss_pct': gain_loss_pct,
                'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M')
            })
            
            updated_holdings.append(updated_holding)
            total_current_value += total_value
    
    # Update portfolio
    portfolio['holdings'] = updated_holdings
    portfolio['alerts'] = alerts
    portfolio['last_updated'] = datetime.now().strftime('%Y-%m-%d %H:%M')
    
    # Add snapshot
    portfolio['snapshots'].append({
        'date': datetime.now().strftime('%Y-%m-%d'),
        'total_value': total_current_value,
        'note': 'Daily update'
    })
    
    # Generate rebalancing suggestions
    rebalancing = generate_rebalancing_suggestions(portfolio)
    portfolio['rebalancing_suggestions'] = rebalancing
    
    st.session_state.saved_portfolios[portfolio_id] = portfolio
    
    return {
        'status': 'updated',
        'message': f'Portfolio updated successfully',
        'alerts': alerts,
        'current_value': total_current_value,
        'rebalancing': rebalancing
    }

def generate_rebalancing_suggestions(portfolio):
    """Generate smart rebalancing suggestions based on current allocation"""
    suggestions = []
    
    # Calculate current allocation %
    total_value = sum(h.get('current_value', h['lumpsum_amount']) 
                     for h in portfolio['holdings'])
    
    if total_value == 0:
        return suggestions
    
    # Group by sector
    sector_values = {}
    for holding in portfolio['holdings']:
        sector = holding['sector']
        value = holding.get('current_value', holding['lumpsum_amount'])
        sector_values[sector] = sector_values.get(sector, 0) + value
    
    # Compare with target allocation
    target_allocation = portfolio.get('sector_allocations', {})
    
    for sector, target_pct in target_allocation.items():
        current_value = sector_values.get(sector, 0)
        current_pct = (current_value / total_value * 100) if total_value > 0 else 0
        
        deviation = current_pct - target_pct
        
        if abs(deviation) > 5:  # More than 5% deviation
            if deviation > 0:
                suggestions.append({
                    'sector': sector,
                    'action': 'reduce',
                    'current': current_pct,
                    'target': target_pct,
                    'deviation': deviation,
                    'message': f"🔻 {sector}: {current_pct:.1f}% (target {target_pct:.1f}%). Consider reducing by {abs(deviation):.1f}%."
                })
            else:
                suggestions.append({
                    'sector': sector,
                    'action': 'increase',
                    'current': current_pct,
                    'target': target_pct,
                    'deviation': deviation,
                    'message': f"🔺 {sector}: {current_pct:.1f}% (target {target_pct:.1f}%). Consider increasing by {abs(deviation):.1f}%."
                })
    
    return suggestions

def render_fund_selector(sector_name, sector_allocation_pct, total_portfolio_value, available_funds):
    """
    Render UI for selecting funds within a sector
    
    Args:
        sector_name: Name of the sector
        sector_allocation_pct: % of portfolio for this sector
        total_portfolio_value: Total lumpsum + SIP value
        available_funds: List of fund dicts for this sector
    
    Returns:
        List of selected fund holdings
    """
    st.markdown(f"#### 📂 {sector_name} ({sector_allocation_pct}% allocation)")
    
    sector_amount = total_portfolio_value * sector_allocation_pct / 100
    st.caption(f"Total for this sector: ₹{sector_amount:,.0f}")
    
    # How many funds to invest in this sector
    num_funds = st.number_input(
        f"How many funds in {sector_name}?",
        min_value=1,
        max_value=min(5, len(available_funds)),
        value=min(2, len(available_funds)),
        key=f"num_funds_{sector_name}"
    )
    
    selected_holdings = []
    
    for i in range(num_funds):
        with st.expander(f"Fund {i+1} in {sector_name}", expanded=(i==0)):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Fund selection
                fund_options = [f['name'] for f in available_funds]
                selected_fund_name = st.selectbox(
                    "Select Fund",
                    options=fund_options,
                    key=f"fund_select_{sector_name}_{i}"
                )
                
                selected_fund = next(f for f in available_funds if f['name'] == selected_fund_name)
            
            with col2:
                # Allocation within sector
                fund_allocation = st.number_input(
                    "% of sector",
                    min_value=0,
                    max_value=100,
                    value=100 // num_funds,
                    key=f"fund_alloc_{sector_name}_{i}"
                )
            
            # Amount distribution
            fund_sector_amount = sector_amount * fund_allocation / 100
            
            col3, col4 = st.columns(2)
            
            with col3:
                lumpsum = st.number_input(
                    "Lumpsum (₹)",
                    min_value=0,
                    value=int(fund_sector_amount * 0.3),
                    step=1000,
                    key=f"lumpsum_{sector_name}_{i}"
                )
            
            with col4:
                monthly_sip = st.number_input(
                    "Monthly SIP (₹)",
                    min_value=0,
                    value=int(fund_sector_amount * 0.7 / 12),
                    step=500,
                    key=f"sip_{sector_name}_{i}"
                )
            
            # Store holding
            selected_holdings.append({
                'sector': sector_name,
                'fund_name': selected_fund['name'],
                'fund_code': selected_fund['code'],
                'fund_manager': selected_fund.get('manager', 'N/A'),
                'allocation_%': (fund_sector_amount / total_portfolio_value * 100),
                'lumpsum_amount': lumpsum,
                'monthly_sip': monthly_sip,
                'start_date': datetime.now().strftime('%Y-%m-%d'),
                'current_nav': 0  # Will be fetched on first update
            })
    
    return selected_holdings

# Example usage in main app would look like:
"""
# In Portfolio Allocator tab, after showing recommended allocation:

st.markdown("### 🎯 Customize Your Fund Selection")

all_holdings = []
for sector, alloc_pct in sector_allocation.items():
    if sector in PORTFOLIO_SECTORS:
        available_funds = PORTFOLIO_SECTORS[sector]['funds']
        holdings = render_fund_selector(
            sector_name=sector,
            sector_allocation_pct=alloc_pct,
            total_portfolio_value=port_principal + (port_sip * port_years * 12),
            available_funds=available_funds
        )
        all_holdings.extend(holdings)

# Save button
if st.button("💾 Save Portfolio with Selected Funds"):
    portfolio_id = save_enhanced_portfolio(
        portfolio_name=portfolio_name,
        sector_allocations=sector_allocation,
        selected_funds_data=all_holdings
    )
    st.success(f"✅ Saved with {len(all_holdings)} fund selections!")
"""
