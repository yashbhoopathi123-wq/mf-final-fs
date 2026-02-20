"""
Portfolio Saving & Tracking Module
Handles portfolio persistence, alerts, and rebalancing suggestions
"""

import streamlit as st
import hashlib
from datetime import datetime
import pandas as pd

def init_portfolio_storage():
    """Initialize session state for portfolio tracking"""
    if 'saved_portfolios' not in st.session_state:
        st.session_state.saved_portfolios = {}
    if 'active_portfolio_id' not in st.session_state:
        st.session_state.active_portfolio_id = None

def save_portfolio(portfolio_name, allocation_data, investment_params):
    """
    Save portfolio allocation with all parameters
    
    Args:
        portfolio_name: User-given name
        allocation_data: Dict of {sector: allocation_%}
        investment_params: Dict with principal, monthly_sip, years, risk_profile
    
    Returns:
        portfolio_id: Unique ID for this portfolio
    """
    portfolio_id = hashlib.md5(
        f"{portfolio_name}{datetime.now().isoformat()}".encode()
    ).hexdigest()[:8]
    
    portfolio = {
        'id': portfolio_id,
        'name': portfolio_name,
        'created_date': datetime.now().isoformat(),
        'last_reviewed': datetime.now().isoformat(),
        'investment_params': investment_params,
        'allocation': allocation_data,
        'initial_value': investment_params['principal'],
        'snapshots': [{
            'date': datetime.now().isoformat(),
            'total_value': investment_params['principal'],
            'note': 'Portfolio created'
        }],
        'alerts': [],
        'manual_notes': ''
    }
    
    st.session_state.saved_portfolios[portfolio_id] = portfolio
    st.session_state.active_portfolio_id = portfolio_id
    
    return portfolio_id

def load_portfolio(portfolio_id):
    """Load a saved portfolio"""
    if portfolio_id in st.session_state.saved_portfolios:
        return st.session_state.saved_portfolios[portfolio_id]
    return None

def delete_portfolio(portfolio_id):
    """Delete a portfolio"""
    if portfolio_id in st.session_state.saved_portfolios:
        del st.session_state.saved_portfolios[portfolio_id]
        if st.session_state.active_portfolio_id == portfolio_id:
            st.session_state.active_portfolio_id = None

def export_portfolio_json(portfolio_id):
    """Export portfolio as JSON for backup"""
    import json
    portfolio = load_portfolio(portfolio_id)
    if portfolio:
        return json.dumps(portfolio, indent=2)
    return None

def import_portfolio_json(json_string):
    """Import portfolio from JSON"""
    import json
    try:
        portfolio = json.loads(json_string)
        portfolio_id = portfolio['id']
        st.session_state.saved_portfolios[portfolio_id] = portfolio
        return portfolio_id
    except:
        return None

def render_portfolio_manager():
    """Render portfolio management UI in sidebar"""
    st.sidebar.markdown("---")
    st.sidebar.subheader("💼 My Portfolios")
    
    saved = st.session_state.saved_portfolios
    
    if not saved:
        st.sidebar.info("No saved portfolios yet. Create one in the Allocator tab!")
        return None
    
    # Portfolio selector
    portfolio_names = {pid: p['name'] for pid, p in saved.items()}
    selected_name = st.sidebar.selectbox(
        "Select Portfolio",
        options=list(portfolio_names.values()),
        key='portfolio_selector'
    )
    
    # Find ID from name
    selected_id = [pid for pid, name in portfolio_names.items() if name == selected_name][0]
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("📂 Load", key='load_portfolio'):
            st.session_state.active_portfolio_id = selected_id
            st.success("Portfolio loaded!")
    
    with col2:
        if st.button("🗑️ Delete", key='delete_portfolio'):
            delete_portfolio(selected_id)
            st.rerun()
    
    # Show quick stats
    if st.session_state.active_portfolio_id:
        active = saved[st.session_state.active_portfolio_id]
        st.sidebar.markdown(f"**Active:** {active['name']}")
        st.sidebar.caption(f"Created: {active['created_date'][:10]}")
        
        # Export button
        if st.sidebar.button("💾 Export JSON"):
            json_str = export_portfolio_json(st.session_state.active_portfolio_id)
            st.sidebar.download_button(
                "Download Backup",
                json_str,
                file_name=f"{active['name']}.json",
                mime="application/json"
            )
    
    return selected_id
