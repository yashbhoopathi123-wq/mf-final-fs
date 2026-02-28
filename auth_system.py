"""
User Authentication & Database System for Mutual Fund Analyzer
Handles user accounts, login/signup, and portfolio persistence
"""

import streamlit as st
import sqlite3
import hashlib
import json
from datetime import datetime
import pandas as pd
import os

# ═════════════════════════════════════════════════════════════════════════════
# DATABASE SETUP
# ═════════════════════════════════════════════════════════════════════════════

def init_database():
    """Initialize SQLite database with users and portfolios tables"""
    
    # Create database file in same directory as app
    db_path = 'mutual_fund_app.db'
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Users table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            user_id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            email TEXT,
            created_date TEXT NOT NULL,
            last_login TEXT
        )
    ''')
    
    # Portfolios table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS portfolios (
            portfolio_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            portfolio_name TEXT NOT NULL,
            created_date TEXT NOT NULL,
            last_reviewed TEXT,
            last_updated TEXT,
            investment_params TEXT,
            sector_allocations TEXT,
            holdings TEXT,
            snapshots TEXT,
            alerts TEXT,
            notes TEXT,
            FOREIGN KEY (user_id) REFERENCES users (user_id)
        )
    ''')
    
    conn.commit()
    conn.close()
    
    return db_path

# ═════════════════════════════════════════════════════════════════════════════
# PASSWORD HASHING
# ═════════════════════════════════════════════════════════════════════════════

def hash_password(password):
    """Hash password using SHA-256"""
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(password, password_hash):
    """Verify password against hash"""
    return hash_password(password) == password_hash

# ═════════════════════════════════════════════════════════════════════════════
# USER MANAGEMENT
# ═════════════════════════════════════════════════════════════════════════════

def create_user(username, password, email=None):
    """Create a new user account"""
    try:
        conn = sqlite3.connect('mutual_fund_app.db')
        cursor = conn.cursor()
        
        # Check if username exists
        cursor.execute('SELECT username FROM users WHERE username = ?', (username,))
        if cursor.fetchone():
            conn.close()
            return {'success': False, 'message': 'Username already exists'}
        
        # Insert new user
        password_hash = hash_password(password)
        created_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        cursor.execute('''
            INSERT INTO users (username, password_hash, email, created_date, last_login)
            VALUES (?, ?, ?, ?, ?)
        ''', (username, password_hash, email, created_date, created_date))
        
        user_id = cursor.lastrowid
        
        conn.commit()
        conn.close()
        
        return {'success': True, 'message': 'Account created successfully!', 'user_id': user_id}
    
    except Exception as e:
        return {'success': False, 'message': f'Error: {str(e)}'}

def authenticate_user(username, password):
    """Authenticate user login"""
    try:
        conn = sqlite3.connect('mutual_fund_app.db')
        cursor = conn.cursor()
        
        cursor.execute('SELECT user_id, password_hash FROM users WHERE username = ?', (username,))
        result = cursor.fetchone()
        
        if not result:
            conn.close()
            return {'success': False, 'message': 'Username not found'}
        
        user_id, password_hash = result
        
        if not verify_password(password, password_hash):
            conn.close()
            return {'success': False, 'message': 'Incorrect password'}
        
        # Update last login
        cursor.execute('UPDATE users SET last_login = ? WHERE user_id = ?',
                      (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), user_id))
        conn.commit()
        conn.close()
        
        return {'success': True, 'message': 'Login successful!', 'user_id': user_id, 'username': username}
    
    except Exception as e:
        return {'success': False, 'message': f'Error: {str(e)}'}

# ═════════════════════════════════════════════════════════════════════════════
# PORTFOLIO MANAGEMENT
# ═════════════════════════════════════════════════════════════════════════════

def save_portfolio_to_db(user_id, portfolio_data):
    """Save portfolio to database"""
    try:
        conn = sqlite3.connect('mutual_fund_app.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO portfolios (
                user_id, portfolio_name, created_date, last_reviewed, last_updated,
                investment_params, sector_allocations, holdings, snapshots, alerts, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            user_id,
            portfolio_data['name'],
            portfolio_data['created_date'],
            portfolio_data.get('last_reviewed', portfolio_data['created_date']),
            portfolio_data.get('last_updated', portfolio_data['created_date']),
            json.dumps(portfolio_data.get('investment_params', {})),
            json.dumps(portfolio_data.get('sector_allocations', {})),
            json.dumps(portfolio_data.get('holdings', [])),
            json.dumps(portfolio_data.get('snapshots', [])),
            json.dumps(portfolio_data.get('alerts', [])),
            portfolio_data.get('notes', '')
        ))
        
        portfolio_id = cursor.lastrowid
        
        conn.commit()
        conn.close()
        
        return {'success': True, 'message': 'Portfolio saved!', 'portfolio_id': portfolio_id}
    
    except Exception as e:
        return {'success': False, 'message': f'Error: {str(e)}'}

def load_user_portfolios(user_id):
    """Load all portfolios for a user"""
    try:
        conn = sqlite3.connect('mutual_fund_app.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT portfolio_id, portfolio_name, created_date, last_updated,
                   investment_params, sector_allocations, holdings, snapshots, alerts
            FROM portfolios
            WHERE user_id = ?
            ORDER BY last_updated DESC
        ''', (user_id,))
        
        portfolios = []
        for row in cursor.fetchall():
            portfolios.append({
                'portfolio_id': row[0],
                'name': row[1],
                'created_date': row[2],
                'last_updated': row[3],
                'investment_params': json.loads(row[4]) if row[4] else {},
                'sector_allocations': json.loads(row[5]) if row[5] else {},
                'holdings': json.loads(row[6]) if row[6] else [],
                'snapshots': json.loads(row[7]) if row[7] else [],
                'alerts': json.loads(row[8]) if row[8] else []
            })
        
        conn.close()
        
        return {'success': True, 'portfolios': portfolios}
    
    except Exception as e:
        return {'success': False, 'message': f'Error: {str(e)}', 'portfolios': []}

def update_portfolio_in_db(portfolio_id, user_id, updated_data):
    """Update existing portfolio"""
    try:
        conn = sqlite3.connect('mutual_fund_app.db')
        cursor = conn.cursor()
        
        # Verify ownership
        cursor.execute('SELECT user_id FROM portfolios WHERE portfolio_id = ?', (portfolio_id,))
        result = cursor.fetchone()
        
        if not result or result[0] != user_id:
            conn.close()
            return {'success': False, 'message': 'Portfolio not found or access denied'}
        
        cursor.execute('''
            UPDATE portfolios
            SET portfolio_name = ?, last_updated = ?, holdings = ?, snapshots = ?, alerts = ?
            WHERE portfolio_id = ?
        ''', (
            updated_data.get('name'),
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            json.dumps(updated_data.get('holdings', [])),
            json.dumps(updated_data.get('snapshots', [])),
            json.dumps(updated_data.get('alerts', [])),
            portfolio_id
        ))
        
        conn.commit()
        conn.close()
        
        return {'success': True, 'message': 'Portfolio updated!'}
    
    except Exception as e:
        return {'success': False, 'message': f'Error: {str(e)}'}

def delete_portfolio_from_db(portfolio_id, user_id):
    """Delete a portfolio"""
    try:
        conn = sqlite3.connect('mutual_fund_app.db')
        cursor = conn.cursor()
        
        # Verify ownership
        cursor.execute('SELECT user_id FROM portfolios WHERE portfolio_id = ?', (portfolio_id,))
        result = cursor.fetchone()
        
        if not result or result[0] != user_id:
            conn.close()
            return {'success': False, 'message': 'Portfolio not found or access denied'}
        
        cursor.execute('DELETE FROM portfolios WHERE portfolio_id = ?', (portfolio_id,))
        
        conn.commit()
        conn.close()
        
        return {'success': True, 'message': 'Portfolio deleted!'}
    
    except Exception as e:
        return {'success': False, 'message': f'Error: {str(e)}'}

# ═════════════════════════════════════════════════════════════════════════════
# SESSION MANAGEMENT
# ═════════════════════════════════════════════════════════════════════════════

def init_session_state():
    """Initialize session state for authentication"""
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if 'user_id' not in st.session_state:
        st.session_state.user_id = None
    if 'username' not in st.session_state:
        st.session_state.username = None

def logout():
    """Logout user"""
    st.session_state.logged_in = False
    st.session_state.user_id = None
    st.session_state.username = None
    st.session_state.active_portfolio_id = None

# ═════════════════════════════════════════════════════════════════════════════
# LOGIN/SIGNUP UI
# ═════════════════════════════════════════════════════════════════════════════

def render_login_page():
    """Render login/signup page"""
    
    st.title("🔐 Mutual Fund Analyzer - Login")
    
    # Center the login form
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("### Welcome Back!")
        
        # Tab for Login vs Signup
        tab1, tab2 = st.tabs(["🔑 Login", "📝 Sign Up"])
        
        with tab1:
            st.markdown("#### Login to Your Account")
            
            login_username = st.text_input("Username", key='login_username')
            login_password = st.text_input("Password", type='password', key='login_password')
            
            col_a, col_b = st.columns(2)
            
            with col_a:
                if st.button("🚀 Login", type='primary', use_container_width=True):
                    if not login_username or not login_password:
                        st.error("Please enter both username and password")
                    else:
                        result = authenticate_user(login_username, login_password)
                        
                        if result['success']:
                            st.session_state.logged_in = True
                            st.session_state.user_id = result['user_id']
                            st.session_state.username = result['username']
                            st.success(result['message'])
                            st.balloons()
                            st.rerun()
                        else:
                            st.error(result['message'])
        
        with tab2:
            st.markdown("#### Create New Account")
            
            signup_username = st.text_input("Choose Username", key='signup_username',
                                           help="Must be unique")
            signup_email = st.text_input("Email (optional)", key='signup_email')
            signup_password = st.text_input("Create Password", type='password', key='signup_password',
                                           help="At least 6 characters")
            signup_password_confirm = st.text_input("Confirm Password", type='password', 
                                                   key='signup_password_confirm')
            
            if st.button("✨ Create Account", type='primary', use_container_width=True):
                # Validation
                if not signup_username or not signup_password:
                    st.error("Username and password are required")
                elif len(signup_password) < 6:
                    st.error("Password must be at least 6 characters")
                elif signup_password != signup_password_confirm:
                    st.error("Passwords don't match")
                else:
                    result = create_user(signup_username, signup_password, signup_email)
                    
                    if result['success']:
                        st.success(result['message'])
                        st.info("✅ Account created! Please login with your credentials.")
                        st.balloons()
                    else:
                        st.error(result['message'])
        
        # Info box
        st.markdown("---")
        st.info("""
        **📊 Track Your Investments**
        - Save multiple portfolios
        - Track real-time performance
        - Get daily rebalancing alerts
        - Export portfolio data
        """)

def render_user_sidebar():
    """Render logged-in user sidebar"""
    
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"### 👤 {st.session_state.username}")
    
    if st.sidebar.button("🚪 Logout", use_container_width=True):
        logout()
        st.rerun()
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("💼 My Portfolios")
    
    # Load user portfolios
    result = load_user_portfolios(st.session_state.user_id)
    
    if result['success'] and result['portfolios']:
        st.sidebar.success(f"📁 {len(result['portfolios'])} portfolio(s)")
        
        # Portfolio selector
        portfolio_names = {p['portfolio_id']: f"{p['name']} (Updated: {p['last_updated'][:10]})" 
                          for p in result['portfolios']}
        
        selected_display = st.sidebar.selectbox(
            "Select Portfolio",
            options=list(portfolio_names.values()),
            key='portfolio_selector_sidebar'
        )
        
        # Find selected portfolio ID
        selected_id = [pid for pid, display in portfolio_names.items() if display == selected_display][0]
        selected_portfolio = next(p for p in result['portfolios'] if p['portfolio_id'] == selected_id)
        
        col1, col2 = st.sidebar.columns(2)
        
        with col1:
            if st.button("📂 Load", key='load_portfolio_btn', use_container_width=True):
                st.session_state.active_portfolio = selected_portfolio
                st.session_state.active_portfolio_id = selected_id
                st.success("Portfolio loaded!")
                st.rerun()
        
        with col2:
            if st.button("🗑️ Delete", key='delete_portfolio_btn', use_container_width=True):
                del_result = delete_portfolio_from_db(selected_id, st.session_state.user_id)
                if del_result['success']:
                    st.success("Deleted!")
                    st.rerun()
                else:
                    st.error(del_result['message'])
        
        # Show active portfolio info
        if 'active_portfolio' in st.session_state and st.session_state.active_portfolio:
            active = st.session_state.active_portfolio
            st.sidebar.markdown(f"**🎯 Active:** {active['name']}")
            
            if active.get('investment_params'):
                params = active['investment_params']
                st.sidebar.caption(f"Principal: ₹{params.get('principal', 0):,}")
                st.sidebar.caption(f"SIP: ₹{params.get('monthly_sip', 0):,}/mo")
                st.sidebar.caption(f"Tenure: {params.get('years', 0)} yrs")
            
            # Holdings preview
            if active.get('holdings'):
                with st.sidebar.expander("📊 Holdings", expanded=False):
                    for h in active['holdings'][:5]:
                        st.write(f"**{h.get('fund_name', 'Unknown')[:25]}**")
                        st.caption(f"₹{h.get('lumpsum_amount', 0):,} + ₹{h.get('monthly_sip', 0)}/mo")
    
    else:
        st.sidebar.info("No portfolios yet. Create one in the Portfolio Allocator!")

# Initialize database on module load
init_database()
