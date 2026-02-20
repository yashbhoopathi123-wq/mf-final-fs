"""
Dynamic Recommendation Engine
Adjusts allocation based on investment amount, tenure, and SIP ratio
"""

def get_dynamic_allocation(principal, monthly_sip, years, base_risk_profile):
    """
    Calculate personalized allocation based on investor profile
    
    Args:
        principal: Lumpsum amount
        monthly_sip: Monthly SIP
        years: Investment horizon
        base_risk_profile: "Conservative", "Moderate", or "Aggressive"
    
    Returns:
        Dict of {sector_name: allocation_%}
    """
    
    # Base allocations by risk profile
    BASE_ALLOCATIONS = {
        "Conservative (Low Risk)": {
            "Debt / Liquid": 30,
            "Gilt / Government Securities": 20,
            "Gold / Commodity": 15,
            "Index / Passive": 20,
            "Hybrid / Balanced": 10,
            "Global / International": 5,
        },
        "Moderate (Balanced Risk)": {
            "Index / Passive": 25,
            "Large Cap": 20,
            "Flexi Cap": 15,
            "Hybrid / Balanced": 10,
            "US Tech / NASDAQ": 10,
            "Debt / Liquid": 10,
            "Gold / Commodity": 5,
            "Global / International": 5,
        },
        "Aggressive (High Risk)": {
            "Small Cap": 20,
            "Mid Cap": 20,
            "US Tech / NASDAQ": 15,
            "Sector – Technology": 10,
            "Sector – Healthcare / Pharma": 10,
            "Large Cap": 10,
            "Flexi Cap": 10,
            "Thematic – ESG / Sustainability": 5,
        },
    }
    
    allocation = BASE_ALLOCATIONS[base_risk_profile].copy()
    
    # Calculate derived metrics
    total_first_year = principal + (monthly_sip * 12)
    total_investment = principal + (monthly_sip * years * 12)
    sip_ratio = (monthly_sip * years * 12) / total_investment if total_investment > 0 else 0
    
    # ══════════════════════════════════════════════════════════════════════════
    # ADJUSTMENT 1: Based on Investment Amount
    # ══════════════════════════════════════════════════════════════════════════
    
    if total_first_year < 50000:
        # Small investor: Focus on simplicity and lower costs
        adjustments = {
            'Index / Passive': +10,
            'Large Cap': +5,
            'Small Cap': -10,
            'Sector – Technology': -5
        }
    
    elif total_first_year > 500000:
        # Large investor: Can afford diversification and specialized sectors
        adjustments = {
            'Global / International': +5,
            'US Tech / NASDAQ': +5,
            'Sector – Technology': +5,
            'Debt / Liquid': -10,
            'Gilt / Government Securities': -5
        }
    
    else:
        # Medium investor: Balanced approach
        adjustments = {}
    
    # Apply amount-based adjustments
    for sector, adj in adjustments.items():
        if sector in allocation:
            allocation[sector] = max(0, allocation[sector] + adj)
    
    # ══════════════════════════════════════════════════════════════════════════
    # ADJUSTMENT 2: Based on Investment Horizon (Tenure)
    # ══════════════════════════════════════════════════════════════════════════
    
    if years <= 2:
        # Very short horizon: Drastically reduce equity, boost debt
        for sector in list(allocation.keys()):
            if any(x in sector for x in ['Small', 'Mid', 'Sector', 'Thematic']):
                reduction = allocation[sector] * 0.8  # Remove 80%
                allocation[sector] -= reduction
                allocation['Debt / Liquid'] = allocation.get('Debt / Liquid', 0) + reduction * 0.7
                allocation['Hybrid / Balanced'] = allocation.get('Hybrid / Balanced', 0) + reduction * 0.3
    
    elif years <= 5:
        # Medium horizon: Moderate equity, balanced debt
        for sector in list(allocation.keys()):
            if 'Small' in sector:
                reduction = allocation[sector] * 0.4
                allocation[sector] -= reduction
                allocation['Large Cap'] = allocation.get('Large Cap', 0) + reduction * 0.5
                allocation['Hybrid / Balanced'] = allocation.get('Hybrid / Balanced', 0) + reduction * 0.5
    
    elif years >= 10:
        # Long horizon: Can afford higher risk
        adjustments_long = {
            'Small Cap': +5,
            'Mid Cap': +5,
            'Sector – Technology': +3,
            'Debt / Liquid': -8,
            'Gilt / Government Securities': -5
        }
        for sector, adj in adjustments_long.items():
            if sector in allocation:
                allocation[sector] = max(0, allocation[sector] + adj)
    
    # ══════════════════════════════════════════════════════════════════════════
    # ADJUSTMENT 3: Based on SIP Ratio
    # ══════════════════════════════════════════════════════════════════════════
    
    if sip_ratio > 0.7:
        # Heavy SIP: Rupee cost averaging reduces volatility risk
        # Can invest more aggressively in small/mid caps
        adjustments_sip = {
            'Small Cap': +5,
            'Mid Cap': +5,
            'Large Cap': -5,
            'Debt / Liquid': -5
        }
        for sector, adj in adjustments_sip.items():
            if sector in allocation:
                allocation[sector] = max(0, allocation[sector] + adj)
    
    elif sip_ratio < 0.3:
        # Heavy lumpsum: One-time investment needs stability
        adjustments_lumpsum = {
            'Large Cap': +5,
            'Index / Passive': +5,
            'Hybrid / Balanced': +5,
            'Small Cap': -10,
            'Sector – Technology': -5
        }
        for sector, adj in adjustments_lumpsum.items():
            if sector in allocation:
                allocation[sector] = max(0, allocation[sector] + adj)
    
    # ══════════════════════════════════════════════════════════════════════════
    # NORMALIZATION: Ensure allocations sum to 100%
    # ══════════════════════════════════════════════════════════════════════════
    
    # Remove zero/negative allocations
    allocation = {k: v for k, v in allocation.items() if v > 0}
    
    # Normalize to 100
    total = sum(allocation.values())
    if total > 0:
        allocation = {k: (v / total) * 100 for k, v in allocation.items()}
    
    # Round to 1 decimal
    allocation = {k: round(v, 1) for k, v in allocation.items()}
    
    return allocation

# Example usage:
if __name__ == "__main__":
    # Test case 1: Small investor, long tenure, heavy SIP
    alloc1 = get_dynamic_allocation(
        principal=10000,
        monthly_sip=5000,
        years=15,
        base_risk_profile="Moderate (Balanced Risk)"
    )
    print("Small investor, 15 years, heavy SIP:")
    for sector, pct in sorted(alloc1.items(), key=lambda x: -x[1]):
        print(f"  {sector}: {pct}%")
    
    # Test case 2: Large investor, short tenure, heavy lumpsum
    alloc2 = get_dynamic_allocation(
        principal=1000000,
        monthly_sip=10000,
        years=3,
        base_risk_profile="Moderate (Balanced Risk)"
    )
    print("\nLarge investor, 3 years, heavy lumpsum:")
    for sector, pct in sorted(alloc2.items(), key=lambda x: -x[1]):
        print(f"  {sector}: {pct}%")
