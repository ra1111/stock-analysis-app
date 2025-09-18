import streamlit as st
import pandas as pd
from utils.ipo_analyzer import IPOAnalyzer
import plotly.graph_objects as go
import plotly.express as px

def render():
    st.header("🚀 IPO Radar & Analysis - Automated Recommendations")
    
    analyzer = IPOAnalyzer()
    
    # Get user budget for personalized recommendations
    st.sidebar.header("💰 Investment Preferences")
    budget = st.sidebar.number_input(
        "Your IPO Investment Budget (₹)",
        min_value=10000,
        max_value=10000000,
        value=100000,
        step=10000,
        help="Set your budget for IPO investments to get personalized recommendations"
    )
    
    risk_tolerance = st.sidebar.selectbox(
        "Risk Tolerance",
        ["conservative", "moderate", "aggressive"],
        index=1,
        help="Your risk tolerance affects lot size suggestions"
    )
    
    investment_category = st.sidebar.selectbox(
        "Investment Category",
        ["Retail Individual Investor", "HNI (High Net Worth)", "Institutional"],
        help="Your investor category affects allotment probability calculations"
    )
    
    # Map investment category for analyzer
    category_mapping = {
        "Retail Individual Investor": "retail",
        "HNI (High Net Worth)": "hni", 
        "Institutional": "institutional"
    }
    category = category_mapping.get(investment_category, "retail")
    
    # Main content area with tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🎯 Live Recommendations", "📈 Current IPOs", "🔮 Upcoming IPOs", "📊 Market Insights"])
    
    with tab1:
        st.subheader("🎯 Personalized IPO Recommendations")
        st.write(f"**Budget:** ₹{budget:,} | **Risk Tolerance:** {risk_tolerance.title()} | **Category:** {investment_category}")
        
        with st.spinner("Analyzing current IPOs and generating recommendations..."):
            recommendations = analyzer.generate_investment_recommendations(budget)
        
        if recommendations['top_recommendations']:
            st.success(f"✅ Found {len(recommendations['all_recommendations'])} recommended IPOs from {recommendations['total_analyzed']} analyzed")
            
            # Top Recommendations
            for i, rec in enumerate(recommendations['top_recommendations'], 1):
                with st.container():
                    # Header with recommendation
                    col1, col2, col3 = st.columns([3, 1, 1])
                    
                    with col1:
                        if rec['recommendation_color'] == 'green':
                            st.success(f"**#{i} - {rec['company_name']} ({rec['ticker']})**")
                        elif rec['recommendation_color'] == 'blue':
                            st.info(f"**#{i} - {rec['company_name']} ({rec['ticker']})**")
                        elif rec['recommendation_color'] == 'orange':
                            st.warning(f"**#{i} - {rec['company_name']} ({rec['ticker']})**")
                        else:
                            st.error(f"**#{i} - {rec['company_name']} ({rec['ticker']})**")
                    
                    with col2:
                        st.metric("Recommendation", rec['recommendation'])
                    
                    with col3:
                        st.metric("Score", f"{rec['score']:.1f}/8")
                    
                    # Details
                    detail_col1, detail_col2, detail_col3 = st.columns(3)
                    
                    with detail_col1:
                        st.write("**📋 IPO Details**")
                        st.write(f"• Price Band: {rec['price_band']}")
                        st.write(f"• Type: {rec['ipo_type']}")
                        st.write(f"• Days Left: {rec['days_remaining']}")
                        st.write(f"• Listing: {rec['listing_date']}")
                    
                    with detail_col2:
                        st.write("**💰 Investment Suggestion**")
                        st.write(f"• Min Investment: ₹{rec['min_investment']:,}")
                        st.write(f"• Suggested Lots: {rec['suggested_lots']}")
                        st.write(f"• Total Amount: ₹{rec['suggested_investment']:,}")
                        st.write(f"• Allotment Odds: {rec['allotment_probability']:.0%}")
                        
                        # Add GMP data if available
                        if 'gmp' in rec and rec['gmp'] is not None:
                            gmp_color = "🟢" if rec['gmp'] > 0 else "🔴" if rec['gmp'] < 0 else "🟡"
                            st.write(f"• **GMP:** {gmp_color} ₹{rec['gmp']:+.0f} ({rec['gmp_percentage']:+.1f}%)")
                            potential_profit = (rec['expected_listing_price'] - rec['price_band_max']) * rec['suggested_lots']
                            profit_color = "🟢" if potential_profit > 0 else "🔴" if potential_profit < 0 else "🟡"
                            st.write(f"• **Expected Profit:** {profit_color} ₹{potential_profit:+,.0f}")
                    
                    with detail_col3:
                        st.write("**✅ Why Recommended**")
                        for reason in rec['reasons'][:3]:  # Show top 3 reasons
                            st.write(f"• {reason}")
                        
                        if rec['risk_factors']:
                            st.write("**⚠️ Risk Factors**")
                            for risk in rec['risk_factors'][:2]:  # Show top 2 risks
                                st.write(f"• {risk}")
                    
                    # Action button
                    col_button1, col_button2, col_button3 = st.columns([1, 1, 2])
                    with col_button1:
                        if st.button(f"📊 Detailed Analysis", key=f"analyze_{rec['ticker']}"):
                            st.session_state[f'show_analysis_{rec["ticker"]}'] = True
                    
                    with col_button2:
                        if st.button(f"🎯 Lot Size Calculator", key=f"lots_{rec['ticker']}"):
                            st.session_state[f'show_lots_{rec["ticker"]}'] = True
                    
                    # Show detailed analysis if requested
                    if st.session_state.get(f'show_analysis_{rec["ticker"]}', False):
                        with st.expander(f"🔍 Detailed Analysis - {rec['company_name']}", expanded=True):
                            ipo_data = analyzer.get_ipo_info(rec['company_name'])
                            analysis = analyzer.analyze_ipo_fundamentals(ipo_data)
                            
                            st.write(f"**Risk Assessment:** {analysis['risk_assessment']}")
                            st.write(f"**Overall Recommendation:** {analysis['recommendation']}")
                            
                            for factor in analysis['factors_to_consider'][:3]:  # Show top 3 factors
                                st.write(f"**{factor['factor']}:** {factor['description']}")
                    
                    # Show lot size calculator if requested
                    if st.session_state.get(f'show_lots_{rec["ticker"]}', False):
                        with st.expander(f"🎯 Lot Size Calculator - {rec['company_name']}", expanded=True):
                            ipo_data = analyzer.get_ipo_info(rec['company_name'])
                            lot_analysis = analyzer.suggest_optimal_lot_size(ipo_data, budget, risk_tolerance)
                            
                            scenario_col1, scenario_col2, scenario_col3 = st.columns(3)
                            
                            scenarios = lot_analysis['scenarios']
                            
                            with scenario_col1:
                                st.write("**Conservative Approach**")
                                st.metric("Lots", scenarios['conservative']['lots'])
                                st.metric("Investment", f"₹{scenarios['conservative']['investment']:,}")
                                st.write(scenarios['conservative']['description'])
                            
                            with scenario_col2:
                                st.write("**Moderate Approach**")
                                st.metric("Lots", scenarios['moderate']['lots'])
                                st.metric("Investment", f"₹{scenarios['moderate']['investment']:,}")
                                st.write(scenarios['moderate']['description'])
                            
                            with scenario_col3:
                                st.write("**Aggressive Approach**")
                                st.metric("Lots", scenarios['aggressive']['lots'])
                                st.metric("Investment", f"₹{scenarios['aggressive']['investment']:,}")
                                st.write(scenarios['aggressive']['description'])
                    
                    st.divider()
        
        else:
            st.warning("⚠️ No IPOs meet the investment criteria at this time. Check upcoming IPOs or adjust your budget.")
    
    with tab2:
        st.subheader("📈 Current Live IPOs")
        
        with st.spinner("Fetching live IPO data..."):
            ipo_data = analyzer.fetch_live_ipos()
        
        if ipo_data['current_ipos']:
            st.success(f"✅ {len(ipo_data['current_ipos'])} live IPOs found")
            
            # Create a summary table
            current_df = pd.DataFrame(ipo_data['current_ipos'])
            
            # Display summary table with GMP data
            display_df = current_df.copy()
            
            # Add GMP display columns
            if 'gmp' in display_df.columns:
                display_df['gmp_display'] = display_df.apply(
                    lambda row: f"₹{row['gmp']:+.0f}" if pd.notna(row['gmp']) else "N/A", axis=1
                )
                display_df['gmp_percentage_display'] = display_df.apply(
                    lambda row: f"{row['gmp_percentage']:+.1f}%" if pd.notna(row['gmp_percentage']) else "N/A", axis=1
                )
                display_df['expected_listing_display'] = display_df.apply(
                    lambda row: f"₹{row['expected_listing_price']:.0f}" if pd.notna(row['expected_listing_price']) else "N/A", axis=1
                )
            
            display_cols = ['company_name', 'ticker', 'price_band', 'gmp_display', 'gmp_percentage_display', 'expected_listing_display', 'type', 'days_remaining']
            
            if all(col in display_df.columns for col in display_cols):
                renamed_df = display_df[display_cols].copy()
                renamed_df.columns = ['Company', 'Ticker', 'Price Band', 'GMP (₹)', 'GMP %', 'Expected Listing', 'Type', 'Days Left']
                st.dataframe(renamed_df, use_container_width=True)
            
            # Detailed view for each IPO
            st.subheader("📋 Detailed IPO Information")
            
            for ipo in ipo_data['current_ipos']:
                with st.expander(f"📊 {ipo['company_name']} ({ipo['ticker']}) - {ipo['price_band']}"):
                    info_col1, info_col2, info_col3 = st.columns(3)
                    
                    with info_col1:
                        st.write("**Basic Details**")
                        st.write(f"Company: {ipo['company_name']}")
                        st.write(f"Ticker: {ipo['ticker']}")
                        st.write(f"Type: {ipo['type']}")
                        st.write(f"Issue Size: {ipo['issue_size']}")
                    
                    with info_col2:
                        st.write("**Pricing & Timeline**")
                        st.write(f"Price Band: {ipo['price_band']}")
                        st.write(f"Lot Size: {ipo['lot_size']}")
                        st.write(f"IPO Dates: {ipo['ipo_dates']}")
                        st.write(f"Days Remaining: {ipo['days_remaining']}")
                    
                    with info_col3:
                        st.write("**Investment & GMP Analysis**")
                        min_investment = ipo['price_band_max'] * ipo['lot_size']
                        st.write(f"Min Investment: ₹{min_investment:,}")
                        st.write(f"Listing Date: {ipo['listing_date']}")
                        
                        # GMP Information
                        if 'gmp' in ipo and ipo['gmp'] is not None:
                            gmp_color = "🟢" if ipo['gmp'] > 0 else "🔴" if ipo['gmp'] < 0 else "🟡"
                            st.write(f"**GMP:** {gmp_color} ₹{ipo['gmp']:+.0f} ({ipo['gmp_percentage']:+.1f}%)")
                            st.write(f"**Expected Listing:** ₹{ipo['expected_listing_price']:.0f}")
                            
                            # Potential profit calculation
                            potential_profit = (ipo['expected_listing_price'] - ipo['price_band_max']) * ipo['lot_size']
                            profit_color = "🟢" if potential_profit > 0 else "🔴" if potential_profit < 0 else "🟡"
                            st.write(f"**Potential Profit:** {profit_color} ₹{potential_profit:+,.0f}")
                        else:
                            st.write("**GMP:** Not Available")
                        
                        # Calculate allotment probability
                        allotment = analyzer.calculate_realistic_allotment_probability(ipo, min_investment, category)
                        st.write(f"**Allotment Odds:** {allotment['probability']:.0%}")
        
        else:
            st.info("ℹ️ No live IPOs available at the moment. Check upcoming IPOs.")
    
    with tab3:
        st.subheader("🔮 Upcoming IPOs")
        
        upcoming = ipo_data.get('upcoming_ipos', [])
        
        if upcoming:
            st.info(f"📅 {len(upcoming)} upcoming IPOs found")
            
            for ipo in upcoming:
                with st.container():
                    if ipo['open_date']:  # Has specific dates
                        col1, col2, col3 = st.columns([2, 1, 1])
                        
                        with col1:
                            st.write(f"**{ipo['company_name']} ({ipo['ticker']})**")
                            st.write(f"Price Band: {ipo['price_band']}")
                            st.write(f"Type: {ipo['type']}")
                        
                        with col2:
                            st.write(f"**Opens:** {ipo['ipo_dates']}")
                            if ipo.get('days_to_open'):
                                st.write(f"**In:** {ipo['days_to_open']} days")
                        
                        with col3:
                            st.write(f"**Issue Size:** {ipo['issue_size']}")
                            if ipo['lot_size']:
                                min_investment = ipo['price_band_max'] * ipo['lot_size']
                                st.write(f"**Min Investment:** ₹{min_investment:,}")
                            
                            # Show GMP for upcoming IPOs
                            if 'gmp' in ipo and ipo['gmp'] is not None:
                                gmp_color = "🟢" if ipo['gmp'] > 0 else "🔴" if ipo['gmp'] < 0 else "🟡"
                                st.write(f"**GMP:** {gmp_color} ₹{ipo['gmp']:+.0f} ({ipo['gmp_percentage']:+.1f}%)")
                                st.write(f"**Expected Listing:** ₹{ipo['expected_listing_price']:.0f}")
                    
                    else:  # TBA dates
                        col1, col2 = st.columns([3, 1])
                        
                        with col1:
                            st.write(f"**{ipo['company_name']} ({ipo['ticker']})** - Major IPO")
                            st.write(f"Status: {ipo['subscription_status']}")
                        
                        with col2:
                            st.write(f"**Issue Size:** {ipo['issue_size']}")
                    
                    st.divider()
        
        else:
            st.info("ℹ️ No upcoming IPOs scheduled at the moment.")
    
    with tab4:
        st.subheader("📊 Market Insights & Performance")
        
        # Get market performance data
        with st.spinner("Analyzing market performance..."):
            performance = analyzer.get_recent_ipo_performance()
        
        # Market sentiment
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if performance['sentiment_color'] == 'green':
                st.success(f"🟢 **Market Sentiment:** {performance['market_sentiment']}")
            elif performance['sentiment_color'] == 'blue':
                st.info(f"🔵 **Market Sentiment:** {performance['market_sentiment']}")
            elif performance['sentiment_color'] == 'orange':
                st.warning(f"🟠 **Market Sentiment:** {performance['market_sentiment']}")
            else:
                st.error(f"🔴 **Market Sentiment:** {performance['market_sentiment']}")
            
            st.write(f"**Recommendation:** {performance['recommendation']}")
        
        with col2:
            st.metric("Success Rate", f"{performance['success_rate']:.0f}%")
            st.metric("Avg Listing Gain", f"{performance['average_listing_gain']:.1f}%")
        
        # Recent IPO performance chart
        st.subheader("📈 Recent IPO Performance")
        
        perf_df = pd.DataFrame(performance['recent_ipos'])
        
        # Create performance chart
        fig = px.bar(
            perf_df, 
            x='company', 
            y='listing_gain',
            color='listing_gain',
            color_continuous_scale=['red', 'yellow', 'green'],
            title="Recent IPO Listing Performance (%)",
            labels={'listing_gain': 'Listing Gain (%)', 'company': 'Company'}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Key insights
        st.subheader("💡 Key Market Insights")
        for insight in performance['key_insights']:
            st.write(f"• {insight}")
        
        # Sector performance
        st.subheader("🏭 Sector Performance Summary")
        sector_data = {}
        for ipo in performance['recent_ipos']:
            sector = ipo['sector']
            if sector not in sector_data:
                sector_data[sector] = []
            sector_data[sector].append(ipo['listing_gain'])
        
        sector_summary = []
        for sector, gains in sector_data.items():
            avg_gain = sum(gains) / len(gains)
            sector_summary.append({'Sector': sector, 'Avg Listing Gain (%)': f"{avg_gain:.1f}%", 'IPOs': len(gains)})
        
        sector_df = pd.DataFrame(sector_summary)
        st.dataframe(sector_df, use_container_width=True)
    
    # Educational Content
    with st.expander("📚 IPO Investment Guide"):
        st.markdown("""
        ### 🎯 How This Automated System Works
        
        **Automatic IPO Discovery:**
        - Real-time data from NSE/BSE and broker platforms
        - Automatic parsing of price bands, lot sizes, and dates
        - Live subscription status monitoring
        
        **Investment Recommendations:**
        - Multi-factor scoring algorithm (pricing, timing, market conditions)
        - Budget-based lot size suggestions
        - Risk-adjusted recommendations based on your profile
        
        **Allotment Probability Calculation:**
        - Based on IPO characteristics (price, size, type)
        - Historical subscription pattern analysis
        - Category-specific probability estimates
        
        ### 📊 Investment Categories
        - **Retail**: Individual investors (up to ₹2 lakhs)
        - **HNI**: High Net Worth Individuals (₹2 lakhs to ₹10 lakhs)  
        - **QIB**: Qualified Institutional Buyers (₹10 lakhs+)
        
        ### 🎲 Understanding Allotment Odds
        - **High (>50%)**: Good chances, likely undersubscribed
        - **Medium (20-50%)**: Moderate chances, selective allotment
        - **Low (<20%)**: Tough competition, oversubscribed IPO
        
        ### 💡 Investment Tips
        1. **Diversify**: Don't put all budget in one IPO
        2. **Apply Early**: Avoid last-minute technical issues
        3. **Use UPI**: Fastest application method
        4. **Check Fundamentals**: Look beyond just pricing
        5. **Set Budget Limits**: Don't exceed your risk capacity
        """)
    
    # Disclaimer
    st.warning("""
    ⚠️ **Investment Disclaimer**: 
    IPO investments carry market risks. This automated analysis is for educational purposes only and should not be considered as investment advice. 
    Always consult with financial advisors and do your own research before investing. Past performance doesn't guarantee future results.
    """)