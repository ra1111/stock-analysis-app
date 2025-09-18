import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from utils.portfolio_analyzer import PortfolioAnalyzer
from utils.data_fetcher import DataFetcher

def render():
    """Render the comprehensive Portfolio Tracker tab"""
    st.header("📊 Portfolio Performance Tracker")
    st.markdown("Upload your portfolio CSV and get comprehensive analysis with benchmark comparison, sector allocation, and performance metrics.")
    
    analyzer = PortfolioAnalyzer()
    data_fetcher = DataFetcher()
    
    # File upload section
    st.subheader("📁 Upload Your Portfolio")
    
    # CSV format guide
    with st.expander("📋 CSV Format Guide", expanded=False):
        st.write("""
        **Required columns (any of these names work):**
        - **Symbol/Ticker/Stock**: Stock symbol (e.g., INFY, TCS, RELIANCE)
        - **Quantity/Qty/Shares**: Number of shares you own
        - **Avg_Price/Average_Price/Price**: Your average purchase price per share
        
        **Optional columns:**
        - **Purchase_Date/Date**: When you bought the stock (defaults to 1 year ago)
        - **Invested_Value**: Total invested amount (auto-calculated if not provided)
        
        **Example CSV:**
        ```
        Symbol,Quantity,Avg_Price,Purchase_Date
        INFY,100,1500,2023-01-15
        TCS,50,3200,2023-03-20
        RELIANCE,25,2400,2023-06-10
        ```
        """)
    
    uploaded_file = st.file_uploader("Choose your portfolio CSV file", type=['csv'])
    
    if uploaded_file is not None:
        # Load portfolio data
        with st.spinner("Loading portfolio data..."):
            portfolio_df = analyzer.load_portfolio_from_csv(uploaded_file)
        
        if portfolio_df is not None and not portfolio_df.empty:
            st.success(f"✅ Portfolio loaded successfully! Found {len(portfolio_df)} holdings.")
            
            # Calculate comprehensive performance
            with st.spinner("Analyzing portfolio performance..."):
                performance = analyzer.calculate_portfolio_performance(portfolio_df)
                
                if performance:
                    # Get benchmark data
                    earliest_date = portfolio_df['Purchase_Date'].min()
                    benchmarks = analyzer.get_benchmark_performance(earliest_date)
                    
                    # Portfolio Overview Section
                    st.subheader("💰 Portfolio Overview")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric(
                            "Total Investment", 
                            f"₹{performance['total_investment']:,.0f}"
                        )
                    
                    with col2:
                        st.metric(
                            "Current Value", 
                            f"₹{performance['total_current_value']:,.0f}",
                            f"₹{performance['total_pnl']:,.0f}"
                        )
                    
                    with col3:
                        st.metric(
                            "Total P&L", 
                            f"{performance['total_pnl_pct']:.1f}%",
                            help="Total profit/loss percentage"
                        )
                    
                    with col4:
                        st.metric(
                            "Portfolio CAGR", 
                            f"{performance['weighted_cagr']:.1f}%",
                            help="Compound Annual Growth Rate (weighted average)"
                        )
                    
                    # Data availability indicator
                    if performance['data_availability'] < 100:
                        st.warning(f"⚠️ Live price data available for {performance['data_availability']:.0f}% of holdings. Some prices may be stale.")
                    
                    # Performance vs Benchmarks
                    if benchmarks:
                        st.subheader("📈 Performance vs Benchmarks")
                        
                        # Create performance comparison chart
                        perf_chart = analyzer.create_performance_chart(performance, benchmarks)
                        if perf_chart:
                            st.plotly_chart(perf_chart, use_container_width=True)
                        
                        # Benchmark comparison table
                        benchmark_data = []
                        benchmark_data.append({
                            'Asset': 'Your Portfolio',
                            'CAGR (%)': f"{performance['weighted_cagr']:.1f}%",
                            'Outperformance vs Nifty50': f"{performance['weighted_cagr'] - benchmarks.get('Nifty50', {}).get('cagr', 0):.1f}%" if 'Nifty50' in benchmarks else "N/A",
                            'Outperformance vs S&P500': f"{performance['weighted_cagr'] - benchmarks.get('S&P500', {}).get('cagr', 0):.1f}%" if 'S&P500' in benchmarks else "N/A"
                        })
                        
                        for name, data in benchmarks.items():
                            benchmark_data.append({
                                'Asset': name,
                                'CAGR (%)': f"{data['cagr']:.1f}%",
                                'Outperformance vs Nifty50': "Benchmark",
                                'Outperformance vs S&P500': "Benchmark"
                            })
                        
                        st.dataframe(pd.DataFrame(benchmark_data), use_container_width=True)
                    
                    # Portfolio Allocation Analysis
                    st.subheader("🥧 Portfolio Allocation Analysis")
                    
                    allocation_data = analyzer.analyze_portfolio_allocation(performance['portfolio_df'])
                    
                    if allocation_data:
                        # Risk analysis summary
                        risk = allocation_data['risk_analysis']
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            score_color = "🟢" if risk['diversification_score'] >= 80 else "🟡" if risk['diversification_score'] >= 60 else "🔴"
                            st.metric(
                                "Diversification Score", 
                                f"{score_color} {risk['diversification_score']:.0f}/100",
                                help="Score based on stock and sector concentration"
                            )
                        
                        with col2:
                            st.metric(
                                "Top 5 Concentration", 
                                f"{risk['top_5_stocks_concentration']:.1f}%",
                                help="Percentage of portfolio in top 5 stocks"
                            )
                        
                        with col3:
                            st.metric(
                                "Number of Sectors", 
                                f"{len(allocation_data['sector_allocation'])}",
                                help="Sector diversification count"
                            )
                        
                        # Risk warnings
                        if not risk['overconcentrated_stocks'].empty:
                            st.warning(f"⚠️ **High Stock Concentration**: {len(risk['overconcentrated_stocks'])} stocks have >25% allocation")
                            for _, stock in risk['overconcentrated_stocks'].iterrows():
                                st.write(f"• {stock['Symbol']}: {stock['Allocation_%']:.1f}%")
                        
                        if not risk['overconcentrated_sectors'].empty:
                            st.warning(f"⚠️ **High Sector Concentration**: {len(risk['overconcentrated_sectors'])} sectors have >40% allocation")
                            for _, sector in risk['overconcentrated_sectors'].iterrows():
                                st.write(f"• {sector['Sector']}: {sector['Allocation_%']:.1f}%")
                        
                        # Allocation charts
                        col1, col2 = st.columns(2)
                        
                        fig_stocks, fig_sectors = analyzer.create_allocation_charts(allocation_data)
                        
                        with col1:
                            st.plotly_chart(fig_stocks, use_container_width=True)
                        
                        with col2:
                            st.plotly_chart(fig_sectors, use_container_width=True)
                    
                    # Detailed Holdings Table
                    st.subheader("📋 Detailed Holdings")
                    
                    # Add filters
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        sector_filter = st.selectbox(
                            "Filter by Sector", 
                            ['All'] + list(performance['portfolio_df']['Sector'].unique())
                        )
                    
                    with col2:
                        sort_by = st.selectbox(
                            "Sort by", 
                            ['Current_Value', 'P&L_%', 'CAGR', 'Symbol']
                        )
                    
                    with col3:
                        sort_order = st.selectbox("Order", ['Descending', 'Ascending'])
                    
                    # Apply filters
                    filtered_df = performance['portfolio_df'].copy()
                    if sector_filter != 'All':
                        filtered_df = filtered_df[filtered_df['Sector'] == sector_filter]
                    
                    # Sort
                    ascending = sort_order == 'Ascending'
                    filtered_df = filtered_df.sort_values(sort_by, ascending=ascending)
                    
                    # Display table with formatted columns
                    display_columns = [
                        'Symbol', 'Sector', 'Quantity', 'Avg_Price', 'Current_Price', 
                        'Investment', 'Current_Value', 'P&L', 'P&L_%', 'CAGR', 'Days_Held'
                    ]
                    
                    # Format the dataframe for display
                    display_df = filtered_df[display_columns].copy()
                    
                    # Format currency columns
                    for col in ['Avg_Price', 'Current_Price', 'Investment', 'Current_Value', 'P&L']:
                        if col in display_df.columns:
                            display_df[col] = display_df[col].apply(lambda x: f"₹{x:,.0f}" if pd.notna(x) else "N/A")
                    
                    # Format percentage columns
                    for col in ['P&L_%', 'CAGR']:
                        if col in display_df.columns:
                            display_df[col] = display_df[col].apply(lambda x: f"{x:.1f}%" if pd.notna(x) else "N/A")
                    
                    st.dataframe(display_df, use_container_width=True)
                    
                    # Financial Ratios Analysis
                    st.subheader("📊 Financial Ratios Analysis")
                    
                    with st.spinner("Analyzing financial ratios..."):
                        ratio_analysis = analyzer.get_portfolio_ratios_analysis(performance['portfolio_df'])
                    
                    if ratio_analysis:
                        # Ratio summary charts
                        col1, col2 = st.columns(2)
                        
                        fig_flags, fig_actions = analyzer.create_ratio_summary_chart(ratio_analysis)
                        
                        with col1:
                            if fig_flags:
                                st.plotly_chart(fig_flags, use_container_width=True)
                        
                        with col2:
                            if fig_actions:
                                st.plotly_chart(fig_actions, use_container_width=True)
                        
                        # Detailed ratio analysis by stock
                        st.subheader("🔍 Detailed Ratio Analysis by Stock")
                        
                        for stock_data in ratio_analysis:
                            symbol = stock_data['Symbol']
                            ratios = stock_data['ratios']
                            analysis = stock_data['analysis']
                            
                            # Determine overall flag color
                            flag_color = {
                                'POSITIVE': '🟢',
                                'NEUTRAL': '🟡', 
                                'NEGATIVE': '🔴'
                            }.get(analysis['overall_flag'], '⚪')
                            
                            # Action color
                            action_color = {
                                'BUY MORE': '🟢',
                                'HOLD & ACCUMULATE': '🔵',
                                'HOLD': '🟡',
                                'MONITOR CLOSELY': '🟠',
                                'CONSIDER SELLING': '🔴'
                            }.get(analysis['action'], '⚪')
                            
                            with st.expander(f"{flag_color} {symbol} - {analysis['overall_flag']} | {action_color} {analysis['action']}", expanded=False):
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    st.write("**Valuation Ratios**")
                                    
                                    # P/E Ratio
                                    if ratios.get('pe_ratio'):
                                        pe_flag = analysis['flags'].get('pe_ratio', 'NEUTRAL')
                                        pe_color = {'POSITIVE': '🟢', 'NEUTRAL': '🟡', 'NEGATIVE': '🔴'}.get(pe_flag, '⚪')
                                        st.write(f"{pe_color} **P/E Ratio:** {ratios['pe_ratio']:.1f}")
                                    
                                    # P/B Ratio
                                    if ratios.get('pb_ratio'):
                                        pb_flag = analysis['flags'].get('pb_ratio', 'NEUTRAL')
                                        pb_color = {'POSITIVE': '🟢', 'NEUTRAL': '🟡', 'NEGATIVE': '🔴'}.get(pb_flag, '⚪')
                                        st.write(f"{pb_color} **P/B Ratio:** {ratios['pb_ratio']:.1f}")
                                    
                                    # PEG Ratio
                                    if ratios.get('peg_ratio'):
                                        peg_flag = analysis['flags'].get('peg_ratio', 'NEUTRAL')
                                        peg_color = {'POSITIVE': '🟢', 'NEUTRAL': '🟡', 'NEGATIVE': '🔴'}.get(peg_flag, '⚪')
                                        st.write(f"{peg_color} **PEG Ratio:** {ratios['peg_ratio']:.1f}")
                                
                                with col2:
                                    st.write("**Profitability & Financial Health**")
                                    
                                    # ROE
                                    if ratios.get('roe'):
                                        roe_flag = analysis['flags'].get('roe', 'NEUTRAL')
                                        roe_color = {'POSITIVE': '🟢', 'NEUTRAL': '🟡', 'NEGATIVE': '🔴'}.get(roe_flag, '⚪')
                                        st.write(f"{roe_color} **ROE:** {ratios['roe']:.1f}%")
                                    
                                    # Debt-to-Equity
                                    if ratios.get('debt_to_equity') is not None:
                                        de_flag = analysis['flags'].get('debt_to_equity', 'NEUTRAL')
                                        de_color = {'POSITIVE': '🟢', 'NEUTRAL': '🟡', 'NEGATIVE': '🔴'}.get(de_flag, '⚪')
                                        st.write(f"{de_color} **Debt/Equity:** {ratios['debt_to_equity']:.1f}")
                                    
                                    # Dividend Yield
                                    if ratios.get('dividend_yield'):
                                        dy_flag = analysis['flags'].get('dividend_yield', 'NEUTRAL')
                                        dy_color = {'POSITIVE': '🟢', 'NEUTRAL': '🟡', 'NEGATIVE': '🔴'}.get(dy_flag, '⚪')
                                        st.write(f"{dy_color} **Dividend Yield:** {ratios['dividend_yield']:.1f}%")
                                
                                with col3:
                                    st.write("**Action & Score**")
                                    st.write(f"**Overall Score:** {analysis['overall_score']:.2f}")
                                    st.write(f"**Recommended Action:** {action_color} **{analysis['action']}**")
                                    
                                    # Market Cap
                                    if ratios.get('market_cap'):
                                        market_cap_cr = ratios['market_cap'] / 10000000  # Convert to Crores
                                        st.write(f"**Market Cap:** ₹{market_cap_cr:,.0f} Cr")
                                
                                # Key recommendations
                                if analysis['recommendations']:
                                    st.write("**📋 Key Analysis Points:**")
                                    for rec in analysis['recommendations'][:4]:  # Show top 4 recommendations
                                        st.write(f"• {rec}")
                    
                    # Performance insights
                    st.subheader("💡 Key Insights")
                    
                    insights = []
                    
                    # Performance insights
                    if performance['weighted_cagr'] > 15:
                        insights.append("🎯 **Strong Performance**: Your portfolio CAGR exceeds 15%, indicating excellent returns.")
                    elif performance['weighted_cagr'] > 10:
                        insights.append("👍 **Good Performance**: Your portfolio is generating solid returns above 10% CAGR.")
                    elif performance['weighted_cagr'] < 5:
                        insights.append("🔍 **Review Needed**: Portfolio CAGR is below 5%, consider reviewing your stock selection.")
                    
                    # Benchmark comparison insights
                    if benchmarks and 'Nifty50' in benchmarks:
                        nifty_outperformance = performance['weighted_cagr'] - benchmarks['Nifty50']['cagr']
                        if nifty_outperformance > 2:
                            insights.append(f"🏆 **Beating the Market**: You're outperforming Nifty50 by {nifty_outperformance:.1f}%!")
                        elif nifty_outperformance < -2:
                            insights.append(f"📊 **Consider Index Funds**: You're underperforming Nifty50 by {abs(nifty_outperformance):.1f}%.")
                    
                    # Diversification insights
                    if allocation_data:
                        div_score = allocation_data['risk_analysis']['diversification_score']
                        if div_score >= 80:
                            insights.append("✅ **Well Diversified**: Your portfolio has excellent diversification across stocks and sectors.")
                        elif div_score >= 60:
                            insights.append("⚖️ **Moderate Diversification**: Consider adding more sectors or reducing concentration.")
                        else:
                            insights.append("⚠️ **High Concentration Risk**: Your portfolio lacks diversification - consider spreading investments.")
                    
                    # Ratio-based insights
                    if ratio_analysis:
                        positive_stocks = sum(1 for stock in ratio_analysis if stock['analysis']['overall_flag'] == 'POSITIVE')
                        negative_stocks = sum(1 for stock in ratio_analysis if stock['analysis']['overall_flag'] == 'NEGATIVE')
                        buy_stocks = sum(1 for stock in ratio_analysis if stock['analysis']['action'] in ['BUY MORE', 'HOLD & ACCUMULATE'])
                        sell_stocks = sum(1 for stock in ratio_analysis if stock['analysis']['action'] == 'CONSIDER SELLING')
                        
                        if positive_stocks > len(ratio_analysis) * 0.6:
                            insights.append(f"💎 **Strong Fundamentals**: {positive_stocks}/{len(ratio_analysis)} stocks show positive ratio analysis")
                        elif negative_stocks > len(ratio_analysis) * 0.4:
                            insights.append(f"⚠️ **Review Portfolio**: {negative_stocks}/{len(ratio_analysis)} stocks show concerning ratios")
                        
                        if buy_stocks > 0:
                            insights.append(f"🎯 **Accumulation Opportunities**: {buy_stocks} stocks recommended for buying/accumulating")
                        
                        if sell_stocks > 0:
                            insights.append(f"🔍 **Review Required**: {sell_stocks} stocks recommended for potential selling")
                    
                    # Display insights
                    for insight in insights:
                        st.info(insight)
                    
                    # Export enhanced data
                    st.subheader("💾 Export Enhanced Data")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        # Portfolio summary CSV
                        summary_csv = filtered_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Portfolio Analysis",
                            data=summary_csv,
                            file_name=f"portfolio_analysis_{datetime.now().strftime('%Y%m%d')}.csv",
                            mime="text/csv"
                        )
                    
                    with col2:
                        # Allocation data CSV
                        if allocation_data:
                            allocation_csv = allocation_data['stock_allocation'].to_csv(index=False)
                            st.download_button(
                                label="📥 Download Allocation Data",
                                data=allocation_csv,
                                file_name=f"portfolio_allocation_{datetime.now().strftime('%Y%m%d')}.csv",
                                mime="text/csv"
                            )
                    
                    with col3:
                        # Ratio analysis CSV
                        if ratio_analysis:
                            ratio_data = []
                            for stock in ratio_analysis:
                                ratio_data.append({
                                    'Symbol': stock['Symbol'],
                                    'Sector': stock['Sector'],
                                    'Overall_Flag': stock['analysis']['overall_flag'],
                                    'Action': stock['analysis']['action'],
                                    'P/E_Ratio': stock['ratios'].get('pe_ratio', 'N/A'),
                                    'P/B_Ratio': stock['ratios'].get('pb_ratio', 'N/A'),
                                    'ROE': stock['ratios'].get('roe', 'N/A'),
                                    'Debt_Equity': stock['ratios'].get('debt_to_equity', 'N/A'),
                                    'Dividend_Yield': stock['ratios'].get('dividend_yield', 'N/A')
                                })
                            
                            ratio_df = pd.DataFrame(ratio_data)
                            ratio_csv = ratio_df.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Ratio Analysis",
                                data=ratio_csv,
                                file_name=f"portfolio_ratios_{datetime.now().strftime('%Y%m%d')}.csv",
                                mime="text/csv"
                            )
                else:
                    st.error("Could not calculate portfolio performance. Please check your data.")
        else:
            st.error("Could not load portfolio data. Please check your CSV format.")
    
    else:
        # Show sample portfolio for demo
        st.subheader("📖 Sample Portfolio Analysis")
        st.info("Upload your CSV to see live analysis, or view this sample portfolio:")
        
        # Create sample data
        sample_data = {
            'Symbol': ['INFY', 'TCS', 'RELIANCE', 'HDFCBANK', 'ITC'],
            'Quantity': [100, 50, 25, 40, 200],
            'Avg_Price': [1500, 3200, 2400, 1600, 220],
            'Current_Value': [165000, 185000, 67500, 70400, 49600],
            'P&L_%': [10.0, 15.6, 12.5, 10.0, 12.7],
            'Sector': ['Technology', 'Technology', 'Energy', 'Banking', 'FMCG']
        }
        
        sample_df = pd.DataFrame(sample_data)
        st.dataframe(sample_df, use_container_width=True)
        
        st.markdown("""
        **What you'll get with your portfolio:**
        - 📊 **Real-time performance metrics** (Current Value, P&L, CAGR)
        - 📈 **Benchmark comparison** vs Nifty50 and S&P500  
        - 🥧 **Sector allocation analysis** with risk assessment
        - 🎯 **Diversification score** and concentration warnings
        - 💡 **Actionable insights** based on your holdings
        - 📥 **Export capabilities** for further analysis
        """)

if __name__ == "__main__":
    render()