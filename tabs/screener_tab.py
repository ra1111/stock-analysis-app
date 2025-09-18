import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from utils.stock_screener import StockScreener

def render():
    st.header("🔍 Stock Screener")
    
    screener = StockScreener()
    
    # Strategy Selection
    st.subheader("📈 Investment Strategy")
    
    strategy_col1, strategy_col2 = st.columns(2)
    
    with strategy_col1:
        strategy = st.selectbox(
            "Select Investment Strategy",
            ["Value", "Moonshot", "Growth"],
            help="Choose your preferred investment approach"
        )
    
    with strategy_col2:
        market = st.selectbox(
            "Select Market",
            ["both", "indian", "us"],
            format_func=lambda x: {
                "both": "Both Indian & US Markets", 
                "indian": "Indian Market Only", 
                "us": "US Market Only"
            }[x]
        )
    
    # Display strategy description
    st.markdown(screener.get_strategy_description(strategy))
    
    # Screening Section
    st.subheader("🎯 Stock Screening Results")
    
    if st.button("🔍 Screen Stocks", type="primary"):
        with st.spinner("🔎 Screening stocks based on your criteria..."):
            results_df = screener.screen_stocks(strategy, market)
        
        # Get transparency report for additional insights
        transparency_report = screener.get_screening_report()
        
        if not results_df.empty:
            st.success(f"✅ Found {len(results_df)} stocks matching your criteria!")
            
            # Enhanced Summary Statistics with Data Quality
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric("📊 Total Matches", len(results_df))
            with col2:
                if 'Market' in results_df.columns:
                    indian_count = len(results_df[results_df['Market'] == 'Indian'])
                    st.metric("🇮🇳 Indian Stocks", indian_count)
                else:
                    st.metric("🇮🇳 Indian Stocks", 0)
            with col3:
                if 'Market' in results_df.columns:
                    us_count = len(results_df[results_df['Market'] == 'US'])
                    st.metric("🇺🇸 US Stocks", us_count)
                else:
                    st.metric("🇺🇸 US Stocks", 0)
            with col4:
                avg_score = results_df['Score'].mean() if 'Score' in results_df.columns else 0
                st.metric("⭐ Avg Score", f"{avg_score:.1f}")
            with col5:
                if not transparency_report.empty and 'Data Quality %' in transparency_report.columns:
                    avg_quality = transparency_report['Data Quality %'].mean()
                    st.metric("📊 Avg Data Quality", f"{avg_quality:.1f}%")
                else:
                    st.metric("📊 Avg Data Quality", "N/A")
            
            # Results Table
            st.subheader("📋 Screening Results")
            
            # Format the results for better display
            display_df = results_df.copy()
            
            # Format numeric columns with enhanced None handling
            numeric_columns = ['Current Price', 'Market Cap', 'PE Ratio', 'PB Ratio', 
                             'ROE %', 'Debt/Equity', 'Dividend Yield %', 'Profit Margin %', 
                             'Revenue Growth %', 'PEG Ratio']
            
            for col in numeric_columns:
                if col in display_df.columns:
                    if col == 'Market Cap':
                        display_df[col] = display_df[col].apply(
                            lambda x: f"₹{x/1e9:.1f}B" if x is not None and x > 0 else "N/A"
                        )
                    elif col == 'Current Price':
                        display_df[col] = display_df[col].apply(
                            lambda x: f"₹{x:.2f}" if x is not None and x > 0 else "N/A"
                        )
                    elif col in ['PE Ratio', 'PB Ratio', 'PEG Ratio']:
                        display_df[col] = display_df[col].apply(
                            lambda x: f"{x:.2f}" if x is not None and x > 0 else "N/A"
                        )
                    elif '%' in col:
                        display_df[col] = display_df[col].apply(
                            lambda x: f"{x:.1f}%" if x is not None else "N/A"
                        )
                    else:
                        display_df[col] = display_df[col].apply(
                            lambda x: f"{x:.2f}" if x is not None else "N/A"
                        )
            
            # Format additional columns for transparency
            if 'Data Quality %' in display_df.columns:
                display_df['Data Quality %'] = display_df['Data Quality %'].apply(
                    lambda x: f"{x:.1f}%" if x is not None else "N/A"
                )
            
            # Display with enhanced sorting options
            sort_options = ["Score", "Data Quality %", "Market Cap", "PE Ratio", "ROE %", "Revenue Growth %"]
            available_sort_options = [opt for opt in sort_options if opt in display_df.columns]
            
            sort_by = st.selectbox(
                "Sort by",
                available_sort_options,
                index=0 if "Score" in available_sort_options else 0
            )
            
            if sort_by in display_df.columns:
                if sort_by == "Score":
                    display_df_sorted = display_df.sort_values(sort_by, ascending=False)
                else:
                    # For other columns, sort by original numeric values
                    display_df_sorted = display_df.loc[results_df.sort_values(sort_by, ascending=False).index]
            else:
                display_df_sorted = display_df
            
            # Enhanced dataframe with data quality information
            enhanced_columns = {
                "Ticker": st.column_config.TextColumn("Ticker", width="small"),
                "Market": st.column_config.TextColumn("Market", width="small"),
                "Score": st.column_config.NumberColumn("Score", width="small"),
            }
            
            if 'Data Quality %' in display_df_sorted.columns:
                enhanced_columns["Data Quality %"] = st.column_config.TextColumn("Data Quality", width="small")
            if 'Status' in display_df_sorted.columns:
                enhanced_columns["Status"] = st.column_config.TextColumn("Status", width="small")
            
            st.dataframe(
                display_df_sorted,
                width='stretch',
                column_config=enhanced_columns
            )
            
            # Top Picks Section with Enhanced Information
            st.subheader("🏆 Top Recommendations")
            top_picks = results_df.nlargest(5, 'Score') if 'Score' in results_df.columns else results_df.head(5)
            
            for i, (_, stock) in enumerate(top_picks.iterrows(), 1):
                # Enhanced header with data quality indicator
                data_quality = stock.get('Data Quality %', 0)
                data_quality = data_quality if data_quality is not None else 0
                quality_icon = "🟢" if data_quality >= 70 else "🟡" if data_quality >= 50 else "🔴"
                
                with st.expander(f"#{i} {stock['Ticker']} ({stock.get('Market', 'Unknown')}) - Score: {stock.get('Score', 0):.1f} {quality_icon}"):
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.write("**Valuation Metrics:**")
                        current_price = stock.get('Current Price')
                        market_cap = stock.get('Market Cap')
                        pe_ratio = stock.get('PE Ratio')
                        pb_ratio = stock.get('PB Ratio')
                        
                        st.write(f"Current Price: {f'₹{current_price:.2f}' if current_price is not None and current_price > 0 else 'N/A'}")
                        st.write(f"Market Cap: {f'₹{market_cap/1e9:.1f}B' if market_cap is not None and market_cap > 0 else 'N/A'}")
                        st.write(f"PE Ratio: {f'{pe_ratio:.2f}' if pe_ratio is not None and pe_ratio > 0 else 'N/A'}")
                        st.write(f"PB Ratio: {f'{pb_ratio:.2f}' if pb_ratio is not None and pb_ratio > 0 else 'N/A'}")
                    
                    with col2:
                        st.write("**Profitability Metrics:**")
                        roe = stock.get('ROE %')
                        profit_margin = stock.get('Profit Margin %')
                        revenue_growth = stock.get('Revenue Growth %')
                        dividend_yield = stock.get('Dividend Yield %')
                        
                        st.write(f"ROE: {f'{roe:.1f}%' if roe is not None else 'N/A'}")
                        st.write(f"Profit Margin: {f'{profit_margin:.1f}%' if profit_margin is not None else 'N/A'}")
                        st.write(f"Revenue Growth: {f'{revenue_growth:.1f}%' if revenue_growth is not None else 'N/A'}")
                        st.write(f"Dividend Yield: {f'{dividend_yield:.1f}%' if dividend_yield is not None else 'N/A'}")
                    
                    with col3:
                        st.write("**Risk & Quality:**")
                        debt_equity = stock.get('Debt/Equity')
                        peg_ratio = stock.get('PEG Ratio')
                        
                        st.write(f"Debt/Equity: {f'{debt_equity:.2f}' if debt_equity is not None else 'N/A'}")
                        st.write(f"PEG Ratio: {f'{peg_ratio:.2f}' if peg_ratio is not None else 'N/A'}")
                        st.write(f"Data Quality: {data_quality:.1f}%")
                        
                        # Missing fields info
                        missing_fields = stock.get('Missing Fields', 0)
                        if missing_fields > 0:
                            st.write(f"⚠️ Missing {missing_fields} fields")
                    
                    with col4:
                        st.write("**Decision Factors:**")
                        
                        # Investment recommendation
                        score = stock.get('Score', 0)
                        if score >= 50:
                            st.success("🟢 Strong Buy Candidate")
                        elif score >= 30:
                            st.warning("🟡 Moderate Buy Candidate")
                        else:
                            st.info("🔵 Consider with Caution")
                        
                        # Show reasons if available
                        reasons = stock.get('Reasons', '')
                        if reasons:
                            st.write("**Key Factors:**")
                            # Show first 3 reasons to avoid clutter
                            reason_list = reasons.split(';')[:3]
                            for reason in reason_list:
                                if reason.strip():
                                    st.write(f"• {reason.strip()}")
                            if len(reasons.split(';')) > 3:
                                st.write("• ...and more")
            
            # Visualization Section
            st.subheader("📊 Data Visualization")
            
            viz_col1, viz_col2 = st.columns(2)
            
            with viz_col1:
                # Score distribution
                if 'Score' in results_df.columns and len(results_df) > 1:
                    fig_score = px.histogram(
                        results_df, 
                        x='Score', 
                        nbins=10,
                        title="Score Distribution",
                        color_discrete_sequence=['#1f77b4']
                    )
                    fig_score.update_layout(
                        xaxis_title="Score",
                        yaxis_title="Number of Stocks"
                    )
                    st.plotly_chart(fig_score, use_container_width=True)
            
            with viz_col2:
                # Market distribution
                if 'Market' in results_df.columns:
                    market_counts = results_df['Market'].value_counts()
                    fig_market = px.pie(
                        values=market_counts.values,
                        names=market_counts.index,
                        title="Market Distribution",
                        color_discrete_sequence=px.colors.qualitative.Set2
                    )
                    st.plotly_chart(fig_market, use_container_width=True)
            
            # Scatter plot analysis
            if len(results_df) > 1:
                st.subheader("🎯 Risk-Return Analysis")
                
                x_axis = st.selectbox("X-Axis", ['PE Ratio', 'PB Ratio', 'Debt/Equity', 'Market Cap'], key='x_axis')
                y_axis = st.selectbox("Y-Axis", ['ROE %', 'Revenue Growth %', 'Profit Margin %', 'Score'], key='y_axis')
                
                if x_axis in results_df.columns and y_axis in results_df.columns:
                    # Filter out zero values for better visualization
                    plot_df = results_df[(results_df[x_axis] > 0) & (results_df[y_axis] > 0)]
                    
                    if len(plot_df) > 0:
                        fig_scatter = px.scatter(
                            plot_df,
                            x=x_axis,
                            y=y_axis,
                            color='Market' if 'Market' in plot_df.columns else None,
                            size='Score' if 'Score' in plot_df.columns else None,
                            hover_data=['Ticker'],
                            title=f"{y_axis} vs {x_axis}",
                            size_max=20
                        )
                        st.plotly_chart(fig_scatter, use_container_width=True)
            
            # Transparency Report Section
            if not transparency_report.empty:
                st.subheader("🔍 Screening Transparency Report")
                
                with st.expander("📊 View Complete Analysis Details", expanded=False):
                    st.write("This report shows all stocks analyzed, including those excluded, with detailed reasons.")
                    
                    # Summary of transparency report
                    total_analyzed = len(transparency_report)
                    included_count = len(transparency_report[transparency_report.get('Status', '') == 'Included'])
                    excluded_count = len(transparency_report[transparency_report.get('Status', '') == 'Excluded'])
                    error_count = len(transparency_report[transparency_report.get('Status', '') == 'Error'])
                    
                    trans_col1, trans_col2, trans_col3, trans_col4 = st.columns(4)
                    with trans_col1:
                        st.metric("📊 Total Analyzed", total_analyzed)
                    with trans_col2:
                        st.metric("✅ Included", included_count)
                    with trans_col3:
                        st.metric("❌ Excluded", excluded_count)
                    with trans_col4:
                        st.metric("⚠️ Errors", error_count)
                    
                    # Filter options for transparency report
                    status_filter = st.selectbox(
                        "Filter by Status:",
                        ["All", "Included", "Excluded", "Error"],
                        key="transparency_filter"
                    )
                    
                    if status_filter != "All":
                        filtered_report = transparency_report[transparency_report.get('Status', '') == status_filter]
                    else:
                        filtered_report = transparency_report
                    
                    # Display filtered transparency report
                    if not filtered_report.empty:
                        # Format the transparency report for display
                        display_columns = ['Ticker', 'Market', 'Status', 'Score', 'Data Quality %', 'Reasons']
                        available_columns = [col for col in display_columns if col in filtered_report.columns]
                        
                        st.dataframe(
                            filtered_report[available_columns],
                            width='stretch',
                            column_config={
                                "Ticker": st.column_config.TextColumn("Ticker", width="small"),
                                "Market": st.column_config.TextColumn("Market", width="small"),
                                "Status": st.column_config.TextColumn("Status", width="small"),
                                "Score": st.column_config.NumberColumn("Score", width="small"),
                                "Data Quality %": st.column_config.NumberColumn("Data Quality", width="small"),
                                "Reasons": st.column_config.TextColumn("Reasons", width="large")
                            }
                        )
            
            # Export functionality
            st.subheader("💾 Export Results")
            
            export_col1, export_col2, export_col3 = st.columns(3)
            
            with export_col1:
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="📁 Download Results CSV",
                    data=csv,
                    file_name=f"{strategy.lower()}_screener_results.csv",
                    mime="text/csv"
                )
            
            with export_col2:
                if not transparency_report.empty:
                    transparency_csv = transparency_report.to_csv(index=False)
                    st.download_button(
                        label="📋 Download Full Report CSV",
                        data=transparency_csv,
                        file_name=f"{strategy.lower()}_transparency_report.csv",
                        mime="text/csv"
                    )
            
            with export_col3:
                st.info(f"💡 Found {len(results_df)} qualifying stocks from {len(transparency_report) if not transparency_report.empty else 'unknown'} analyzed")
        
        else:
            st.warning("🔍 No stocks found matching the current criteria. Try adjusting your strategy or market selection.")
            
            # Show transparency report even when no matches
            if not transparency_report.empty:
                st.subheader("🔍 Why No Matches Found?")
                
                # Analysis of why stocks were excluded
                excluded_stocks = transparency_report[transparency_report.get('Status', '') == 'Excluded']
                error_stocks = transparency_report[transparency_report.get('Status', '') == 'Error']
                
                analysis_col1, analysis_col2 = st.columns(2)
                
                with analysis_col1:
                    st.write("**Exclusion Summary:**")
                    if not excluded_stocks.empty:
                        st.write(f"• {len(excluded_stocks)} stocks excluded due to criteria")
                        
                        # Most common exclusion reasons
                        if 'Reasons' in excluded_stocks.columns:
                            all_reasons = []
                            reasons_series = excluded_stocks['Reasons']
                            if hasattr(reasons_series, 'dropna'):
                                for reasons_str in reasons_series.dropna():
                                    all_reasons.extend([r.strip() for r in str(reasons_str).split(';') if r.strip()])
                            else:
                                # Handle case where it might be a different data structure
                                for reasons_str in excluded_stocks['Reasons']:
                                    if reasons_str is not None:
                                        all_reasons.extend([r.strip() for r in str(reasons_str).split(';') if r.strip()])
                            
                            if all_reasons:
                                from collections import Counter
                                common_reasons = Counter(all_reasons).most_common(3)
                                st.write("**Top exclusion reasons:**")
                                for reason, count in common_reasons:
                                    if reason and not reason.startswith('?'):
                                        st.write(f"• {reason} ({count} stocks)")
                    
                    if not error_stocks.empty:
                        st.write(f"• {len(error_stocks)} stocks had data errors")
                
                with analysis_col2:
                    st.write("**Data Quality Issues:**")
                    if 'Data Quality %' in transparency_report.columns:
                        avg_quality = transparency_report['Data Quality %'].mean()
                        low_quality_count = len(transparency_report[transparency_report['Data Quality %'] < 50])
                        
                        st.write(f"• Average data quality: {avg_quality:.1f}%")
                        st.write(f"• {low_quality_count} stocks with poor data quality (<50%)")
                    
                    if 'Missing Fields' in transparency_report.columns:
                        high_missing = len(transparency_report[transparency_report['Missing Fields'] > 5])
                        st.write(f"• {high_missing} stocks missing >5 key metrics")
                
                # Show sample of excluded stocks
                with st.expander("📋 Sample of Excluded Stocks", expanded=False):
                    sample_excluded = excluded_stocks.head(10) if not excluded_stocks.empty else pd.DataFrame()
                    if not sample_excluded.empty:
                        display_cols = ['Ticker', 'Market', 'Data Quality %', 'Reasons']
                        available_cols = [col for col in display_cols if col in sample_excluded.columns]
                        st.dataframe(sample_excluded[available_cols], width='stretch')
            
            st.info("""
            **Possible solutions:**
            - Try a different strategy with more flexible criteria
            - Expand market selection to include both Indian and US markets
            - Check the transparency report above to understand specific exclusion reasons
            - Consider that current market conditions may not favor the selected strategy
            - Some stocks may have insufficient data quality for reliable screening
            """)
    
    # Data Quality Information
    with st.expander("📊 Understanding Data Quality"):
        st.markdown("""
        ### 🎯 Data Quality Scoring
        Our screener evaluates data quality for each stock:
        
        - **🟢 High Quality (70%+):** Most fundamental metrics available
        - **🟡 Medium Quality (50-70%):** Some key metrics missing but reliable analysis possible
        - **🔴 Low Quality (<50%):** Too many missing metrics for reliable screening
        
        ### 📈 Fallback Scoring System
        When some metrics are missing, we:
        
        1. **Calculate partial scores** based on available data
        2. **Normalize scores** to account for missing information
        3. **Apply quality penalties** (10% per missing key metric)
        4. **Ensure minimum threshold** (30% of full score maintained)
        
        ### 🔍 Transparency Features
        - **Inclusion/Exclusion Reasons:** See why each stock passed or failed
        - **Data Quality Metrics:** Understand data availability for each stock
        - **Criteria Flexibility:** Stocks need 70% of criteria to pass when data is available
        - **Fallback Handling:** Graceful handling of missing data instead of immediate exclusion
        
        ### ⚠️ Important Notes
        - Stocks with <30% data quality are excluded for safety
        - Missing market cap data typically indicates delisted or invalid tickers
        - Some metrics may be temporarily unavailable due to reporting delays
        """)
    
    # Strategy Comparison
    st.subheader("📈 Strategy Comparison Guide")
    
    with st.expander("💡 Which Strategy Should I Choose?"):
        st.markdown("""
        ### 🏛️ Value Strategy (Benjamin Graham Style)
        **Best For:** Conservative investors seeking undervalued, stable companies
        - Focus on established companies with strong balance sheets
        - Lower risk, steady returns over long term
        - Good for defensive portfolios
        - **Time Horizon:** 3-5+ years
        
        ### 🚀 Moonshot Strategy  
        **Best For:** Aggressive investors seeking high growth potential
        - Small to mid-cap companies with explosive growth potential
        - Higher risk, potentially higher returns
        - Good for growth-oriented portfolios
        - **Time Horizon:** 2-5 years with regular monitoring
        
        ### 📈 Growth Strategy
        **Best For:** Investors seeking companies with sustainable growth
        - Companies with consistent growth track record
        - Moderate to high risk with good return potential  
        - Balanced approach between value and moonshot
        - **Time Horizon:** 2-4 years
        
        ### 🎯 Portfolio Allocation Suggestion:
        - **Conservative:** 70% Value, 20% Growth, 10% Moonshot
        - **Balanced:** 40% Value, 40% Growth, 20% Moonshot  
        - **Aggressive:** 20% Value, 30% Growth, 50% Moonshot
        """)
    
    # Market Insights
    with st.expander("🌍 Market Insights"):
        st.markdown("""
        ### 🇮🇳 Indian Market Characteristics:
        - **High Growth Potential:** Emerging market with demographic dividend
        - **Volatility:** Higher volatility compared to developed markets
        - **Currency Risk:** For foreign investors
        - **Regulatory Environment:** Evolving regulatory framework
        
        ### 🇺🇸 US Market Characteristics:  
        - **Mature Market:** Established companies with global presence
        - **Lower Volatility:** Generally more stable than emerging markets
        - **Currency Stability:** USD as global reserve currency
        - **High Valuations:** Premium valuations in many sectors
        
        ### 💡 Diversification Benefits:
        - **Geographic Diversification:** Spread risk across different economies
        - **Currency Hedging:** Natural hedge against currency fluctuations
        - **Sector Exposure:** Access to different industry dynamics
        - **Economic Cycles:** Different markets may be in different phases
        """)
