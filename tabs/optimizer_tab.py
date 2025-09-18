import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from utils.portfolio_optimizer import PortfolioOptimizer
from utils.portfolio_analyzer import PortfolioAnalyzer

def render():
    st.header("⚖️ Portfolio Optimizer")
    
    optimizer = PortfolioOptimizer()
    analyzer = PortfolioAnalyzer()
    
    # Portfolio Input Methods
    st.subheader("📊 Portfolio Input")
    
    input_method = st.radio(
        "How would you like to input your portfolio?",
        ["Upload CSV File", "Manual Entry", "Select from Screener Results"]
    )
    
    tickers = []
    current_portfolio = pd.DataFrame()
    
    if input_method == "Upload CSV File":
        uploaded_file = st.file_uploader(
            "Upload your portfolio CSV",
            type=['csv'],
            key="optimizer_csv_uploader",
            help="Same format as Portfolio tab: Ticker, Quantity, AvgPrice, InvestedValue"
        )
        
        if uploaded_file:
            file_content = uploaded_file.getvalue().decode('utf-8')
            current_portfolio = analyzer.parse_portfolio_csv(file_content)
            
            if not current_portfolio.empty:
                st.success(f"✅ Portfolio loaded! {len(current_portfolio)} stocks found.")
                tickers = current_portfolio['Ticker'].tolist()
                
                # Analyze current portfolio for optimization
                analyzed_portfolio = analyzer.analyze_portfolio(current_portfolio)
                if not analyzed_portfolio.empty:
                    st.subheader("📈 Current Portfolio Analysis")
                    current_metrics = optimizer.calculate_portfolio_metrics(analyzed_portfolio)
                    
                    if current_metrics:
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Annual Return", f"{current_metrics['annual_return']:.1%}")
                        with col2:
                            st.metric("Volatility", f"{current_metrics['annual_volatility']:.1%}")
                        with col3:
                            st.metric("Sharpe Ratio", f"{current_metrics['sharpe_ratio']:.2f}")
                        with col4:
                            st.metric("Max Drawdown", f"{current_metrics['max_drawdown']:.1%}")
                        
                        # Current allocation
                        st.subheader("🥧 Current Allocation")
                        weights_df = pd.DataFrame(
                            list(current_metrics['current_weights'].items()),
                            columns=['Ticker', 'Weight']
                        )
                        weights_df['Weight %'] = weights_df['Weight'] * 100
                        
                        fig_current = px.pie(
                            weights_df,
                            values='Weight',
                            names='Ticker',
                            title="Current Portfolio Weights"
                        )
                        st.plotly_chart(fig_current, use_container_width=True)
    
    elif input_method == "Manual Entry":
        st.subheader("✏️ Enter Stock Tickers")
        
        # Text area for multiple tickers
        ticker_input = st.text_area(
            "Enter stock tickers (one per line)",
            placeholder="RELIANCE\nTCS\nINFY\nHDFC\nAAPL\nMSFT",
            height=150
        )
        
        if ticker_input:
            tickers = [ticker.strip().upper() for ticker in ticker_input.split('\n') if ticker.strip()]
            st.success(f"✅ {len(tickers)} tickers entered: {', '.join(tickers)}")
    
    else:  # Select from Screener Results
        st.info("💡 First run the Stock Screener to get recommendations, then return here to optimize allocation.")
        
        # Manual input as fallback
        st.subheader("✏️ Or Enter Tickers Manually")
        ticker_input = st.text_input(
            "Enter tickers separated by commas",
            placeholder="RELIANCE, TCS, INFY, AAPL, MSFT"
        )
        
        if ticker_input:
            tickers = [ticker.strip().upper() for ticker in ticker_input.split(',') if ticker.strip()]
            st.success(f"✅ {len(tickers)} tickers entered: {', '.join(tickers)}")
    
    # Optimization Settings
    if tickers and len(tickers) >= 2:
        st.subheader("🎯 Optimization Settings")
        
        opt_col1, opt_col2 = st.columns(2)
        
        with opt_col1:
            optimization_type = st.selectbox(
                "Optimization Objective",
                ["max_sharpe", "min_volatility", "target_return"],
                format_func=lambda x: {
                    "max_sharpe": "Maximum Sharpe Ratio",
                    "min_volatility": "Minimum Volatility", 
                    "target_return": "Target Return"
                }[x],
                help="Choose your optimization goal"
            )
        
        with opt_col2:
            investment_amount = st.number_input(
                "Total Investment Amount (₹)",
                min_value=10000,
                value=100000,
                step=10000,
                help="Total amount to invest/rebalance"
            )
        
        # Risk tolerance settings
        st.subheader("⚠️ Risk Preferences")
        
        risk_col1, risk_col2 = st.columns(2)
        
        with risk_col1:
            max_weight = st.slider(
                "Maximum Weight per Stock (%)",
                min_value=5,
                max_value=50,
                value=25,
                help="Maximum allocation to any single stock"
            ) / 100
        
        with risk_col2:
            min_weight = st.slider(
                "Minimum Weight per Stock (%)",
                min_value=0,
                max_value=10,
                value=2,
                help="Minimum allocation to include a stock"
            ) / 100
        
        # Run Optimization
        if st.button("⚡ Optimize Portfolio", type="primary"):
            with st.spinner("🔄 Running portfolio optimization..."):
                optimization_result = optimizer.optimize_portfolio(
                    tickers, 
                    investment_amount, 
                    optimization_type
                )
            
            if optimization_result:
                st.success("✅ Portfolio optimization completed!")
                
                # Display Results
                st.subheader("📊 Optimization Results")
                st.markdown(f"**{optimization_result['title']}**")
                
                # Performance Metrics
                perf_col1, perf_col2, perf_col3 = st.columns(3)
                
                with perf_col1:
                    st.metric(
                        "Expected Annual Return",
                        f"{optimization_result['performance']['expected_return']:.1%}"
                    )
                with perf_col2:
                    st.metric(
                        "Annual Volatility", 
                        f"{optimization_result['performance']['volatility']:.1%}"
                    )
                with perf_col3:
                    st.metric(
                        "Sharpe Ratio",
                        f"{optimization_result['performance']['sharpe_ratio']:.2f}"
                    )
                
                # Optimal Weights
                st.subheader("🎯 Optimal Allocation")
                
                weights_data = []
                for ticker, weight in optimization_result['weights'].items():
                    if weight > 0.001:  # Only show meaningful allocations
                        weights_data.append({
                            'Ticker': ticker,
                            'Weight': weight,
                            'Weight %': f"{weight * 100:.1f}%",
                            'Amount (₹)': f"₹{weight * investment_amount:,.0f}"
                        })
                
                weights_df = pd.DataFrame(weights_data)
                weights_df = weights_df.sort_values('Weight', ascending=False)
                
                st.dataframe(weights_df, use_container_width=True)
                
                # Allocation Visualization
                allocation_col1, allocation_col2 = st.columns(2)
                
                with allocation_col1:
                    # Pie chart of optimal weights
                    fig_optimal = px.pie(
                        weights_df,
                        values='Weight',
                        names='Ticker',
                        title="Optimal Portfolio Allocation"
                    )
                    st.plotly_chart(fig_optimal, use_container_width=True)
                
                with allocation_col2:
                    # Bar chart of weights
                    fig_bar = px.bar(
                        weights_df,
                        x='Ticker',
                        y='Weight',
                        title="Allocation by Stock",
                        text='Weight %'
                    )
                    fig_bar.update_traces(textposition='outside')
                    st.plotly_chart(fig_bar, use_container_width=True)
                
                # Discrete Allocation (Actual Shares)
                st.subheader("📋 Discrete Allocation")
                st.info(f"Based on ₹{investment_amount:,} investment amount")
                
                allocation_data = []
                total_allocated = 0
                
                for ticker, shares in optimization_result['allocation'].items():
                    if shares > 0:
                        # Get current price (simplified)
                        weight = optimization_result['weights'].get(ticker, 0)
                        estimated_price = (weight * investment_amount) / shares if shares > 0 else 0
                        allocation_value = shares * estimated_price
                        
                        allocation_data.append({
                            'Ticker': ticker,
                            'Shares': shares,
                            'Est. Price (₹)': f"₹{estimated_price:.2f}",
                            'Total Value (₹)': f"₹{allocation_value:,.0f}",
                            'Weight %': f"{weight * 100:.1f}%"
                        })
                        
                        total_allocated += allocation_value
                
                if allocation_data:
                    allocation_df = pd.DataFrame(allocation_data)
                    st.dataframe(allocation_df, use_container_width=True)
                    
                    leftover = optimization_result.get('leftover', 0)
                    
                    summary_col1, summary_col2, summary_col3 = st.columns(3)
                    
                    with summary_col1:
                        st.metric("💰 Total Allocated", f"₹{total_allocated:,.0f}")
                    with summary_col2:
                        st.metric("💸 Cash Leftover", f"₹{leftover:,.0f}")
                    with summary_col3:
                        allocation_rate = (total_allocated / investment_amount) * 100
                        st.metric("📊 Allocation Rate", f"{allocation_rate:.1f}%")
                
                # Rebalancing Recommendations
                if not current_portfolio.empty:
                    st.subheader("🔄 Rebalancing Recommendations")
                    
                    analyzed_current = analyzer.analyze_portfolio(current_portfolio)
                    if not analyzed_current.empty:
                        rebalancing_suggestions = optimizer.suggest_rebalancing(
                            analyzed_current, 
                            optimization_result['weights'],
                            investment_amount
                        )
                        
                        if not rebalancing_suggestions.empty:
                            st.dataframe(
                                rebalancing_suggestions,
                                use_container_width=True,
                                column_config={
                                    "Action": st.column_config.TextColumn(
                                        "Action",
                                        width="small"
                                    ),
                                    "Shares": st.column_config.NumberColumn(
                                        "Shares",
                                        format="%.2f"
                                    ),
                                    "Value": st.column_config.NumberColumn(
                                        "Value (₹)",
                                        format="₹%.0f"
                                    )
                                }
                            )
                        else:
                            st.info("✅ Your current portfolio is well-balanced according to the optimization!")
                
                # Efficient Frontier
                st.subheader("📈 Efficient Frontier Analysis")
                
                if len(tickers) >= 3:  # Need at least 3 assets for meaningful frontier
                    if st.button("📊 Generate Efficient Frontier"):
                        with st.spinner("Calculating efficient frontier..."):
                            frontier_data = optimizer.generate_efficient_frontier(tickers)
                        
                        if frontier_data is not None and not frontier_data.empty:
                            fig_frontier = go.Figure()
                            
                            # Plot efficient frontier
                            fig_frontier.add_trace(go.Scatter(
                                x=frontier_data['volatility'],
                                y=frontier_data['return'],
                                mode='lines+markers',
                                name='Efficient Frontier',
                                line=dict(color='blue', width=2)
                            ))
                            
                            # Mark optimal portfolio
                            opt_return = optimization_result['performance']['expected_return']
                            opt_vol = optimization_result['performance']['volatility']
                            
                            fig_frontier.add_trace(go.Scatter(
                                x=[opt_vol],
                                y=[opt_return],
                                mode='markers',
                                marker=dict(color='red', size=12, symbol='star'),
                                name='Optimal Portfolio'
                            ))
                            
                            fig_frontier.update_layout(
                                title='Efficient Frontier Analysis',
                                xaxis_title='Volatility (Risk)',
                                yaxis_title='Expected Return',
                                hovermode='closest'
                            )
                            
                            st.plotly_chart(fig_frontier, use_container_width=True)
                            
                            st.info("💡 The red star shows your optimized portfolio on the efficient frontier.")
                        else:
                            st.error("Unable to generate efficient frontier. Need more historical data.")
                else:
                    st.info("💡 Add more stocks (minimum 3) to generate efficient frontier analysis.")
                
                # Export Optimization Results
                st.subheader("💾 Export Results")
                
                export_col1, export_col2 = st.columns(2)
                
                with export_col1:
                    if not weights_df.empty:
                        csv_weights = weights_df.to_csv(index=False)
                        st.download_button(
                            label="📁 Download Weights CSV",
                            data=csv_weights,
                            file_name="optimized_portfolio_weights.csv",
                            mime="text/csv"
                        )
                
                with export_col2:
                    if allocation_data:
                        csv_allocation = pd.DataFrame(allocation_data).to_csv(index=False)
                        st.download_button(
                            label="📁 Download Allocation CSV",
                            data=csv_allocation,
                            file_name="portfolio_allocation.csv",
                            mime="text/csv"
                        )
            
            else:
                st.error("❌ Optimization failed. Please check your stock tickers and try again.")
    
    elif tickers and len(tickers) == 1:
        st.warning("⚠️ Portfolio optimization requires at least 2 stocks. Please add more tickers.")
    
    else:
        st.info("👆 Please input your portfolio using one of the methods above to start optimization.")
    
    # Educational Content
    with st.expander("📚 Portfolio Optimization Guide"):
        st.markdown("""
        ### What is Portfolio Optimization?
        Portfolio optimization is the process of selecting the best mix of investments to achieve your financial goals while managing risk.
        
        ### Optimization Methods:
        
        #### 🎯 Maximum Sharpe Ratio
        - **Goal:** Maximize risk-adjusted returns
        - **Best For:** Balanced investors seeking optimal risk-return trade-off
        - **Formula:** (Return - Risk-free rate) / Volatility
        
        #### 🛡️ Minimum Volatility  
        - **Goal:** Minimize portfolio risk
        - **Best For:** Conservative investors prioritizing capital preservation
        - **Result:** Lowest possible portfolio volatility
        
        #### 📈 Target Return
        - **Goal:** Achieve specific return with minimum risk
        - **Best For:** Investors with specific return requirements
        - **Input:** Desired annual return percentage
        
        ### Key Concepts:
        
        **Diversification:** Spreading investments across different assets to reduce risk
        
        **Correlation:** How stocks move relative to each other
        - Positive correlation: Move in same direction
        - Negative correlation: Move in opposite directions
        - Zero correlation: Independent movements
        
        **Efficient Frontier:** The set of optimal portfolios offering the highest expected return for each level of risk
        
        **Sharpe Ratio:** Risk-adjusted return measure (higher is better)
        
        ### Important Notes:
        - Optimization is based on historical data
        - Past performance doesn't guarantee future results
        - Regular rebalancing may be required
        - Consider transaction costs in real implementation
        - Review and adjust periodically based on market conditions
        """)
    
    # Risk Warning
    st.warning("""
    ⚠️ **Risk Disclaimer:** 
    Portfolio optimization is based on historical data and mathematical models. Market conditions can change, and past performance 
    does not guarantee future results. Always consider your risk tolerance, investment timeline, and financial goals before making 
    investment decisions. Consider consulting with a financial advisor for personalized advice.
    """)
