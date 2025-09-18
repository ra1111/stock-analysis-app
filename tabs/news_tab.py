import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from utils.news_analyzer import NewsAnalyzer
from utils.portfolio_analyzer import PortfolioAnalyzer
from datetime import datetime, timedelta

def render():
    st.header("📰 News Analysis")
    
    news_analyzer = NewsAnalyzer()
    portfolio_analyzer = PortfolioAnalyzer()
    
    # Input Methods
    st.subheader("📊 Select Stocks for News Analysis")
    
    input_method = st.radio(
        "How would you like to input stocks for news analysis?",
        ["Upload Portfolio CSV", "Manual Entry"]
    )
    
    tickers = []
    
    if input_method == "Upload Portfolio CSV":
        uploaded_file = st.file_uploader(
            "Upload your portfolio CSV",
            type=['csv'],
            key="news_csv_uploader",
            help="Same format as Portfolio tab: Ticker, Quantity, AvgPrice, InvestedValue"
        )
        
        if uploaded_file:
            portfolio_df = portfolio_analyzer.load_portfolio_from_csv(uploaded_file)
            
            if portfolio_df is not None and not portfolio_df.empty:
                st.success(f"✅ Portfolio loaded! {len(portfolio_df)} stocks found.")
                tickers = portfolio_df['Symbol'].tolist()
                
                # Display loaded portfolio
                st.subheader("📈 Loaded Portfolio")
                display_portfolio = portfolio_df[['Symbol', 'Quantity', 'Invested_Value']].copy()
                display_portfolio['Invested_Value'] = display_portfolio['Invested_Value'].apply(lambda x: f"₹{x:,.0f}")
                display_portfolio = display_portfolio.rename(columns={'Symbol': 'Ticker', 'Invested_Value': 'InvestedValue'})
                st.dataframe(display_portfolio, use_container_width=True)
            else:
                st.error("❌ Unable to parse the uploaded CSV. Please check the format.")
    
    else:  # Manual Entry
        st.subheader("✏️ Enter Stock Tickers")
        
        ticker_input = st.text_area(
            "Enter stock tickers (one per line)",
            placeholder="RELIANCE\nTCS\nINFY\nHDFC\nAAPL\nMSFT\nGOOGL",
            height=150,
            help="Enter stock symbols, one per line. Indian stocks will automatically get .NS suffix if needed."
        )
        
        if ticker_input:
            tickers = [ticker.strip().upper() for ticker in ticker_input.split('\n') if ticker.strip()]
            st.success(f"✅ {len(tickers)} tickers entered: {', '.join(tickers)}")
    
    # News Analysis Section
    if tickers:
        st.subheader("🔍 News Analysis Options")
        
        analysis_col1, analysis_col2 = st.columns(2)
        
        with analysis_col1:
            analysis_scope = st.selectbox(
                "Analysis Scope",
                ["All Stocks", "Selected Stocks"],
                help="Choose whether to analyze all stocks or select specific ones"
            )
        
        with analysis_col2:
            max_news_per_stock = st.slider(
                "Max News per Stock",
                min_value=1,
                max_value=20,
                value=5,
                help="Maximum number of news articles to analyze per stock"
            )
        
        # Add days back filter
        days_col1, days_col2 = st.columns(2)
        
        with days_col1:
            days_back = st.slider(
                "Days Back to Search",
                min_value=1,
                max_value=30,
                value=7,
                help="How many days back to search for news"
            )
        
        with days_col2:
            show_critical_only = st.checkbox(
                "Show Critical News Only",
                value=False,
                help="Filter to show only high-impact, critical news"
            )
        
        # Stock Selection for "Selected Stocks" option
        selected_tickers = tickers
        if analysis_scope == "Selected Stocks":
            selected_tickers = st.multiselect(
                "Choose specific stocks to analyze",
                tickers,
                default=tickers[:3] if len(tickers) > 3 else tickers,
                help="Select specific stocks for focused news analysis"
            )
        
        # Filter Options
        st.subheader("🎛️ Analysis Filters")
        
        filter_col1, filter_col2, filter_col3 = st.columns(3)
        
        with filter_col1:
            materiality_filter = st.multiselect(
                "Materiality Level",
                ["High", "Medium", "Low"],
                default=["High", "Medium", "Low"],
                help="Filter news by materiality level"
            )
        
        with filter_col2:
            sentiment_filter = st.multiselect(
                "Sentiment",
                ["Positive", "Negative", "Neutral"],
                default=["Positive", "Negative", "Neutral"],
                help="Filter news by sentiment"
            )
        
        with filter_col3:
            signal_filter = st.multiselect(
                "Trading Signal",
                ["BUY", "SELL", "HOLD"],
                default=["BUY", "SELL", "HOLD"],
                help="Filter news by trading signal"
            )
        
        # Run News Analysis
        if st.button("📰 Analyze News", type="primary"):
            if selected_tickers:
                with st.spinner(f"🔍 Analyzing news for {len(selected_tickers)} stocks... (searching {days_back} days back)"):
                    news_df = news_analyzer.analyze_portfolio_news(
                        selected_tickers, 
                        max_news_per_stock=max_news_per_stock,
                        days_back=days_back
                    )
                
                if not news_df.empty:
                    # Apply filters
                    filtered_df = news_df[
                        (news_df['Materiality'].isin(materiality_filter)) &
                        (news_df['Sentiment'].isin(sentiment_filter)) &
                        (news_df['Signal'].isin(signal_filter))
                    ]
                    
                    # Apply critical news filter if enabled
                    if show_critical_only and 'Is_Critical' in filtered_df.columns:
                        filtered_df = filtered_df[filtered_df['Is_Critical'] == True]
                    
                    if not filtered_df.empty:
                        st.success(f"✅ Found {len(filtered_df)} relevant news articles!")
                        
                        # News Summary Dashboard
                        st.subheader("📊 News Analysis Summary")
                        
                        summary = news_analyzer.generate_news_summary(filtered_df)
                        
                        # Summary Metrics
                        summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
                        
                        with summary_col1:
                            st.metric("📰 Total News", summary['total_news'])
                        
                        with summary_col2:
                            critical_news = summary.get('critical_news', 0)
                            st.metric("🚨 Critical News", critical_news)
                        
                        with summary_col3:
                            buy_signals = summary['by_signal'].get('BUY', 0)
                            sell_signals = summary['by_signal'].get('SELL', 0)
                            net_signals = buy_signals - sell_signals
                            st.metric("📈 Net Signals", f"+{net_signals}" if net_signals >= 0 else str(net_signals))
                        
                        with summary_col4:
                            recent_24h = summary.get('recent_news_24h', 0)
                            st.metric("⏰ Last 24h", recent_24h)
                        
                        # Summary Charts
                        st.subheader("📈 News Analysis Charts")
                        
                        chart_col1, chart_col2 = st.columns(2)
                        
                        with chart_col1:
                            # Sentiment Distribution
                            sentiment_counts = filtered_df['Sentiment'].value_counts()
                            fig_sentiment = px.pie(
                                values=sentiment_counts.values,
                                names=sentiment_counts.index,
                                title="News Sentiment Distribution",
                                color_discrete_map={
                                    'Positive': '#2E8B57',
                                    'Negative': '#DC143C', 
                                    'Neutral': '#808080'
                                }
                            )
                            st.plotly_chart(fig_sentiment, use_container_width=True)
                        
                        with chart_col2:
                            # Trading Signals Distribution
                            signal_counts = filtered_df['Signal'].value_counts()
                            fig_signals = px.bar(
                                x=signal_counts.index,
                                y=signal_counts.values,
                                title="Trading Signals Distribution",
                                color=signal_counts.index,
                                color_discrete_map={
                                    'BUY': '#228B22',
                                    'SELL': '#FF4500',
                                    'HOLD': '#4682B4'
                                }
                            )
                            fig_signals.update_layout(showlegend=False)
                            st.plotly_chart(fig_signals, use_container_width=True)
                        
                        # Topic and Materiality Analysis
                        topic_col1, topic_col2 = st.columns(2)
                        
                        with topic_col1:
                            # Topic Distribution
                            topic_counts = filtered_df['Topic'].value_counts()
                            fig_topics = px.bar(
                                x=topic_counts.values,
                                y=topic_counts.index,
                                orientation='h',
                                title="News Topics Distribution",
                                color=topic_counts.values,
                                color_continuous_scale="viridis"
                            )
                            st.plotly_chart(fig_topics, use_container_width=True)
                        
                        with topic_col2:
                            # Materiality vs Sentiment Heatmap
                            materiality_sentiment = pd.crosstab(
                                filtered_df['Materiality'], 
                                filtered_df['Sentiment']
                            )
                            
                            fig_heatmap = px.imshow(
                                materiality_sentiment.values,
                                x=materiality_sentiment.columns,
                                y=materiality_sentiment.index,
                                color_continuous_scale="RdYlBu_r",
                                title="Materiality vs Sentiment Heatmap",
                                text_auto=True
                            )
                            st.plotly_chart(fig_heatmap, use_container_width=True)
                        
                        # Critical News Alerts - Enhanced Display
                        if 'Is_Critical' in filtered_df.columns:
                            critical_news = filtered_df[filtered_df['Is_Critical'] == True]
                        else:
                            critical_news = filtered_df[
                                (filtered_df['Materiality'] == 'High') | 
                                (filtered_df['Signal'].isin(['BUY', 'SELL']))
                            ]
                        
                        if not critical_news.empty:
                            st.subheader("🚨 Critical News Alerts")
                            st.info(f"Found {len(critical_news)} critical news items that require immediate attention!")
                            
                            # Sort critical news by materiality score if available
                            if 'Materiality_Score' in critical_news.columns:
                                critical_news = critical_news.sort_values('Materiality_Score', ascending=False)
                            
                            for _, news in critical_news.head(5).iterrows():
                                # Use full title if available
                                title = news.get('Full_Title', news['Title'])
                                time_display = news.get('Time_Display', news['Time'])
                                
                                with st.expander(f"🚨 {news['Ticker']} - {news['Signal']} Signal ({time_display})"):
                                    st.write(f"**Title:** {title}")
                                    st.write(f"**Publisher:** {news['Publisher']} | **Source:** {news.get('Source', 'Unknown')}")
                                    st.write(f"**Published:** {news['Time']} ({time_display})")
                                    
                                    alert_col1, alert_col2, alert_col3 = st.columns(3)
                                    
                                    with alert_col1:
                                        sentiment_color = {
                                            'Positive': '🟢', 
                                            'Negative': '🔴', 
                                            'Neutral': '🟡'
                                        }
                                        st.write(f"**Sentiment:** {sentiment_color.get(news['Sentiment'], '⚪')} {news['Sentiment']}")
                                    
                                    with alert_col2:
                                        materiality_color = {
                                            'High': '🔴', 
                                            'Medium': '🟡', 
                                            'Low': '🟢'
                                        }
                                        st.write(f"**Materiality:** {materiality_color.get(news['Materiality'], '⚪')} {news['Materiality']}")
                                    
                                    with alert_col3:
                                        signal_color = {
                                            'BUY': '🟢', 
                                            'SELL': '🔴', 
                                            'HOLD': '🟡'
                                        }
                                        st.write(f"**Signal:** {signal_color.get(news['Signal'], '⚪')} {news['Signal']}")
                                    
                                    st.write(f"**Analysis:** {news['Rationale']}")
                                    
                                    if news['Link']:
                                        st.markdown(f"[📖 Read Full Article]({news['Link']})")
                        
                        # Stock-wise Analysis
                        st.subheader("📊 Stock-wise News Analysis")
                        
                        stock_analysis = filtered_df.groupby('Ticker').agg({
                            'Sentiment': lambda x: x.value_counts().to_dict(),
                            'Signal': lambda x: x.value_counts().to_dict(),
                            'Materiality': lambda x: x.value_counts().to_dict(),
                            'Confidence': lambda x: pd.to_numeric(x, errors='coerce').mean()
                        }).reset_index()
                        
                        for _, stock_data in stock_analysis.iterrows():
                            ticker = stock_data['Ticker']
                            stock_news = filtered_df[filtered_df['Ticker'] == ticker]
                            
                            with st.expander(f"📈 {ticker} - {len(stock_news)} news articles"):
                                stock_col1, stock_col2, stock_col3 = st.columns(3)
                                
                                with stock_col1:
                                    st.write("**Sentiment Breakdown:**")
                                    sentiment_data = stock_data['Sentiment']
                                    for sentiment, count in sentiment_data.items():
                                        st.write(f"• {sentiment}: {count}")
                                
                                with stock_col2:
                                    st.write("**Trading Signals:**")
                                    signal_data = stock_data['Signal']
                                    for signal, count in signal_data.items():
                                        st.write(f"• {signal}: {count}")
                                
                                with stock_col3:
                                    avg_confidence = stock_data['Confidence']
                                    st.metric("Avg Confidence", f"{avg_confidence:.2f}")
                                    
                                    # Overall recommendation for the stock
                                    buy_count = signal_data.get('BUY', 0)
                                    sell_count = signal_data.get('SELL', 0)
                                    hold_count = signal_data.get('HOLD', 0)
                                    
                                    if buy_count > sell_count and buy_count > hold_count:
                                        st.success("📈 Overall: BUY Tilt")
                                    elif sell_count > buy_count and sell_count > hold_count:
                                        st.error("📉 Overall: SELL Tilt")
                                    else:
                                        st.info("➡️ Overall: HOLD/Neutral")
                                
                                # Show recent news for this stock
                                st.write("**Recent News:**")
                                recent_stock_news = stock_news.head(3)
                                for _, article in recent_stock_news.iterrows():
                                    st.write(f"• {article['Title'][:100]}...")
                        
                        # Detailed News Feed
                        st.subheader("📋 Detailed News Feed")
                        
                        # Sorting options
                        sort_col1, sort_col2 = st.columns(2)
                        
                        with sort_col1:
                            sort_by = st.selectbox(
                                "Sort by",
                                ["Time", "Materiality", "Confidence", "Ticker"],
                                index=0
                            )
                        
                        with sort_col2:
                            sort_order = st.selectbox(
                                "Order",
                                ["Descending", "Ascending"],
                                index=0
                            )
                        
                        # Apply sorting
                        ascending = sort_order == "Ascending"
                        if sort_by == "Materiality":
                            # Custom sort for materiality
                            materiality_order = {"High": 3, "Medium": 2, "Low": 1}
                            filtered_df['materiality_rank'] = filtered_df['Materiality'].map(materiality_order)
                            sorted_df = filtered_df.sort_values('materiality_rank', ascending=ascending)
                            sorted_df = sorted_df.drop('materiality_rank', axis=1)
                        elif sort_by == "Confidence":
                            sorted_df = filtered_df.sort_values('Confidence', ascending=ascending)
                        else:
                            sorted_df = filtered_df.sort_values(sort_by, ascending=ascending)
                        
                        # Display news feed table with enhanced columns
                        display_columns = ['Ticker', 'Title', 'Publisher', 'Time_Display', 'Topic', 
                                         'Materiality', 'Sentiment', 'Signal', 'Confidence']
                        
                        # Add critical indicator if available
                        if 'Is_Critical' in sorted_df.columns:
                            sorted_df['Critical'] = sorted_df['Is_Critical'].apply(lambda x: '🚨' if x else '')
                            display_columns.insert(1, 'Critical')
                        
                        # Use Time_Display if available, otherwise fall back to Time
                        if 'Time_Display' not in sorted_df.columns:
                            display_columns = [col if col != 'Time_Display' else 'Time' for col in display_columns]
                        
                        st.dataframe(
                            sorted_df[display_columns],
                            use_container_width=True,
                            column_config={
                                "Title": st.column_config.TextColumn(
                                    "Title",
                                    width="large"
                                ),
                                "Time_Display": st.column_config.TextColumn(
                                    "Recency",
                                    width="small"
                                ),
                                "Time": st.column_config.TextColumn(
                                    "Time",
                                    width="medium"
                                ),
                                "Critical": st.column_config.TextColumn(
                                    "❗",
                                    width="small"
                                ),
                                "Confidence": st.column_config.NumberColumn(
                                    "Confidence",
                                    format="%.2f"
                                )
                            }
                        )
                        
                        # Export Options
                        st.subheader("💾 Export Analysis")
                        
                        export_col1, export_col2 = st.columns(2)
                        
                        with export_col1:
                            # Export filtered news
                            csv_news = filtered_df.to_csv(index=False)
                            st.download_button(
                                label="📁 Download News Analysis CSV",
                                data=csv_news,
                                file_name=f"news_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv"
                            )
                        
                        with export_col2:
                            # Export summary
                            summary_data = {
                                'Total News': [summary['total_news']],
                                'High Priority': [summary['high_priority_news']],
                                'Positive Sentiment': [summary['by_sentiment'].get('Positive', 0)],
                                'Negative Sentiment': [summary['by_sentiment'].get('Negative', 0)],
                                'Buy Signals': [summary['by_signal'].get('BUY', 0)],
                                'Sell Signals': [summary['by_signal'].get('SELL', 0)]
                            }
                            
                            csv_summary = pd.DataFrame(summary_data).to_csv(index=False)
                            st.download_button(
                                label="📊 Download Summary CSV",
                                data=csv_summary,
                                file_name=f"news_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv"
                            )
                    
                    else:
                        st.warning("🔍 No news articles match your current filters. Try adjusting the filter criteria.")
                
                else:
                    st.warning(f"📰 No recent news found for the selected stocks in the last {days_back} days. This could be due to:")
                    st.write("• Limited news coverage for some stocks")
                    st.write("• API rate limits or temporary issues")
                    st.write("• Weekend/holiday periods with reduced news flow")
                    st.write("• Stock symbols not recognized by the news service")
                    st.write(f"• No news in the selected {days_back}-day period")
                    
                    st.info("💡 Try selecting different stocks, increasing the search period, or check back later for updated news.")
            
            else:
                st.warning("⚠️ Please select at least one stock for news analysis.")
    
    else:
        st.info("👆 Please input your portfolio or stock tickers to start news analysis.")
        
        # Instructions and Help
        st.subheader("📖 How to Use News Analysis")
        
        with st.expander("🎯 Feature Overview"):
            st.markdown("""
            ### What does News Analysis do?
            
            **📰 News Fetching:** Automatically retrieves latest news for your stocks from multiple sources:
            - Yahoo Finance for financial news
            - Google News RSS for broader coverage
            - Automatic deduplication and recency filtering
            
            **🔍 Topic Classification:** Categorizes news into topics like:
            - Earnings & Financial Results
            - Management Changes
            - Regulatory & Legal Issues
            - Mergers & Acquisitions
            - Product & Operational Updates
            
            **⚖️ Materiality Rating:** Rates news impact as High/Medium/Low based on:
            - Keywords indicating significant events (earnings, mergers, lawsuits)
            - Historical impact of similar news
            - Company-specific context
            - **🚨 Critical News Highlighting:** Auto-identifies news requiring immediate attention
            
            **😊 Sentiment Analysis:** Uses VADER sentiment analysis to determine:
            - Positive/Negative/Neutral sentiment
            - Confidence scores for sentiment classification
            
            **📈 Trading Signals:** Generates actionable recommendations:
            - **BUY:** Positive news with high materiality
            - **SELL:** Negative news with significant impact
            - **HOLD:** Mixed signals or low materiality news
            - **Recency Tracking:** Shows how recent each news item is (hours/days ago)
            """)
        
        with st.expander("💡 Best Practices"):
            st.markdown("""
            ### Getting the Most from News Analysis
            
            **🎯 Focus on High Priority:**
            - Pay attention to High materiality news
            - Look for consistent signal patterns across multiple articles
            - Consider the source credibility and timing
            
            **📊 Use in Context:**
            - Combine with fundamental analysis from other tabs
            - Consider overall market conditions
            - Don't rely solely on short-term news sentiment
            
            **⏰ Timing Considerations:**
            - News impact is often immediate but short-lived
            - Long-term investors should focus on fundamental news
            - Day traders might act on sentiment-driven news
            
            **🔄 Regular Monitoring:**
            - Check news analysis weekly for portfolio holdings
            - Set up alerts for high-priority news
            - Track sentiment trends over time
            
            **⚠️ Risk Management:**
            - Verify important news from multiple sources
            - Consider the broader context beyond individual articles
            - Use position sizing to manage news-driven volatility
            """)
        
        with st.expander("🔧 Technical Details"):
            st.markdown("""
            ### How the Analysis Works
            
            **Data Source:** Yahoo Finance News API via yfinance library
            
            **Sentiment Engine:** VADER (Valence Aware Dictionary and sEntiment Reasoner)
            - Optimized for social media and financial text
            - Provides compound scores from -1 (negative) to +1 (positive)
            
            **Topic Classification:** Keyword-based classification using financial terms dictionary
            
            **Materiality Scoring:** Rule-based system considering:
            - Keyword significance weights
            - News category importance
            - Historical impact patterns
            
            **Signal Generation:** Multi-factor model combining:
            - Sentiment strength and direction
            - Materiality level
            - Topic category risk/opportunity profile
            - Confidence thresholds
            
            **Limitations:**
            - Analysis is based on English language news
            - May miss context requiring human interpretation
            - Historical data dependency for pattern recognition
            - API rate limits may affect real-time updates
            """)
    
    # Disclaimer
    st.warning("""
    ⚠️ **News Analysis Disclaimer:** 
    This news analysis is for informational purposes only and should not be considered as investment advice. 
    News sentiment can be volatile and may not reflect long-term investment fundamentals. Always verify important 
    news from multiple sources and consider your overall investment strategy before making trading decisions.
    """)
