# Stock Analysis Dashboard

## Overview

This is a comprehensive stock analysis dashboard built with Streamlit that serves long-term investors who need automated portfolio tracking, stock screening, IPO analysis, and news monitoring. The application follows Benjamin Graham's value investing principles while also supporting growth and moonshot investment strategies. It provides portfolio optimization, news sentiment analysis, and stock screening capabilities for both Indian and US markets.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Framework**: Streamlit with a multi-tab interface
- **Layout**: Wide layout with expandable sidebar
- **Tabs Structure**: Five main functional tabs (Portfolio, IPO Radar, Stock Screener, Portfolio Optimizer, News Analysis)
- **Visualization**: Plotly for interactive charts and graphs
- **Data Input**: CSV file upload with validation and manual entry options

### Backend Architecture
- **Core Engine**: Python-based modular architecture with utils package
- **Data Processing**: Pandas for data manipulation and NumPy for numerical computations
- **Analysis Modules**:
  - PortfolioAnalyzer: Portfolio performance tracking and analysis
  - StockScreener: Multi-strategy stock filtering (Value, Growth, Moonshot)
  - IPOAnalyzer: IPO evaluation and analysis framework
  - NewsAnalyzer: News sentiment analysis and categorization
  - PortfolioOptimizer: Modern Portfolio Theory optimization
  - DataFetcher: Centralized data retrieval with caching

### Investment Strategies Implementation
- **Value Strategy**: Graham-based criteria (Market Cap > 5000, P/E < 16, P/B < 3, ROE > 15, etc.)
- **Moonshot Strategy**: Small-cap growth criteria (Market Cap < 2000, P/E < 30, PEG < 1, etc.)
- **Growth Strategy**: Growth-focused metrics (PEG < 1, Sales Growth > 25%, ROE > 20%, etc.)

### Data Management
- **Caching Strategy**: Streamlit's @st.cache_data with 5-minute TTL for API calls
- **Portfolio Data**: CSV-based portfolio input with RTF format handling
- **Data Validation**: Comprehensive input validation and error handling
- **Market Support**: Dual market support (Indian NSE/BSE and US markets)

### Analysis Capabilities
- **Portfolio Performance**: P&L tracking, CAGR calculation, benchmark comparison
- **Risk Analysis**: Sector allocation, concentration analysis, volatility metrics
- **Optimization**: Scipy-based portfolio optimization using Modern Portfolio Theory
- **News Processing**: Multi-source news aggregation with sentiment analysis using VADER

## External Dependencies

### Financial Data APIs
- **yfinance**: Primary data source for stock prices, historical data, and company information
- **Yahoo Finance**: Real-time market data and fundamental metrics
- **Google News RSS**: News aggregation for sentiment analysis

### Python Libraries
- **streamlit**: Web application framework
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computations
- **plotly**: Interactive visualization and charting
- **scipy**: Portfolio optimization algorithms
- **vaderSentiment**: News sentiment analysis
- **feedparser**: RSS feed processing for news
- **trafilatura**: Web content extraction
- **requests**: HTTP client for API calls

### Data Sources
- **NSE/BSE**: Indian stock market data (through yfinance)
- **NYSE/NASDAQ**: US stock market data (through yfinance)
- **News Sources**: Google News, Yahoo Finance News
- **IPO Data**: Framework for NSE/BSE IPO APIs (requires authentication)

### Deployment Requirements
- **Platform**: Streamlit Cloud or similar free hosting
- **Python Version**: 3.7+
- **Memory**: Moderate usage due to data caching
- **Network**: Requires internet access for real-time data fetching