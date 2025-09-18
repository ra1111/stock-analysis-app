import pandas as pd
import numpy as np
from io import StringIO
from utils.data_fetcher import DataFetcher
import streamlit as st
import yfinance as yf
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests

class PortfolioAnalyzer:
    def __init__(self):
        self.data_fetcher = DataFetcher()
        self.sector_mapping = {
            # Technology & IT
            'INFY': 'Technology', 'TCS': 'Technology', 'WIPRO': 'Technology', 'HCLTECH': 'Technology',
            'TECHM': 'Technology', 'LTI': 'Technology', 'MINDTREE': 'Technology', 'INFIBEAM': 'Technology',
            'TANLA': 'Technology', 'ROUTE': 'Technology',
            
            # Banking & Financial Services
            'HDFCBANK': 'Banking', 'ICICIBANK': 'Banking', 'SBIN': 'Banking', 'KOTAKBANK': 'Banking',
            'AXISBANK': 'Banking', 'INDUSINDBK': 'Banking', 'BAJFINANCE': 'NBFC', 'HDFCLIFE': 'Insurance',
            
            # Energy & Oil
            'RELIANCE': 'Energy', 'ONGC': 'Energy', 'IOC': 'Energy', 'BPCL': 'Energy', 'HPCL': 'Energy',
            'COALINDIA': 'Energy', 'NTPC': 'Power', 'POWERGRID': 'Power',
            
            # Automobiles
            'MARUTI': 'Automobile', 'TATAMOTORS': 'Automobile', 'M&M': 'Automobile', 'BAJAJ-AUTO': 'Automobile',
            'HEROMOTOCO': 'Automobile', 'EICHERMOT': 'Automobile',
            
            # FMCG & Consumer
            'HINDUNILVR': 'FMCG', 'ITC': 'FMCG', 'NESTLEIND': 'FMCG', 'DABUR': 'FMCG', 'MARICO': 'FMCG',
            'GODREJCP': 'FMCG', 'BRITANNIA': 'FMCG',
            
            # Pharmaceuticals
            'SUNPHARMA': 'Pharma', 'DRREDDY': 'Pharma', 'CIPLA': 'Pharma', 'LUPIN': 'Pharma',
            'BIOCON': 'Pharma', 'AUROPHARMA': 'Pharma',
            
            # Metals & Mining
            'TATASTEEL': 'Metals', 'JSWSTEEL': 'Metals', 'HINDALCO': 'Metals', 'NATIONALUM': 'Metals',
            'VEDL': 'Metals', 'HINDZINC': 'Metals',
            
            # Infrastructure & Construction
            'LT': 'Infrastructure', 'ULTRACEMCO': 'Cement', 'SHREECEM': 'Cement', 'ACC': 'Cement',
            'AMBUJACEMENT': 'Cement', 'JKCEMENT': 'Cement'
        }
    
    def load_portfolio_from_csv(self, uploaded_file):
        """Load portfolio data from uploaded CSV file with flexible column handling"""
        try:
            # Read CSV
            df = pd.read_csv(uploaded_file)
            
            # Standardize column names (handle different formats)
            column_mapping = {
                'symbol': 'Symbol', 'Symbol': 'Symbol', 'SYMBOL': 'Symbol',
                'stock': 'Symbol', 'Stock': 'Symbol', 'ticker': 'Symbol', 'Ticker': 'Symbol',
                'quantity': 'Quantity', 'Quantity': 'Quantity', 'QUANTITY': 'Quantity',
                'qty': 'Quantity', 'shares': 'Quantity', 'Shares': 'Quantity',
                'avg_price': 'Avg_Price', 'average_price': 'Avg_Price', 'price': 'Avg_Price',
                'Average Price': 'Avg_Price', 'Avg Price': 'Avg_Price', 'Price': 'Avg_Price',
                'AvgPrice': 'Avg_Price', 'avgprice': 'Avg_Price',
                'purchase_date': 'Purchase_Date', 'date': 'Purchase_Date', 'Date': 'Purchase_Date',
                'Purchase Date': 'Purchase_Date', 'buy_date': 'Purchase_Date', 'invested_value': 'Invested_Value',
                'InvestedValue': 'Invested_Value', 'investment': 'Invested_Value'
            }
            
            # Apply column mapping
            df = df.rename(columns=column_mapping)
            
            # Ensure required columns exist
            required_cols = ['Symbol', 'Quantity', 'Avg_Price']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                st.error(f"Missing required columns: {missing_cols}. Please ensure your CSV has Symbol, Quantity, and Avg_Price columns.")
                return None
            
            # Clean and validate data
            df['Symbol'] = df['Symbol'].str.upper().str.strip()
            df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce')
            df['Avg_Price'] = pd.to_numeric(df['Avg_Price'], errors='coerce')
            
            # Calculate Invested_Value if not provided
            if 'Invested_Value' not in df.columns:
                df['Invested_Value'] = df['Quantity'] * df['Avg_Price']
            else:
                df['Invested_Value'] = pd.to_numeric(df['Invested_Value'], errors='coerce')
            
            # Handle purchase date if available
            if 'Purchase_Date' in df.columns:
                df['Purchase_Date'] = pd.to_datetime(df['Purchase_Date'], errors='coerce')
            else:
                # Default to 1 year ago if no date provided
                df['Purchase_Date'] = datetime.now() - timedelta(days=365)
            
            # Remove rows with invalid data
            df = df.dropna(subset=['Symbol', 'Quantity', 'Avg_Price'])
            df = df[df['Quantity'] > 0]
            df = df[df['Avg_Price'] > 0]
            
            # Add .NS suffix for Indian stocks if not present
            df['Symbol_NSE'] = df['Symbol'].str.upper().apply(lambda x: f"{x}.NS" if not x.endswith('.NS') and not any(x.endswith(suffix) for suffix in ['.BO', '.L', '.NYSE']) else x)
            
            return df
            
        except Exception as e:
            st.error(f"Error loading portfolio: {str(e)}")
            return None
    
    def analyze_portfolio(self, portfolio_df):
        """Analyze portfolio with current prices and recommendations"""
        if portfolio_df.empty:
            return pd.DataFrame()
        
        results = []
        
        for _, row in portfolio_df.iterrows():
            ticker = row['Symbol']
            quantity = row['Quantity']
            avg_price = row['Avg_Price']
            invested_value = row['Invested_Value']
            
            # Get current price
            current_price = self.data_fetcher.get_current_price(ticker)
            
            # Calculate P&L
            current_value = quantity * current_price
            pnl = current_value - invested_value
            pnl_percent = (pnl / invested_value) * 100 if invested_value > 0 else 0
            
            # Get fundamentals for recommendation
            fundamentals = self.data_fetcher.get_fundamentals(ticker)
            recommendation = self._get_recommendation(fundamentals, pnl_percent, ticker)
            
            results.append({
                'Ticker': ticker,
                'Quantity': quantity,
                'Avg Price': avg_price,
                'Current Price': current_price,
                'Invested Value': invested_value,
                'Current Value': current_value,
                'P&L': pnl,
                'P&L %': pnl_percent,
                'Recommendation': recommendation['action'],
                'Reason': recommendation['reason'],
                'Confidence': f"{recommendation.get('confidence', 50)}%",
                'Analysis': recommendation.get('analysis', {})
            })
        
        return pd.DataFrame(results)
    
    def _get_recommendation(self, fundamentals, pnl_percent, ticker):
        """Generate buy/hold/sell recommendation with transparent logic"""
        
        # Initialize detailed analysis
        analysis_details = {
            'data_available': False,
            'value_metrics': {},
            'growth_metrics': {},
            'risk_metrics': {},
            'value_score': 0,
            'growth_score': 0,
            'risk_score': 0,
            'final_score': 0,
            'reasoning_parts': []
        }
        
        # Check if we have fundamental data
        if not fundamentals or fundamentals.get('market_cap', 0) == 0:
            return self._fallback_recommendation(pnl_percent, ticker, analysis_details)
        
        analysis_details['data_available'] = True
        
        # Extract fundamental metrics with safe None handling
        pe = fundamentals.get('pe_ratio', 0) or 0
        pb = fundamentals.get('pb_ratio', 0) or 0
        roe = fundamentals.get('roe', 0) or 0
        debt_equity = fundamentals.get('debt_to_equity', 0) or 0
        dividend_yield = fundamentals.get('dividend_yield', 0) or 0
        revenue_growth = fundamentals.get('revenue_growth', 0) or 0
        earnings_growth = fundamentals.get('earnings_growth', 0) or 0
        peg_ratio = fundamentals.get('peg_ratio', 0) or 0
        
        # Store metrics for transparency
        analysis_details['value_metrics'] = {
            'pe_ratio': pe,
            'pb_ratio': pb,
            'roe': roe,
            'dividend_yield': dividend_yield
        }
        
        analysis_details['growth_metrics'] = {
            'revenue_growth': revenue_growth,
            'earnings_growth': earnings_growth,
            'peg_ratio': peg_ratio
        }
        
        analysis_details['risk_metrics'] = {
            'debt_to_equity': debt_equity,
            'current_pnl': pnl_percent
        }
        
        # VALUE ANALYSIS
        value_score = 0
        value_reasons = []
        
        if pe > 0:
            if pe < 12:
                value_score += 2
                value_reasons.append(f"Excellent PE ({pe:.1f}) - Very undervalued")
            elif pe < 16:
                value_score += 1
                value_reasons.append(f"Good PE ({pe:.1f}) - Fairly valued")
            elif pe > 25:
                value_score -= 1
                value_reasons.append(f"High PE ({pe:.1f}) - Potentially overvalued")
            else:
                value_reasons.append(f"Moderate PE ({pe:.1f}) - Neutral")
        
        if pb > 0:
            if pb < 1.5:
                value_score += 1
                value_reasons.append(f"Low PB ({pb:.1f}) - Trading below book value")
            elif pb > 4:
                value_score -= 1
                value_reasons.append(f"High PB ({pb:.1f}) - Premium to book value")
            else:
                value_reasons.append(f"Moderate PB ({pb:.1f})")
        
        if roe > 0:
            if roe > 20:
                value_score += 2
                value_reasons.append(f"Excellent ROE ({roe:.1f}%) - Highly profitable")
            elif roe > 15:
                value_score += 1
                value_reasons.append(f"Good ROE ({roe:.1f}%) - Solid profitability")
            elif roe < 10:
                value_score -= 1
                value_reasons.append(f"Low ROE ({roe:.1f}%) - Poor profitability")
            else:
                value_reasons.append(f"Moderate ROE ({roe:.1f}%)")
        
        if dividend_yield > 0:
            if dividend_yield > 3:
                value_score += 1
                value_reasons.append(f"High dividend yield ({dividend_yield:.1f}%) - Good income")
            elif dividend_yield > 1.5:
                value_reasons.append(f"Moderate dividend yield ({dividend_yield:.1f}%)")
            else:
                value_reasons.append(f"Low dividend yield ({dividend_yield:.1f}%)")
        
        analysis_details['value_score'] = value_score
        
        # GROWTH ANALYSIS
        growth_score = 0
        growth_reasons = []
        
        if revenue_growth != 0:
            if revenue_growth > 20:
                growth_score += 2
                growth_reasons.append(f"Excellent revenue growth ({revenue_growth:.1f}%)")
            elif revenue_growth > 10:
                growth_score += 1
                growth_reasons.append(f"Good revenue growth ({revenue_growth:.1f}%)")
            elif revenue_growth < 0:
                growth_score -= 1
                growth_reasons.append(f"Declining revenue ({revenue_growth:.1f}%)")
            else:
                growth_reasons.append(f"Slow revenue growth ({revenue_growth:.1f}%)")
        
        if earnings_growth != 0:
            if earnings_growth > 20:
                growth_score += 2
                growth_reasons.append(f"Excellent earnings growth ({earnings_growth:.1f}%)")
            elif earnings_growth > 10:
                growth_score += 1
                growth_reasons.append(f"Good earnings growth ({earnings_growth:.1f}%)")
            elif earnings_growth < 0:
                growth_score -= 1
                growth_reasons.append(f"Declining earnings ({earnings_growth:.1f}%)")
            else:
                growth_reasons.append(f"Slow earnings growth ({earnings_growth:.1f}%)")
        
        if peg_ratio > 0:
            if peg_ratio < 1:
                growth_score += 1
                growth_reasons.append(f"Attractive PEG ratio ({peg_ratio:.1f}) - Growth at reasonable price")
            elif peg_ratio > 2:
                growth_score -= 1
                growth_reasons.append(f"High PEG ratio ({peg_ratio:.1f}) - Expensive for growth")
            else:
                growth_reasons.append(f"Moderate PEG ratio ({peg_ratio:.1f})")
        
        analysis_details['growth_score'] = growth_score
        
        # RISK ANALYSIS
        risk_score = 0
        risk_reasons = []
        
        if debt_equity > 0:
            if debt_equity < 30:
                risk_score += 1
                risk_reasons.append(f"Low debt-to-equity ({debt_equity:.1f}%) - Conservative capital structure")
            elif debt_equity > 80:
                risk_score -= 2
                risk_reasons.append(f"High debt-to-equity ({debt_equity:.1f}%) - High financial risk")
            elif debt_equity > 50:
                risk_score -= 1
                risk_reasons.append(f"Moderate debt-to-equity ({debt_equity:.1f}%) - Some financial risk")
            else:
                risk_reasons.append(f"Moderate debt-to-equity ({debt_equity:.1f}%)")
        
        # Current P&L consideration
        if pnl_percent < -20:
            risk_reasons.append(f"Currently at significant loss ({pnl_percent:.1f}%) - High risk")
        elif pnl_percent < -10:
            risk_reasons.append(f"Currently at moderate loss ({pnl_percent:.1f}%) - Consider averaging")
        elif pnl_percent > 30:
            risk_reasons.append(f"Currently at good profit ({pnl_percent:.1f}%) - Consider booking")
        else:
            risk_reasons.append(f"Current P&L: {pnl_percent:.1f}%")
        
        analysis_details['risk_score'] = risk_score
        
        # FINAL RECOMMENDATION
        final_score = value_score + growth_score + risk_score
        analysis_details['final_score'] = final_score
        
        # Combine all reasoning
        all_reasons = value_reasons + growth_reasons + risk_reasons
        analysis_details['reasoning_parts'] = all_reasons
        
        # Decision logic with clear thresholds
        if final_score >= 4 and pnl_percent < -5:
            action = 'BUY MORE'
            main_reason = f"Strong fundamentals (Score: {final_score}) + currently undervalued"
        elif final_score >= 3:
            action = 'HOLD'
            main_reason = f"Good fundamentals (Score: {final_score}) - maintain position"
        elif final_score <= -2:
            action = 'SELL'
            main_reason = f"Weak fundamentals (Score: {final_score}) - consider exit"
        elif pnl_percent > 25:
            action = 'SELL'
            main_reason = f"Good profit booking opportunity ({pnl_percent:.1f}% gain)"
        else:
            action = 'HOLD'
            main_reason = f"Mixed signals (Score: {final_score}) - wait and watch"
        
        # Create detailed reason
        detailed_reason = f"{main_reason}. {'; '.join(all_reasons[:3])}"  # Top 3 reasons
        if len(all_reasons) > 3:
            detailed_reason += f" + {len(all_reasons) - 3} more factors"
        
        return {
            'action': action,
            'reason': detailed_reason,
            'analysis': analysis_details,
            'confidence': min(100, max(10, 50 + abs(final_score) * 10))  # Confidence based on score strength
        }
    
    def _fallback_recommendation(self, pnl_percent, ticker, analysis_details):
        """Fallback recommendation when fundamental data is missing"""
        analysis_details['reasoning_parts'] = ["Using technical analysis due to missing fundamental data"]
        
        # Use simple technical rules when fundamentals are missing
        if pnl_percent > 30:
            return {
                'action': 'SELL',
                'reason': f'No fundamental data available. Consider profit booking at {pnl_percent:.1f}% gain',
                'analysis': analysis_details,
                'confidence': 60
            }
        elif pnl_percent < -25:
            return {
                'action': 'SELL',
                'reason': f'No fundamental data + heavy loss ({pnl_percent:.1f}%). Consider cutting losses',
                'analysis': analysis_details,
                'confidence': 70
            }
        elif pnl_percent < -10:
            return {
                'action': 'HOLD',
                'reason': f'No fundamental data. Monitor closely due to {pnl_percent:.1f}% loss',
                'analysis': analysis_details,
                'confidence': 40
            }
        else:
            return {
                'action': 'HOLD',
                'reason': f'No fundamental data available for analysis. Current P&L: {pnl_percent:.1f}%',
                'analysis': analysis_details,
                'confidence': 30
            }
    
    def simulate_additional_investment(self, portfolio_df, ticker, additional_amount):
        """Simulate impact of additional investment"""
        current_price = self.data_fetcher.get_current_price(ticker)
        
        if current_price == 0:
            return None
        
        additional_shares = additional_amount / current_price
        
        # Find existing holding
        existing_row = portfolio_df[portfolio_df['Ticker'] == ticker]
        
        if existing_row.empty:
            return {
                'new_position': True,
                'additional_shares': additional_shares,
                'additional_amount': additional_amount,
                'new_avg_price': current_price
            }
        else:
            existing = existing_row.iloc[0]
            new_quantity = existing['Quantity'] + additional_shares
            new_invested_value = existing['InvestedValue'] + additional_amount
            new_avg_price = new_invested_value / new_quantity
            
            return {
                'new_position': False,
                'original_quantity': existing['Quantity'],
                'additional_shares': additional_shares,
                'new_quantity': new_quantity,
                'original_invested': existing['InvestedValue'],
                'additional_amount': additional_amount,
                'new_invested_value': new_invested_value,
                'original_avg_price': existing['Avg_Price'],
                'new_avg_price': new_avg_price
            }
    
    def get_current_prices(self, symbols):
        """Fetch current prices for portfolio symbols"""
        current_prices = {}
        
        for symbol in symbols:
            try:
                # Try multiple data sources
                price = None
                
                # Try yfinance first
                if symbol.endswith('.NS') or symbol.endswith('.BO'):
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(period='1d')
                    if not hist.empty:
                        price = hist['Close'].iloc[-1]
                
                # Fallback to data fetcher
                if price is None:
                    base_symbol = symbol.replace('.NS', '').replace('.BO', '')
                    price = self.data_fetcher.get_current_price(base_symbol)
                
                # Store result
                current_prices[symbol] = price if price and price > 0 else None
                
            except Exception as e:
                current_prices[symbol] = None
                
        return current_prices
    
    def calculate_portfolio_performance(self, portfolio_df):
        """Calculate comprehensive portfolio performance metrics"""
        if portfolio_df is None or portfolio_df.empty:
            return None
        
        # Get current prices
        symbols = portfolio_df['Symbol_NSE'].unique()
        current_prices = self.get_current_prices(symbols)
        
        # Add current prices to dataframe
        portfolio_df['Current_Price'] = portfolio_df['Symbol_NSE'].map(current_prices)
        
        # Calculate metrics
        portfolio_df['Investment'] = portfolio_df['Quantity'] * portfolio_df['Avg_Price']
        portfolio_df['Current_Value'] = portfolio_df['Quantity'] * portfolio_df['Current_Price'].fillna(portfolio_df['Avg_Price'])
        portfolio_df['P&L'] = portfolio_df['Current_Value'] - portfolio_df['Investment']
        portfolio_df['P&L_%'] = (portfolio_df['P&L'] / portfolio_df['Investment']) * 100
        
        # Calculate holding period for CAGR
        portfolio_df['Days_Held'] = (datetime.now() - portfolio_df['Purchase_Date']).dt.days
        portfolio_df['Years_Held'] = portfolio_df['Days_Held'] / 365.25
        
        # Calculate CAGR for each holding
        portfolio_df['CAGR'] = ((portfolio_df['Current_Value'] / portfolio_df['Investment']) ** (1 / portfolio_df['Years_Held'].clip(lower=0.1))) - 1
        portfolio_df['CAGR'] = portfolio_df['CAGR'] * 100  # Convert to percentage
        
        # Add sector classification
        portfolio_df['Sector'] = portfolio_df['Symbol'].map(self.sector_mapping).fillna('Others')
        
        # Portfolio-level metrics
        total_investment = portfolio_df['Investment'].sum()
        total_current_value = portfolio_df['Current_Value'].sum()
        total_pnl = total_current_value - total_investment
        total_pnl_pct = (total_pnl / total_investment) * 100 if total_investment > 0 else 0
        
        # Weighted average CAGR
        portfolio_df['Weight'] = portfolio_df['Investment'] / total_investment
        weighted_cagr = (portfolio_df['CAGR'] * portfolio_df['Weight']).sum()
        
        return {
            'portfolio_df': portfolio_df,
            'total_investment': total_investment,
            'total_current_value': total_current_value,
            'total_pnl': total_pnl,
            'total_pnl_pct': total_pnl_pct,
            'weighted_cagr': weighted_cagr,
            'num_stocks': len(portfolio_df),
            'data_availability': (portfolio_df['Current_Price'].notna().sum() / len(portfolio_df)) * 100
        }
    
    def get_benchmark_performance(self, start_date):
        """Get Nifty50 and S&P500 performance for comparison"""
        try:
            end_date = datetime.now()
            
            # Fetch Nifty50 (^NSEI)
            nifty = yf.Ticker("^NSEI")
            nifty_hist = nifty.history(start=start_date, end=end_date)
            
            # Fetch S&P500 (^GSPC)
            sp500 = yf.Ticker("^GSPC")
            sp500_hist = sp500.history(start=start_date, end=end_date)
            
            benchmarks = {}
            
            # Calculate Nifty50 performance
            if not nifty_hist.empty:
                nifty_start = nifty_hist['Close'].iloc[0]
                nifty_end = nifty_hist['Close'].iloc[-1]
                years_elapsed = (end_date - start_date).days / 365.25
                nifty_cagr = ((nifty_end / nifty_start) ** (1 / years_elapsed)) - 1
                benchmarks['Nifty50'] = {
                    'start_price': nifty_start,
                    'end_price': nifty_end,
                    'total_return': ((nifty_end / nifty_start) - 1) * 100,
                    'cagr': nifty_cagr * 100
                }
            
            # Calculate S&P500 performance
            if not sp500_hist.empty:
                sp500_start = sp500_hist['Close'].iloc[0]
                sp500_end = sp500_hist['Close'].iloc[-1]
                years_elapsed = (end_date - start_date).days / 365.25
                sp500_cagr = ((sp500_end / sp500_start) ** (1 / years_elapsed)) - 1
                benchmarks['S&P500'] = {
                    'start_price': sp500_start,
                    'end_price': sp500_end,
                    'total_return': ((sp500_end / sp500_start) - 1) * 100,
                    'cagr': sp500_cagr * 100
                }
            
            return benchmarks
            
        except Exception as e:
            st.warning(f"Could not fetch benchmark data: {str(e)}")
            return {}
    
    def analyze_portfolio_allocation(self, portfolio_df):
        """Analyze portfolio allocation by stock and sector"""
        if portfolio_df is None or portfolio_df.empty:
            return None
        
        # Stock-wise allocation
        stock_allocation = portfolio_df.groupby('Symbol').agg({
            'Current_Value': 'sum',
            'Investment': 'sum',
            'P&L': 'sum',
            'Sector': 'first'
        }).reset_index()
        
        total_value = stock_allocation['Current_Value'].sum()
        stock_allocation['Allocation_%'] = (stock_allocation['Current_Value'] / total_value) * 100
        stock_allocation = stock_allocation.sort_values('Allocation_%', ascending=False)
        
        # Sector-wise allocation
        sector_allocation = portfolio_df.groupby('Sector').agg({
            'Current_Value': 'sum',
            'Investment': 'sum',
            'P&L': 'sum'
        }).reset_index()
        
        sector_allocation['Allocation_%'] = (sector_allocation['Current_Value'] / total_value) * 100
        sector_allocation = sector_allocation.sort_values('Allocation_%', ascending=False)
        
        # Risk analysis
        risk_analysis = {
            'overconcentrated_stocks': stock_allocation[stock_allocation['Allocation_%'] > 25],
            'overconcentrated_sectors': sector_allocation[sector_allocation['Allocation_%'] > 40],
            'top_5_stocks_concentration': stock_allocation.head(5)['Allocation_%'].sum(),
            'diversification_score': self._calculate_diversification_score(stock_allocation, sector_allocation)
        }
        
        return {
            'stock_allocation': stock_allocation,
            'sector_allocation': sector_allocation,
            'risk_analysis': risk_analysis
        }
    
    def _calculate_diversification_score(self, stock_allocation, sector_allocation):
        """Calculate diversification score (0-100)"""
        score = 100
        
        # Penalize high concentration in single stocks
        max_stock_allocation = stock_allocation['Allocation_%'].max()
        if max_stock_allocation > 25:
            score -= (max_stock_allocation - 25) * 2  # Penalty for concentration > 25%
        
        # Penalize high concentration in single sectors
        max_sector_allocation = sector_allocation['Allocation_%'].max()
        if max_sector_allocation > 40:
            score -= (max_sector_allocation - 40) * 1.5  # Penalty for sector concentration > 40%
        
        # Reward having multiple sectors
        num_sectors = len(sector_allocation)
        if num_sectors >= 5:
            score += 10
        elif num_sectors >= 3:
            score += 5
        
        # Reward having reasonable number of stocks (not too few, not too many)
        num_stocks = len(stock_allocation)
        if 8 <= num_stocks <= 20:
            score += 10
        elif 5 <= num_stocks <= 30:
            score += 5
        
        return max(0, min(100, score))
    
    def create_allocation_charts(self, allocation_data):
        """Create pie charts for portfolio allocation"""
        stock_allocation = allocation_data['stock_allocation']
        sector_allocation = allocation_data['sector_allocation']
        
        # Stock allocation chart (top 10)
        top_stocks = stock_allocation.head(10)
        fig_stocks = px.pie(
            top_stocks, 
            values='Allocation_%', 
            names='Symbol',
            title='Top 10 Stock Allocation (%)',
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig_stocks.update_traces(textposition='inside', textinfo='percent+label')
        
        # Sector allocation chart
        fig_sectors = px.pie(
            sector_allocation, 
            values='Allocation_%', 
            names='Sector',
            title='Sector Allocation (%)',
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig_sectors.update_traces(textposition='inside', textinfo='percent+label')
        
        return fig_stocks, fig_sectors
    
    def create_performance_chart(self, portfolio_performance, benchmarks):
        """Create performance comparison chart"""
        if portfolio_performance is None:
            return None
        
        # Create performance summary
        performance_data = []
        
        # Portfolio performance
        weighted_cagr = portfolio_performance['weighted_cagr']
        performance_data.append({
            'Asset': 'Your Portfolio',
            'CAGR (%)': weighted_cagr,
            'Color': '#1f77b4'
        })
        
        # Benchmark performance
        for benchmark_name, benchmark_data in benchmarks.items():
            performance_data.append({
                'Asset': benchmark_name,
                'CAGR (%)': benchmark_data['cagr'],
                'Color': '#ff7f0e' if benchmark_name == 'Nifty50' else '#2ca02c'
            })
        
        df_perf = pd.DataFrame(performance_data)
        
        # Create bar chart
        fig = px.bar(
            df_perf, 
            x='Asset', 
            y='CAGR (%)',
            title='Portfolio vs Benchmark Performance (CAGR)',
            color='Asset',
            color_discrete_map={row['Asset']: row['Color'] for row in performance_data}
        )
        
        fig.update_layout(
            yaxis_title="CAGR (%)",
            xaxis_title="",
            showlegend=False
        )
        
        return fig
    
    def get_stock_ratios(self, symbol):
        """Fetch key financial ratios for a stock"""
        try:
            # Try yfinance first for comprehensive data
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            ratios = {
                'pe_ratio': info.get('trailingPE'),
                'forward_pe': info.get('forwardPE'),
                'pb_ratio': info.get('priceToBook'),
                'roe': info.get('returnOnEquity'),
                'debt_to_equity': info.get('debtToEquity'),
                'dividend_yield': info.get('dividendYield'),
                'peg_ratio': info.get('pegRatio'),
                'current_ratio': info.get('currentRatio'),
                'quick_ratio': info.get('quickRatio'),
                'gross_margin': info.get('grossMargins'),
                'operating_margin': info.get('operatingMargins'),
                'profit_margin': info.get('profitMargins'),
                'revenue_growth': info.get('revenueGrowth'),
                'earnings_growth': info.get('earningsGrowth'),
                'book_value': info.get('bookValue'),
                'market_cap': info.get('marketCap')
            }
            
            # Convert percentages where needed
            if ratios['roe']:
                ratios['roe'] *= 100  # Convert to percentage
            if ratios['dividend_yield']:
                ratios['dividend_yield'] *= 100  # Convert to percentage
            if ratios['revenue_growth']:
                ratios['revenue_growth'] *= 100
            if ratios['earnings_growth']:
                ratios['earnings_growth'] *= 100
            if ratios['gross_margin']:
                ratios['gross_margin'] *= 100
            if ratios['operating_margin']:
                ratios['operating_margin'] *= 100
            if ratios['profit_margin']:
                ratios['profit_margin'] *= 100
            
            return ratios
            
        except Exception as e:
            # Fallback to basic ratios from data fetcher
            try:
                base_symbol = symbol.replace('.NS', '').replace('.BO', '')
                fundamentals = self.data_fetcher.get_fundamentals(base_symbol)
                
                return {
                    'pe_ratio': fundamentals.get('pe_ratio'),
                    'pb_ratio': fundamentals.get('pb_ratio'),
                    'roe': fundamentals.get('roe'),
                    'debt_to_equity': fundamentals.get('debt_to_equity'),
                    'dividend_yield': fundamentals.get('dividend_yield'),
                    'peg_ratio': fundamentals.get('peg_ratio'),
                    'forward_pe': None,
                    'current_ratio': None,
                    'quick_ratio': None,
                    'gross_margin': None,
                    'operating_margin': None,
                    'profit_margin': None,
                    'revenue_growth': fundamentals.get('revenue_growth'),
                    'earnings_growth': fundamentals.get('earnings_growth'),
                    'book_value': None,
                    'market_cap': fundamentals.get('market_cap')
                }
            except:
                return {}
    
    def analyze_ratios(self, ratios):
        """Analyze ratios and provide flags and recommendations"""
        analysis = {
            'flags': {},
            'recommendations': [],
            'overall_score': 0,
            'overall_flag': 'NEUTRAL',
            'action': 'HOLD'
        }
        
        score = 0
        total_ratios = 0
        
        # P/E Ratio Analysis
        if ratios.get('pe_ratio') and ratios['pe_ratio'] > 0:
            pe = ratios['pe_ratio']
            total_ratios += 1
            if pe < 12:
                analysis['flags']['pe_ratio'] = 'POSITIVE'
                analysis['recommendations'].append(f"✅ Excellent P/E ({pe:.1f}) - Stock appears undervalued")
                score += 2
            elif pe < 18:
                analysis['flags']['pe_ratio'] = 'POSITIVE'
                analysis['recommendations'].append(f"👍 Good P/E ({pe:.1f}) - Fairly valued stock")
                score += 1
            elif pe < 25:
                analysis['flags']['pe_ratio'] = 'NEUTRAL'
                analysis['recommendations'].append(f"⚖️ Moderate P/E ({pe:.1f}) - Watch for growth justification")
            elif pe < 35:
                analysis['flags']['pe_ratio'] = 'NEGATIVE'
                analysis['recommendations'].append(f"⚠️ High P/E ({pe:.1f}) - Expensive, ensure strong growth")
                score -= 1
            else:
                analysis['flags']['pe_ratio'] = 'NEGATIVE'
                analysis['recommendations'].append(f"🔴 Very High P/E ({pe:.1f}) - Significantly overvalued")
                score -= 2
        
        # P/B Ratio Analysis
        if ratios.get('pb_ratio') and ratios['pb_ratio'] > 0:
            pb = ratios['pb_ratio']
            total_ratios += 1
            if pb < 1:
                analysis['flags']['pb_ratio'] = 'POSITIVE'
                analysis['recommendations'].append(f"💎 Excellent P/B ({pb:.1f}) - Trading below book value")
                score += 2
            elif pb < 2:
                analysis['flags']['pb_ratio'] = 'POSITIVE'
                analysis['recommendations'].append(f"👍 Good P/B ({pb:.1f}) - Reasonable valuation")
                score += 1
            elif pb < 4:
                analysis['flags']['pb_ratio'] = 'NEUTRAL'
                analysis['recommendations'].append(f"⚖️ Moderate P/B ({pb:.1f}) - Fair premium to book")
            else:
                analysis['flags']['pb_ratio'] = 'NEGATIVE'
                analysis['recommendations'].append(f"🔴 High P/B ({pb:.1f}) - Expensive relative to assets")
                score -= 1
        
        # ROE Analysis
        if ratios.get('roe') and ratios['roe'] > 0:
            roe = ratios['roe']
            total_ratios += 1
            if roe > 25:
                analysis['flags']['roe'] = 'POSITIVE'
                analysis['recommendations'].append(f"🚀 Excellent ROE ({roe:.1f}%) - Highly profitable company")
                score += 2
            elif roe > 15:
                analysis['flags']['roe'] = 'POSITIVE'
                analysis['recommendations'].append(f"✅ Good ROE ({roe:.1f}%) - Strong profitability")
                score += 1
            elif roe > 10:
                analysis['flags']['roe'] = 'NEUTRAL'
                analysis['recommendations'].append(f"⚖️ Moderate ROE ({roe:.1f}%) - Acceptable returns")
            else:
                analysis['flags']['roe'] = 'NEGATIVE'
                analysis['recommendations'].append(f"🔴 Low ROE ({roe:.1f}%) - Poor capital efficiency")
                score -= 1
        
        # Debt-to-Equity Analysis
        if ratios.get('debt_to_equity') is not None:
            de = ratios['debt_to_equity']
            total_ratios += 1
            if de < 0.3:
                analysis['flags']['debt_to_equity'] = 'POSITIVE'
                analysis['recommendations'].append(f"💪 Low Debt ({de:.1f}) - Strong balance sheet")
                score += 1
            elif de < 0.6:
                analysis['flags']['debt_to_equity'] = 'NEUTRAL'
                analysis['recommendations'].append(f"⚖️ Moderate Debt ({de:.1f}) - Manageable leverage")
            elif de < 1.0:
                analysis['flags']['debt_to_equity'] = 'NEGATIVE'
                analysis['recommendations'].append(f"⚠️ High Debt ({de:.1f}) - Monitor financial health")
                score -= 1
            else:
                analysis['flags']['debt_to_equity'] = 'NEGATIVE'
                analysis['recommendations'].append(f"🔴 Very High Debt ({de:.1f}) - Significant financial risk")
                score -= 2
        
        # Dividend Yield Analysis
        if ratios.get('dividend_yield') and ratios['dividend_yield'] > 0:
            dy = ratios['dividend_yield']
            if dy > 4:
                analysis['flags']['dividend_yield'] = 'POSITIVE'
                analysis['recommendations'].append(f"💰 High Dividend ({dy:.1f}%) - Good income potential")
                score += 1
            elif dy > 2:
                analysis['flags']['dividend_yield'] = 'POSITIVE'
                analysis['recommendations'].append(f"👍 Good Dividend ({dy:.1f}%) - Decent income")
            elif dy > 1:
                analysis['flags']['dividend_yield'] = 'NEUTRAL'
                analysis['recommendations'].append(f"⚖️ Moderate Dividend ({dy:.1f}%) - Some income")
            else:
                analysis['flags']['dividend_yield'] = 'NEGATIVE'
                analysis['recommendations'].append(f"📉 Low Dividend ({dy:.1f}%) - Limited income")
        
        # PEG Ratio Analysis
        if ratios.get('peg_ratio') and ratios['peg_ratio'] > 0:
            peg = ratios['peg_ratio']
            total_ratios += 1
            if peg < 1:
                analysis['flags']['peg_ratio'] = 'POSITIVE'
                analysis['recommendations'].append(f"🎯 Excellent PEG ({peg:.1f}) - Growth at reasonable price")
                score += 1
            elif peg < 1.5:
                analysis['flags']['peg_ratio'] = 'NEUTRAL'
                analysis['recommendations'].append(f"⚖️ Fair PEG ({peg:.1f}) - Reasonable growth premium")
            else:
                analysis['flags']['peg_ratio'] = 'NEGATIVE'
                analysis['recommendations'].append(f"🔴 High PEG ({peg:.1f}) - Expensive for growth rate")
                score -= 1
        
        # Calculate overall score and action
        if total_ratios > 0:
            avg_score = score / total_ratios
            analysis['overall_score'] = avg_score
            
            if avg_score >= 1:
                analysis['overall_flag'] = 'POSITIVE'
                analysis['action'] = 'BUY MORE'
            elif avg_score >= 0.3:
                analysis['overall_flag'] = 'POSITIVE'  
                analysis['action'] = 'HOLD & ACCUMULATE'
            elif avg_score >= -0.3:
                analysis['overall_flag'] = 'NEUTRAL'
                analysis['action'] = 'HOLD'
            elif avg_score >= -1:
                analysis['overall_flag'] = 'NEGATIVE'
                analysis['action'] = 'MONITOR CLOSELY'
            else:
                analysis['overall_flag'] = 'NEGATIVE'
                analysis['action'] = 'CONSIDER SELLING'
        
        return analysis
    
    def get_portfolio_ratios_analysis(self, portfolio_df):
        """Get ratio analysis for entire portfolio"""
        if portfolio_df is None or portfolio_df.empty:
            return None
        
        portfolio_ratios = []
        
        for _, row in portfolio_df.iterrows():
            symbol = row['Symbol_NSE']
            
            # Get ratios for this stock
            ratios = self.get_stock_ratios(symbol)
            
            # Analyze ratios
            ratio_analysis = self.analyze_ratios(ratios)
            
            # Combine with portfolio data
            stock_analysis = {
                'Symbol': row['Symbol'],
                'Sector': row.get('Sector', 'Others'),
                'Current_Value': row.get('Current_Value', 0),
                'P&L_%': row.get('P&L_%', 0),
                'ratios': ratios,
                'analysis': ratio_analysis
            }
            
            portfolio_ratios.append(stock_analysis)
        
        return portfolio_ratios
    
    def create_ratio_summary_chart(self, portfolio_ratios):
        """Create summary chart of ratio flags across portfolio"""
        if not portfolio_ratios:
            return None
        
        # Count flags by type
        flag_counts = {'POSITIVE': 0, 'NEUTRAL': 0, 'NEGATIVE': 0}
        action_counts = {}
        
        for stock in portfolio_ratios:
            overall_flag = stock['analysis']['overall_flag']
            action = stock['analysis']['action']
            
            flag_counts[overall_flag] += 1
            action_counts[action] = action_counts.get(action, 0) + 1
        
        # Create flag distribution chart
        fig_flags = px.pie(
            values=list(flag_counts.values()),
            names=list(flag_counts.keys()),
            title='Portfolio Ratio Analysis Distribution',
            color_discrete_map={
                'POSITIVE': '#00C851',
                'NEUTRAL': '#ffbb33', 
                'NEGATIVE': '#ff4444'
            }
        )
        fig_flags.update_traces(textposition='inside', textinfo='percent+label')
        
        # Create action recommendation chart
        fig_actions = px.bar(
            x=list(action_counts.keys()),
            y=list(action_counts.values()),
            title='Recommended Actions Distribution',
            color=list(action_counts.keys()),
            color_discrete_map={
                'BUY MORE': '#00C851',
                'HOLD & ACCUMULATE': '#2BBBAD',
                'HOLD': '#ffbb33',
                'MONITOR CLOSELY': '#ff8800',
                'CONSIDER SELLING': '#ff4444'
            }
        )
        fig_actions.update_layout(showlegend=False, xaxis_title="Action", yaxis_title="Number of Stocks")
        
        return fig_flags, fig_actions
