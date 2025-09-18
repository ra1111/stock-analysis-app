import pandas as pd
import numpy as np
from utils.data_fetcher import DataFetcher
import streamlit as st
from datetime import datetime

class StockScreener:
    def __init__(self):
        self.data_fetcher = DataFetcher()
        # Common Indian and US stock tickers for screening
        self.indian_stocks = [
            'RELIANCE', 'TCS', 'INFY', 'HDFC', 'HDFCBANK', 'ICICIBANK', 'SBIN', 
            'BHARTIARTL', 'ITC', 'KOTAKBANK', 'LT', 'ASIANPAINT', 'AXISBANK',
            'MARUTI', 'HCLTECH', 'WIPRO', 'ULTRACEMCO', 'TATASTEEL', 'ONGC',
            'NTPC', 'POWERGRID', 'NESTLEIND', 'HINDALCO', 'COALINDIA', 'IOC',
            'GRASIM', 'JSWSTEEL', 'TECHM', 'BAJFINANCE', 'BAJAJFINSV', 'ADANIPORTS',
            'TITAN', 'DIVISLAB', 'DRREDDY', 'SUNPHARMA', 'CIPLA', 'BIOCON',
            'VEDL', 'TATAMOTORS', 'M&M', 'HEROMOTOCO', 'BAJAJ-AUTO', 'TVSMOTOR',
            'INDIGO', 'SPICEJET', 'GODREJCP', 'BRITANNIA', 'DABUR', 'HINDUNILVR',
            'MARICO', 'COLPAL', 'PGHH', 'PIDILITIND', 'BERGEPAINT', 'KANSAINER',
            'DMART', 'TRENT', 'JUBLFOOD', 'VBL', 'MCDOWELL-N', 'UBL'
        ]
        
        self.us_stocks = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'BRK-B',
            'JPM', 'JNJ', 'V', 'PG', 'UNH', 'HD', 'MA', 'DIS', 'PYPL', 'BAC',
            'NFLX', 'ADBE', 'CRM', 'CMCSA', 'XOM', 'VZ', 'KO', 'ABT', 'PFE',
            'PEP', 'T', 'CSCO', 'INTC', 'WMT', 'CVX', 'MRK', 'ABBV', 'NKE',
            'TMO', 'ACN', 'COST', 'MDT', 'NEE', 'BMY', 'TXN', 'QCOM', 'LIN',
            'HON', 'UNP', 'UPS', 'PM', 'LOW', 'IBM', 'AMGN', 'RTX', 'SPGI',
            'GS', 'BLK', 'CAT', 'INTU', 'ISRG', 'BKNG', 'AXP', 'DE', 'GILD'
        ]
    
    def screen_stocks(self, strategy, market='both'):
        """Screen stocks based on strategy"""
        if market == 'indian' or market == 'both':
            indian_results = self._screen_market(self.indian_stocks, strategy, 'Indian')
        else:
            indian_results = pd.DataFrame()
        
        if market == 'us' or market == 'both':
            us_results = self._screen_market(self.us_stocks, strategy, 'US')
        else:
            us_results = pd.DataFrame()
        
        # Combine results
        if not indian_results.empty and not us_results.empty:
            return pd.concat([indian_results, us_results], ignore_index=True)
        elif not indian_results.empty:
            return indian_results
        elif not us_results.empty:
            return us_results
        else:
            return pd.DataFrame()
    
    def _screen_market(self, tickers, strategy, market_name):
        """Screen a specific market with enhanced transparency"""
        results = []
        detailed_results = []  # For transparency report
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, ticker in enumerate(tickers):
            status_text.text(f'Analyzing {ticker} ({market_name})...')
            progress_bar.progress((i + 1) / len(tickers))
            
            try:
                # Use enhanced data fetcher with retry
                fundamentals = self.data_fetcher.get_fundamentals_with_retry(ticker)
                data_quality = self.data_fetcher.get_data_quality(ticker)
                
                # Evaluate criteria and get detailed reasons
                meets_criteria, reasons, fallback_score = self._evaluate_stock(fundamentals, strategy, data_quality)
                current_price = self.data_fetcher.get_current_price(ticker)
                
                stock_info = {
                    'Ticker': ticker,
                    'Market': market_name,
                    'Current Price': current_price if current_price > 0 else None,
                    'Market Cap': fundamentals.get('market_cap'),
                    'PE Ratio': fundamentals.get('pe_ratio'),
                    'PB Ratio': fundamentals.get('pb_ratio'),
                    'ROE %': fundamentals.get('roe'),
                    'Debt/Equity': fundamentals.get('debt_to_equity'),
                    'Dividend Yield %': fundamentals.get('dividend_yield'),
                    'Profit Margin %': fundamentals.get('profit_margin'),
                    'Revenue Growth %': fundamentals.get('revenue_growth'),
                    'PEG Ratio': fundamentals.get('peg_ratio'),
                    'Score': fallback_score,
                    'Data Quality %': data_quality.get('quality_score', 0),
                    'Status': 'Included' if meets_criteria else 'Excluded',
                    'Reasons': '; '.join(reasons),
                    'Missing Fields': len(data_quality.get('missing_fields', [])),
                    'Fallback Applied': fallback_score != self._calculate_score(fundamentals, strategy)
                }
                
                # Add to detailed results for transparency
                detailed_results.append(stock_info)
                
                # Only include qualifying stocks in main results
                if meets_criteria:
                    results.append(stock_info)
                    
            except Exception as e:
                # Track failures for transparency
                detailed_results.append({
                    'Ticker': ticker,
                    'Market': market_name,
                    'Status': 'Error',
                    'Reasons': f'Data fetch failed: {str(e)}',
                    'Data Quality %': 0,
                    'Score': 0
                })
                continue
        
        progress_bar.empty()
        status_text.empty()
        
        # Store detailed results for transparency report
        if not hasattr(self, 'screening_report'):
            self.screening_report = []
        self.screening_report.extend(detailed_results)
        
        df = pd.DataFrame(results)
        if not df.empty:
            df = df.sort_values('Score', ascending=False)
        
        return df
    
    def _evaluate_stock(self, fundamentals, strategy, data_quality):
        """Evaluate stock with detailed reasons and fallback scoring"""
        reasons = []
        meets_criteria = False
        
        # Check data availability
        quality_score = data_quality.get('quality_score', 0)
        missing_fields = data_quality.get('missing_fields', [])
        
        if quality_score < 30:  # Less than 30% data available
            reasons.append(f"Insufficient data quality ({quality_score:.1f}%)")
            fallback_score = 0
            return False, reasons, fallback_score
        
        # Extract metrics with None handling
        # Safe extraction with None handling
        market_cap = fundamentals.get('market_cap') or 0
        pe_ratio = fundamentals.get('pe_ratio') or 0
        pb_ratio = fundamentals.get('pb_ratio') or 0
        roe = fundamentals.get('roe') or 0
        debt_equity = fundamentals.get('debt_to_equity') or 0
        dividend_yield = fundamentals.get('dividend_yield') or 0
        profit_margin = fundamentals.get('profit_margin') or 0
        revenue_growth = fundamentals.get('revenue_growth') or 0
        peg_ratio = fundamentals.get('peg_ratio') or 0
        price_to_sales = fundamentals.get('price_to_sales') or 0
        
        # Convert market cap to billions if available
        market_cap_billions = market_cap / 1e9 if market_cap else None
        
        # Strategy-specific evaluation with graceful handling
        if strategy == 'Value':
            meets_criteria, reasons = self._evaluate_value_criteria(
                market_cap_billions, pe_ratio, pb_ratio, roe, debt_equity,
                dividend_yield, profit_margin, revenue_growth, missing_fields
            )
        elif strategy == 'Moonshot':
            meets_criteria, reasons = self._evaluate_moonshot_criteria(
                market_cap_billions, pe_ratio, peg_ratio, roe, debt_equity,
                revenue_growth, profit_margin, missing_fields
            )
        elif strategy == 'Growth':
            meets_criteria, reasons = self._evaluate_growth_criteria(
                peg_ratio, revenue_growth, roe, price_to_sales, debt_equity,
                market_cap_billions, missing_fields
            )
        
        # Calculate fallback score
        fallback_score = self._calculate_fallback_score(fundamentals, strategy, missing_fields)
        
        # Add data quality information to reasons
        if quality_score < 70:
            reasons.append(f"Limited data quality ({quality_score:.1f}%)")
        if missing_fields:
            reasons.append(f"Missing: {', '.join(missing_fields[:3])}{'...' if len(missing_fields) > 3 else ''}")
        
        return meets_criteria, reasons, fallback_score
    
    def _evaluate_value_criteria(self, market_cap, pe_ratio, pb_ratio, roe, debt_equity, dividend_yield, profit_margin, revenue_growth, missing_fields):
        """Evaluate Value strategy criteria with graceful handling"""
        reasons = []
        passes = 0
        total_criteria = 8
        
        # Market Cap check
        if market_cap is not None:
            if market_cap > 5:
                passes += 1
                reasons.append("✓ Large cap (>₹5000Cr)")
            else:
                reasons.append("✗ Too small cap")
        else:
            reasons.append("? Market cap unknown")
            total_criteria -= 1
        
        # PE Ratio check
        if pe_ratio is not None:
            if 0 < pe_ratio < 16:
                passes += 1
                reasons.append("✓ Reasonable PE (<16)")
            else:
                reasons.append(f"✗ High PE ({pe_ratio:.1f})")
        else:
            reasons.append("? PE ratio unknown")
            total_criteria -= 1
        
        # PB Ratio check
        if pb_ratio is not None:
            if 0 < pb_ratio < 3:
                passes += 1
                reasons.append("✓ Good PB (<3)")
            else:
                reasons.append(f"✗ High PB ({pb_ratio:.1f})")
        else:
            reasons.append("? PB ratio unknown")
            total_criteria -= 1
        
        # ROE check
        if roe is not None:
            if roe > 15:
                passes += 1
                reasons.append("✓ Strong ROE (>15%)")
            else:
                reasons.append(f"✗ Weak ROE ({roe:.1f}%)")
        else:
            reasons.append("? ROE unknown")
            total_criteria -= 1
        
        # Debt/Equity check
        if debt_equity is not None:
            if debt_equity < 50:
                passes += 1
                reasons.append("✓ Low debt (<0.5)")
            else:
                reasons.append(f"✗ High debt ({debt_equity:.1f})")
        else:
            reasons.append("? Debt/Equity unknown")
            total_criteria -= 1
        
        # Dividend Yield check
        if dividend_yield is not None:
            if dividend_yield > 1.5:
                passes += 1
                reasons.append("✓ Good dividend (>1.5%)")
            else:
                reasons.append(f"✗ Low dividend ({dividend_yield:.1f}%)")
        else:
            reasons.append("? Dividend yield unknown")
            total_criteria -= 1
        
        # Profit Margin check
        if profit_margin is not None:
            if profit_margin > 15:
                passes += 1
                reasons.append("✓ High margin (>15%)")
            else:
                reasons.append(f"✗ Low margin ({profit_margin:.1f}%)")
        else:
            reasons.append("? Profit margin unknown")
            total_criteria -= 1
        
        # Revenue Growth check
        if revenue_growth is not None:
            if revenue_growth > 10:
                passes += 1
                reasons.append("✓ Growing revenue (>10%)")
            else:
                reasons.append(f"✗ Slow growth ({revenue_growth:.1f}%)")
        else:
            reasons.append("? Revenue growth unknown")
            total_criteria -= 1
        
        # Require at least 70% of criteria to pass when data is available
        meets_criteria = total_criteria > 0 and (passes / total_criteria) >= 0.7
        
        return meets_criteria, reasons
    
    def _evaluate_moonshot_criteria(self, market_cap, pe_ratio, peg_ratio, roe, debt_equity, revenue_growth, profit_margin, missing_fields):
        """Evaluate Moonshot strategy criteria with graceful handling"""
        reasons = []
        passes = 0
        total_criteria = 7
        
        # Similar pattern as value criteria but with moonshot thresholds
        if market_cap is not None:
            if market_cap < 2:
                passes += 1
                reasons.append("✓ Small cap (<₹2000Cr)")
            else:
                reasons.append("✗ Too large cap")
        else:
            reasons.append("? Market cap unknown")
            total_criteria -= 1
        
        if pe_ratio is not None:
            if 0 < pe_ratio < 30:
                passes += 1
                reasons.append("✓ Reasonable PE (<30)")
            else:
                reasons.append(f"✗ High PE ({pe_ratio:.1f})")
        else:
            reasons.append("? PE ratio unknown")
            total_criteria -= 1
        
        if peg_ratio is not None:
            if 0 < peg_ratio < 1:
                passes += 1
                reasons.append("✓ Good PEG (<1)")
            else:
                reasons.append(f"✗ High PEG ({peg_ratio:.1f})")
        else:
            reasons.append("? PEG ratio unknown")
            total_criteria -= 1
        
        if roe is not None:
            if roe > 20:
                passes += 1
                reasons.append("✓ Excellent ROE (>20%)")
            else:
                reasons.append(f"✗ Weak ROE ({roe:.1f}%)")
        else:
            reasons.append("? ROE unknown")
            total_criteria -= 1
        
        if debt_equity is not None:
            if debt_equity < 50:
                passes += 1
                reasons.append("✓ Low debt (<0.5)")
            else:
                reasons.append(f"✗ High debt ({debt_equity:.1f})")
        else:
            reasons.append("? Debt/Equity unknown")
            total_criteria -= 1
        
        if revenue_growth is not None:
            if revenue_growth > 25:
                passes += 1
                reasons.append("✓ High growth (>25%)")
            else:
                reasons.append(f"✗ Slow growth ({revenue_growth:.1f}%)")
        else:
            reasons.append("? Revenue growth unknown")
            total_criteria -= 1
        
        if profit_margin is not None:
            if profit_margin > 15:
                passes += 1
                reasons.append("✓ High margin (>15%)")
            else:
                reasons.append(f"✗ Low margin ({profit_margin:.1f}%)")
        else:
            reasons.append("? Profit margin unknown")
            total_criteria -= 1
        
        meets_criteria = total_criteria > 0 and (passes / total_criteria) >= 0.7
        
        return meets_criteria, reasons
    
    def _evaluate_growth_criteria(self, peg_ratio, revenue_growth, roe, price_to_sales, debt_equity, market_cap, missing_fields):
        """Evaluate Growth strategy criteria with graceful handling"""
        reasons = []
        passes = 0
        total_criteria = 6
        
        if peg_ratio is not None:
            if 0 < peg_ratio < 1:
                passes += 1
                reasons.append("✓ Good PEG (<1)")
            else:
                reasons.append(f"✗ High PEG ({peg_ratio:.1f})")
        else:
            reasons.append("? PEG ratio unknown")
            total_criteria -= 1
        
        if revenue_growth is not None:
            if revenue_growth > 25:
                passes += 1
                reasons.append("✓ High growth (>25%)")
            else:
                reasons.append(f"✗ Slow growth ({revenue_growth:.1f}%)")
        else:
            reasons.append("? Revenue growth unknown")
            total_criteria -= 1
        
        if roe is not None:
            if roe > 20:
                passes += 1
                reasons.append("✓ Strong ROE (>20%)")
            else:
                reasons.append(f"✗ Weak ROE ({roe:.1f}%)")
        else:
            reasons.append("? ROE unknown")
            total_criteria -= 1
        
        if price_to_sales is not None:
            if 0 < price_to_sales < 5:
                passes += 1
                reasons.append("✓ Reasonable P/S (<5)")
            else:
                reasons.append(f"✗ High P/S ({price_to_sales:.1f})")
        else:
            reasons.append("? Price/Sales unknown")
            total_criteria -= 1
        
        if debt_equity is not None:
            if debt_equity < 50:
                passes += 1
                reasons.append("✓ Low debt (<0.5)")
            else:
                reasons.append(f"✗ High debt ({debt_equity:.1f})")
        else:
            reasons.append("? Debt/Equity unknown")
            total_criteria -= 1
        
        if market_cap is not None:
            if market_cap > 1:
                passes += 1
                reasons.append("✓ Adequate size (>₹1000Cr)")
            else:
                reasons.append("✗ Too small cap")
        else:
            reasons.append("? Market cap unknown")
            total_criteria -= 1
        
        meets_criteria = total_criteria > 0 and (passes / total_criteria) >= 0.7
        
        return meets_criteria, reasons
    
    def _calculate_score(self, fundamentals, strategy):
        """Calculate a score for ranking stocks (traditional method)"""
        score = 0
        
        pe_ratio = fundamentals.get('pe_ratio', 0) or 0
        pb_ratio = fundamentals.get('pb_ratio', 0) or 0
        roe = fundamentals.get('roe', 0) or 0
        debt_equity = fundamentals.get('debt_to_equity', 0) or 0
        dividend_yield = fundamentals.get('dividend_yield', 0) or 0
        profit_margin = fundamentals.get('profit_margin', 0) or 0
        revenue_growth = fundamentals.get('revenue_growth', 0) or 0
        peg_ratio = fundamentals.get('peg_ratio', 0) or 0
        
        if strategy == 'Value':
            # Lower PE is better
            if pe_ratio > 0:
                score += max(0, 20 - pe_ratio)
            # Lower PB is better
            if pb_ratio > 0:
                score += max(0, 5 - pb_ratio)
            # Higher ROE is better
            score += min(roe, 50)
            # Higher dividend yield is better
            score += min(dividend_yield * 2, 10)
            # Lower debt is better
            score += max(0, 100 - debt_equity)
            
        elif strategy == 'Moonshot':
            # Higher growth is better
            score += min(revenue_growth, 100)
            score += min(roe, 50)
            score += min(profit_margin, 50)
            if peg_ratio > 0:
                score += max(0, 10 - peg_ratio * 10)
                
        elif strategy == 'Growth':
            score += min(revenue_growth, 100)
            score += min(roe, 50)
            if peg_ratio > 0:
                score += max(0, 10 - peg_ratio * 10)
        
        return round(score, 2)
    
    def _calculate_fallback_score(self, fundamentals, strategy, missing_fields):
        """Calculate score with fallback when data is missing"""
        score = 0
        available_metrics = 0
        total_possible_score = 0
        
        # Extract values, treating None as missing
        pe_ratio = fundamentals.get('pe_ratio') or 0
        pb_ratio = fundamentals.get('pb_ratio') or 0
        roe = fundamentals.get('roe') or 0
        debt_equity = fundamentals.get('debt_to_equity') or 0
        dividend_yield = fundamentals.get('dividend_yield') or 0
        profit_margin = fundamentals.get('profit_margin') or 0
        revenue_growth = fundamentals.get('revenue_growth') or 0
        peg_ratio = fundamentals.get('peg_ratio') or 0
        
        if strategy == 'Value':
            # PE component
            if pe_ratio is not None and pe_ratio > 0:
                component_score = max(0, 20 - pe_ratio)
                score += component_score
                total_possible_score += 20
                available_metrics += 1
            
            # PB component
            if pb_ratio is not None and pb_ratio > 0:
                component_score = max(0, 5 - pb_ratio)
                score += component_score
                total_possible_score += 5
                available_metrics += 1
            
            # ROE component
            if roe is not None:
                component_score = min(roe, 50)
                score += component_score
                total_possible_score += 50
                available_metrics += 1
            
            # Dividend component
            if dividend_yield is not None:
                component_score = min(dividend_yield * 2, 10)
                score += component_score
                total_possible_score += 10
                available_metrics += 1
            
            # Debt component
            if debt_equity is not None:
                component_score = max(0, 100 - debt_equity)
                score += component_score
                total_possible_score += 100
                available_metrics += 1
                
        elif strategy == 'Moonshot':
            # Revenue growth component
            if revenue_growth is not None:
                component_score = min(revenue_growth, 100)
                score += component_score
                total_possible_score += 100
                available_metrics += 1
            
            # ROE component
            if roe is not None:
                component_score = min(roe, 50)
                score += component_score
                total_possible_score += 50
                available_metrics += 1
            
            # Profit margin component
            if profit_margin is not None:
                component_score = min(profit_margin, 50)
                score += component_score
                total_possible_score += 50
                available_metrics += 1
            
            # PEG component
            if peg_ratio is not None and peg_ratio > 0:
                component_score = max(0, 10 - peg_ratio * 10)
                score += component_score
                total_possible_score += 10
                available_metrics += 1
                
        elif strategy == 'Growth':
            # Revenue growth component
            if revenue_growth is not None:
                component_score = min(revenue_growth, 100)
                score += component_score
                total_possible_score += 100
                available_metrics += 1
            
            # ROE component
            if roe is not None:
                component_score = min(roe, 50)
                score += component_score
                total_possible_score += 50
                available_metrics += 1
            
            # PEG component
            if peg_ratio is not None and peg_ratio > 0:
                component_score = max(0, 10 - peg_ratio * 10)
                score += component_score
                total_possible_score += 10
                available_metrics += 1
        
        # Normalize score based on available data
        if available_metrics > 0 and total_possible_score > 0:
            # Scale score to account for missing data
            normalized_score = (score / total_possible_score) * 100
            # Apply penalty for missing data
            data_completeness_penalty = 1 - (len(missing_fields) * 0.1)  # 10% penalty per missing field
            final_score = normalized_score * max(0.3, data_completeness_penalty)  # Minimum 30% of score
        else:
            final_score = 0
        
        return round(final_score, 2)
    
    def get_screening_report(self):
        """Get detailed screening transparency report"""
        if not hasattr(self, 'screening_report') or not self.screening_report:
            return pd.DataFrame()
        
        report_df = pd.DataFrame(self.screening_report)
        return report_df
    
    def get_strategy_description(self, strategy):
        """Get description of screening criteria"""
        descriptions = {
            'Value': """
            **Value Investing Criteria (Graham Style):**
            - Market Cap > ₹5,000 Cr
            - P/E Ratio < 16
            - P/B Ratio < 3
            - ROE > 15%
            - Debt/Equity < 0.5
            - Dividend Yield > 1.5%
            - Profit Margin > 15%
            - Revenue Growth > 10%
            """,
            
            'Moonshot': """
            **Moonshot Stock Criteria:**
            - Market Cap < ₹2,000 Cr
            - P/E Ratio < 30
            - PEG Ratio < 1
            - ROE > 20%
            - Debt/Equity < 0.5
            - Revenue Growth (5Y) > 25%
            - Profit Margin > 15%
            - High growth potential
            """,
            
            'Growth': """
            **Growth Stock Criteria:**
            - PEG Ratio < 1
            - Revenue Growth (5Y) > 25%
            - ROE > 20%
            - Price/Sales < 5
            - Debt/Equity < 0.5
            - Market Cap > ₹1,000 Cr
            - Sustainable growth model
            """
        }
        
        return descriptions.get(strategy, "")
