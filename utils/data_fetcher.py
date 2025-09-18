import yfinance as yf
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta, date
import streamlit as st
import time
import logging
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.fundamentaldata import FundamentalData
import nsepython as nse
import diskcache as dc
from jugaad_data.nse import NSELive, stock_df
from typing import Optional, Dict, List, Tuple, Any
import os
import hashlib
import json

class DataFetcher:
    def __init__(self):
        # Cache configuration
        self.cache_timeout = 300  # 5 minutes
        self.max_retries = 3
        self.retry_delay = 1  # seconds
        
        # Initialize persistent cache
        cache_dir = os.path.join(os.getcwd(), '.cache')
        os.makedirs(cache_dir, exist_ok=True)
        self.cache = dc.Cache(cache_dir, size_limit=int(100e6))  # 100MB cache
        
        # Data quality and rate limiting
        self.data_quality_cache = {}
        self.api_call_counts = {'jugaad': 0, 'nsepython': 0, 'alphavantage': 0, 'yfinance': 0}
        self.last_api_call = {'jugaad': 0, 'nsepython': 0, 'alphavantage': 0, 'yfinance': 0}
        
        # Rate limits (calls per minute)
        self.rate_limits = {
            'jugaad': 30,  # Conservative limit for scraping
            'nsepython': 60,  # Higher limit
            'alphavantage': 5,  # Free tier limit
            'yfinance': 60  # Generally reliable
        }
        
        # Data sources priority for Indian stocks
        self.indian_sources = ['jugaad', 'nsepython', 'yfinance']
        self.us_sources = ['yfinance', 'alphavantage']
        
        # Initialize NSE Live API
        try:
            self.nse_live = NSELive()
        except Exception as e:
            logging.warning(f"NSE Live initialization failed: {e}")
            self.nse_live = None
        
        # Indian stock exchanges and their suffixes
        self.indian_exchanges = {
            'NSE': '.NS',
            'BSE': '.BO'
        }
        
        # Alpha Vantage setup (using free tier)
        self.alpha_vantage_key = None  # Will be set if API key is available
        self.alpha_vantage_ts = None
        self.alpha_vantage_fd = None
    
    def get_stock_data(self, ticker: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
        """Fetch stock data with multi-source fallback and caching"""
        # Generate cache key
        cache_key = self._generate_cache_key('stock_data', ticker, period, interval)
        
        # Check cache first
        cached_data = self._get_from_cache(cache_key)
        if cached_data is not None:
            return cached_data
        
        # Normalize ticker and determine market
        normalized_ticker, market = self._normalize_ticker(ticker)
        
        # Select appropriate data sources based on market
        sources = self.indian_sources if market == 'indian' else self.us_sources
        
        data = pd.DataFrame()
        errors = []
        
        for source in sources:
            try:
                if self._check_rate_limit(source):
                    if source == 'jugaad' and market == 'indian':
                        data = self._fetch_jugaad_data(normalized_ticker, period, interval)
                    elif source == 'nsepython' and market == 'indian':
                        data = self._fetch_nsepython_data(normalized_ticker, period, interval)
                    elif source == 'yfinance':
                        data = self._fetch_yfinance_data(normalized_ticker, period, interval)
                    elif source == 'alphavantage' and market == 'us':
                        data = self._fetch_alphavantage_data(normalized_ticker, period, interval)
                    
                    if not data.empty:
                        # Cache successful result
                        self._save_to_cache(cache_key, data)
                        self._update_api_call_count(source)
                        break
                        
            except Exception as e:
                errors.append(f"{source}: {str(e)}")
                continue
        
        if data.empty:
            logging.error(f"All data sources failed for {ticker}. Errors: {errors}")
        
        return data
    
    def get_stock_info(self, ticker: str) -> Dict[str, Any]:
        """Get stock information and fundamentals with multi-source fallback"""
        # Generate cache key
        cache_key = self._generate_cache_key('stock_info', ticker)
        
        # Check cache first
        cached_info = self._get_from_cache(cache_key)
        if cached_info is not None:
            return cached_info
        
        # Normalize ticker and determine market
        normalized_ticker, market = self._normalize_ticker(ticker)
        
        # Select appropriate data sources based on market
        sources = self.indian_sources if market == 'indian' else self.us_sources
        
        info = {}
        errors = []
        
        for source in sources:
            try:
                if self._check_rate_limit(source):
                    if source == 'nsepython' and market == 'indian':
                        info = self._fetch_nsepython_info(normalized_ticker)
                    elif source == 'yfinance':
                        info = self._fetch_yfinance_info(normalized_ticker)
                    elif source == 'alphavantage' and market == 'us':
                        info = self._fetch_alphavantage_info(normalized_ticker)
                    
                    if info and 'regularMarketPrice' in info or 'currentPrice' in info:
                        # Cache successful result
                        self._save_to_cache(cache_key, info)
                        self._update_api_call_count(source)
                        break
                        
            except Exception as e:
                errors.append(f"{source}: {str(e)}")
                continue
        
        if not info:
            logging.error(f"All info sources failed for {ticker}. Errors: {errors}")
        
        return info
    
    def get_multiple_stocks_data(self, tickers: List[str], period: str = "1y") -> Dict[str, pd.DataFrame]:
        """Fetch data for multiple stocks with optimized caching"""
        data = {}
        for ticker in tickers:
            stock_data = self.get_stock_data(ticker, period)
            if not stock_data.empty:
                data[ticker] = stock_data
        return data
    
    def get_current_price(self, ticker: str) -> float:
        """Get current price for a stock with multiple fallbacks"""
        # Try live NSE API first for Indian stocks
        normalized_ticker, market = self._normalize_ticker(ticker)
        
        if market == 'indian' and self.nse_live:
            try:
                clean_ticker = normalized_ticker.replace('.NS', '').replace('.BO', '')
                quote = self.nse_live.stock_quote(clean_ticker)
                if quote and 'priceInfo' in quote:
                    return float(quote['priceInfo']['lastPrice'])
            except Exception as e:
                logging.debug(f"NSE Live failed for {ticker}: {e}")
        
        # Fallback to regular info fetching
        try:
            info = self.get_stock_info(ticker)
            return float(info.get('regularMarketPrice', info.get('currentPrice', 0)))
        except:
            return 0.0
    
    def get_fundamentals_with_retry(self, ticker: str) -> Dict[str, Any]:
        """Get fundamentals with retry mechanism and data quality tracking"""
        for attempt in range(self.max_retries):
            try:
                fundamentals = self.get_fundamentals(ticker)
                
                # Track data quality
                self._assess_data_quality(ticker, fundamentals)
                
                return fundamentals
            except Exception as e:
                if attempt == self.max_retries - 1:
                    # Last attempt failed
                    self.data_quality_cache[ticker] = {
                        'quality_score': 0,
                        'missing_fields': [],
                        'error': str(e),
                        'last_updated': datetime.now()
                    }
                    return self._get_empty_fundamentals()
                time.sleep(self.retry_delay)
        
        return self._get_empty_fundamentals()
    
    def get_fundamentals(self, ticker: str) -> Dict[str, Any]:
        """Extract key fundamental metrics with enhanced multi-source support"""
        info = self.get_stock_info(ticker)
        
        if not info:
            return self._get_empty_fundamentals()
        
        fundamentals = {
            'market_cap': self._safe_get(info, 'marketCap'),
            'pe_ratio': self._safe_get(info, 'trailingPE'),
            'pb_ratio': self._safe_get(info, 'priceToBook'),
            'roe': self._safe_get(info, 'returnOnEquity'),
            'debt_to_equity': self._safe_get(info, 'debtToEquity'),
            'dividend_yield': self._safe_get(info, 'dividendYield'),
            'profit_margin': self._safe_get(info, 'profitMargins'),
            'revenue_growth': self._safe_get(info, 'revenueGrowth'),
            'earnings_growth': self._safe_get(info, 'earningsGrowth'),
            'peg_ratio': self._safe_get(info, 'pegRatio'),
            'price_to_sales': self._safe_get(info, 'priceToSalesTrailing12Months'),
            'current_ratio': self._safe_get(info, 'currentRatio'),
            'book_value': self._safe_get(info, 'bookValue'),
            'enterprise_value': self._safe_get(info, 'enterpriseValue'),
            'ebitda': self._safe_get(info, 'ebitda')
        }
        
        # Convert percentage values (only if not None)
        percentage_fields = ['roe', 'dividend_yield', 'profit_margin', 'revenue_growth', 'earnings_growth']
        for field in percentage_fields:
            if fundamentals[field] is not None and isinstance(fundamentals[field], (int, float)):
                fundamentals[field] = fundamentals[field] * 100
        
        return fundamentals
    
    def _safe_get(self, data: Dict[str, Any], key: str, default: Any = None) -> Any:
        """Safely get value from dict, handling various edge cases"""
        value = data.get(key, default)
        
        # Handle infinite values and NaN
        if value is not None:
            if isinstance(value, (int, float)):
                if np.isnan(value) or np.isinf(value):
                    return None
                if value < 0 and key in ['market_cap', 'pe_ratio', 'pb_ratio']:  # These shouldn't be negative
                    return None
                # Handle very large or suspicious values
                if key == 'pe_ratio' and value > 1000:  # Unrealistic P/E
                    return None
                if key == 'pb_ratio' and value > 100:   # Unrealistic P/B
                    return None
        
        return value
    
    def _get_empty_fundamentals(self) -> Dict[str, Any]:
        """Return fundamentals dict with None values"""
        return {
            'market_cap': None,
            'pe_ratio': None,
            'pb_ratio': None,
            'roe': None,
            'debt_to_equity': None,
            'dividend_yield': None,
            'profit_margin': None,
            'revenue_growth': None,
            'earnings_growth': None,
            'peg_ratio': None,
            'price_to_sales': None,
            'current_ratio': None,
            'book_value': None,
            'enterprise_value': None,
            'ebitda': None
        }
    
    # ============================================================================
    # CACHE MANAGEMENT AND CLEANUP
    # ============================================================================
    
    def clear_cache(self) -> None:
        """Clear all cached data"""
        try:
            self.cache.clear()
            logging.info("Cache cleared successfully")
        except Exception as e:
            logging.error(f"Error clearing cache: {e}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        try:
            return {
                'size': self.cache.size if hasattr(self.cache, 'size') else 0,
                'volume': self.cache.volume() if hasattr(self.cache, 'volume') else 0,
                'api_calls': self.api_call_counts.copy(),
                'last_calls': {k: datetime.fromtimestamp(v) if v > 0 else None 
                              for k, v in self.last_api_call.items()}
            }
        except Exception as e:
            logging.error(f"Error getting cache stats: {e}")
            return {}
    
    def cleanup_old_cache(self, max_age_days: int = 7) -> None:
        """Remove cache entries older than specified days"""
        try:
            cutoff_date = datetime.now() - timedelta(days=max_age_days)
            keys_to_remove = []
            
            for key in list(self.cache.iterkeys()) if hasattr(self.cache, 'iterkeys') else list(self.cache):
                try:
                    cached_item = self.cache.get(key)
                    if cached_item and isinstance(cached_item, (list, tuple)) and len(cached_item) >= 2:
                        data, timestamp = cached_item[0], cached_item[1]
                        if isinstance(timestamp, datetime) and timestamp < cutoff_date:
                            keys_to_remove.append(key)
                except Exception:
                    keys_to_remove.append(key)  # Remove corrupted entries
            
            for key in keys_to_remove:
                try:
                    del self.cache[key]
                except Exception:
                    pass
                
            logging.info(f"Cleaned up {len(keys_to_remove)} old cache entries")
            
        except Exception as e:
            logging.error(f"Error cleaning up cache: {e}")
    
    def _assess_data_quality(self, ticker: str, fundamentals: Dict[str, Any]) -> None:
        """Assess and cache data quality metrics with enhanced validation"""
        total_fields = len(fundamentals)
        available_fields = sum(1 for v in fundamentals.values() if v is not None)
        missing_fields = [k for k, v in fundamentals.items() if v is None]
        
        # Enhanced quality assessment
        critical_fields = ['market_cap', 'pe_ratio', 'pb_ratio', 'roe']  # Most important fields
        critical_available = sum(1 for field in critical_fields if fundamentals.get(field) is not None)
        critical_score = (critical_available / len(critical_fields)) * 100
        
        # Overall quality score (weighted)
        overall_score = (available_fields / total_fields) * 70 + critical_score * 30 / 100
        
        self.data_quality_cache[ticker] = {
            'quality_score': overall_score,
            'available_fields': available_fields,
            'total_fields': total_fields,
            'missing_fields': missing_fields,
            'critical_score': critical_score,
            'data_freshness': 'fresh',  # Can be 'fresh', 'stale', 'very_old'
            'last_updated': datetime.now(),
            'sources_tried': len([s for s in self.api_call_counts if self.api_call_counts[s] > 0])
        }
    
    def get_data_quality(self, ticker: str) -> Dict[str, Any]:
        """Get data quality information for a ticker"""
        return self.data_quality_cache.get(ticker, {
            'quality_score': 0,
            'missing_fields': [],
            'error': 'No data available'
        })
    
    # ============================================================================
    # TICKER NORMALIZATION AND MARKET DETECTION
    # ============================================================================
    
    def _normalize_ticker(self, ticker: str) -> Tuple[str, str]:
        """Enhanced ticker normalization with proper NSE/BSE handling"""
        ticker = ticker.upper().strip()
        
        # Already has suffix - determine market
        if any(suffix in ticker for suffix in ['.NS', '.BO']):
            return ticker, 'indian'
        elif any(suffix in ticker for suffix in ['.TO', '.L']):
            return ticker, 'international'
        elif '.' not in ticker:
            # US market stocks typically don't need suffixes
            if self._is_likely_us_stock(ticker):
                return ticker, 'us'
            else:
                # Assume Indian stock, default to NSE
                return f"{ticker}.NS", 'indian'
        
        return ticker, 'us'  # Default to US market
    
    def _is_likely_us_stock(self, ticker: str) -> bool:
        """Heuristic to determine if ticker is likely US stock"""
        # Common patterns for US stocks
        us_patterns = [
            len(ticker) <= 4 and ticker.isalpha(),  # Short alphabetic tickers
            ticker in ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN', 'META'],  # Known US stocks
            '.' in ticker and ticker.split('.')[1] in ['US', 'O', 'K']  # US suffixes
        ]
        return any(us_patterns)
    
    # ============================================================================
    # CACHE MANAGEMENT
    # ============================================================================
    
    def _generate_cache_key(self, operation: str, ticker: str, *args) -> str:
        """Generate consistent cache keys"""
        key_data = f"{operation}_{ticker}_{'-'.join(map(str, args))}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _get_from_cache(self, cache_key: str) -> Optional[Any]:
        """Get data from cache with freshness check"""
        try:
            cached_item = self.cache.get(cache_key)
            if cached_item is not None and isinstance(cached_item, (list, tuple)) and len(cached_item) >= 2:
                data, timestamp = cached_item[0], cached_item[1]
                # Check if cache is still fresh
                if isinstance(timestamp, datetime) and (datetime.now() - timestamp).seconds < self.cache_timeout:
                    return data
                else:
                    # Cache expired, remove it
                    try:
                        del self.cache[cache_key]
                    except Exception:
                        pass
            return None
        except Exception as e:
            logging.debug(f"Cache read error: {e}")
            return None
    
    def _save_to_cache(self, cache_key: str, data: Any) -> None:
        """Save data to cache with timestamp"""
        try:
            self.cache[cache_key] = (data, datetime.now())
        except Exception as e:
            logging.debug(f"Cache write error: {e}")
    
    # ============================================================================
    # RATE LIMITING
    # ============================================================================
    
    def _check_rate_limit(self, source: str) -> bool:
        """Check if API call is within rate limits"""
        current_time = time.time()
        last_call = self.last_api_call.get(source, 0)
        
        # Check if enough time has passed since last call
        min_interval = 60.0 / self.rate_limits.get(source, 60)  # seconds between calls
        
        if current_time - last_call >= min_interval:
            return True
        else:
            # Wait if necessary
            wait_time = min_interval - (current_time - last_call)
            if wait_time < 2:  # Only wait if less than 2 seconds
                time.sleep(wait_time)
                return True
            return False
    
    def _update_api_call_count(self, source: str) -> None:
        """Update API call tracking"""
        self.api_call_counts[source] += 1
        self.last_api_call[source] = time.time()
    
    # ============================================================================
    # DATA SOURCE IMPLEMENTATIONS
    # ============================================================================
    
    def _fetch_jugaad_data(self, ticker: str, period: str, interval: str) -> pd.DataFrame:
        """Fetch data using jugaad-data (NSE direct)"""
        try:
            # Convert period to date range
            end_date = date.today()
            if period == "1y":
                start_date = end_date - timedelta(days=365)
            elif period == "6mo":
                start_date = end_date - timedelta(days=180)
            elif period == "3mo":
                start_date = end_date - timedelta(days=90)
            else:
                start_date = end_date - timedelta(days=365)
            
            # Remove .NS/.BO suffix for jugaad
            clean_ticker = ticker.replace('.NS', '').replace('.BO', '')
            
            # Fetch data
            data = stock_df(symbol=clean_ticker, from_date=start_date, to_date=end_date, series="EQ")
            
            if data is not None and not data.empty:
                # Standardize column names to match yfinance format
                column_mapping = {
                    'OPEN': 'Open',
                    'HIGH': 'High', 
                    'LOW': 'Low',
                    'CLOSE': 'Close',
                    'VOLUME': 'Volume'
                }
                data = data.rename(columns=column_mapping)
                # Set date as index
                if 'DATE' in data.columns:
                    data['DATE'] = pd.to_datetime(data['DATE'])
                    data.set_index('DATE', inplace=True)
                
                return data
            
            return pd.DataFrame()
            
        except Exception as e:
            logging.debug(f"Jugaad fetch failed for {ticker}: {e}")
            return pd.DataFrame()
    
    def _fetch_nsepython_data(self, ticker: str, period: str, interval: str) -> pd.DataFrame:
        """Fetch data using nsepython"""
        try:
            # nsepython doesn't have historical data API in the same way
            # We'll use it primarily for current quotes
            clean_ticker = ticker.replace('.NS', '').replace('.BO', '')
            
            # For now, return empty as nsepython is better for real-time quotes
            return pd.DataFrame()
            
        except Exception as e:
            logging.debug(f"NSEPython fetch failed for {ticker}: {e}")
            return pd.DataFrame()
    
    def _fetch_yfinance_data(self, ticker: str, period: str, interval: str) -> pd.DataFrame:
        """Fetch data using yfinance with enhanced Indian stock support"""
        try:
            stock = yf.Ticker(ticker)
            data = stock.history(period=period, interval=interval)
            
            if data.empty and '.NS' in ticker:
                # Try with .BO suffix for BSE
                ticker_bo = ticker.replace('.NS', '.BO')
                stock = yf.Ticker(ticker_bo)
                data = stock.history(period=period, interval=interval)
            
            return data
            
        except Exception as e:
            logging.debug(f"YFinance fetch failed for {ticker}: {e}")
            return pd.DataFrame()
    
    def _fetch_alphavantage_data(self, ticker: str, period: str, interval: str) -> pd.DataFrame:
        """Fetch data using Alpha Vantage (for US stocks)"""
        try:
            if self.alpha_vantage_ts is None:
                return pd.DataFrame()  # No API key configured
            
            # Alpha Vantage has different interval mapping
            av_interval = 'daily'  # Default to daily
            if interval == '1wk':
                av_interval = 'weekly'
            elif interval == '1mo':
                av_interval = 'monthly'
            
            if av_interval == 'daily':
                data, _ = self.alpha_vantage_ts.get_daily(symbol=ticker, outputsize='full')
            elif av_interval == 'weekly':
                data, _ = self.alpha_vantage_ts.get_weekly(symbol=ticker)
            else:
                data, _ = self.alpha_vantage_ts.get_monthly(symbol=ticker)
            
            if data is not None and not data.empty:
                # Standardize column names
                av_column_mapping = {
                    '1. open': 'Open',
                    '2. high': 'High',
                    '3. low': 'Low', 
                    '4. close': 'Close',
                    '5. volume': 'Volume'
                }
                data = data.rename(columns=av_column_mapping)
                # Convert index to datetime
                data.index = pd.to_datetime(data.index)
                return data
            
            return pd.DataFrame()
            
        except Exception as e:
            logging.debug(f"Alpha Vantage fetch failed for {ticker}: {e}")
            return pd.DataFrame()
    
    def _fetch_yfinance_info(self, ticker: str) -> Dict[str, Any]:
        """Fetch stock info using yfinance"""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            if not info or 'regularMarketPrice' not in info:
                # Try with .BO suffix for Indian stocks
                if '.NS' in ticker:
                    ticker_bo = ticker.replace('.NS', '.BO')
                    stock = yf.Ticker(ticker_bo)
                    info = stock.info
            
            return info
            
        except Exception as e:
            logging.debug(f"YFinance info fetch failed for {ticker}: {e}")
            return {}
    
    def _fetch_nsepython_info(self, ticker: str) -> Dict[str, Any]:
        """Fetch stock info using nsepython"""
        try:
            clean_ticker = ticker.replace('.NS', '').replace('.BO', '')
            
            # Get quote data from nsepython
            quote = nse.nsefetch(f'https://www.nseindia.com/api/quote-equity?symbol={clean_ticker}')
            
            if quote and 'priceInfo' in quote:
                # Convert to yfinance-like format
                info = {
                    'regularMarketPrice': quote['priceInfo'].get('lastPrice'),
                    'currentPrice': quote['priceInfo'].get('lastPrice'),
                    'marketCap': quote.get('marketDeptOrderBook', {}).get('totalTradedValue'),
                    'symbol': clean_ticker
                }
                return info
            
            return {}
            
        except Exception as e:
            logging.debug(f"NSEPython info fetch failed for {ticker}: {e}")
            return {}
    
    def _fetch_alphavantage_info(self, ticker: str) -> Dict[str, Any]:
        """Fetch stock info using Alpha Vantage"""
        try:
            if self.alpha_vantage_fd is None:
                return {}
            
            # Get company overview
            data, _ = self.alpha_vantage_fd.get_company_overview(symbol=ticker)
            
            if data and 'Symbol' in data:
                # Convert to yfinance-like format
                info = {
                    'regularMarketPrice': float(data.get('50DayMovingAverage', 0)),
                    'marketCap': int(data.get('MarketCapitalization', 0)) if data.get('MarketCapitalization') != 'None' else None,
                    'trailingPE': float(data.get('PERatio', 0)) if data.get('PERatio') != 'None' else None,
                    'priceToBook': float(data.get('PriceToBookRatio', 0)) if data.get('PriceToBookRatio') != 'None' else None,
                    'symbol': ticker
                }
                return info
            
            return {}
            
        except Exception as e:
            logging.debug(f"Alpha Vantage info fetch failed for {ticker}: {e}")
            return {}
