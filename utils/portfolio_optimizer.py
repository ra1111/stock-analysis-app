import pandas as pd
import numpy as np
from scipy.optimize import minimize
from utils.data_fetcher import DataFetcher
import streamlit as st

class PortfolioOptimizer:
    def __init__(self):
        self.data_fetcher = DataFetcher()
    
    def optimize_portfolio(self, tickers, investment_amount=100000, optimization_type='max_sharpe'):
        """Optimize portfolio allocation"""
        if len(tickers) < 2:
            st.error("Need at least 2 stocks for optimization")
            return None
        
        # Get price data
        price_data = {}
        for ticker in tickers:
            data = self.data_fetcher.get_stock_data(ticker, period="2y")
            if not data.empty:
                price_data[ticker] = data['Close']
        
        if len(price_data) < 2:
            st.error("Could not fetch sufficient price data for optimization")
            return None
        
        # Create price DataFrame
        prices_df = pd.DataFrame(price_data)
        prices_df = prices_df.dropna()
        
        if prices_df.empty:
            st.error("No overlapping price data found")
            return None
        
        try:
            # Calculate expected returns and covariance matrix
            returns = prices_df.pct_change().dropna()
            mu = returns.mean() * 252  # Annualized expected returns
            cov_matrix = returns.cov() * 252  # Annualized covariance matrix
            
            # Number of assets
            n_assets = len(mu)
            
            # Optimization function for Sharpe ratio
            def negative_sharpe(weights, mu, cov_matrix, risk_free_rate=0.02):
                portfolio_return = np.sum(mu * weights)
                portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_std
                return -sharpe_ratio
            
            # Optimization function for portfolio variance
            def portfolio_volatility(weights, cov_matrix):
                return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            
            # Constraints and bounds
            constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
            bounds = tuple((0, 1) for _ in range(n_assets))
            
            # Initial guess (equal weights)
            initial_guess = np.array([1/n_assets] * n_assets)
            
            if optimization_type == 'max_sharpe':
                result = minimize(negative_sharpe, initial_guess, args=(mu, cov_matrix),
                                method='SLSQP', bounds=bounds, constraints=constraints)
                title = "Maximum Sharpe Ratio Portfolio"
                
            elif optimization_type == 'min_volatility':
                result = minimize(portfolio_volatility, initial_guess, args=(cov_matrix,),
                                method='SLSQP', bounds=bounds, constraints=constraints)
                title = "Minimum Volatility Portfolio"
                
            elif optimization_type == 'target_return':
                target_return = st.sidebar.slider("Target Annual Return", 0.05, 0.30, 0.15, 0.01)
                constraints = [
                    {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                    {'type': 'eq', 'fun': lambda x: np.sum(x * mu) - target_return}
                ]
                result = minimize(portfolio_volatility, initial_guess, args=(cov_matrix,),
                                method='SLSQP', bounds=bounds, constraints=constraints)
                title = f"Target Return Portfolio ({target_return:.1%})"
            
            else:
                result = minimize(negative_sharpe, initial_guess, args=(mu, cov_matrix),
                                method='SLSQP', bounds=bounds, constraints=constraints)
                title = "Maximum Sharpe Ratio Portfolio"
            
            if not result.success:
                st.error("Optimization failed to converge")
                return None
            
            # Extract optimal weights
            optimal_weights = result.x
            
            # Clean weights (remove tiny allocations)
            cleaned_weights = {}
            for i, ticker in enumerate(prices_df.columns):
                if optimal_weights[i] > 0.001:  # Only include weights > 0.1%
                    cleaned_weights[ticker] = optimal_weights[i]
            
            # Normalize cleaned weights
            total_weight = sum(cleaned_weights.values())
            cleaned_weights = {k: v/total_weight for k, v in cleaned_weights.items()}
            
            # Calculate portfolio performance
            portfolio_return = np.sum(mu * optimal_weights)
            portfolio_volatility = np.sqrt(np.dot(optimal_weights.T, np.dot(cov_matrix, optimal_weights)))
            sharpe_ratio = (portfolio_return - 0.02) / portfolio_volatility
            
            performance = (portfolio_return, portfolio_volatility, sharpe_ratio)
            
            # Get latest prices for discrete allocation
            latest_prices = prices_df.iloc[-1].to_dict()
            
            # Simple discrete allocation
            allocation = {}
            total_allocated = 0
            
            for ticker, weight in cleaned_weights.items():
                target_value = weight * investment_amount
                price = latest_prices[ticker]
                shares = int(target_value / price)
                if shares > 0:
                    allocation[ticker] = shares
                    total_allocated += shares * price
            
            leftover = investment_amount - total_allocated
            
            return {
                'title': title,
                'weights': cleaned_weights,
                'performance': {
                    'expected_return': performance[0],
                    'volatility': performance[1],
                    'sharpe_ratio': performance[2]
                },
                'allocation': allocation,
                'leftover': leftover,
                'investment_amount': investment_amount,
                'prices_df': prices_df
            }
            
        except Exception as e:
            st.error(f"Optimization failed: {str(e)}")
            return None
    
    def calculate_portfolio_metrics(self, portfolio_df):
        """Calculate current portfolio metrics"""
        if portfolio_df.empty:
            return None
        
        tickers = portfolio_df['Ticker'].tolist()
        
        # Get price data
        price_data = {}
        for ticker in tickers:
            data = self.data_fetcher.get_stock_data(ticker, period="1y")
            if not data.empty:
                price_data[ticker] = data['Close']
        
        if not price_data:
            return None
        
        prices_df = pd.DataFrame(price_data)
        prices_df = prices_df.dropna()
        
        if prices_df.empty:
            return None
        
        # Calculate current weights
        total_value = portfolio_df['Current Value'].sum()
        current_weights = {}
        
        for _, row in portfolio_df.iterrows():
            ticker = row['Ticker']
            if ticker in prices_df.columns:
                weight = row['Current Value'] / total_value
                current_weights[ticker] = weight
        
        try:
            # Calculate returns and risk metrics
            returns = prices_df.pct_change().dropna()
            
            # Portfolio returns
            portfolio_returns = np.zeros(len(returns))
            for ticker, weight in current_weights.items():
                if ticker in returns.columns:
                    portfolio_returns += weight * returns[ticker]
            
            # Calculate metrics
            annual_return = np.mean(portfolio_returns) * 252
            annual_volatility = np.std(portfolio_returns) * np.sqrt(252)
            sharpe_ratio = annual_return / annual_volatility if annual_volatility > 0 else 0
            
            # Maximum drawdown
            cumulative_returns = (1 + pd.Series(portfolio_returns)).cumprod()
            rolling_max = cumulative_returns.expanding().max()
            drawdowns = (cumulative_returns - rolling_max) / rolling_max
            max_drawdown = drawdowns.min()
            
            # VaR (Value at Risk) - 95% confidence
            var_95 = np.percentile(portfolio_returns, 5)
            
            return {
                'annual_return': annual_return,
                'annual_volatility': annual_volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'var_95': var_95,
                'current_weights': current_weights,
                'total_value': total_value
            }
            
        except Exception as e:
            st.error(f"Failed to calculate portfolio metrics: {str(e)}")
            return None
    
    def suggest_rebalancing(self, current_portfolio, optimized_weights, investment_amount):
        """Suggest rebalancing actions"""
        suggestions = []
        
        current_total = current_portfolio['Current Value'].sum()
        target_total = investment_amount
        
        for ticker, target_weight in optimized_weights.items():
            current_row = current_portfolio[current_portfolio['Ticker'] == ticker]
            
            target_value = target_weight * target_total
            current_value = current_row['Current Value'].iloc[0] if not current_row.empty else 0
            
            difference = target_value - current_value
            current_price = self.data_fetcher.get_current_price(ticker)
            
            if current_price > 0:
                shares_difference = difference / current_price
                
                if abs(difference) > target_total * 0.01:  # Only suggest if difference > 1%
                    action = "BUY" if difference > 0 else "SELL"
                    suggestions.append({
                        'Ticker': ticker,
                        'Action': action,
                        'Shares': abs(shares_difference),
                        'Value': abs(difference),
                        'Current Weight': (current_value / current_total) * 100 if current_total > 0 else 0,
                        'Target Weight': target_weight * 100,
                        'Difference': difference
                    })
        
        return pd.DataFrame(suggestions)
    
    def generate_efficient_frontier(self, tickers, num_portfolios=100):
        """Generate efficient frontier data for plotting"""
        if len(tickers) < 2:
            return None
        
        price_data = {}
        for ticker in tickers:
            data = self.data_fetcher.get_stock_data(ticker, period="2y")
            if not data.empty:
                price_data[ticker] = data['Close']
        
        if len(price_data) < 2:
            return None
        
        prices_df = pd.DataFrame(price_data)
        prices_df = prices_df.dropna()
        
        if prices_df.empty:
            return None
        
        try:
            mu = expected_returns.mean_historical_return(prices_df)
            S = risk_models.sample_cov(prices_df)
            
            # Generate range of target returns
            min_ret = mu.min()
            max_ret = mu.max()
            target_returns = np.linspace(min_ret, max_ret, num_portfolios)
            
            efficient_portfolios = []
            
            for target_return in target_returns:
                try:
                    ef = EfficientFrontier(mu, S)
                    ef.efficient_return(target_return)
                    ret, vol, sharpe = ef.portfolio_performance()
                    
                    efficient_portfolios.append({
                        'return': ret,
                        'volatility': vol,
                        'sharpe': sharpe
                    })
                except:
                    continue
            
            return pd.DataFrame(efficient_portfolios)
            
        except Exception as e:
            st.error(f"Failed to generate efficient frontier: {str(e)}")
            return None
