import yfinance as yf
import pandas as pd
import requests
import feedparser
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime, timedelta
import re
import streamlit as st
import time
from urllib.parse import quote

class NewsAnalyzer:
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()
        self.topic_keywords = {
            'earnings': ['earnings', 'revenue', 'profit', 'loss', 'quarterly', 'q1', 'q2', 'q3', 'q4', 'results', 'financial'],
            'guidance': ['guidance', 'outlook', 'forecast', 'projection', 'expects', 'anticipated', 'target'],
            'dividend': ['dividend', 'buyback', 'payout', 'distribution', 'shareholder', 'return'],
            'management': ['ceo', 'cfo', 'management', 'board', 'director', 'appointment', 'resignation', 'leadership'],
            'regulation': ['regulation', 'compliance', 'lawsuit', 'legal', 'court', 'fine', 'penalty', 'investigation'],
            'merger': ['merger', 'acquisition', 'takeover', 'deal', 'partnership', 'joint venture', 'alliance'],
            'debt': ['debt', 'loan', 'credit', 'financing', 'leverage', 'borrowing', 'refinance'],
            'expansion': ['expansion', 'capex', 'investment', 'facility', 'plant', 'capacity', 'growth'],
            'macro': ['inflation', 'interest rate', 'gdp', 'economic', 'policy', 'government', 'election'],
            'operational': ['production', 'sales', 'launch', 'product', 'service', 'operations', 'contract']
        }
    
    def get_stock_news(self, ticker, max_news=10, days_back=14):
        """Fetch news for a specific stock from multiple sources"""
        all_news = []
        
        try:
            # Get news from yfinance first
            yf_news = self._get_yfinance_news(ticker, max_news)
            all_news.extend(yf_news)
            
            # Get news from Google News RSS
            google_news = self._get_google_news(ticker, max_news, days_back)
            all_news.extend(google_news)
            
            # Remove duplicates and filter by recency
            filtered_news = self._filter_and_deduplicate_news(all_news, days_back)
            
            # Sort by publish time (most recent first) and limit
            filtered_news.sort(key=lambda x: x.get('providerPublishTime', 0), reverse=True)
            
            return filtered_news[:max_news]
            
        except Exception as e:
            st.error(f"Error fetching news for {ticker}: {str(e)}")
            return []
    
    def _get_yfinance_news(self, ticker, max_news):
        """Get news from yfinance"""
        try:
            # Handle Indian stocks
            if not any(suffix in ticker for suffix in ['.NS', '.BO', '.TO', '.L']):
                if ticker.isupper() and '.' not in ticker:
                    ticker = f"{ticker}.NS"
            
            stock = yf.Ticker(ticker)
            news = stock.news
            
            if not news:
                return []
            
            processed_news = []
            for item in news[:max_news]:
                processed_item = {
                    'title': item.get('title', ''),
                    'link': item.get('link', ''),
                    'publisher': item.get('publisher', 'Yahoo Finance'),
                    'providerPublishTime': item.get('providerPublishTime', 0),
                    'type': item.get('type', ''),
                    'summary': item.get('summary', ''),
                    'source': 'yfinance'
                }
                processed_news.append(processed_item)
            
            return processed_news
        except Exception as e:
            return []
    
    def _get_google_news(self, ticker, max_news, days_back):
        """Get news from Google News RSS"""
        try:
            # Clean ticker for search
            search_ticker = ticker.replace('.NS', '').replace('.BO', '')
            
            # Create search query with company name variations
            search_queries = [
                f"{search_ticker} stock",
                f"{search_ticker} shares", 
                f"{search_ticker} company"
            ]
            
            all_google_news = []
            
            for query in search_queries[:1]:  # Limit to 1 query to avoid rate limits
                try:
                    # Google News RSS URL
                    encoded_query = quote(query)
                    url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-US&gl=US&ceid=US:en"
                    
                    # Add headers to avoid blocking
                    headers = {
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                    }
                    
                    response = requests.get(url, headers=headers, timeout=10)
                    
                    if response.status_code == 200:
                        feed = feedparser.parse(response.content)
                        
                        for entry in feed.entries[:max_news]:
                            # Parse publish time
                            pub_time = 0
                            if hasattr(entry, 'published_parsed'):
                                try:
                                    pub_time = int(time.mktime(entry.published_parsed))
                                except:
                                    pub_time = int(time.time()) - 86400  # Default to 1 day ago
                            
                            processed_item = {
                                'title': entry.get('title', ''),
                                'link': entry.get('link', ''),
                                'publisher': entry.get('source', {}).get('title', 'Google News'),
                                'providerPublishTime': pub_time,
                                'type': 'news',
                                'summary': entry.get('summary', ''),
                                'source': 'google_news'
                            }
                            all_google_news.append(processed_item)
                    
                    # Small delay to be respectful
                    time.sleep(0.5)
                    
                except Exception as e:
                    continue  # Skip this query if it fails
            
            return all_google_news
            
        except Exception as e:
            return []
    
    def _filter_and_deduplicate_news(self, news_list, days_back):
        """Filter news by recency and remove duplicates"""
        if not news_list:
            return []
        
        # Calculate cutoff time
        cutoff_time = int(time.time()) - (days_back * 24 * 60 * 60)
        
        # Filter by recency
        recent_news = []
        for item in news_list:
            pub_time = item.get('providerPublishTime', 0)
            if pub_time >= cutoff_time:
                recent_news.append(item)
        
        # Remove duplicates based on title similarity
        unique_news = []
        seen_titles = set()
        
        for item in recent_news:
            title = item.get('title', '').lower()
            # Create a simplified title for comparison
            simple_title = re.sub(r'[^a-zA-Z0-9\s]', '', title)[:50]
            
            if simple_title not in seen_titles:
                seen_titles.add(simple_title)
                unique_news.append(item)
        
        return unique_news
    
    def classify_news_topic(self, title, summary=""):
        """Classify news into topics"""
        text = (title + " " + summary).lower()
        
        topic_scores = {}
        for topic, keywords in self.topic_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text)
            if score > 0:
                topic_scores[topic] = score
        
        if not topic_scores:
            return 'general'
        
        # Return topic with highest score
        return max(topic_scores.items(), key=lambda x: x[1])[0]
    
    def rate_materiality(self, title, summary="", topic="general"):
        """Rate the materiality of news as High/Medium/Low"""
        text = (title + " " + summary).lower()
        
        high_materiality_keywords = [
            'earnings', 'revenue', 'profit', 'loss', 'guidance', 'outlook',
            'merger', 'acquisition', 'lawsuit', 'investigation', 'bankruptcy',
            'ceo', 'resignation', 'dividend', 'buyback', 'restructuring',
            'regulatory action', 'approval', 'rejection', 'partnership'
        ]
        
        medium_materiality_keywords = [
            'contract', 'expansion', 'investment', 'launch', 'upgrade',
            'downgrade', 'rating', 'analyst', 'recommendation', 'target'
        ]
        
        high_score = sum(1 for keyword in high_materiality_keywords if keyword in text)
        medium_score = sum(1 for keyword in medium_materiality_keywords if keyword in text)
        
        if high_score >= 2 or any(keyword in text for keyword in ['earnings', 'merger', 'lawsuit', 'bankruptcy']):
            return 'High'
        elif high_score >= 1 or medium_score >= 2:
            return 'Medium'
        else:
            return 'Low'
    
    def analyze_sentiment(self, title, summary=""):
        """Analyze sentiment of news using VADER"""
        text = title + " " + summary
        
        # Get VADER sentiment scores
        scores = self.analyzer.polarity_scores(text)
        
        # Classify sentiment
        if scores['compound'] >= 0.05:
            sentiment = 'Positive'
            confidence = abs(scores['compound'])
        elif scores['compound'] <= -0.05:
            sentiment = 'Negative'
            confidence = abs(scores['compound'])
        else:
            sentiment = 'Neutral'
            confidence = 1 - abs(scores['compound'])
        
        return {
            'sentiment': sentiment,
            'confidence': confidence,
            'scores': scores
        }
    
    def generate_trading_signal(self, sentiment_data, materiality, topic):
        """Generate buy/hold/sell signal based on sentiment and context"""
        sentiment = sentiment_data['sentiment']
        confidence = sentiment_data['confidence']
        
        # Base signal on sentiment
        if sentiment == 'Positive':
            base_signal = 'BUY'
        elif sentiment == 'Negative':
            base_signal = 'SELL'
        else:
            base_signal = 'HOLD'
        
        # Adjust based on materiality
        if materiality == 'Low':
            # Low materiality news shouldn't drive major decisions
            if base_signal in ['BUY', 'SELL']:
                base_signal = 'HOLD'
        
        # Adjust based on topic
        if topic in ['earnings', 'guidance', 'merger']:
            # High impact topics - maintain signal strength
            pass
        elif topic in ['regulation', 'debt']:
            # Risk factors - be more cautious
            if base_signal == 'BUY':
                base_signal = 'HOLD'
        elif topic in ['operational', 'expansion']:
            # Medium impact - moderate signals
            if base_signal == 'SELL' and confidence < 0.7:
                base_signal = 'HOLD'
        
        # Generate rationale
        rationale = self._generate_rationale(sentiment, materiality, topic, confidence)
        
        return {
            'signal': base_signal,
            'confidence': confidence,
            'rationale': rationale
        }
    
    def _generate_rationale(self, sentiment, materiality, topic, confidence):
        """Generate rationale for the trading signal"""
        rationale_parts = []
        
        # Sentiment component
        if sentiment == 'Positive':
            rationale_parts.append(f"Positive sentiment detected (confidence: {confidence:.2f})")
        elif sentiment == 'Negative':
            rationale_parts.append(f"Negative sentiment detected (confidence: {confidence:.2f})")
        else:
            rationale_parts.append(f"Neutral sentiment (confidence: {confidence:.2f})")
        
        # Materiality component
        if materiality == 'High':
            rationale_parts.append("High materiality - significant impact expected")
        elif materiality == 'Medium':
            rationale_parts.append("Medium materiality - moderate impact expected")
        else:
            rationale_parts.append("Low materiality - limited impact expected")
        
        # Topic component
        topic_impact = {
            'earnings': 'Earnings-related news can drive significant price movements',
            'guidance': 'Management guidance affects future expectations',
            'dividend': 'Dividend news impacts income-focused investors',
            'management': 'Leadership changes can affect company direction',
            'regulation': 'Regulatory issues pose compliance risks',
            'merger': 'M&A activity can create value or uncertainty',
            'debt': 'Credit-related news affects financial stability',
            'expansion': 'Growth initiatives signal future potential',
            'macro': 'Macroeconomic factors affect sector-wide performance',
            'operational': 'Operational updates reflect business health'
        }
        
        if topic in topic_impact:
            rationale_parts.append(topic_impact[topic])
        
        return ". ".join(rationale_parts)
    
    def analyze_portfolio_news(self, portfolio_tickers, max_news_per_stock=10, days_back=14):
        """Analyze news for entire portfolio"""
        portfolio_news_analysis = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, ticker in enumerate(portfolio_tickers):
            status_text.text(f'Analyzing news for {ticker}... ({i+1}/{len(portfolio_tickers)})')
            progress_bar.progress((i + 1) / len(portfolio_tickers))
            
            # Fetch news with user-specified limits
            news_items = self.get_stock_news(ticker, max_news_per_stock, days_back)
            
            for item in news_items:
                # Classify and analyze each news item
                topic = self.classify_news_topic(item['title'], item.get('summary', ''))
                materiality = self.rate_materiality(item['title'], item.get('summary', ''), topic)
                sentiment_data = self.analyze_sentiment(item['title'], item.get('summary', ''))
                trading_signal = self.generate_trading_signal(sentiment_data, materiality, topic)
                
                # Convert timestamp with better handling
                try:
                    if item['providerPublishTime'] > 0:
                        publish_time = datetime.fromtimestamp(item['providerPublishTime'])
                        time_str = publish_time.strftime("%Y-%m-%d %H:%M")
                        # Calculate hours ago for recency
                        hours_ago = int((datetime.now() - publish_time).total_seconds() / 3600)
                        if hours_ago < 24:
                            time_display = f"{hours_ago}h ago"
                        else:
                            days_ago = hours_ago // 24
                            time_display = f"{days_ago}d ago"
                    else:
                        time_str = "Unknown"
                        time_display = "Unknown"
                except:
                    time_str = "Unknown"
                    time_display = "Unknown"
                
                # Enhanced materiality scoring for critical news highlighting
                materiality_score = self._calculate_materiality_score(item['title'], item.get('summary', ''), topic, sentiment_data)
                
                portfolio_news_analysis.append({
                    'Ticker': ticker,
                    'Title': item['title'][:100] + "..." if len(item['title']) > 100 else item['title'],
                    'Full_Title': item['title'],  # Keep full title for display
                    'Publisher': item.get('publisher', 'Unknown'),
                    'Source': item.get('source', 'unknown'),
                    'Time': time_str,
                    'Time_Display': time_display,
                    'Topic': topic.title(),
                    'Materiality': materiality,
                    'Materiality_Score': materiality_score,  # Numeric score for sorting
                    'Sentiment': sentiment_data['sentiment'],
                    'Signal': trading_signal['signal'],
                    'Confidence': f"{sentiment_data['confidence']:.2f}",
                    'Confidence_Numeric': sentiment_data['confidence'],
                    'Rationale': trading_signal['rationale'][:200] + "..." if len(trading_signal['rationale']) > 200 else trading_signal['rationale'],
                    'Link': item.get('link', ''),
                    'Is_Critical': materiality == 'High' and (trading_signal['signal'] in ['BUY', 'SELL'] or sentiment_data['confidence'] > 0.7)
                })
            
            # Small delay to be respectful to APIs
            time.sleep(0.2)
        
        progress_bar.empty()
        status_text.empty()
        
        return pd.DataFrame(portfolio_news_analysis)
    
    def _calculate_materiality_score(self, title, summary, topic, sentiment_data):
        """Calculate numeric materiality score for sorting"""
        text = (title + " " + summary).lower()
        score = 0
        
        # Base score from topic
        topic_scores = {
            'earnings': 10, 'guidance': 9, 'merger': 10, 'regulation': 8,
            'management': 7, 'dividend': 6, 'debt': 7, 'expansion': 5,
            'operational': 4, 'macro': 6, 'general': 2
        }
        score += topic_scores.get(topic, 2)
        
        # Boost for high-impact keywords
        high_impact_keywords = [
            'earnings', 'bankruptcy', 'merger', 'acquisition', 'lawsuit',
            'investigation', 'ceo', 'resignation', 'dividend', 'buyback',
            'guidance', 'outlook', 'approval', 'rejection'
        ]
        
        for keyword in high_impact_keywords:
            if keyword in text:
                score += 3
        
        # Sentiment confidence boost
        if sentiment_data['confidence'] > 0.7:
            score += 2
        
        # Signal strength boost
        if sentiment_data['sentiment'] in ['Positive', 'Negative']:
            score += 1
        
        return min(score, 20)  # Cap at 20
    
    def generate_news_summary(self, news_df):
        """Generate enhanced summary of news analysis"""
        if news_df.empty:
            return {
                'total_news': 0,
                'by_sentiment': {},
                'by_signal': {},
                'by_materiality': {},
                'by_topic': {},
                'high_priority_news': 0,
                'critical_news': 0,
                'recent_news_24h': 0,
                'avg_materiality_score': 0
            }
        
        # Calculate recent news (last 24 hours)
        recent_24h = 0
        if 'Time_Display' in news_df.columns:
            recent_24h = len(news_df[news_df['Time_Display'].str.contains('h ago', na=False)])
        
        # Calculate average materiality score
        avg_materiality = 0
        if 'Materiality_Score' in news_df.columns:
            avg_materiality = news_df['Materiality_Score'].mean()
        
        # Count critical news
        critical_count = 0
        if 'Is_Critical' in news_df.columns:
            critical_count = news_df['Is_Critical'].sum()
        
        summary = {
            'total_news': len(news_df),
            'by_sentiment': news_df['Sentiment'].value_counts().to_dict(),
            'by_signal': news_df['Signal'].value_counts().to_dict(),
            'by_materiality': news_df['Materiality'].value_counts().to_dict(),
            'by_topic': news_df['Topic'].value_counts().to_dict(),
            'high_priority_news': news_df[
                (news_df['Materiality'] == 'High') | 
                (news_df['Signal'].isin(['BUY', 'SELL']))
            ].shape[0],
            'critical_news': critical_count,
            'recent_news_24h': recent_24h,
            'avg_materiality_score': avg_materiality
        }
        
        return summary
