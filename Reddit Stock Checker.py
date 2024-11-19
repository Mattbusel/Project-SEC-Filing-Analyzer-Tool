import praw
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import feedparser
import requests
from textblob import TextBlob
from collections import defaultdict
import yfinance as yf
from bs4 import BeautifulSoup
import logging
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import json
from concurrent.futures import ThreadPoolExecutor
import asyncio
import aiohttp

# Custom exceptions
class RedditAuthError(Exception):
    """Raised when Reddit authentication fails."""
    pass

class DataFetchError(Exception):
    """Raised when data fetching from any source fails."""
    pass

@dataclass
class SentimentScore:
    """Data class to store sentiment analysis results."""
    value: float
    confidence: float
    source: str
    timestamp: datetime

class MarketSentimentAnalyzer:
    """
    A class to analyze market sentiment from multiple sources including Reddit,
    news feeds, and market data.
    """
    
    def __init__(self, client_id: str, client_secret: str, username: str):
        """
        Initialize the Market Sentiment Analyzer.
        
        Args:
            client_id: Reddit API client ID
            client_secret: Reddit API client secret
            username: Reddit username
            
        Raises:
            RedditAuthError: If Reddit authentication fails
        """
        # Set up logging with more detailed format
        self.logger = logging.getLogger(__name__)
        logging_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        logging.basicConfig(level=logging.INFO, format=logging_format)
        
        # Initialize Reddit client
        user_agent = f"python:sec_sentiment_analyzer:v1.0 (by /u/{username})"
        try:
            self.reddit = praw.Reddit(
                client_id=client_id,
                client_secret=client_secret,
                user_agent=user_agent
            )
            self.reddit.user.me()
            self.logger.info("Successfully authenticated with Reddit API")
        except Exception as e:
            raise RedditAuthError(f"Failed to authenticate with Reddit: {str(e)}")
        
        # Enhanced news sources
        self.news_feeds = {
            'yahoo_finance': 'https://finance.yahoo.com/news/rssindex',
            'marketwatch': 'http://feeds.marketwatch.com/marketwatch/topstories',
            'reuters': 'https://www.reutersagency.com/feed/',
            'seeking_alpha': 'https://seekingalpha.com/market_currents.xml'
        }
        
        # Initialize sentiment data storage with TTL cache
        self.sentiment_cache = {}
        self.cache_ttl = timedelta(hours=1)
        
        # Initialize async session
        self.session = None

    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
            
    def test_reddit_connection(self) -> bool:
        """Test the Reddit API connection."""
        try:
            list(self.reddit.subreddit('stocks').hot(limit=1))
            return True
        except Exception as e:
            self.logger.error(f"Reddit connection test failed: {str(e)}")
            return False

    async def fetch_reddit_sentiment_async(self, ticker: str, subreddits: List[str], 
                                         timeframe: int = 7) -> Dict:
        """
        Asynchronously fetch and analyze Reddit posts/comments for a given ticker.
        """
        sentiment_scores = []
        post_volumes = 0
        posts_analyzed = []
        
        async def process_post(post):
            nonlocal post_volumes
            post_volumes += 1
            
            # Analyze post title and body
            text = f"{post.title} {post.selftext}"
            sentiment = await self._analyze_text_async(text)
            
            # Store post details
            post_data = {
                'title': post.title,
                'score': post.score,
                'sentiment': sentiment.value,
                'confidence': sentiment.confidence,
                'created_utc': datetime.fromtimestamp(post.created_utc),
                'subreddit': post.subreddit.display_name
            }
            
            sentiment_scores.append(sentiment.value)
            posts_analyzed.append(post_data)
            
            # Process comments asynchronously
            try:
                post.comments.replace_more(limit=0)
                comment_tasks = []
                for comment in post.comments.list()[:10]:
                    comment_tasks.append(self._analyze_text_async(comment.body))
                comment_sentiments = await asyncio.gather(*comment_tasks)
                sentiment_scores.extend([s.value for s in comment_sentiments])
            except Exception as e:
                self.logger.warning(f"Error processing comments: {str(e)}")

        tasks = []
        for subreddit_name in subreddits:
            try:
                subreddit = self.reddit.subreddit(subreddit_name)
                posts = subreddit.search(f"{ticker}", time_filter='week', limit=100)
                tasks.extend([process_post(post) for post in posts])
            except Exception as e:
                self.logger.error(f"Error fetching from r/{subreddit_name}: {str(e)}")
                continue

        await asyncio.gather(*tasks)
        
        return {
            'avg_sentiment': np.mean(sentiment_scores) if sentiment_scores else 0,
            'sentiment_confidence': np.mean([p.get('confidence', 0) for p in posts_analyzed]),
            'post_volume': post_volumes,
            'sentiment_stddev': np.std(sentiment_scores) if sentiment_scores else 0,
            'posts': posts_analyzed,
            'timestamp': datetime.now()
        }

    async def _analyze_text_async(self, text: str) -> SentimentScore:
        """
        Asynchronously analyze text sentiment using TextBlob.
        """
        blob = TextBlob(text)
        return SentimentScore(
            value=blob.sentiment.polarity,
            confidence=abs(blob.sentiment.subjectivity),
            source="textblob",
            timestamp=datetime.now()
        )

    async def fetch_news_sentiment_async(self, ticker: str) -> Dict:
        """
        Asynchronously fetch and analyze news articles from RSS feeds.
        """
        news_data = []
        
        async def process_feed(source: str, feed_url: str):
            try:
                async with self.session.get(feed_url) as response:
                    feed_content = await response.text()
                    feed = feedparser.parse(feed_content)
                    
                    for entry in feed.entries:
                        # Get title and description/summary safely
                        title = entry.get('title', '')
                        description = entry.get('description', '')
                        if not description:
                            description = entry.get('summary', '')
                            
                        # Check if article is relevant to the ticker
                        if ticker.lower() in title.lower() or ticker.lower() in description.lower():
                            sentiment = await self._analyze_text_async(f"{title} {description}")
                            news_data.append({
                                'source': source,
                                'date': entry.get('published', datetime.now().isoformat()),
                                'sentiment': sentiment.value,
                                'confidence': sentiment.confidence,
                                'title': title,
                                'summary': description[:200] + '...' if len(description) > 200 else description
                            })
            except Exception as e:
                self.logger.error(f"Error fetching news from {source}: {str(e)}")
                
        # Update news feed URLs to more reliable sources
        self.news_feeds = {
            'marketwatch': 'http://feeds.marketwatch.com/marketwatch/topstories',
            'investing': 'https://www.investing.com/rss/news.rss',
            'benzinga': 'https://www.benzinga.com/feed',
            'finviz': 'https://finviz.com/news.ashx'
        }

        tasks = [process_feed(source, url) 
                for source, url in self.news_feeds.items()]
        await asyncio.gather(*tasks)
        
        return {
            'article_count': len(news_data),
            'avg_sentiment': np.mean([d['sentiment'] for d in news_data]) if news_data else 0,
            'avg_confidence': np.mean([d['confidence'] for d in news_data]) if news_data else 0,
            'recent_articles': sorted(news_data, key=lambda x: x['date'])[-5:] if news_data else [],
            'all_articles': news_data,
            'timestamp': datetime.now()
        }
        
    async def get_stock_metrics_async(self, ticker: str) -> Dict:
        """
        Asynchronously fetch basic stock metrics using yfinance.
        """
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period="1mo")
            
            metrics = {
                'volume': hist['Volume'].mean(),
                'volatility': hist['Close'].pct_change().std(),
                'price_change': (hist['Close'][-1] - hist['Close'][0]) / hist['Close'][0],
                'current_price': hist['Close'][-1],
                'volume_change': (hist['Volume'][-1] - hist['Volume'].mean()) / hist['Volume'].mean(),
                'avg_volume': hist['Volume'].mean(),
                'rsi': self._calculate_rsi(hist['Close']),
                'timestamp': datetime.now()
            }
            
            # Add technical indicators
            metrics.update(self._calculate_technical_indicators(hist))
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error fetching stock metrics for {ticker}: {str(e)}")
            return {}

    def _calculate_rsi(self, prices: pd.Series, periods: int = 14) -> float:
        """Calculate the Relative Strength Index."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
        
        rs = gain / loss
        return 100 - (100 / (1 + rs)).iloc[-1]

    def _calculate_technical_indicators(self, hist: pd.DataFrame) -> Dict:
        """Calculate additional technical indicators."""
        close = hist['Close']
        
        # Calculate moving averages
        ma20 = close.rolling(window=20).mean().iloc[-1]
        ma50 = close.rolling(window=50).mean().iloc[-1]
        
        # Calculate MACD
        exp1 = close.ewm(span=12, adjust=False).mean()
        exp2 = close.ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        
        return {
            'ma20': ma20,
            'ma50': ma50,
            'macd': macd.iloc[-1],
            'macd_signal': signal.iloc[-1],
            'macd_histogram': (macd - signal).iloc[-1]
        }

    async def analyze_company_async(self, ticker: str) -> Dict:
        """
        Perform comprehensive asynchronous sentiment analysis for a company.
        """
        try:
            self.logger.info(f"Starting analysis for {ticker}...")
            
            # Fetch all data concurrently
            reddit_task = self.fetch_reddit_sentiment_async(
                ticker, 
                ['wallstreetbets', 'stocks', 'investing', 'SecurityAnalysis']
            )
            news_task = self.fetch_news_sentiment_async(ticker)
            stock_task = self.get_stock_metrics_async(ticker)
            
            reddit_data, news_data, stock_data = await asyncio.gather(
                reddit_task, news_task, stock_task
            )
            
            # Debug prints
            self.logger.info(f"Reddit sentiment: {reddit_data.get('avg_sentiment')}")
            self.logger.info(f"News sentiment: {news_data.get('avg_sentiment')}")
            if stock_data:
                self.logger.info(f"Technical score: {self._calculate_technical_score(stock_data)}")
            
            # Calculate composite score
            composite_score = await self._calculate_composite_score_async(
                reddit_data, news_data, stock_data
            )
            self.logger.info(f"Calculated composite score: {composite_score}")
            
            # Combine all data sources
            analysis = {
                'ticker': ticker,
                'timestamp': datetime.now(),
                'social_sentiment': {
                    'reddit': reddit_data
                },
                'news_sentiment': news_data,
                'market_metrics': stock_data,
                'composite_score': composite_score,
                'analysis_summary': self._generate_analysis_summary(
                    reddit_data, news_data, stock_data
                )
            }
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error in analyze_company for {ticker}: {str(e)}")
            raise

    async def _calculate_composite_score_async(self, reddit_data: Dict, 
                                             news_data: Dict, stock_data: Dict) -> float:
        """
        Calculate a weighted composite sentiment score based on all available data.
        """
        scores = []
        weights = []
        confidences = []
        
        # Reddit sentiment (30% weight)
        if reddit_data.get('avg_sentiment') is not None:
            reddit_sentiment = float(reddit_data['avg_sentiment'])
            if not np.isnan(reddit_sentiment):
                scores.append(reddit_sentiment)
                weights.append(0.3)
                confidences.append(reddit_data.get('sentiment_confidence', 0.5))
        
        # News sentiment (40% weight)
        if news_data.get('avg_sentiment') is not None:
            news_sentiment = float(news_data['avg_sentiment'])
            if not np.isnan(news_sentiment):
                scores.append(news_sentiment)
                weights.append(0.4)
                confidences.append(news_data.get('avg_confidence', 0.5))
        
        # Technical indicators (30% weight)
        if stock_data:
            technical_score = self._calculate_technical_score(stock_data)
            if not np.isnan(technical_score):
                scores.append(technical_score)
                weights.append(0.3)
                confidences.append(0.8)  # Higher confidence in technical data
        
        if not scores:
            return 0.0
            
        # Weight the scores by both their predetermined weights and confidence
        weights = np.array(weights) * np.array(confidences)
        total_weight = np.sum(weights)
        
        if total_weight == 0:
            return 0.0
            
        # Normalize weights
        weights = weights / total_weight
        
        # Calculate weighted average and ensure it's not NaN
        composite_score = float(np.dot(scores, weights))
        return composite_score if not np.isnan(composite_score) else 0.0

    def _calculate_technical_score(self, stock_data: Dict) -> float:
        """Calculate a technical analysis score from -1 to 1."""
        try:
            signals = []
            
            # Price momentum
            if 'price_change' in stock_data:
                price_signal = np.clip(float(stock_data['price_change']), -1, 1)
                if not np.isnan(price_signal):
                    signals.append(price_signal)
            
            # RSI signals
            if 'rsi' in stock_data and not np.isnan(stock_data['rsi']):
                rsi = float(stock_data['rsi'])
                if rsi < 30:
                    signals.append(1)  # Oversold
                elif rsi > 70:
                    signals.append(-1)  # Overbought
                else:
                    signals.append(0)
            
            # MACD signals
            if all(k in stock_data and not np.isnan(stock_data[k]) 
                  for k in ['macd', 'macd_signal']):
                macd_diff = stock_data['macd'] - stock_data['macd_signal']
                if abs(stock_data['macd_signal']) > 0:  # Avoid division by zero
                    macd_signal = np.clip(macd_diff / abs(stock_data['macd_signal']), -1, 1)
                    signals.append(float(macd_signal))
            
            # Moving average signals
            if all(k in stock_data and not np.isnan(stock_data[k]) 
                  for k in ['ma20', 'ma50', 'current_price']):
                if stock_data['current_price'] > 0:  # Avoid division by zero
                    ma_diff = (stock_data['ma20'] - stock_data['ma50']) / stock_data['current_price']
                    signals.append(np.clip(float(ma_diff * 10), -1, 1))
            
            if not signals:
                return 0.0
                
            return float(np.mean(signals))
            
        except Exception as e:
            self.logger.error(f"Error calculating technical score: {str(e)}")
            return 0.0

    async def analyze_company_async(self, ticker: str) -> Dict:
        """
        Perform comprehensive asynchronous sentiment analysis for a company.
        """
        try:
            self.logger.info(f"Starting analysis for {ticker}...")
            
            # Fetch all data concurrently
            reddit_task = self.fetch_reddit_sentiment_async(
                ticker, 
                ['wallstreetbets', 'stocks', 'investing', 'SecurityAnalysis']
            )
            news_task = self.fetch_news_sentiment_async(ticker)
            stock_task = self.get_stock_metrics_async(ticker)
            
            reddit_data, news_data, stock_data = await asyncio.gather(
                reddit_task, news_task, stock_task
            )
            
            # Calculate volume metrics
            volume_metrics = self._analyze_volume_trends(stock_data)
            
            # Add market context
            market_context = await self._get_market_context(ticker)
            
            # Calculate composite score with all components
            composite_score = await self._calculate_composite_score_async(
                reddit_data, news_data, stock_data
            )
            
            analysis = {
                'ticker': ticker,
                'timestamp': datetime.now(),
                'social_sentiment': {
                    'reddit': reddit_data
                },
                'news_sentiment': news_data,
                'market_metrics': {
                    **stock_data,
                    'volume_analysis': volume_metrics,
                    'market_context': market_context
                },
                'composite_score': composite_score,
                'analysis_summary': self._generate_analysis_summary(
                    reddit_data, news_data, stock_data
                ),
                'score_components': {
                    'social_score': reddit_data.get('avg_sentiment', 0),
                    'news_score': news_data.get('avg_sentiment', 0),
                    'technical_score': self._calculate_technical_score(stock_data)
                }
            }
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error in analyze_company for {ticker}: {str(e)}")
            raise
            
    def _analyze_volume_trends(self, stock_data: Dict) -> Dict:
        """Analyze trading volume trends."""
        if not stock_data or 'volume' not in stock_data:
            return {}
            
        avg_volume = stock_data['avg_volume']
        current_volume = stock_data['volume']
        
        return {
            'volume_ratio': current_volume / avg_volume if avg_volume > 0 else 1.0,
            'volume_trend': 'increasing' if current_volume > avg_volume * 1.1 else
                          'decreasing' if current_volume < avg_volume * 0.9 else
                          'stable',
            'significant_volume': current_volume > avg_volume * 1.5
        }
    def _generate_analysis_summary(self, reddit_data: Dict, 
                                 news_data: Dict, stock_data: Dict) -> str:
        """
        Generate a human-readable summary of the analysis.
        """
        summary_parts = []
        
        # Social sentiment summary with more detail
        if reddit_data.get('post_volume', 0) > 0:
            sentiment = "positive" if reddit_data['avg_sentiment'] > 0 else "negative"
            strength = "strong" if abs(reddit_data['avg_sentiment']) > 0.5 else "moderate"
            confidence = reddit_data.get('sentiment_confidence', 0)
            summary_parts.append(
                f"Social media sentiment is {strength}ly {sentiment} "
                f"(score: {reddit_data['avg_sentiment']:.2f}, confidence: {confidence:.2f}) "
                f"based on {reddit_data['post_volume']} relevant posts."
            )
            
            if 'sentiment_stddev' in reddit_data:
                summary_parts.append(
                    f"Social sentiment volatility: {reddit_data['sentiment_stddev']:.2f}"
                )

        # News sentiment summary
        if news_data.get('article_count', 0) > 0:
            sentiment = "positive" if news_data['avg_sentiment'] > 0 else "negative"
            strength = "strong" if abs(news_data['avg_sentiment']) > 0.5 else "moderate"
            summary_parts.append(
                f"News sentiment is {strength}ly {sentiment} "
                f"(score: {news_data['avg_sentiment']:.2f}) based on "
                f"{news_data['article_count']} recent articles."
            )

        # Technical analysis summary with more metrics
        if stock_data:
            summary_parts.append("\nPrice Analysis:")
            price_change = stock_data.get('price_change', 0) * 100
            summary_parts.append(
                f"• Stock has moved {abs(price_change):.1f}% "
                f"{'up' if price_change > 0 else 'down'} over the past month"
            )

            # Technical indicators
            summary_parts.append("\nTechnical Indicators:")
            
            if 'rsi' in stock_data:
                rsi = stock_data['rsi']
                rsi_status = (
                    "oversold" if rsi < 30 else 
                    "overbought" if rsi > 70 else 
                    "neutral"
                )
                summary_parts.append(f"• RSI: {rsi:.1f} ({rsi_status})")

            if all(k in stock_data for k in ['macd', 'macd_signal']):
                macd_diff = stock_data['macd'] - stock_data['macd_signal']
                momentum = "bullish" if macd_diff > 0 else "bearish"
                summary_parts.append(f"• MACD shows {momentum} momentum")

            if all(k in stock_data for k in ['ma20', 'ma50']):
                trend = (
                    "upward" if stock_data['ma20'] > stock_data['ma50']
                    else "downward"
                )
                summary_parts.append(f"• Moving averages indicate {trend} trend")
                
        # Add composite score interpretation
        technical_score = self._calculate_technical_score(stock_data)
        sentiment_str = (
            "strongly bullish" if technical_score > 0.5 else
            "moderately bullish" if technical_score > 0.1 else
            "neutral" if -0.1 <= technical_score <= 0.1 else
            "moderately bearish" if technical_score > -0.5 else
            "strongly bearish"
        )
        summary_parts.append(
            f"\nOverall Market Sentiment: {sentiment_str} "
            f"(Score: {technical_score:.2f})"
        )

        # Combine all parts
        if summary_parts:
            return "\n".join(summary_parts)
        else:
            return "Insufficient data for meaningful analysis."
     
    async def _get_market_context(self, ticker: str) -> Dict:
        """Get broader market context."""
        try:
            # Get S&P 500 data as market reference
            spy = yf.Ticker("SPY")
            spy_hist = spy.history(period="1mo")
            
            return {
                'market_trend': (spy_hist['Close'][-1] - spy_hist['Close'][0]) / spy_hist['Close'][0],
                'market_volatility': spy_hist['Close'].pct_change().std(),
                'relative_strength': stock_data.get('price_change', 0) - 
                                   ((spy_hist['Close'][-1] - spy_hist['Close'][0]) / spy_hist['Close'][0])
            }
        except Exception as e:
            self.logger.error(f"Error getting market context: {str(e)}")
            return {}

def main():
    """
    Main function to demonstrate the usage of the MarketSentimentAnalyzer class.
    """
    # Your Reddit credentials
    CLIENT_ID = ""
    CLIENT_SECRET = ""
    REDDIT_USERNAME = ""

    async def run_analysis():
        try:
            async with MarketSentimentAnalyzer(
                client_id=CLIENT_ID,
                client_secret=CLIENT_SECRET,
                username=REDDIT_USERNAME
            ) as analyzer:
                # Test the analyzer with a sample ticker
                analysis = await analyzer.analyze_company_async("AAPL")
                
                print("\nAnalysis Results:")
                print("-" * 50)
                print(f"Ticker: {analysis['ticker']}")
                print(f"\nComposite Score: {analysis['composite_score']:.2f}")
                print("\nDetailed Analysis:")
                print(analysis['analysis_summary'])
                
                if analysis['news_sentiment']['recent_articles']:
                    print("\nRecent News Headlines:")
                    for article in analysis['news_sentiment']['recent_articles']:
                        print(f"- {article['title']} (Sentiment: {article['sentiment']:.2f})")

        except RedditAuthError as e:
            print(f"Authentication Error: {str(e)}")
            print("\nPlease ensure you have:")
            print("1. Correct client_id")
            print("2. Correct client_secret")
            print("3. Your actual Reddit username")
            print("\nYou can get these credentials from: https://www.reddit.com/prefs/apps")
            
        except Exception as e:
            print(f"Unexpected error: {str(e)}")
            logging.exception("Detailed error information:")

    # Run the async analysis
    asyncio.run(run_analysis())

if __name__ == "__main__":
    main()
