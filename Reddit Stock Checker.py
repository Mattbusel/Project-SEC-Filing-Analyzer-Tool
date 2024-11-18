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
from typing import Dict, List, Tuple


class RedditAuthError(Exception):
    
    pass

class MarketSentimentAnalyzer:
    def __init__(self, client_id: str, client_secret: str, username: str):
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)
        
        
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
        
        # Store RSS feed URLs
        self.news_feeds = {
            'yahoo_finance': 'https://finance.yahoo.com/news/rssindex',
            'marketwatch': 'http://feeds.marketwatch.com/marketwatch/topstories',
        }
        
       
        self.sentiment_data = defaultdict(list)

    def test_reddit_connection(self) -> bool:
        
        try:
            
            list(self.reddit.subreddit('stocks').hot(limit=5))
            return True
        except Exception as e:
            self.logger.error(f"Reddit connection test failed: {str(e)}")
            return False

    def fetch_reddit_sentiment(self, ticker: str, subreddits: List[str], 
                             timeframe: int = 7) -> Dict:
        """
        Fetch and analyze Reddit posts/comments for a given ticker.
        """
        sentiment_scores = []
        post_volumes = 0
        posts_analyzed = []
        
        for subreddit_name in subreddits:
            try:
                subreddit = self.reddit.subreddit(subreddit_name)
                
                # Search for posts containing the ticker
                for post in subreddit.search(f"{ticker}", time_filter='week', limit=100):
                    post_volumes += 1
                    
                    # Analyze post title and body
                    text = f"{post.title} {post.selftext}"
                    blob = TextBlob(text)
                    sentiment_score = blob.sentiment.polarity
                    sentiment_scores.append(sentiment_score)
                    
                    # Store post details
                    posts_analyzed.append({
                        'title': post.title,
                        'score': post.score,
                        'sentiment': sentiment_score,
                        'created_utc': datetime.fromtimestamp(post.created_utc),
                        'subreddit': subreddit_name
                    })
                    
                    # Get top comments
                    try:
                        post.comments.replace_more(limit=0)
                        for comment in post.comments.list()[:10]:
                            blob = TextBlob(comment.body)
                            sentiment_scores.append(blob.sentiment.polarity)
                    except Exception as e:
                        self.logger.warning(f"Error fetching comments: {str(e)}")
                        continue
            
            except Exception as e:
                self.logger.error(f"Error fetching from r/{subreddit_name}: {str(e)}")
                continue
        
        return {
            'avg_sentiment': np.mean(sentiment_scores) if sentiment_scores else 0,
            'post_volume': post_volumes,
            'sentiment_stddev': np.std(sentiment_scores) if sentiment_scores else 0,
            'posts': posts_analyzed
        }

    def fetch_news_sentiment(self, ticker: str) -> Dict:
        """
        Fetch and analyze news articles from RSS feeds.
        """
        news_data = []
        
        for source, feed_url in self.news_feeds.items():
            try:
                feed = feedparser.parse(feed_url)
                
                for entry in feed.entries:
                    if ticker.lower() in entry.title.lower() or ticker.lower() in entry.summary.lower():
                        blob = TextBlob(f"{entry.title} {entry.summary}")
                        news_data.append({
                            'source': source,
                            'date': entry.published,
                            'sentiment': blob.sentiment.polarity,
                            'title': entry.title,
                            'summary': entry.summary
                        })
            except Exception as e:
                self.logger.error(f"Error fetching news from {source}: {str(e)}")
                continue
        
        return {
            'article_count': len(news_data),
            'avg_sentiment': np.mean([d['sentiment'] for d in news_data]) if news_data else 0,
            'recent_articles': news_data[-5:] if news_data else [],
            'all_articles': news_data
        }

    def get_stock_metrics(self, ticker: str) -> Dict:
        """
        Fetch basic stock metrics using yfinance.
        """
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period="1mo")
            
            return {
                'volume': hist['Volume'].mean(),
                'volatility': hist['Close'].pct_change().std(),
                'price_change': (hist['Close'][-1] - hist['Close'][0]) / hist['Close'][0],
                'current_price': hist['Close'][-1],
                'volume_change': (hist['Volume'][-1] - hist['Volume'].mean()) / hist['Volume'].mean(),
                'avg_volume': hist['Volume'].mean()
            }
        except Exception as e:
            self.logger.error(f"Error fetching stock metrics for {ticker}: {str(e)}")
            return {}

    def analyze_company(self, ticker: str) -> Dict:
        """
        Perform comprehensive sentiment analysis for a company.
        """
        try:
            # Fetch data from all sources
            self.logger.info(f"Analyzing {ticker}...")
            
            reddit_data = self.fetch_reddit_sentiment(
                ticker, 
                ['wallstreetbets', 'stocks', 'investing']
            )
            self.logger.info(f"Reddit analysis complete for {ticker}")
            
            news_data = self.fetch_news_sentiment(ticker)
            self.logger.info(f"News analysis complete for {ticker}")
            
            stock_data = self.get_stock_metrics(ticker)
            self.logger.info(f"Stock metrics complete for {ticker}")
            
            # Combine all data sources
            analysis = {
                'ticker': ticker,
                'timestamp': datetime.now().isoformat(),
                'social_sentiment': {
                    'reddit': reddit_data
                },
                'news_sentiment': news_data,
                'market_metrics': stock_data,
                'composite_score': self._calculate_composite_score(
                    reddit_data, news_data, stock_data
                )
            }
            
            # Store historical data
            self.sentiment_data[ticker].append(analysis)
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error in analyze_company for {ticker}: {str(e)}")
            raise

    def _calculate_composite_score(self, reddit_data: Dict, 
                                 news_data: Dict, stock_data: Dict) -> float:
        """
        Calculate a composite sentiment score based on all available data.
        """
        scores = []
        weights = []
        
        # Reddit sentiment
        if reddit_data.get('avg_sentiment') is not None:
            scores.append(reddit_data['avg_sentiment'])
            weights.append(0.3)
        
        # News sentiment
        if news_data.get('avg_sentiment') is not None:
            scores.append(news_data['avg_sentiment'])
            weights.append(0.4)
        
        # Stock momentum
        if stock_data.get('price_change') is not None:
            scores.append(np.clip(stock_data['price_change'], -1, 1))
            weights.append(0.3)
        
        if not scores:
            return 0
            
        # Normalize weights
        weights = np.array(weights) / sum(weights)
        return float(np.dot(scores, weights))

    def get_historical_trends(self, ticker: str, days: int = 30) -> pd.DataFrame:
        """
        Get historical sentiment trends for a company.
        """
        data = self.sentiment_data.get(ticker, [])
        if not data:
            return pd.DataFrame()
        
        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df.set_index('timestamp').last(f"{days}D")

def main():
    # Your Reddit credentials
    CLIENT_ID = "
    CLIENT_SECRET = ""
    REDDIT_USERNAME = ""  # Replace with your Reddit username

    try:
        # Initialize analyzer with credentials
        analyzer = MarketSentimentAnalyzer(
            client_id=CLIENT_ID,
            client_secret=CLIENT_SECRET,
            username=REDDIT_USERNAME
        )

        # Test Reddit connection
        if analyzer.test_reddit_connection():
            print("Successfully connected to Reddit API!")
            
            # Test analysis
            print("\nTesting sentiment analysis for AAPL...")
            analysis = analyzer.analyze_company("AAPL")
            
            print(f"\nResults:")
            print(f"Composite Score: {analysis['composite_score']:.2f}")
            print(f"Reddit Post Volume: {analysis['social_sentiment']['reddit']['post_volume']}")
            print(f"Reddit Sentiment: {analysis['social_sentiment']['reddit']['avg_sentiment']:.2f}")
            
            if analysis['news_sentiment']['article_count'] > 0:
                print("\nRecent News Headlines:")
                for article in analysis['news_sentiment']['recent_articles']:
                    print(f"- {article['title']} (Sentiment: {article['sentiment']:.2f})")
            
            print("\nMarket Metrics:")
            for metric, value in analysis['market_metrics'].items():
                if isinstance(value, float):
                    print(f"- {metric}: {value:.2f}")
                else:
                    print(f"- {metric}: {value}")
            
        else:
            print("Failed to connect to Reddit API. Please check your credentials.")
            
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

if __name__ == "__main__":
    main()
    
