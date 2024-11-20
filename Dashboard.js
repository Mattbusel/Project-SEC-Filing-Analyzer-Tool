import React, { useState } from 'react';
import './AnalysisDashboard.css';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, LineChart, Line } from 'recharts';

const AnalysisDashboard = () => {
  const [ticker, setTicker] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [analysis, setAnalysis] = useState(null);

  const handleAnalyze = async () => {
    if (!ticker) return;
    setLoading(true);
    setError('');
    
    try {
      const response = await fetch(`http://localhost:5000/api/analyze/${ticker}`);
      const data = await response.json();
      
      if (!response.ok) {
        throw new Error(data.error || 'Analysis failed');
      }
      
      setAnalysis(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const getSentimentColor = (score) => {
    if (score > 0.5) return '#22c55e';  // Strong positive
    if (score > 0) return '#86efac';    // Positive
    if (score < -0.5) return '#ef4444'; // Strong negative
    if (score < 0) return '#fca5a5';    // Negative
    return '#94a3b8';                   // Neutral
  };

  const renderSentimentChart = () => {
    if (!analysis) return null;

    const data = [
      {
        name: 'Social',
        score: analysis.social_sentiment.reddit.avg_sentiment,
        confidence: analysis.social_sentiment.reddit.sentiment_confidence
      },
      {
        name: 'Technical',
        score: analysis.composite_score,
        confidence: 0.8
      }
    ];

    return (
      <div className="chart-container">
        <h3>Sentiment Analysis</h3>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={data}>
            <XAxis dataKey="name" />
            <YAxis domain={[-1, 1]} />
            <Tooltip />
            <Bar dataKey="score" fill="#3b82f6" name="Sentiment Score" />
            <Bar dataKey="confidence" fill="#93c5fd" name="Confidence" />
          </BarChart>
        </ResponsiveContainer>
      </div>
    );
  };

  return (
    <div className="dashboard-container">
      <div className="header">
        <h1>Market Sentiment Analyzer</h1>
        <p className="subtitle">Analyze market sentiment from social media, news, and technical indicators</p>
      </div>

      <div className="search-container">
        <input
          type="text"
          value={ticker}
          onChange={(e) => setTicker(e.target.value.toUpperCase())}
          placeholder="Enter stock ticker (e.g., AAPL)"
          className="ticker-input"
        />
        <button 
          onClick={handleAnalyze} 
          className="analyze-button"
          disabled={loading || !ticker}
        >
          {loading ? 'Analyzing...' : 'Analyze'}
        </button>
      </div>

      {error && <div className="error-message">{error}</div>}

      <div className="info-section">
        <div className="info-card">
          <h3>Social Sentiment</h3>
          <p>Analyze Reddit discussions and social media trends</p>
        </div>
        <div className="info-card">
          <h3>News Analysis</h3>
          <p>Process recent news and media coverage</p>
        </div>
        <div className="info-card">
          <h3>Technical Indicators</h3>
          <p>Evaluate market trends and technical signals</p>
        </div>
      </div>

      {analysis && (
        <div className="results-container">
          <h2>Analysis Results for {ticker}</h2>
          
          {renderSentimentChart()}

          <div className="metrics-grid">
            <div className="metric-card">
              <h4>Composite Score</h4>
              <div className="metric-value" style={{ color: getSentimentColor(analysis.composite_score) }}>
                {analysis.composite_score.toFixed(2)}
              </div>
            </div>
            
            <div className="metric-card">
              <h4>Social Sentiment</h4>
              <div className="metric-value" style={{ color: getSentimentColor(analysis.social_sentiment.reddit.avg_sentiment) }}>
                {analysis.social_sentiment.reddit.avg_sentiment.toFixed(2)}
              </div>
              <div className="metric-subtitle">Confidence: {(analysis.social_sentiment.reddit.sentiment_confidence * 100).toFixed(1)}%</div>
            </div>
            
            <div className="metric-card">
              <h4>Post Volume</h4>
              <div className="metric-value">
                {analysis.social_sentiment.reddit.post_volume}
              </div>
              <div className="metric-subtitle">Reddit Posts Analyzed</div>
            </div>
          </div>

          <div className="analysis-details">
            <h3>Detailed Analysis</h3>
            <div className="detail-section">
              <div className="detail-item">
                <h4>Social Media Sentiment</h4>
                <p>{`Social media sentiment is ${analysis.social_sentiment.reddit.avg_sentiment > 0 ? 'positive' : 'negative'} with a score of ${analysis.social_sentiment.reddit.avg_sentiment.toFixed(2)}`}</p>
                <p>Volatility: {analysis.social_sentiment.reddit.sentiment_stddev.toFixed(2)}</p>
              </div>
              
              <div className="detail-item">
                <h4>Technical Analysis</h4>
                <ul>
                  <li>RSI: {analysis.market_metrics.rsi ? analysis.market_metrics.rsi.toFixed(1) : 'N/A'}</li>
                  <li>Price Change: {(analysis.market_metrics.price_change * 100).toFixed(1)}%</li>
                  <li>Volume Change: {(analysis.market_metrics.volume_change * 100).toFixed(1)}%</li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default AnalysisDashboard;
