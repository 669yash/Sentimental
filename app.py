from flask import Flask, render_template, request, jsonify
import yfinance as yf
import feedparser
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
import numpy as np
import base64
import io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# Company sectors for recommendations
COMPANY_SECTORS = {
    'AAPL': {'sector': 'Technology', 'companies': ['MSFT', 'GOOGL', 'META', 'NVDA']},
    'MSFT': {'sector': 'Technology', 'companies': ['AAPL', 'GOOGL', 'META', 'NVDA']},
    'GOOGL': {'sector': 'Technology', 'companies': ['AAPL', 'MSFT', 'META', 'NVDA']},
    'META': {'sector': 'Technology', 'companies': ['AAPL', 'MSFT', 'GOOGL', 'NVDA']},
    'NVDA': {'sector': 'Technology', 'companies': ['AAPL', 'MSFT', 'GOOGL', 'META']},
    'TSLA': {'sector': 'Automotive', 'companies': ['F', 'GM', 'NIO', 'RIVN']},
    'F': {'sector': 'Automotive', 'companies': ['TSLA', 'GM', 'NIO', 'RIVN']},
    'GM': {'sector': 'Automotive', 'companies': ['TSLA', 'F', 'NIO', 'RIVN']},
    'AMZN': {'sector': 'E-commerce', 'companies': ['SHOP', 'EBAY', 'WMT', 'TGT']},
    'WMT': {'sector': 'Retail', 'companies': ['AMZN', 'TGT', 'COST', 'HD']},
    'JPM': {'sector': 'Banking', 'companies': ['BAC', 'WFC', 'C', 'GS']},
    'BAC': {'sector': 'Banking', 'companies': ['JPM', 'WFC', 'C', 'GS']},
    'JNJ': {'sector': 'Healthcare', 'companies': ['PFE', 'UNH', 'ABT', 'MRK']},
    'PFE': {'sector': 'Healthcare', 'companies': ['JNJ', 'UNH', 'ABT', 'MRK']}
}

def fetch_news_sentiment(ticker):
    """Fetch news and calculate sentiment scores"""
    try:
        rss_url = f"https://news.google.com/rss/search?q={ticker}&hl=en-US&gl=US&ceid=US:en"
        feed = feedparser.parse(rss_url)
        
        news_items = []
        analyzer = SentimentIntensityAnalyzer()
        
        for entry in feed.entries:
            try:
                published = entry.published if 'published' in entry else entry.updated
                published_date = datetime.strptime(published, '%a, %d %b %Y %H:%M:%S %Z').date()
                
                sentiment_score = analyzer.polarity_scores(entry.title)['compound']
                
                news_items.append({
                    'Date': published_date,
                    'Headline': entry.title,
                    'Sentiment': sentiment_score
                })
            except:
                continue
        
        if not news_items:
            return pd.DataFrame(), []
            
        news_df = pd.DataFrame(news_items)
        
        sentiment_daily = news_df.groupby('Date').agg(
            AvgSentiment=('Sentiment', 'mean'),
            NewsCount=('Sentiment', 'count')
        ).reset_index()
        
        news_list = []
        for _, row in news_df.iterrows():
            news_list.append({
                'date': row['Date'].isoformat(),
                'headline': row['Headline'],
                'sentiment': round(row['Sentiment'], 3)
            })
        
        return sentiment_daily, news_list
    except Exception as e:
        print(f"Error fetching news: {e}")
        return pd.DataFrame(), []

def fetch_stock_data(ticker, period="1mo"):
    """Fetch stock price data"""
    try:
        stock_data = yf.download(ticker, period=period, interval="1d", progress=False)
        
        if stock_data.empty:
            return pd.DataFrame()
        
        if isinstance(stock_data.columns, pd.MultiIndex):
            stock_data.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in stock_data.columns.values]
        
        stock_data.reset_index(inplace=True)
        
        if 'Date' not in stock_data.columns:
            stock_data['Date'] = stock_data.index
        
        stock_data['Date'] = pd.to_datetime(stock_data['Date']).dt.date
        
        close_col = None
        for col in stock_data.columns:
            if 'Close' in col:
                close_col = col
                break
        
        if close_col is None:
            return pd.DataFrame()
        
        stock_price_df = stock_data[['Date', close_col]].copy()
        stock_price_df.rename(columns={close_col: 'Close'}, inplace=True)
        
        return stock_price_df
    except Exception as e:
        print(f"Error fetching stock data: {e}")
        return pd.DataFrame()

def create_charts(merged_df, ticker):
    """Create charts and return base64 encoded images"""
    charts = {}
    plt.style.use('default')
    
    try:
        # Chart 1: Price vs Sentiment
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        line1 = ax1.plot(merged_df['Date'], merged_df['Close'], label='Close Price', 
                        color='#2563eb', linewidth=3, marker='o', markersize=4)
        ax1.set_xlabel('Date', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Price ($)', color='#2563eb', fontsize=12, fontweight='bold')
        ax1.tick_params(axis='y', labelcolor='#2563eb')
        
        ax2 = ax1.twinx()
        line2 = ax2.plot(merged_df['Date'], merged_df['AvgSentiment'], label='Avg Sentiment', 
                        color='#dc2626', linewidth=3, marker='s', markersize=4)
        ax2.set_ylabel('Sentiment Score', color='#dc2626', fontsize=12, fontweight='bold')
        ax2.tick_params(axis='y', labelcolor='#dc2626')
        
        plt.title(f'{ticker} Stock Price vs News Sentiment', fontsize=16, fontweight='bold', pad=20)
        
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper left', frameon=True, shadow=True)
        
        ax1.grid(True, alpha=0.3)
        plt.tight_layout()
        
        img = io.BytesIO()
        plt.savefig(img, format='png', dpi=100, bbox_inches='tight', facecolor='white')
        img.seek(0)
        charts['price_sentiment'] = base64.b64encode(img.getvalue()).decode()
        plt.close()
        
        # Chart 2: Sentiment Bar Chart
        fig, ax = plt.subplots(figsize=(12, 6))
        colors = ['#dc2626' if x < 0 else '#16a34a' if x > 0 else '#6b7280' for x in merged_df['AvgSentiment']]
        bars = ax.bar(merged_df['Date'], merged_df['AvgSentiment'], color=colors, alpha=0.8, edgecolor='white', linewidth=1)
        
        ax.set_xlabel('Date', fontsize=12, fontweight='bold')
        ax.set_ylabel('Average Sentiment', fontsize=12, fontweight='bold')
        ax.set_title('Daily Average Sentiment Analysis', fontsize=16, fontweight='bold', pad=20)
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=2)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        img = io.BytesIO()
        plt.savefig(img, format='png', dpi=100, bbox_inches='tight', facecolor='white')
        img.seek(0)
        charts['sentiment_bar'] = base64.b64encode(img.getvalue()).decode()
        plt.close()
        
        # Chart 3: Moving Averages
        merged_df['Sentiment_MA7'] = merged_df['AvgSentiment'].rolling(window=7, min_periods=1).mean()
        merged_df['Close_MA7'] = merged_df['Close'].rolling(window=7, min_periods=1).mean()
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(merged_df['Date'], merged_df['Close_MA7'], label='7-Day MA Price', 
               color='#2563eb', linewidth=3, marker='o', markersize=3)
        ax2 = ax.twinx()
        ax2.plot(merged_df['Date'], merged_df['Sentiment_MA7'], label='7-Day MA Sentiment', 
                color='#dc2626', linewidth=3, marker='s', markersize=3)
        
        ax.set_xlabel('Date', fontsize=12, fontweight='bold')
        ax.set_ylabel('Price ($)', color='#2563eb', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Sentiment', color='#dc2626', fontsize=12, fontweight='bold')
        ax.set_title('7-Day Moving Averages: Price and Sentiment', fontsize=16, fontweight='bold', pad=20)
        
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left', frameon=True, shadow=True)
        
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        img = io.BytesIO()
        plt.savefig(img, format='png', dpi=100, bbox_inches='tight', facecolor='white')
        img.seek(0)
        charts['moving_avg'] = base64.b64encode(img.getvalue()).decode()
        plt.close()
        
        # Chart 4: Price Prediction
        df_for_model = merged_df.dropna(subset=['Close', 'AvgSentiment']).copy()
        
        if len(df_for_model) > 1:
            X = df_for_model['AvgSentiment'].values.reshape(-1, 1)
            y = df_for_model['Close'].values
            
            model = LinearRegression()
            model.fit(X, y)
            
            df_for_model['Predicted_Close'] = model.predict(X)
            
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(df_for_model['Date'], df_for_model['Close'], label='Actual Close Price', 
                   color='#2563eb', linewidth=3, marker='o', markersize=4)
            ax.plot(df_for_model['Date'], df_for_model['Predicted_Close'], 
                   label='Predicted from Sentiment', color='#dc2626', linestyle='--', linewidth=3, marker='s', markersize=4)
            
            ax.set_xlabel('Date', fontsize=12, fontweight='bold')
            ax.set_ylabel('Price ($)', fontsize=12, fontweight='bold')
            ax.set_title('Stock Price Prediction from Sentiment Analysis', fontsize=16, fontweight='bold', pad=20)
            ax.legend(frameon=True, shadow=True)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            
            img = io.BytesIO()
            plt.savefig(img, format='png', dpi=100, bbox_inches='tight', facecolor='white')
            img.seek(0)
            charts['prediction'] = base64.b64encode(img.getvalue()).decode()
            plt.close()
        
    except Exception as e:
        print(f"Error creating charts: {e}")
    
    return charts

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        ticker = request.json.get('ticker', '').upper().strip()
        
        if not ticker:
            return jsonify({'error': 'Please provide a valid ticker symbol'}), 400
        
        # Fetch data
        sentiment_daily, news_list = fetch_news_sentiment(ticker)
        stock_df = fetch_stock_data(ticker)
        
        if stock_df.empty:
            return jsonify({'error': f'Could not fetch stock data for {ticker}. Please check the ticker symbol.'}), 400
        
        if sentiment_daily.empty:
            return jsonify({'error': f'Could not fetch news data for {ticker}. Limited news coverage may be available.'}), 400
        
        # Merge data
        merged = pd.merge(stock_df, sentiment_daily, on='Date', how='left')
        merged['AvgSentiment'].fillna(method='ffill', inplace=True)
        merged = merged.dropna()
        
        if merged.empty:
            return jsonify({'error': 'No overlapping data found between stock prices and news sentiment.'}), 400
        
        # Create charts
        charts = create_charts(merged, ticker)
        
        # Calculate summary stats
        avg_sentiment = float(merged['AvgSentiment'].mean())
        current_price = float(merged['Close'].iloc[-1])
        price_change = float(merged['Close'].iloc[-1] - merged['Close'].iloc[0])
        price_change_pct = (price_change / merged['Close'].iloc[0]) * 100
        
        # Sentiment interpretation
        if avg_sentiment > 0.1:
            sentiment_label = "Positive"
        elif avg_sentiment < -0.1:
            sentiment_label = "Negative"
        else:
            sentiment_label = "Neutral"
        
        # Get recommendations
        recommendations = None
        if ticker in COMPANY_SECTORS:
            recommendations = {
                'sector': COMPANY_SECTORS[ticker]['sector'],
                'companies': COMPANY_SECTORS[ticker]['companies']
            }
        
        return jsonify({
            'success': True,
            'ticker': ticker,
            'charts': charts,
            'news': news_list[:10],
            'recommendations': recommendations,
            'summary': {
                'current_price': round(current_price, 2),
                'price_change': round(price_change, 2),
                'price_change_pct': round(price_change_pct, 2),
                'avg_sentiment': round(avg_sentiment, 3),
                'sentiment_label': sentiment_label,
                'total_news': len(news_list),
                'analysis_period': f"{merged['Date'].min()} to {merged['Date'].max()}"
            }
        })
        
    except Exception as e:
        print(f"Analysis error: {e}")
        return jsonify({'error': f'An error occurred during analysis: {str(e)}'}), 500

if __name__ == "__main__":
    app.run(debug=True, port=8000)