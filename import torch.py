import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from collections import Counter
import re
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

class SECFilingAnalyzer:
    def __init__(self, directory):
       
        self.directory = directory
        self.df = None
        self.summary_stats = None
        
    def load_and_clean_data(self):
       
       
        file_list = glob.glob(self.directory + "/*.csv")
        dfs = [pd.read_csv(file) for file in file_list]
        self.df = pd.concat(dfs, ignore_index=True)
        
        
        self.df['Filing date'] = pd.to_datetime(self.df['Filing date'])
        self.df['Reporting date'] = pd.to_datetime(self.df['Reporting date'], errors='coerce')
        
        
        self.df['Year'] = self.df['Filing date'].dt.year
        self.df['Month'] = self.df['Filing date'].dt.month
        self.df['Quarter'] = self.df['Filing date'].dt.quarter
        self.df['WeekDay'] = self.df['Filing date'].dt.day_name()
        
        
        self.df['Delay'] = self.df['Reporting date'] - self.df['Filing date']
        self.df['Delay_in_days'] = self.df['Delay'].dt.days
        
        return self.df
    
    def detect_anomalies(self, column='Delay_in_days', threshold=3):
        """Detect anomalies in specified column using z-score"""
        z_scores = np.abs(stats.zscore(self.df[column].dropna()))
        anomalies = self.df[column].dropna()[z_scores > threshold]
        return anomalies
    
    def analyze_filing_patterns(self):
        """Analyze patterns in filing behavior"""
        patterns = {
            'weekday_distribution': self.df['WeekDay'].value_counts(),
            'quarter_distribution': self.df['Quarter'].value_counts(),
            'yearly_growth': self.df.groupby('Year').size().pct_change() * 100,
            'seasonal_patterns': self.df.groupby(['Year', 'Month']).size().unstack().mean()
        }
        return patterns
    
    def generate_visualizations(self):
        """Generate comprehensive visualizations"""
        plt.style.use('classic')
        
        # 1. Enhanced Monthly Filing Trends with Trend Line
        plt.figure(figsize=(15, 8))
        monthly_counts = self.df.groupby(self.df['Filing date'].dt.to_period('M')).size()
        ax = monthly_counts.plot(kind='bar', color='skyblue', alpha=0.7)
        
        # Add trend line
        z = np.polyfit(range(len(monthly_counts)), monthly_counts, 1)
        p = np.poly1d(z)
        plt.plot(range(len(monthly_counts)), p(range(len(monthly_counts))), "r--", alpha=0.8)
        
        plt.title('Monthly Filing Frequency with Trend Line', pad=20, fontsize=12)
        plt.xlabel('Month', labelpad=10)
        plt.ylabel('Number of Filings', labelpad=10)
        plt.xticks(rotation=45)
        ax.grid(True, axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()
        
        # 2. Filing Delays Boxplot by Form Type
        plt.figure(figsize=(15, 8))
        sns.boxplot(x='Form type', y='Delay_in_days', data=self.df)
        plt.title('Filing Delays by Form Type', pad=20, fontsize=12)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        
        # 3. Weekly Filing Pattern Heatmap
        weekly_pattern = self.df.pivot_table(
            index=self.df['Filing date'].dt.day_name(),
            columns=self.df['Filing date'].dt.month,
            aggfunc='size',
            fill_value=0
        )
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(weekly_pattern, cmap='YlOrRd', annot=True, fmt='g')
        plt.title('Filing Patterns: Day of Week vs Month', pad=20)
        plt.tight_layout()
        plt.show()
        
        # 4. Form Type Distribution Over Time
        plt.figure(figsize=(15, 8))
        form_type_by_year = self.df.pivot_table(
            index='Year',
            columns='Form type',
            aggfunc='size',
            fill_value=0
        )
        form_type_by_year.plot(kind='bar', stacked=True)
        plt.title('Form Type Distribution Over Years', pad=20)
        plt.xlabel('Year')
        plt.ylabel('Number of Filings')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()
        
        # 5. Delay Distribution Analysis
        plt.figure(figsize=(12, 6))
        delay_stats = self.df.groupby('Form type')['Delay_in_days'].agg(['mean', 'std']).round(2)
        delay_stats.plot(kind='bar', yerr='std', capsize=5)
        plt.title('Average Filing Delays by Form Type with Standard Deviation')
        plt.xlabel('Form Type')
        plt.ylabel('Days')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        
    def calculate_complex_metrics(self):
        """Calculate complex metrics and statistics"""
        metrics = {
            'filing_velocity': self._calculate_filing_velocity(),
            'compliance_score': self._calculate_compliance_score(),
            'form_complexity': self._analyze_form_complexity()
        }
        return metrics
    
    def _calculate_filing_velocity(self):
        """Calculate the rate of change in filing frequency"""
        monthly_counts = self.df.groupby(self.df['Filing date'].dt.to_period('M')).size()
        velocity = monthly_counts.pct_change()
        return {
            'average_velocity': velocity.mean(),
            'volatility': velocity.std(),
            'max_increase': velocity.max(),
            'max_decrease': velocity.min()
        }
    
    def _calculate_compliance_score(self):
        """Calculate compliance score based on filing delays"""
        delays = self.df['Delay_in_days'].dropna()
        score = 100 * (1 - (delays / delays.max()))
        return {
            'average_score': score.mean(),
            'score_std': score.std(),
            'compliance_rate': (delays <= 0).mean() * 100
        }
    
    def _analyze_form_complexity(self):
        """Analyze complexity of different form types"""
        return self.df.groupby('Form type').agg({
            'Delay_in_days': ['mean', 'std', 'count'],
        }).round(2)
    
    def analyze_filing_trends(self):
        """Analyze trends in filing patterns"""
        # Monthly trend analysis
        monthly_filings = self.df.groupby([self.df['Filing date'].dt.year, 
                                         self.df['Filing date'].dt.month]).size()
        # Calculate moving averages
        ma_3 = monthly_filings.rolling(window=3).mean()
        ma_6 = monthly_filings.rolling(window=6).mean()
        
        return {
            'monthly_filings': monthly_filings,
            'moving_avg_3month': ma_3,
            'moving_avg_6month': ma_6
        }
    
    def generate_report(self):
        """Generate a comprehensive analysis report"""
        metrics = self.calculate_complex_metrics()
        patterns = self.analyze_filing_patterns()
        
        report = f"""
SEC Filings Advanced Analysis Report
==================================

Dataset Overview:
----------------
Total Filings: {len(self.df):,}
Date Range: {self.df['Filing date'].min().strftime('%Y-%m-%d')} to {self.df['Filing date'].max().strftime('%Y-%m-%d')}
Number of Unique Form Types: {self.df['Form type'].nunique()}

Filing Velocity Metrics:
----------------------
Average Monthly Change: {metrics['filing_velocity']['average_velocity']:.2f}%
Volatility: {metrics['filing_velocity']['volatility']:.2f}
Maximum Monthly Increase: {metrics['filing_velocity']['max_increase']:.2f}%
Maximum Monthly Decrease: {metrics['filing_velocity']['max_decrease']:.2f}%

Compliance Metrics:
-----------------
Average Compliance Score: {metrics['compliance_score']['average_score']:.2f}
Compliance Rate: {metrics['compliance_score']['compliance_rate']:.2f}%

Filing Patterns:
--------------
Most Common Filing Day: {patterns['weekday_distribution'].index[0]}
Busiest Quarter: Q{patterns['quarter_distribution'].index[0]}
Average Yearly Growth: {patterns['yearly_growth'].mean():.2f}%

Form Type Analysis:
-----------------
{metrics['form_complexity'].to_string()}

Anomaly Detection:
----------------
Number of Anomalous Delays: {len(self.detect_anomalies())}
"""
        return report

def save_analysis_to_excel(self, filename='sec_filing_analysis.xlsx'):
    """Save analysis results to Excel file"""
    with pd.ExcelWriter(filename) as writer:
        # Basic statistics
        self.df.describe().to_excel(writer, sheet_name='Basic Statistics')
        
        # Filing patterns
        patterns = self.analyze_filing_patterns()
        pd.DataFrame(patterns['weekday_distribution']).to_excel(writer, sheet_name='Filing Patterns', startrow=0)
        pd.DataFrame(patterns['quarter_distribution']).to_excel(writer, sheet_name='Filing Patterns', startrow=len(patterns['weekday_distribution'])+2)
        
        # Complex metrics
        metrics = self.calculate_complex_metrics()
        metrics['form_complexity'].to_excel(writer, sheet_name='Complex Metrics')
        
        # Anomalies
        anomalies = self.detect_anomalies()
        pd.DataFrame(anomalies).to_excel(writer, sheet_name='Anomalies')

# Example usage:
if __name__ == "__main__":
    directory = r"C:\Users\mattb\Videos\Desktop\Csv sec"
    analyzer = SECFilingAnalyzer(directory)
    analyzer.load_and_clean_data()
    
    # Generate visualizations
    analyzer.generate_visualizations()
    
    # Generate and print report
    report = analyzer.generate_report()
    print(report)
    
    # Calculate additional metrics
    metrics = analyzer.calculate_complex_metrics()
    patterns = analyzer.analyze_filing_patterns()
    
    # Detect anomalies
    anomalies = analyzer.detect_anomalies()
    if len(anomalies) > 0:
        print("\nAnomaly Detection Results:")
        print(f"Found {len(anomalies)} anomalies in filing delays")

class SECFilingAnalyzer:
    def __init__(self, directory):
        """Initialize the SEC Filing Analyzer with the directory containing CSV files"""
        self.directory = directory
        self.df = None
        self.summary_stats = None
        self.ml_models = {}
        self.label_encoders = {}
        self.scaler = MinMaxScaler()
        
    def load_and_clean_data(self):
        """Load and clean the SEC filing data"""
        # Load data
        file_list = glob.glob(self.directory + "/*.csv")
        dfs = [pd.read_csv(file) for file in file_list]
        self.df = pd.concat(dfs, ignore_index=True)
        
        # Clean dates
        self.df['Filing date'] = pd.to_datetime(self.df['Filing date'])
        self.df['Reporting date'] = pd.to_datetime(self.df['Reporting date'], errors='coerce')
        
        # Add temporal features
        self.df['Year'] = self.df['Filing date'].dt.year
        self.df['Month'] = self.df['Filing date'].dt.month
        self.df['Quarter'] = self.df['Filing date'].dt.quarter
        self.df['WeekDay'] = self.df['Filing date'].dt.day_name()
        
        # Calculate delays
        self.df['Delay'] = self.df['Reporting date'] - self.df['Filing date']
        self.df['Delay_in_days'] = self.df['Delay'].dt.days
        
        # Engineer additional features
        self._engineer_features()
        
        return self.df
    
    def _engineer_features(self):
        """Engineer additional features for ML models"""
        # Time-based features
        self.df['DayOfWeek'] = self.df['Filing date'].dt.dayofweek
        self.df['DayOfYear'] = self.df['Filing date'].dt.dayofyear
        self.df['WeekOfYear'] = self.df['Filing date'].dt.isocalendar().week
        
        # Categorical encoding
        categorical_columns = ['Form type', 'WeekDay']
        for col in categorical_columns:
            le = LabelEncoder()
            self.df[f'{col}_encoded'] = le.fit_transform(self.df[col])
            self.label_encoders[col] = le
        
        # Filing frequency features
        self.df['Monthly_Filing_Count'] = self.df.groupby(
            [self.df['Filing date'].dt.year, self.df['Filing date'].dt.month]
        )['Filing date'].transform('count')
        
        # Normalize numerical features
        numerical_features = ['Delay_in_days', 'Monthly_Filing_Count']
        self.df[numerical_features] = self.scaler.fit_transform(
            self.df[numerical_features].fillna(0)
        )
    
    def train_delay_predictor(self):
        """Train a Random Forest model to predict filing delays"""
        features = [
            'DayOfWeek', 'DayOfYear', 'WeekOfYear', 
            'Form type_encoded', 'Monthly_Filing_Count'
        ]
        
        X = self.df[features].dropna()
        y = self.df['Delay_in_days'].dropna()
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        
        self.ml_models['delay_predictor'] = {
            'model': rf_model,
            'features': features,
            'accuracy': rf_model.score(X_test, y_test)
        }
        
        return rf_model.score(X_test, y_test)
    
    def predict_filing_delay(self, form_type, filing_date):
        """Predict filing delay for a new filing"""
        if 'delay_predictor' not in self.ml_models:
            raise ValueError("Delay predictor model not trained yet")
            
        # Prepare input features
        filing_date = pd.to_datetime(filing_date)
        input_data = pd.DataFrame({
            'DayOfWeek': [filing_date.dayofweek],
            'DayOfYear': [filing_date.dayofyear],
            'WeekOfYear': [filing_date.isocalendar().week],
            'Form type_encoded': [self.label_encoders['Form type'].transform([form_type])[0]],
            'Monthly_Filing_Count': [self.df['Monthly_Filing_Count'].mean()]
        })
        
        prediction = self.ml_models['delay_predictor']['model'].predict(input_data)
        return prediction[0]
    
    def train_anomaly_detector(self):
        """Train an Isolation Forest model for advanced anomaly detection"""
        features = [
            'Delay_in_days', 'Monthly_Filing_Count',
            'DayOfWeek', 'Form type_encoded'
        ]
        
        X = self.df[features].dropna()
        
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        self.ml_models['anomaly_detector'] = {
            'model': iso_forest.fit(X),
            'features': features
        }
        
        # Add anomaly predictions to dataframe
        self.df['is_anomaly'] = iso_forest.predict(X)
        return self.df[self.df['is_anomaly'] == -1]  # Return anomalies
    
    def analyze_filing_patterns_ml(self):
        """Use KMeans clustering to identify filing pattern clusters"""
        features = [
            'DayOfWeek', 'WeekOfYear', 'Monthly_Filing_Count',
            'Delay_in_days', 'Form type_encoded'
        ]
        
        X = self.df[features].dropna()
        
        # Determine optimal number of clusters using elbow method
        inertias = []
        K = range(1, 11)
        for k in K:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(X)
            inertias.append(kmeans.inertia_)
        
        # Plot elbow curve
        plt.figure(figsize=(10, 6))
        plt.plot(K, inertias, 'bx-')
        plt.xlabel('k')
        plt.ylabel('Inertia')
        plt.title('Elbow Method For Optimal k')
        plt.show()
        
        # Perform clustering with optimal k (using 4 as default)
        kmeans = KMeans(n_clusters=4, random_state=42)
        self.df['filing_cluster'] = kmeans.fit_predict(X)
        
        return self.analyze_clusters()
    
    def analyze_clusters(self):
        """Analyze the characteristics of each filing cluster"""
        cluster_analysis = {}
        
        for cluster in self.df['filing_cluster'].unique():
            cluster_data = self.df[self.df['filing_cluster'] == cluster]
            cluster_analysis[f'Cluster_{cluster}'] = {
                'size': len(cluster_data),
                'avg_delay': cluster_data['Delay_in_days'].mean(),
                'common_form_type': cluster_data['Form type'].mode()[0],
                'common_weekday': cluster_data['WeekDay'].mode()[0],
                'avg_monthly_filings': cluster_data['Monthly_Filing_Count'].mean()
            }
        
        return pd.DataFrame(cluster_analysis).T
    
    def predict_future_filing_volume(self, days=30):
        """Predict future filing volumes using Random Forest"""
        # Prepare historical data
        historical_data = self.df.groupby('Filing date').size().reset_index()
        historical_data.columns = ['date', 'filings']
        
        # Create features for prediction
        historical_data['dayofweek'] = historical_data['date'].dt.dayofweek
        historical_data['dayofyear'] = historical_data['date'].dt.dayofyear
        historical_data['weekofyear'] = historical_data['date'].dt.isocalendar().week
        
        # Train model
        features = ['dayofweek', 'dayofyear', 'weekofyear']
        X = historical_data[features]
        y = historical_data['filings']
        
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        # Generate future dates
        last_date = historical_data['date'].max()
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days)
        
        # Create future features
        future_data = pd.DataFrame({
            'date': future_dates,
            'dayofweek': future_dates.dayofweek,
            'dayofyear': future_dates.dayofyear,
            'weekofyear': future_dates.isocalendar().week
        })
        
        # Make predictions
        future_data['predicted_filings'] = model.predict(future_data[features])
        
        return future_data[['date', 'predicted_filings']]
    
    def generate_ml_report(self):
        """Generate a comprehensive ML analysis report"""
        report = f"""
SEC Filings Machine Learning Analysis Report
==========================================

Model Performance Metrics:
------------------------
Filing Delay Predictor Accuracy: {self.ml_models.get('delay_predictor', {}).get('accuracy', 'Not trained')}
Number of Anomalies Detected: {len(self.df[self.df['is_anomaly'] == -1]) if 'is_anomaly' in self.df else 'Not analyzed'}

Clustering Analysis:
------------------
{self.analyze_clusters().to_string() if 'filing_cluster' in self.df else 'Clustering not performed'}

Feature Importance (Delay Predictor):
----------------------------------
{pd.DataFrame({
    'Feature': self.ml_models.get('delay_predictor', {}).get('features', []),
    'Importance': self.ml_models.get('delay_predictor', {}).get('model', RandomForestRegressor()).feature_importances_
}).sort_values('Importance', ascending=False).to_string() if 'delay_predictor' in self.ml_models else 'Model not trained'}
"""
        return report

# Example usage:
if __name__ == "__main__":
    directory = r"C:\Users\mattb\Videos\Desktop\Csv sec"
    analyzer = SECFilingAnalyzer(directory)
    analyzer.load_and_clean_data()
    
    # Train ML models
    print("Training delay predictor...")
    accuracy = analyzer.train_delay_predictor()
    print(f"Delay predictor accuracy: {accuracy:.2f}")
    
    print("\nTraining anomaly detector...")
    anomalies = analyzer.train_anomaly_detector()
    print(f"Found {len(anomalies)} anomalies")
    
    print("\nAnalyzing filing patterns...")
    clusters = analyzer.analyze_filing_patterns_ml()
    print("\nCluster Analysis:")
    print(clusters)
    
    # Generate future predictions
    print("\nPredicting future filing volumes...")
    future_predictions = analyzer.predict_future_filing_volume(days=30)
    print("\nPredicted filing volumes for next 30 days:")
    print(future_predictions)
    
    # Generate and print ML report
    report = analyzer.generate_ml_report()
    print("\nML Analysis Report:")
    print(report)
    
    