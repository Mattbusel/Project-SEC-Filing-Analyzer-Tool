import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import seaborn as sns
from datetime import datetime
import logging
import warnings
warnings.filterwarnings('ignore')


class SECFilingAnalyzer:
    def __init__(self, directory):
        self.directory = directory
        self.df = None
        self.ml_models = {}
        self.label_encoders = {}
        self.scaler = StandardScaler()

    def load_and_clean_data(self):
        """Load and clean the SEC filing data"""
        try:
            # Load files
            file_list = glob.glob(self.directory + "/*.csv")
            if not file_list:
                raise ValueError(f"No CSV files found in directory: {self.directory}")
            
            dfs = []
            for file in file_list:
                try:
                    df = pd.read_csv(file)
                    dfs.append(df)
                except Exception as e:
                    print(f"Error reading file {file}: {str(e)}")
            
            if not dfs:
                raise ValueError("No valid data loaded from CSV files")
            
            self.df = pd.concat(dfs, ignore_index=True)
            
            # Process and create features
            self._clean_dates()
            self._create_temporal_features()
            self._clean_delays()
            self._create_form_features()
            self._create_volume_features()
            self._create_statistical_features()
            
            # Handle missing values
            self.df = self.df.dropna()
            
            print(f"\nDataFrame shape after cleaning: {self.df.shape}")
            return self.df
            
        except Exception as e:
            print(f"Error in load_and_clean_data: {str(e)}")
            raise

    def _clean_dates(self):
        """Clean and validate dates"""
        self.df['Filing date'] = pd.to_datetime(self.df['Filing date'])
        self.df['Reporting date'] = pd.to_datetime(self.df['Reporting date'], errors='coerce')
        
        self.df = self.df.dropna(subset=['Filing date', 'Reporting date'])
        
        current_year = pd.Timestamp.now().year
        self.df = self.df[
            (self.df['Filing date'].dt.year >= 2000) & 
            (self.df['Filing date'].dt.year <= current_year)
        ]

    def _create_temporal_features(self):
        """Create temporal features"""
        # Basic date components
        self.df['Year'] = self.df['Filing date'].dt.year
        self.df['Month'] = self.df['Filing date'].dt.month
        self.df['Quarter'] = self.df['Filing date'].dt.quarter
        self.df['WeekDay'] = self.df['Filing date'].dt.day_name()
        self.df['DayOfWeek'] = self.df['Filing date'].dt.dayofweek
        self.df['DayOfYear'] = self.df['Filing date'].dt.dayofyear
        self.df['WeekOfYear'] = self.df['Filing date'].dt.isocalendar().week
        self.df['DayOfMonth'] = self.df['Filing date'].dt.day
        
        # Calendar events
        self.df['IsMonthEnd'] = self.df['Filing date'].dt.is_month_end.astype(int)
        self.df['IsQuarterEnd'] = self.df['Filing date'].dt.is_quarter_end.astype(int)
        self.df['IsYearEnd'] = self.df['Filing date'].dt.is_year_end.astype(int)
        self.df['IsMonthStart'] = self.df['Filing date'].dt.is_month_start.astype(int)
        self.df['IsQuarterStart'] = self.df['Filing date'].dt.is_quarter_start.astype(int)
        self.df['IsWeekend'] = self.df['DayOfWeek'].isin([5, 6]).astype(int)
        
        # Days remaining calculations
        self.df['DaysInMonth'] = self.df['Filing date'].dt.days_in_month
        self.df['DaysToMonthEnd'] = self.df['DaysInMonth'] - self.df['DayOfMonth']
        self.df['DaysToYearEnd'] = 365 - self.df['DayOfYear']

        # Calculate days to quarter end
        self.df['DaysToQuarterEnd'] = self.df.apply(
            lambda row: (pd.Timestamp(row['Year'], 3 * ((row['Month'] - 1) // 3 + 1), 1) + 
                        pd.Timedelta(days=-1) - row['Filing date']).days,
            axis=1
        )

    def _clean_delays(self):
        """Calculate and clean delay values"""
        self.df['Delay'] = (self.df['Filing date'] - self.df['Reporting date']).dt.days
        
        # Remove outliers using IQR method
        Q1 = self.df['Delay'].quantile(0.25)
        Q3 = self.df['Delay'].quantile(0.75)
        IQR = Q3 - Q1
        
        self.df = self.df[
            (self.df['Delay'] >= (Q1 - 2 * IQR)) & 
            (self.df['Delay'] <= (Q3 + 2 * IQR))
        ]
        
        self.df['Delay_in_days'] = self.df['Delay'].abs()

    def _create_form_features(self):
        """Create form-type related features"""
        # Encode form types
        le = LabelEncoder()
        self.df['Form type_encoded'] = le.fit_transform(self.df['Form type'])
        self.label_encoders['Form type'] = le
        
        # Form type frequency
        form_freq = self.df['Form type'].value_counts(normalize=True)
        self.df['Form type_freq'] = self.df['Form type'].map(form_freq)
        
        # Form type delay statistics
        form_stats = self.df.groupby('Form type')['Delay_in_days'].agg(['mean', 'std']).fillna(0)
        self.df['Form_avg_delay'] = self.df['Form type'].map(form_stats['mean'])
        self.df['Form_std_delay'] = self.df['Form type'].map(form_stats['std'])

    def _create_volume_features(self):
        """Create volume-based features"""
        # Daily volumes
        daily_volumes = self.df.groupby('Filing date').size()
        self.df['DailyVolume'] = self.df['Filing date'].map(daily_volumes)
        
        # Monthly volumes
        monthly_volumes = self.df.groupby(['Year', 'Month']).size()
        self.df['MonthlyVolume'] = self.df.apply(
            lambda x: monthly_volumes.get((x['Year'], x['Month']), 0), 
            axis=1
        )

    def _create_statistical_features(self):
        """Create statistical features"""
        # Rolling statistics
        for window in [7, 30]:
            for col in ['DailyVolume', 'Delay_in_days']:
                self.df[f'{col}_rolling_mean_{window}d'] = self.df.groupby('Form type')[col].transform(
                    lambda x: x.rolling(window=window, min_periods=1).mean()
                )
                self.df[f'{col}_rolling_std_{window}d'] = self.df.groupby('Form type')[col].transform(
                    lambda x: x.rolling(window=window, min_periods=1).std()
                )
        
        # Normalize numerical features
        numerical_features = self.df.select_dtypes(include=['int64', 'float64']).columns
        self.df[numerical_features] = self.scaler.fit_transform(self.df[numerical_features])

    def train_delay_predictor(self):
        """Train the model with proper feature encoding"""
        try:
            # Identify numeric and categorical columns
            numeric_features = self.df.select_dtypes(include=['int64', 'float64']).columns
            categorical_features = self.df.select_dtypes(include=['object']).columns
            
            # Print column types for debugging
            print("\nNumeric features:", len(numeric_features))
            print("Categorical features:", len(categorical_features))
            
            # Select features for the model
            exclude_columns = [
                'Filing date', 'Reporting date', 'Delay', 'Form type',
                'WeekDay', 'Season', 'Delay_in_days'
            ]
            
            features = [col for col in self.df.columns 
                       if col not in exclude_columns 
                       and not any(ex in col for ex in categorical_features)]
            
            # Print selected features for debugging
            print("\nSelected features:", len(features))
            print(features)
            
            # Prepare data
            X = self.df[features].copy()
            y = self.df['Delay_in_days']
            
            # Fill any remaining NaN values
            X = X.fillna(0)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Train initial model
            print("\nTraining initial model...")
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            rf.fit(X_train, y_train)
            initial_score = rf.score(X_test, y_test)
            print(f"Initial model R² score: {initial_score:.4f}")
            
            # Grid search
            print("\nProceeding with GridSearchCV...")
            param_grid = {
                'n_estimators': [200],
                'max_depth': [15, 20],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2],
                'max_features': ['sqrt']
            }
            
            grid_search = GridSearchCV(
                RandomForestRegressor(random_state=42),
                param_grid,
                cv=5,
                scoring='r2',
                n_jobs=-1
            )
            
            grid_search.fit(X_train, y_train)
            
            # Get predictions and scores
            best_model = grid_search.best_estimator_
            y_pred = best_model.predict(X_test)
            
            train_score = best_model.score(X_train, y_train)
            test_score = best_model.score(X_test, y_test)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            
            # Store results
            self.ml_models['delay_predictor'] = {
                'model': best_model,
                'features': features,
                'train_score': train_score,
                'test_score': test_score,
                'rmse': rmse,
                'best_params': grid_search.best_params_,
                'feature_importance': dict(zip(features, best_model.feature_importances_))
            }
            
            # Print metrics
            print("\nModel Performance Metrics:")
            print(f"Train R² Score: {train_score:.4f}")
            print(f"Test R² Score: {test_score:.4f}")
            print(f"RMSE: {rmse:.4f}")
            print("\nBest Parameters:", grid_search.best_params_)
            
            # Print top features
            importance_df = pd.DataFrame(
                self.ml_models['delay_predictor']['feature_importance'].items(),
                columns=['Feature', 'Importance']
            ).sort_values('Importance', ascending=False)
            
            print("\nTop 10 Most Important Features:")
            print(importance_df.head(10))
            
            return test_score
            
        except Exception as e:
            print(f"\nError in train_delay_predictor: {str(e)}")
            print("\nDataFrame info:")
            print(self.df.info())
            raise

    def generate_visualizations(self):
        """Generate visualizations"""
        if self.df is None:
            raise ValueError("No data available. Please load data first.")
        
        try:
            fig = plt.figure(figsize=(20, 15))
            
            # Filing trends
            ax1 = plt.subplot(311)
            self._plot_filing_trends(ax1)
            
            # Delay distribution
            ax2 = plt.subplot(312)
            self._plot_delay_distribution(ax2)
            
            # Weekly patterns
            ax3 = plt.subplot(313)
            self._plot_weekly_patterns(ax3)
            
            plt.tight_layout()
            plt.show()
            
            # Plot feature importance separately
            if self.ml_models:
                self._plot_feature_importance()
                
        except Exception as e:
            print(f"Error in generate_visualizations: {str(e)}")
            raise

    def _plot_filing_trends(self, ax):
        """Plot filing trends"""
        monthly_counts = self.df.groupby(pd.Grouper(key='Filing date', freq='M')).size()
        ax.plot(monthly_counts.index, monthly_counts.values)
        ax.set_title('Monthly Filing Trends')
        ax.set_xlabel('Date')
        ax.set_ylabel('Number of Filings')
        plt.setp(ax.get_xticklabels(), rotation=45)

    def _plot_delay_distribution(self, ax):
        """Plot delay distribution"""
        ax.hist(self.df['Delay_in_days'].dropna(), bins=50)
        ax.set_title('Distribution of Filing Delays')
        ax.set_xlabel('Delay (days)')
        ax.set_ylabel('Count')

    def _plot_weekly_patterns(self, ax):
        """Plot weekly patterns"""
        weekly_pattern = pd.crosstab(
            self.df['Filing date'].dt.day_name(),
            self.df['Filing date'].dt.month
        )
        
        days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        weekly_pattern = weekly_pattern.reindex(days_order)
        
        im = ax.pcolormesh(weekly_pattern.values, cmap='YlOrRd')
        ax.set_yticks(np.arange(len(days_order)))
        ax.set_yticklabels(days_order)
        ax.set_xticks(np.arange(12))
        ax.set_xticklabels(range(1, 13))
        
        ax.set_title('Filing Patterns: Day of Week vs Month')
        ax.set_xlabel('Month')
        ax.set_ylabel('Day of Week')
        plt.colorbar(im, ax=ax)

    def _plot_feature_importance(self):
        """Plot feature importance"""
        if 'delay_predictor' not in self.ml_models:
            return
            
        importance_df = pd.DataFrame(
            self.ml_models['delay_predictor']['feature_importance'].items(),
            columns=['Feature', 'Importance']
        ).sort_values('Importance', ascending=True)
        
        plt.figure(figsize=(10, 6))
        plt.barh(range(len(importance_df)), importance_df['Importance'])

def main():
    def generate_visualizations(self):
        """Generate comprehensive visualizations"""
        if self.df is None:
            raise ValueError("No data available. Please load data first.")
        
        plt.style.use('default')
        
        # Create subplots for all visualizations
        fig = plt.figure(figsize=(20, 15))
        gs = fig.add_gridspec(3, 2)
        
        # Filing trends
        ax1 = fig.add_subplot(gs[0, :])
        self._plot_filing_trends(ax1)
        
        # Delay distribution
        ax2 = fig.add_subplot(gs[1, 0])
        ax3 = fig.add_subplot(gs[1, 1])
        self._plot_delay_distribution(ax2, ax3)
        
        # Weekly patterns
        ax4 = fig.add_subplot(gs[2, 0])
        self._plot_weekly_patterns(ax4)
        
        # Feature importance
        if 'delay_predictor' in self.ml_models:
            ax5 = fig.add_subplot(gs[2, 1])
            self._plot_feature_importance(ax5)
        
        plt.tight_layout()
        plt.show()
    
    def _plot_filing_trends(self, ax):
        """Plot filing trends over time with trend line"""
        monthly_counts = self.df.groupby(
            pd.Grouper(key='Filing date', freq='M')
        ).size().reset_index()
        
        ax.plot(monthly_counts['Filing date'], 
                monthly_counts[0], 
                color='blue', 
                alpha=0.6, 
                label='Actual Filings')
        
        z = np.polyfit(range(len(monthly_counts)), monthly_counts[0], 1)
        p = np.poly1d(z)
        ax.plot(monthly_counts['Filing date'], 
                p(range(len(monthly_counts))), 
                'r--', 
                alpha=0.8, 
                label='Trend Line')
        
        ax.set_title('Monthly Filing Frequency Trend Analysis')
        ax.set_xlabel('Date')
        ax.set_ylabel('Number of Filings')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.setp(ax.get_xticklabels(), rotation=45)
    
    def _plot_delay_distribution(self, ax1, ax2):
        """Plot filing delay distribution"""
        ax1.hist(self.df['Delay_in_days'].dropna(), 
                bins=50, 
                color='blue', 
                alpha=0.6)
        ax1.set_title('Distribution of Filing Delays')
        ax1.set_xlabel('Delay (days)')
        ax1.set_ylabel('Count')
        
        form_types = self.df['Form type'].unique()
        delays_by_form = [self.df[self.df['Form type'] == ft]['Delay_in_days'] 
                         for ft in form_types]
        ax2.boxplot(delays_by_form, labels=form_types)
        ax2.set_title('Filing Delays by Form Type')
        plt.setp(ax2.get_xticklabels(), rotation=45)
        ax2.set_ylabel('Delay (days)')
    
    def _plot_weekly_patterns(self, ax):
        """Create weekly filing pattern visualization"""
        weekly_pattern = pd.crosstab(
            self.df['Filing date'].dt.day_name(),
            self.df['Filing date'].dt.month
        )
        
        days_order = ['Monday', 'Tuesday', 'Wednesday', 
                     'Thursday', 'Friday', 'Saturday', 'Sunday']
        weekly_pattern = weekly_pattern.reindex(days_order)
        
        im = ax.pcolormesh(weekly_pattern.values, cmap='YlOrRd')
        
        ax.set_yticks(np.arange(0.5, len(days_order)))
        ax.set_yticklabels(days_order)
        ax.set_xticks(np.arange(0.5, 13))
        ax.set_xticklabels(range(1, 13))
        
        ax.set_title('Filing Patterns: Day of Week vs Month')
        ax.set_xlabel('Month')
        ax.set_ylabel('Day of Week')
        
        plt.colorbar(im, ax=ax, label='Number of Filings')
    
    def _plot_feature_importance(self, ax):
        """Visualize feature importance from the ML model"""
        importance_df = pd.DataFrame(
            self.ml_models['delay_predictor']['feature_importance'].items(),
            columns=['Feature', 'Importance']
        ).sort_values('Importance', ascending=True)
        
        ax.barh(range(len(importance_df)), 
                importance_df['Importance'], 
                align='center',
                color='blue',
                alpha=0.6)
        
        ax.set_yticks(range(len(importance_df)))
        ax.set_yticklabels(importance_df['Feature'])
        
        ax.set_title('Feature Importance in Delay Prediction')
        ax.set_xlabel('Importance Score')



def main():
    try:
        directory = r"C:\Users\mattb\Videos\Desktop\Csv sec"
        analyzer = SECFilingAnalyzer(directory)
        
        print("Loading and cleaning data...")
        df = analyzer.load_and_clean_data()
        print(f"Successfully loaded and cleaned {len(df)} records")
        
        print("\nTraining model...")
        accuracy = analyzer.train_delay_predictor()
        print(f"Model R² score: {accuracy:.4f}")
        
        print("\nGenerating visualizations...")
        analyzer.generate_visualizations()
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()
)
    
    
