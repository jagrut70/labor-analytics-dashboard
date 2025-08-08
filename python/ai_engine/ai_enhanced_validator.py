"""
AI-Enhanced Validation Engine
Leverages machine learning for intelligent data validation and anomaly detection
"""

import pandas as pd
import numpy as np
import joblib
import pickle
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.ensemble import IsolationForest, RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, mean_squared_error, r2_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import xgboost as xgb
from lightgbm import LGBMRegressor
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.optimizers import Adam

# Advanced Analytics
from scipy import stats
from scipy.stats import zscore
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA

# Visualization
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt

# Custom imports
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_connectors.database_connector import DatabaseConnector
import structlog

logger = structlog.get_logger(__name__)


class AIEnhancedValidator:
    """AI-powered validation engine with ML capabilities"""
    
    def __init__(self, config_path: str = "../config/ai_config.json"):
        """Initialize AI-enhanced validator"""
        self.config = self._load_config(config_path)
        self.db_connector = DatabaseConnector()
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.anomaly_detectors = {}
        self._setup_logging()
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load AI configuration"""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            return config
        except FileNotFoundError:
            # Default AI configuration
            return {
                "anomaly_detection": {
                    "isolation_forest": {"contamination": 0.1, "random_state": 42},
                    "zscore_threshold": 3.0,
                    "iqr_multiplier": 1.5
                },
                "predictive_validation": {
                    "salary_prediction": {"model_type": "xgboost", "features": ["department_id", "position", "experience_years"]},
                    "hours_prediction": {"model_type": "lightgbm", "features": ["employee_id", "project_id", "day_of_week"]}
                },
                "clustering": {
                    "n_clusters": 5,
                    "features": ["salary", "hours_worked", "overtime_hours"]
                },
                "time_series": {
                    "forecast_periods": 30,
                    "seasonality": 7
                }
            }
    
    def _setup_logging(self):
        """Setup structured logging"""
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                structlog.processors.JSONRenderer()
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )
    
    def detect_anomalies_isolation_forest(self, df: pd.DataFrame, features: List[str]) -> Dict[str, Any]:
        """Detect anomalies using Isolation Forest"""
        try:
            # Prepare data
            X = df[features].fillna(df[features].median())
            
            # Train Isolation Forest
            iso_forest = IsolationForest(
                contamination=self.config["anomaly_detection"]["isolation_forest"]["contamination"],
                random_state=self.config["anomaly_detection"]["isolation_forest"]["random_state"]
            )
            
            # Fit and predict
            iso_forest.fit(X)
            predictions = iso_forest.predict(X)
            scores = iso_forest.decision_function(X)
            
            # Identify anomalies
            anomalies = predictions == -1
            anomaly_indices = df[anomalies].index.tolist()
            
            return {
                "anomaly_indices": anomaly_indices,
                "anomaly_scores": scores,
                "anomaly_count": len(anomaly_indices),
                "anomaly_percentage": (len(anomaly_indices) / len(df)) * 100,
                "model": iso_forest
            }
            
        except Exception as e:
            logger.error(f"Error in isolation forest anomaly detection: {e}")
            return {"error": str(e)}
    
    def detect_anomalies_statistical(self, df: pd.DataFrame, column: str) -> Dict[str, Any]:
        """Detect anomalies using statistical methods (Z-score, IQR)"""
        try:
            data = df[column].dropna()
            
            # Z-score method
            z_scores = np.abs(zscore(data))
            z_anomalies = z_scores > self.config["anomaly_detection"]["zscore_threshold"]
            
            # IQR method
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - self.config["anomaly_detection"]["iqr_multiplier"] * IQR
            upper_bound = Q3 + self.config["anomaly_detection"]["iqr_multiplier"] * IQR
            iqr_anomalies = (data < lower_bound) | (data > upper_bound)
            
            # Combine methods
            combined_anomalies = z_anomalies | iqr_anomalies
            anomaly_indices = data[combined_anomalies].index.tolist()
            
            return {
                "anomaly_indices": anomaly_indices,
                "z_score_anomalies": data[z_anomalies].index.tolist(),
                "iqr_anomalies": data[iqr_anomalies].index.tolist(),
                "anomaly_count": len(anomaly_indices),
                "statistics": {
                    "mean": data.mean(),
                    "std": data.std(),
                    "q1": Q1,
                    "q3": Q3,
                    "iqr": IQR
                }
            }
            
        except Exception as e:
            logger.error(f"Error in statistical anomaly detection: {e}")
            return {"error": str(e)}
    
    def predict_salary_anomalies(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Predict salary anomalies using ML models"""
        try:
            # Prepare features
            features = self.config["predictive_validation"]["salary_prediction"]["features"]
            
            # Create feature engineering
            df_features = df.copy()
            
            # Add derived features
            if 'hire_date' in df.columns and 'salary' in df.columns:
                df_features['experience_years'] = (
                    pd.Timestamp.now() - pd.to_datetime(df_features['hire_date'])
                ).dt.days / 365.25
            
            if 'department_id' in df.columns:
                # Department average salary
                dept_avg_salary = df_features.groupby('department_id')['salary'].transform('mean')
                df_features['salary_vs_dept_avg'] = df_features['salary'] - dept_avg_salary
            
            # Prepare training data
            X = df_features[features].fillna(0)
            y = df_features['salary']
            
            # Train model
            if self.config["predictive_validation"]["salary_prediction"]["model_type"] == "xgboost":
                model = xgb.XGBRegressor(random_state=42)
            else:
                model = RandomForestRegressor(n_estimators=100, random_state=42)
            
            model.fit(X, y)
            predictions = model.predict(X)
            
            # Calculate residuals
            residuals = np.abs(y - predictions)
            residual_threshold = np.percentile(residuals, 95)  # 95th percentile
            
            # Identify anomalies
            anomalies = residuals > residual_threshold
            anomaly_indices = df_features[anomalies].index.tolist()
            
            return {
                "anomaly_indices": anomaly_indices,
                "predictions": predictions,
                "residuals": residuals,
                "threshold": residual_threshold,
                "model": model,
                "feature_importance": dict(zip(features, model.feature_importances_))
            }
            
        except Exception as e:
            logger.error(f"Error in salary prediction: {e}")
            return {"error": str(e)}
    
    def cluster_employees(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Cluster employees using K-means"""
        try:
            features = self.config["clustering"]["features"]
            n_clusters = self.config["clustering"]["n_clusters"]
            
            # Prepare data
            X = df[features].fillna(df[features].median())
            
            # Standardize features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Apply PCA for visualization
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_scaled)
            
            # K-means clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = kmeans.fit_predict(X_scaled)
            
            # Analyze clusters
            df_clustered = df.copy()
            df_clustered['cluster'] = clusters
            
            cluster_analysis = {}
            for cluster_id in range(n_clusters):
                cluster_data = df_clustered[df_clustered['cluster'] == cluster_id]
                cluster_analysis[f"cluster_{cluster_id}"] = {
                    "size": len(cluster_data),
                    "avg_salary": cluster_data['salary'].mean() if 'salary' in cluster_data.columns else None,
                    "avg_hours": cluster_data['hours_worked'].mean() if 'hours_worked' in cluster_data.columns else None
                }
            
            return {
                "clusters": clusters,
                "cluster_centers": kmeans.cluster_centers_,
                "cluster_analysis": cluster_analysis,
                "pca_data": X_pca,
                "model": kmeans,
                "scaler": scaler
            }
            
        except Exception as e:
            logger.error(f"Error in employee clustering: {e}")
            return {"error": str(e)}
    
    def forecast_time_series(self, df: pd.DataFrame, date_column: str, value_column: str) -> Dict[str, Any]:
        """Forecast time series data using ARIMA and LSTM"""
        try:
            # Prepare time series data
            df_ts = df.copy()
            df_ts[date_column] = pd.to_datetime(df_ts[date_column])
            df_ts = df_ts.set_index(date_column)
            
            # Resample to daily frequency
            ts_data = df_ts[value_column].resample('D').sum().fillna(0)
            
            # ARIMA Model
            try:
                arima_model = ARIMA(ts_data, order=(1, 1, 1))
                arima_fitted = arima_model.fit()
                arima_forecast = arima_fitted.forecast(steps=self.config["time_series"]["forecast_periods"])
            except:
                arima_forecast = None
            
            # LSTM Model
            try:
                # Prepare LSTM data
                def create_sequences(data, seq_length):
                    X, y = [], []
                    for i in range(len(data) - seq_length):
                        X.append(data[i:(i + seq_length)])
                        y.append(data[i + seq_length])
                    return np.array(X), np.array(y)
                
                # Normalize data
                scaler = StandardScaler()
                scaled_data = scaler.fit_transform(ts_data.values.reshape(-1, 1))
                
                # Create sequences
                seq_length = 7
                X, y = create_sequences(scaled_data, seq_length)
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                # Build LSTM model
                model = Sequential([
                    LSTM(50, return_sequences=True, input_shape=(seq_length, 1)),
                    Dropout(0.2),
                    LSTM(50, return_sequences=False),
                    Dropout(0.2),
                    Dense(1)
                ])
                
                model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
                model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)
                
                # Forecast
                last_sequence = scaled_data[-seq_length:]
                lstm_forecast = []
                
                for _ in range(self.config["time_series"]["forecast_periods"]):
                    next_pred = model.predict(last_sequence.reshape(1, seq_length, 1))
                    lstm_forecast.append(next_pred[0, 0])
                    last_sequence = np.roll(last_sequence, -1)
                    last_sequence[-1] = next_pred[0, 0]
                
                lstm_forecast = scaler.inverse_transform(np.array(lstm_forecast).reshape(-1, 1))
                
            except Exception as e:
                logger.warning(f"LSTM forecast failed: {e}")
                lstm_forecast = None
            
            return {
                "arima_forecast": arima_forecast,
                "lstm_forecast": lstm_forecast,
                "original_data": ts_data,
                "forecast_periods": self.config["time_series"]["forecast_periods"]
            }
            
        except Exception as e:
            logger.error(f"Error in time series forecasting: {e}")
            return {"error": str(e)}
    
    def intelligent_validation_rules(self, df: pd.DataFrame, table_name: str) -> Dict[str, Any]:
        """Generate intelligent validation rules using ML"""
        try:
            intelligent_rules = {}
            
            # Analyze data patterns
            for column in df.columns:
                if df[column].dtype in ['int64', 'float64']:
                    # Numeric column analysis
                    stats_analysis = {
                        "mean": df[column].mean(),
                        "std": df[column].std(),
                        "min": df[column].min(),
                        "max": df[column].max(),
                        "q1": df[column].quantile(0.25),
                        "q3": df[column].quantile(0.75),
                        "iqr": df[column].quantile(0.75) - df[column].quantile(0.25)
                    }
                    
                    # Generate intelligent ranges
                    lower_bound = max(stats_analysis["q1"] - 1.5 * stats_analysis["iqr"], stats_analysis["min"])
                    upper_bound = min(stats_analysis["q3"] + 1.5 * stats_analysis["iqr"], stats_analysis["max"])
                    
                    intelligent_rules[f"{column}_range"] = {
                        "type": "range",
                        "min": lower_bound,
                        "max": upper_bound,
                        "confidence": 0.95,
                        "method": "iqr_based"
                    }
                    
                    # Detect outliers
                    outliers = self.detect_anomalies_statistical(df, column)
                    if outliers.get("anomaly_count", 0) > 0:
                        intelligent_rules[f"{column}_outliers"] = {
                            "type": "outlier_detection",
                            "anomaly_count": outliers["anomaly_count"],
                            "anomaly_percentage": (outliers["anomaly_count"] / len(df)) * 100,
                            "method": "statistical"
                        }
                
                elif df[column].dtype == 'object':
                    # Categorical column analysis
                    value_counts = df[column].value_counts()
                    unique_count = len(value_counts)
                    
                    intelligent_rules[f"{column}_categorical"] = {
                        "type": "categorical",
                        "unique_values": unique_count,
                        "most_common": value_counts.index[0] if len(value_counts) > 0 else None,
                        "frequency": value_counts.iloc[0] if len(value_counts) > 0 else 0
                    }
            
            return {
                "table_name": table_name,
                "intelligent_rules": intelligent_rules,
                "total_rules": len(intelligent_rules),
                "generated_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error generating intelligent validation rules: {e}")
            return {"error": str(e)}
    
    def run_ai_enhanced_validation(self, table_name: str) -> Dict[str, Any]:
        """Run comprehensive AI-enhanced validation"""
        try:
            logger.info(f"Starting AI-enhanced validation for table: {table_name}")
            
            # Get data from database
            query = f"SELECT * FROM labor_analytics.{table_name}"
            df = self.db_connector.get_dataframe(query)
            
            if df.empty:
                return {"error": f"Table {table_name} is empty"}
            
            results = {
                "table_name": table_name,
                "timestamp": datetime.now().isoformat(),
                "total_records": len(df),
                "ai_analysis": {}
            }
            
            # 1. Anomaly Detection
            if 'salary' in df.columns:
                results["ai_analysis"]["salary_anomalies"] = self.detect_anomalies_statistical(df, 'salary')
            
            if 'hours_worked' in df.columns:
                results["ai_analysis"]["hours_anomalies"] = self.detect_anomalies_statistical(df, 'hours_worked')
            
            # 2. Predictive Validation
            if all(col in df.columns for col in ['salary', 'department_id', 'position']):
                results["ai_analysis"]["salary_prediction"] = self.predict_salary_anomalies(df)
            
            # 3. Clustering Analysis
            if all(col in df.columns for col in ['salary', 'hours_worked']):
                results["ai_analysis"]["employee_clustering"] = self.cluster_employees(df)
            
            # 4. Time Series Forecasting (for time_tracking table)
            if table_name == 'time_tracking' and 'date' in df.columns and 'hours_worked' in df.columns:
                results["ai_analysis"]["time_series_forecast"] = self.forecast_time_series(df, 'date', 'hours_worked')
            
            # 5. Intelligent Validation Rules
            results["ai_analysis"]["intelligent_rules"] = self.intelligent_validation_rules(df, table_name)
            
            # 6. Overall AI Score
            ai_score = self._calculate_ai_score(results["ai_analysis"])
            results["ai_score"] = ai_score
            
            logger.info(f"AI-enhanced validation completed for {table_name}. AI Score: {ai_score:.2f}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in AI-enhanced validation: {e}")
            return {"error": str(e)}
    
    def _calculate_ai_score(self, ai_analysis: Dict[str, Any]) -> float:
        """Calculate overall AI confidence score"""
        try:
            score = 0.0
            factors = 0
            
            # Anomaly detection score
            if "salary_anomalies" in ai_analysis:
                anomaly_percentage = ai_analysis["salary_anomalies"].get("anomaly_percentage", 0)
                score += max(0, 100 - anomaly_percentage)  # Lower anomalies = higher score
                factors += 1
            
            # Prediction accuracy score
            if "salary_prediction" in ai_analysis:
                residuals = ai_analysis["salary_prediction"].get("residuals", [])
                if len(residuals) > 0:
                    avg_residual = np.mean(residuals)
                    score += max(0, 100 - avg_residual)
                    factors += 1
            
            # Clustering quality score
            if "employee_clustering" in ai_analysis:
                cluster_analysis = ai_analysis["employee_clustering"].get("cluster_analysis", {})
                if cluster_analysis:
                    # Calculate cluster balance
                    cluster_sizes = [info["size"] for info in cluster_analysis.values()]
                    balance_score = 100 - (max(cluster_sizes) - min(cluster_sizes)) / sum(cluster_sizes) * 100
                    score += balance_score
                    factors += 1
            
            return score / factors if factors > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating AI score: {e}")
            return 0.0
    
    def save_models(self, output_dir: str = "../models"):
        """Save trained models for future use"""
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            for model_name, model in self.models.items():
                model_path = os.path.join(output_dir, f"{model_name}.pkl")
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
            
            logger.info(f"Models saved to {output_dir}")
            
        except Exception as e:
            logger.error(f"Error saving models: {e}")
    
    def load_models(self, models_dir: str = "../models"):
        """Load previously trained models"""
        try:
            if os.path.exists(models_dir):
                for model_file in os.listdir(models_dir):
                    if model_file.endswith('.pkl'):
                        model_name = model_file.replace('.pkl', '')
                        model_path = os.path.join(models_dir, model_file)
                        
                        with open(model_path, 'rb') as f:
                            self.models[model_name] = pickle.load(f)
                
                logger.info(f"Loaded {len(self.models)} models from {models_dir}")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")


if __name__ == "__main__":
    # Test AI-enhanced validator
    ai_validator = AIEnhancedValidator()
    
    # Test with sample data
    sample_data = pd.DataFrame({
        'employee_id': ['EMP001', 'EMP002', 'EMP003', 'EMP004', 'EMP005'],
        'salary': [75000, 85000, 15000, 250000, 65000],  # Includes anomalies
        'hours_worked': [40, 45, 20, 60, 35],
        'department_id': [1, 2, 1, 3, 2],
        'position': ['Manager', 'Developer', 'Intern', 'CEO', 'Analyst']
    })
    
    # Run AI-enhanced validation
    results = ai_validator.run_ai_enhanced_validation('employees')
    
    print("AI-Enhanced Validation Results:")
    print(f"AI Score: {results.get('ai_score', 0):.2f}")
    print(f"Total Records: {results.get('total_records', 0)}")
    
    if 'ai_analysis' in results:
        for analysis_type, analysis_result in results['ai_analysis'].items():
            print(f"\n{analysis_type}:")
            if 'anomaly_count' in analysis_result:
                print(f"  Anomalies: {analysis_result['anomaly_count']}")
            if 'cluster_analysis' in analysis_result:
                print(f"  Clusters: {len(analysis_result['cluster_analysis'])}")
