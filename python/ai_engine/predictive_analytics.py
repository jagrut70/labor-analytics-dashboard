"""
Predictive Analytics Engine
Advanced ML models for labor forecasting and optimization
"""

import pandas as pd
import numpy as np
import json
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVR, SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder, PolynomialFeatures
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, classification_report, confusion_matrix
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
import xgboost as xgb
from lightgbm import LGBMRegressor, LGBMClassifier

# Time Series
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm

# Deep Learning
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Conv1D, MaxPooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Optimization
from scipy.optimize import minimize
import optuna

# Custom imports
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_connectors.database_connector import DatabaseConnector
import structlog

logger = structlog.get_logger(__name__)


class PredictiveAnalytics:
    """Advanced predictive analytics for labor optimization"""
    
    def __init__(self, config_path: str = "../config/ai_config.json"):
        """Initialize predictive analytics engine"""
        self.config = self._load_config(config_path)
        self.db_connector = DatabaseConnector()
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self._setup_logging()
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load predictive analytics configuration"""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            return config.get('predictive_validation', {})
        except FileNotFoundError:
            return {
                "salary_prediction": {"model_type": "xgboost", "features": ["department_id", "position", "experience_years"]},
                "hours_prediction": {"model_type": "lightgbm", "features": ["employee_id", "project_id", "day_of_week"]},
                "overtime_prediction": {"model_type": "random_forest", "features": ["hours_worked", "project_id"]}
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
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create advanced features for predictive modeling"""
        try:
            df_features = df.copy()
            
            # Temporal features
            if 'date' in df_features.columns:
                df_features['date'] = pd.to_datetime(df_features['date'])
                df_features['day_of_week'] = df_features['date'].dt.dayofweek
                df_features['month'] = df_features['date'].dt.month
                df_features['quarter'] = df_features['date'].dt.quarter
                df_features['year'] = df_features['date'].dt.year
                df_features['is_weekend'] = df_features['day_of_week'].isin([5, 6]).astype(int)
                df_features['is_month_end'] = df_features['date'].dt.is_month_end.astype(int)
            
            # Experience features
            if 'hire_date' in df_features.columns:
                df_features['hire_date'] = pd.to_datetime(df_features['hire_date'])
                df_features['experience_years'] = (
                    pd.Timestamp.now() - df_features['hire_date']
                ).dt.days / 365.25
                df_features['experience_months'] = (
                    pd.Timestamp.now() - df_features['hire_date']
                ).dt.days / 30.44
            
            # Salary features
            if 'salary' in df_features.columns and 'department_id' in df_features.columns:
                dept_stats = df_features.groupby('department_id')['salary'].agg(['mean', 'std', 'min', 'max'])
                df_features['salary_vs_dept_mean'] = df_features['salary'] - df_features['department_id'].map(dept_stats['mean'])
                df_features['salary_percentile'] = df_features.groupby('department_id')['salary'].rank(pct=True)
            
            # Hours features
            if 'hours_worked' in df_features.columns:
                df_features['hours_category'] = pd.cut(
                    df_features['hours_worked'], 
                    bins=[0, 4, 8, 12, 24], 
                    labels=['part_time', 'full_time', 'overtime', 'extended']
                )
                df_features['is_overtime'] = (df_features['hours_worked'] > 8).astype(int)
            
            # Project features
            if 'project_id' in df_features.columns:
                project_stats = df_features.groupby('project_id')['hours_worked'].agg(['mean', 'sum', 'count'])
                df_features['project_avg_hours'] = df_features['project_id'].map(project_stats['mean'])
                df_features['project_total_hours'] = df_features['project_id'].map(project_stats['sum'])
                df_features['project_activity'] = df_features['project_id'].map(project_stats['count'])
            
            # Employee features
            if 'employee_id' in df_features.columns:
                emp_stats = df_features.groupby('employee_id').agg({
                    'hours_worked': ['mean', 'std', 'sum'],
                    'overtime_hours': ['mean', 'sum']
                }).fillna(0)
                
                df_features['emp_avg_hours'] = df_features['employee_id'].map(emp_stats[('hours_worked', 'mean')])
                df_features['emp_hours_std'] = df_features['employee_id'].map(emp_stats[('hours_worked', 'std')])
                df_features['emp_total_hours'] = df_features['employee_id'].map(emp_stats[('hours_worked', 'sum')])
                df_features['emp_avg_overtime'] = df_features['employee_id'].map(emp_stats[('overtime_hours', 'mean')])
            
            # Lag features for time series
            if 'date' in df_features.columns and 'employee_id' in df_features.columns:
                df_features = df_features.sort_values(['employee_id', 'date'])
                df_features['hours_lag_1'] = df_features.groupby('employee_id')['hours_worked'].shift(1)
                df_features['hours_lag_7'] = df_features.groupby('employee_id')['hours_worked'].shift(7)
                df_features['hours_rolling_mean_7'] = df_features.groupby('employee_id')['hours_worked'].rolling(7).mean().reset_index(0, drop=True)
            
            return df_features
            
        except Exception as e:
            logger.error(f"Error creating features: {e}")
            return df
    
    def predict_salary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Predict employee salaries using advanced ML models"""
        try:
            # Create features
            df_features = self.create_features(df)
            
            # Prepare features and target
            feature_config = self.config.get('salary_prediction', {})
            features = feature_config.get('features', ['department_id', 'position', 'experience_years'])
            target = 'salary'
            
            # Ensure all features exist
            available_features = [f for f in features if f in df_features.columns]
            if not available_features:
                return {"error": "No valid features found for salary prediction"}
            
            X = df_features[available_features].fillna(0)
            y = df_features[target]
            
            # Encode categorical variables
            for col in X.columns:
                if X[col].dtype == 'object':
                    le = LabelEncoder()
                    X[col] = le.fit_transform(X[col].astype(str))
                    self.encoders[f'salary_{col}'] = le
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            self.scalers['salary'] = scaler
            
            # Train model
            model_type = feature_config.get('model_type', 'xgboost')
            
            if model_type == 'xgboost':
                model = xgb.XGBRegressor(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=6,
                    random_state=42
                )
            elif model_type == 'lightgbm':
                model = LGBMRegressor(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=6,
                    random_state=42
                )
            elif model_type == 'random_forest':
                model = RandomForestRegressor(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42
                )
            else:
                model = LinearRegression()
            
            # Train model
            model.fit(X_train_scaled, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test_scaled)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)
            
            # Feature importance
            if hasattr(model, 'feature_importances_'):
                feature_importance = dict(zip(available_features, model.feature_importances_))
            else:
                feature_importance = dict(zip(available_features, [0] * len(available_features)))
            
            # Store model
            self.models['salary_prediction'] = model
            
            return {
                "model_type": model_type,
                "features": available_features,
                "metrics": {
                    "mse": mse,
                    "rmse": rmse,
                    "r2": r2,
                    "mae": np.mean(np.abs(y_test - y_pred))
                },
                "feature_importance": feature_importance,
                "predictions": y_pred.tolist(),
                "actual": y_test.tolist()
            }
            
        except Exception as e:
            logger.error(f"Error in salary prediction: {e}")
            return {"error": str(e)}
    
    def predict_hours_worked(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Predict hours worked using time series and ML models"""
        try:
            # Create features
            df_features = self.create_features(df)
            
            # Prepare features and target
            feature_config = self.config.get('hours_prediction', {})
            features = feature_config.get('features', ['employee_id', 'project_id', 'day_of_week'])
            target = 'hours_worked'
            
            # Ensure all features exist
            available_features = [f for f in features if f in df_features.columns]
            if not available_features:
                return {"error": "No valid features found for hours prediction"}
            
            X = df_features[available_features].fillna(0)
            y = df_features[target]
            
            # Encode categorical variables
            for col in X.columns:
                if X[col].dtype == 'object':
                    le = LabelEncoder()
                    X[col] = le.fit_transform(X[col].astype(str))
                    self.encoders[f'hours_{col}'] = le
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            self.scalers['hours'] = scaler
            
            # Train model
            model_type = feature_config.get('model_type', 'lightgbm')
            
            if model_type == 'lightgbm':
                model = LGBMRegressor(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=6,
                    random_state=42
                )
            elif model_type == 'xgboost':
                model = xgb.XGBRegressor(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=6,
                    random_state=42
                )
            else:
                model = RandomForestRegressor(n_estimators=100, random_state=42)
            
            # Train model
            model.fit(X_train_scaled, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test_scaled)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)
            
            # Store model
            self.models['hours_prediction'] = model
            
            return {
                "model_type": model_type,
                "features": available_features,
                "metrics": {
                    "mse": mse,
                    "rmse": rmse,
                    "r2": r2,
                    "mae": np.mean(np.abs(y_test - y_pred))
                },
                "predictions": y_pred.tolist(),
                "actual": y_test.tolist()
            }
            
        except Exception as e:
            logger.error(f"Error in hours prediction: {e}")
            return {"error": str(e)}
    
    def forecast_labor_demand(self, df: pd.DataFrame, periods: int = 30) -> Dict[str, Any]:
        """Forecast labor demand using time series analysis"""
        try:
            if 'date' not in df.columns or 'hours_worked' not in df.columns:
                return {"error": "Date and hours_worked columns required for forecasting"}
            
            # Prepare time series data
            df_ts = df.copy()
            df_ts['date'] = pd.to_datetime(df_ts['date'])
            df_ts = df_ts.set_index('date')
            
            # Aggregate by date
            daily_hours = df_ts['hours_worked'].resample('D').sum().fillna(0)
            
            # Check stationarity
            adf_result = adfuller(daily_hours)
            is_stationary = adf_result[1] < 0.05
            
            # Seasonal decomposition
            decomposition = seasonal_decompose(daily_hours, period=7, extrapolate_trend='freq')
            
            # ARIMA model
            try:
                # Determine ARIMA parameters
                if is_stationary:
                    arima_order = (1, 0, 1)
                else:
                    arima_order = (1, 1, 1)
                
                arima_model = ARIMA(daily_hours, order=arima_order)
                arima_fitted = arima_model.fit()
                arima_forecast = arima_fitted.forecast(steps=periods)
                
            except Exception as e:
                logger.warning(f"ARIMA forecast failed: {e}")
                arima_forecast = None
            
            # LSTM model for time series
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
                scaled_data = scaler.fit_transform(daily_hours.values.reshape(-1, 1))
                
                # Create sequences
                seq_length = 7
                X, y = create_sequences(scaled_data, seq_length)
                
                if len(X) > 0:
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
                    
                    for _ in range(periods):
                        next_pred = model.predict(last_sequence.reshape(1, seq_length, 1))
                        lstm_forecast.append(next_pred[0, 0])
                        last_sequence = np.roll(last_sequence, -1)
                        last_sequence[-1] = next_pred[0, 0]
                    
                    lstm_forecast = scaler.inverse_transform(np.array(lstm_forecast).reshape(-1, 1))
                else:
                    lstm_forecast = None
                    
            except Exception as e:
                logger.warning(f"LSTM forecast failed: {e}")
                lstm_forecast = None
            
            return {
                "forecast_periods": periods,
                "is_stationary": is_stationary,
                "arima_forecast": arima_forecast.tolist() if arima_forecast is not None else None,
                "lstm_forecast": lstm_forecast.flatten().tolist() if lstm_forecast is not None else None,
                "decomposition": {
                    "trend": decomposition.trend.tolist(),
                    "seasonal": decomposition.seasonal.tolist(),
                    "residual": decomposition.resid.tolist()
                },
                "historical_data": daily_hours.tolist()
            }
            
        except Exception as e:
            logger.error(f"Error in labor demand forecasting: {e}")
            return {"error": str(e)}
    
    def optimize_resource_allocation(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Optimize resource allocation using mathematical optimization"""
        try:
            # Prepare data
            if 'employee_id' not in df.columns or 'project_id' not in df.columns or 'hours_worked' not in df.columns:
                return {"error": "Required columns missing for optimization"}
            
            # Create employee-project matrix
            allocation_matrix = df.pivot_table(
                index='employee_id', 
                columns='project_id', 
                values='hours_worked', 
                aggfunc='sum'
            ).fillna(0)
            
            # Calculate optimization parameters
            employees = allocation_matrix.index.tolist()
            projects = allocation_matrix.columns.tolist()
            
            # Employee capacity (total hours available)
            employee_capacity = allocation_matrix.sum(axis=1).to_dict()
            
            # Project requirements (total hours needed)
            project_requirements = allocation_matrix.sum(axis=0).to_dict()
            
            # Employee efficiency (hours per project)
            employee_efficiency = allocation_matrix.to_dict()
            
            # Optimization objective: maximize total efficiency
            def objective(x):
                total_efficiency = 0
                for i, emp in enumerate(employees):
                    for j, proj in enumerate(projects):
                        idx = i * len(projects) + j
                        total_efficiency += x[idx] * employee_efficiency.get((emp, proj), 0)
                return -total_efficiency  # Minimize negative efficiency (maximize efficiency)
            
            # Constraints
            constraints = []
            
            # Employee capacity constraints
            for i, emp in enumerate(employees):
                def capacity_constraint(x, emp_idx=i):
                    return employee_capacity[employees[emp_idx]] - sum(x[emp_idx * len(projects) + j] for j in range(len(projects)))
                constraints.append({'type': 'ineq', 'fun': capacity_constraint})
            
            # Project requirement constraints
            for j, proj in enumerate(projects):
                def requirement_constraint(x, proj_idx=j):
                    return sum(x[i * len(projects) + proj_idx] for i in range(len(employees))) - project_requirements[projects[proj_idx]]
                constraints.append({'type': 'ineq', 'fun': requirement_constraint})
            
            # Non-negativity constraints
            bounds = [(0, None)] * (len(employees) * len(projects))
            
            # Initial guess
            x0 = np.ones(len(employees) * len(projects)) * 0.1
            
            # Solve optimization
            result = minimize(
                objective, 
                x0, 
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000}
            )
            
            if result.success:
                # Reshape solution
                optimal_allocation = result.x.reshape(len(employees), len(projects))
                
                # Create results
                allocation_results = {}
                for i, emp in enumerate(employees):
                    allocation_results[emp] = {}
                    for j, proj in enumerate(projects):
                        allocation_results[emp][proj] = optimal_allocation[i, j]
                
                return {
                    "optimization_success": True,
                    "optimal_allocation": allocation_results,
                    "total_efficiency": -result.fun,
                    "employee_utilization": {
                        emp: sum(allocation_results[emp].values()) / employee_capacity[emp]
                        for emp in employees
                    },
                    "project_coverage": {
                        proj: sum(allocation_results[emp][proj] for emp in employees) / project_requirements[proj]
                        for proj in projects
                    }
                }
            else:
                return {
                    "optimization_success": False,
                    "error": "Optimization failed to converge",
                    "message": result.message
                }
            
        except Exception as e:
            logger.error(f"Error in resource optimization: {e}")
            return {"error": str(e)}
    
    def predict_employee_performance(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Predict employee performance using ML models"""
        try:
            # Create features
            df_features = self.create_features(df)
            
            # Define performance metrics
            if 'hours_worked' in df_features.columns and 'overtime_hours' in df_features.columns:
                # Performance score based on efficiency and overtime
                df_features['performance_score'] = (
                    df_features['hours_worked'] / 8.0 * 0.7 +  # Efficiency
                    (1 - df_features['overtime_hours'] / df_features['hours_worked']) * 0.3  # Overtime efficiency
                ).fillna(0)
                
                # Categorize performance
                df_features['performance_category'] = pd.cut(
                    df_features['performance_score'],
                    bins=[0, 0.6, 0.8, 1.0],
                    labels=['low', 'medium', 'high']
                )
                
                target = 'performance_category'
                problem_type = 'classification'
            else:
                return {"error": "Insufficient data for performance prediction"}
            
            # Prepare features
            features = [
                'experience_years', 'salary_vs_dept_mean', 'emp_avg_hours',
                'emp_hours_std', 'emp_avg_overtime', 'day_of_week', 'is_weekend'
            ]
            
            available_features = [f for f in features if f in df_features.columns]
            if not available_features:
                return {"error": "No valid features found for performance prediction"}
            
            X = df_features[available_features].fillna(0)
            y = df_features[target]
            
            # Encode target
            le = LabelEncoder()
            y_encoded = le.fit_transform(y)
            self.encoders['performance_target'] = le
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            self.scalers['performance'] = scaler
            
            # Train model
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train_scaled, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)
            
            # Calculate metrics
            classification_rep = classification_report(y_test, y_pred, output_dict=True)
            conf_matrix = confusion_matrix(y_test, y_pred)
            
            # Feature importance
            feature_importance = dict(zip(available_features, model.feature_importances_))
            
            # Store model
            self.models['performance_prediction'] = model
            
            return {
                "problem_type": problem_type,
                "features": available_features,
                "metrics": {
                    "accuracy": classification_rep['accuracy'],
                    "precision": classification_rep['weighted avg']['precision'],
                    "recall": classification_rep['weighted avg']['recall'],
                    "f1_score": classification_rep['weighted avg']['f1-score']
                },
                "confusion_matrix": conf_matrix.tolist(),
                "feature_importance": feature_importance,
                "predictions": y_pred.tolist(),
                "prediction_probabilities": y_pred_proba.tolist(),
                "actual": y_test.tolist()
            }
            
        except Exception as e:
            logger.error(f"Error in performance prediction: {e}")
            return {"error": str(e)}
    
    def generate_predictive_report(self, results: Dict[str, Any]) -> str:
        """Generate comprehensive predictive analytics report"""
        try:
            report = []
            report.append("=" * 60)
            report.append("PREDICTIVE ANALYTICS REPORT")
            report.append("=" * 60)
            report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            report.append("")
            
            if "error" in results:
                report.append(f"❌ Error: {results['error']}")
                return "\n".join(report)
            
            # Model performance
            if "metrics" in results:
                report.append("📊 MODEL PERFORMANCE")
                report.append("-" * 30)
                for metric, value in results["metrics"].items():
                    report.append(f"  {metric.replace('_', ' ').title()}: {value:.4f}")
            
            # Feature importance
            if "feature_importance" in results:
                report.append("\n🎯 FEATURE IMPORTANCE")
                report.append("-" * 30)
                sorted_features = sorted(
                    results["feature_importance"].items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )
                for feature, importance in sorted_features[:5]:
                    report.append(f"  {feature}: {importance:.4f}")
            
            # Predictions summary
            if "predictions" in results:
                report.append("\n🔮 PREDICTIONS SUMMARY")
                report.append("-" * 30)
                predictions = results["predictions"]
                if isinstance(predictions[0], (int, float)):
                    report.append(f"  Average Prediction: {np.mean(predictions):.2f}")
                    report.append(f"  Prediction Range: {np.min(predictions):.2f} - {np.max(predictions):.2f}")
                else:
                    # Classification results
                    unique_preds, counts = np.unique(predictions, return_counts=True)
                    for pred, count in zip(unique_preds, counts):
                        percentage = (count / len(predictions)) * 100
                        report.append(f"  {pred}: {count} ({percentage:.1f}%)")
            
            return "\n".join(report)
            
        except Exception as e:
            logger.error(f"Error generating predictive report: {e}")
            return f"Error generating report: {str(e)}"


if __name__ == "__main__":
    # Test predictive analytics
    predictor = PredictiveAnalytics()
    
    # Create sample data
    sample_data = pd.DataFrame({
        'employee_id': ['EMP001', 'EMP002', 'EMP003', 'EMP004', 'EMP005'] * 10,
        'project_id': ['PRJ001', 'PRJ002', 'PRJ003'] * 17,
        'date': pd.date_range('2023-01-01', periods=50, freq='D'),
        'hours_worked': np.random.normal(8, 2, 50),
        'overtime_hours': np.random.exponential(1, 50),
        'salary': np.random.normal(75000, 15000, 50),
        'department_id': np.random.randint(1, 6, 50),
        'position': np.random.choice(['Developer', 'Manager', 'Analyst'], 50),
        'hire_date': pd.date_range('2020-01-01', periods=50, freq='D')
    })
    
    print("Predictive Analytics Test Results:")
    print("=" * 50)
    
    # Test salary prediction
    salary_results = predictor.predict_salary(sample_data)
    if "error" not in salary_results:
        print(f"\n💰 Salary Prediction:")
        print(f"  R² Score: {salary_results['metrics']['r2']:.4f}")
        print(f"  RMSE: {salary_results['metrics']['rmse']:.2f}")
    
    # Test hours prediction
    hours_results = predictor.predict_hours_worked(sample_data)
    if "error" not in hours_results:
        print(f"\n⏰ Hours Prediction:")
        print(f"  R² Score: {hours_results['metrics']['r2']:.4f}")
        print(f"  RMSE: {hours_results['metrics']['rmse']:.2f}")
    
    # Test performance prediction
    performance_results = predictor.predict_employee_performance(sample_data)
    if "error" not in performance_results:
        print(f"\n📈 Performance Prediction:")
        print(f"  Accuracy: {performance_results['metrics']['accuracy']:.4f}")
        print(f"  F1 Score: {performance_results['metrics']['f1_score']:.4f}")
    
    # Test labor demand forecasting
    forecast_results = predictor.forecast_labor_demand(sample_data, periods=7)
    if "error" not in forecast_results:
        print(f"\n📊 Labor Demand Forecast:")
        print(f"  Forecast Periods: {forecast_results['forecast_periods']}")
        print(f"  Stationary: {forecast_results['is_stationary']}")
    
    # Test resource optimization
    optimization_results = predictor.optimize_resource_allocation(sample_data)
    if "error" not in optimization_results:
        print(f"\n⚙️  Resource Optimization:")
        print(f"  Success: {optimization_results['optimization_success']}")
        if optimization_results['optimization_success']:
            print(f"  Total Efficiency: {optimization_results['total_efficiency']:.2f}")
