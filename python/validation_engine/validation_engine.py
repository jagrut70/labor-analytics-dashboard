"""
Main Validation Engine for Labor Analytics
Orchestrates bi-directional data validation between Power BI, PostgreSQL, and Excel rules
"""

import pandas as pd
import json
import schedule
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import structlog
from pathlib import Path
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_connectors.database_connector import DatabaseConnector
from data_connectors.excel_validator import ExcelValidator

logger = structlog.get_logger(__name__)


class ValidationEngine:
    """Main validation engine for bi-directional data validation"""
    
    def __init__(self, config_path: str = "../config/validation_config.json"):
        """Initialize validation engine with configuration"""
        self.config = self._load_config(config_path)
        self.db_connector = DatabaseConnector()
        self.excel_validator = ExcelValidator()
        self.validation_history = []
        self._setup_logging()
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load validation configuration"""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            logger.info("Validation configuration loaded successfully")
            return config
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {config_path}")
            raise
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON in configuration file: {config_path}")
            raise
    
    def _setup_logging(self):
        """Setup structured logging"""
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
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
    
    def run_full_validation(self) -> Dict[str, Any]:
        """Run comprehensive validation across all data sources"""
        logger.info("Starting full validation cycle")
        
        validation_results = {
            'timestamp': datetime.now().isoformat(),
            'overall_status': 'PASS',
            'tables_validated': [],
            'cross_validation_results': [],
            'data_quality_metrics': {},
            'errors': [],
            'warnings': []
        }
        
        try:
            # Validate each table
            tables_to_validate = ['employees', 'time_tracking', 'payroll', 'departments', 'projects']
            
            for table_name in tables_to_validate:
                table_result = self._validate_table(table_name)
                validation_results['tables_validated'].append(table_result)
                
                if not table_result['valid']:
                    validation_results['overall_status'] = 'FAIL'
                    validation_results['errors'].extend(table_result['errors'])
            
            # Run cross-validation rules
            cross_validation_results = self._run_cross_validation()
            validation_results['cross_validation_results'] = cross_validation_results
            
            if any(not result['valid'] for result in cross_validation_results):
                validation_results['overall_status'] = 'FAIL'
            
            # Calculate data quality metrics
            quality_metrics = self._calculate_data_quality_metrics()
            validation_results['data_quality_metrics'] = quality_metrics
            
            # Store validation results in database
            self._store_validation_results(validation_results)
            
            # Send notifications if needed
            self._send_notifications(validation_results)
            
            # Add to history
            self.validation_history.append(validation_results)
            
            logger.info(f"Full validation completed. Status: {validation_results['overall_status']}")
            
        except Exception as e:
            logger.error(f"Error during full validation: {e}")
            validation_results['overall_status'] = 'ERROR'
            validation_results['errors'].append(str(e))
        
        return validation_results
    
    def _validate_table(self, table_name: str) -> Dict[str, Any]:
        """Validate a specific table"""
        logger.info(f"Validating table: {table_name}")
        
        try:
            # Get data from database
            query = f"SELECT * FROM labor_analytics.{table_name}"
            df = self.db_connector.get_dataframe(query)
            
            if df.empty:
                logger.warning(f"Table {table_name} is empty")
                return {
                    'table_name': table_name,
                    'valid': True,
                    'total_records': 0,
                    'valid_records': 0,
                    'invalid_records': 0,
                    'errors': [],
                    'warnings': ['Table is empty']
                }
            
            # Validate with Excel rules
            excel_validation = self.excel_validator.validate_dataframe(df, table_name)
            
            # Validate with database constraints
            db_validation = self._validate_database_constraints(df, table_name)
            
            # Combine results
            combined_errors = excel_validation['errors'] + db_validation['errors']
            combined_warnings = excel_validation['warnings'] + db_validation['warnings']
            
            is_valid = excel_validation['valid'] and db_validation['valid']
            
            result = {
                'table_name': table_name,
                'valid': is_valid,
                'total_records': len(df),
                'valid_records': len(df) - len(set([error['record_index'] for error in combined_errors])),
                'invalid_records': len(set([error['record_index'] for error in combined_errors])),
                'errors': combined_errors,
                'warnings': combined_warnings,
                'excel_validation': excel_validation,
                'database_validation': db_validation
            }
            
            logger.info(f"Table {table_name} validation completed. Valid: {is_valid}")
            return result
            
        except Exception as e:
            logger.error(f"Error validating table {table_name}: {e}")
            return {
                'table_name': table_name,
                'valid': False,
                'total_records': 0,
                'valid_records': 0,
                'invalid_records': 0,
                'errors': [str(e)],
                'warnings': []
            }
    
    def _validate_database_constraints(self, df: pd.DataFrame, table_name: str) -> Dict[str, Any]:
        """Validate data against database constraints"""
        errors = []
        warnings = []
        
        try:
            # Get table schema
            schema_df = self.db_connector.get_table_schema(table_name)
            
            for _, column in schema_df.iterrows():
                column_name = column['column_name']
                data_type = column['data_type']
                is_nullable = column['is_nullable'] == 'YES'
                
                if column_name in df.columns:
                    # Check for null values in non-nullable columns
                    if not is_nullable and df[column_name].isnull().any():
                        null_indices = df[df[column_name].isnull()].index.tolist()
                        for idx in null_indices:
                            errors.append({
                                'record_index': idx,
                                'field_name': column_name,
                                'rule_name': 'database_not_null',
                                'error_message': f'Column {column_name} cannot be null',
                                'value': None
                            })
                    
                    # Check data types
                    if data_type in ['integer', 'bigint', 'smallint']:
                        non_numeric = df[~df[column_name].apply(lambda x: pd.isna(x) or isinstance(x, (int, float)))]
                        for idx in non_numeric.index:
                            errors.append({
                                'record_index': idx,
                                'field_name': column_name,
                                'rule_name': 'database_data_type',
                                'error_message': f'Column {column_name} must be numeric',
                                'value': df.loc[idx, column_name]
                            })
            
            return {
                'valid': len(errors) == 0,
                'errors': errors,
                'warnings': warnings
            }
            
        except Exception as e:
            logger.error(f"Error validating database constraints for {table_name}: {e}")
            return {
                'valid': False,
                'errors': [str(e)],
                'warnings': []
            }
    
    def _run_cross_validation(self) -> List[Dict[str, Any]]:
        """Run cross-validation rules between tables"""
        logger.info("Running cross-validation rules")
        
        cross_validation_results = []
        
        try:
            # Get cross-validation rules from config
            cross_rules = self.config.get('cross_validation_rules', [])
            
            for rule in cross_rules:
                rule_name = rule['name']
                rule_description = rule['description']
                formula = rule['formula']
                tolerance = rule.get('tolerance', 0)
                
                logger.info(f"Running cross-validation rule: {rule_name}")
                
                try:
                    # Execute the cross-validation rule
                    result = self._execute_cross_validation_rule(rule_name, formula, tolerance)
                    result['rule_name'] = rule_name
                    result['description'] = rule_description
                    
                    cross_validation_results.append(result)
                    
                except Exception as e:
                    logger.error(f"Error executing cross-validation rule {rule_name}: {e}")
                    cross_validation_results.append({
                        'rule_name': rule_name,
                        'description': rule_description,
                        'valid': False,
                        'error': str(e)
                    })
        
        except Exception as e:
            logger.error(f"Error running cross-validation: {e}")
        
        return cross_validation_results
    
    def _execute_cross_validation_rule(self, rule_name: str, formula: str, tolerance: float) -> Dict[str, Any]:
        """Execute a specific cross-validation rule"""
        if rule_name == 'salary_hours_consistency':
            return self._validate_salary_hours_consistency(tolerance)
        elif rule_name == 'overtime_calculation':
            return self._validate_overtime_calculation(tolerance)
        else:
            raise ValueError(f"Unknown cross-validation rule: {rule_name}")
    
    def _validate_salary_hours_consistency(self, tolerance: float) -> Dict[str, Any]:
        """Validate salary consistency with hours worked and hourly rate"""
        try:
            query = """
            SELECT 
                e.employee_id,
                e.salary,
                e.hourly_rate,
                COALESCE(SUM(tt.hours_worked), 0) as total_hours_worked
            FROM labor_analytics.employees e
            LEFT JOIN labor_analytics.time_tracking tt ON e.employee_id = tt.employee_id
            WHERE e.hourly_rate IS NOT NULL
            GROUP BY e.employee_id, e.salary, e.hourly_rate
            """
            
            df = self.db_connector.get_dataframe(query)
            
            errors = []
            for idx, row in df.iterrows():
                calculated_salary = row['hourly_rate'] * row['total_hours_worked']
                difference = abs(row['salary'] - calculated_salary)
                
                if difference > tolerance:
                    errors.append({
                        'employee_id': row['employee_id'],
                        'actual_salary': row['salary'],
                        'calculated_salary': calculated_salary,
                        'difference': difference,
                        'tolerance': tolerance
                    })
            
            return {
                'valid': len(errors) == 0,
                'errors': errors,
                'total_records': len(df),
                'invalid_records': len(errors)
            }
            
        except Exception as e:
            logger.error(f"Error validating salary-hours consistency: {e}")
            return {
                'valid': False,
                'error': str(e)
            }
    
    def _validate_overtime_calculation(self, tolerance: float) -> Dict[str, Any]:
        """Validate overtime hours calculation"""
        try:
            query = """
            SELECT 
                employee_id,
                date,
                hours_worked,
                overtime_hours,
                CASE 
                    WHEN hours_worked > 8 THEN hours_worked - 8 
                    ELSE 0 
                END as calculated_overtime
            FROM labor_analytics.time_tracking
            """
            
            df = self.db_connector.get_dataframe(query)
            
            errors = []
            for idx, row in df.iterrows():
                difference = abs(row['overtime_hours'] - row['calculated_overtime'])
                
                if difference > tolerance:
                    errors.append({
                        'employee_id': row['employee_id'],
                        'date': row['date'],
                        'actual_overtime': row['overtime_hours'],
                        'calculated_overtime': row['calculated_overtime'],
                        'difference': difference,
                        'tolerance': tolerance
                    })
            
            return {
                'valid': len(errors) == 0,
                'errors': errors,
                'total_records': len(df),
                'invalid_records': len(errors)
            }
            
        except Exception as e:
            logger.error(f"Error validating overtime calculation: {e}")
            return {
                'valid': False,
                'error': str(e)
            }
    
    def _calculate_data_quality_metrics(self) -> Dict[str, Any]:
        """Calculate overall data quality metrics"""
        try:
            metrics = {}
            thresholds = self.config.get('data_quality_thresholds', {})
            
            # Calculate completeness
            completeness_scores = []
            for table_name in ['employees', 'time_tracking', 'payroll']:
                try:
                    schema_df = self.db_connector.get_table_schema(table_name)
                    total_fields = len(schema_df)
                    
                    query = f"SELECT * FROM labor_analytics.{table_name}"
                    df = self.db_connector.get_dataframe(query)
                    
                    if not df.empty:
                        non_null_fields = df.count().sum()
                        total_possible_fields = len(df) * total_fields
                        completeness = non_null_fields / total_possible_fields if total_possible_fields > 0 else 0
                        completeness_scores.append(completeness)
                except Exception as e:
                    logger.warning(f"Error calculating completeness for {table_name}: {e}")
            
            metrics['completeness'] = sum(completeness_scores) / len(completeness_scores) if completeness_scores else 0
            
            # Calculate accuracy (based on validation results)
            total_errors = sum(len(result.get('errors', [])) for result in self.validation_history[-1]['tables_validated']) if self.validation_history else 0
            total_records = sum(result.get('total_records', 0) for result in self.validation_history[-1]['tables_validated']) if self.validation_history else 1
            metrics['accuracy'] = 1 - (total_errors / total_records) if total_records > 0 else 1
            
            # Calculate consistency (based on cross-validation results)
            cross_validation_failures = sum(1 for result in self.validation_history[-1].get('cross_validation_results', []) if not result.get('valid', True)) if self.validation_history else 0
            total_cross_validations = len(self.validation_history[-1].get('cross_validation_results', [])) if self.validation_history else 1
            metrics['consistency'] = 1 - (cross_validation_failures / total_cross_validations) if total_cross_validations > 0 else 1
            
            # Calculate timeliness (based on data freshness)
            try:
                query = "SELECT MAX(updated_at) as latest_update FROM labor_analytics.employees"
                result = self.db_connector.execute_query(query)
                if result and result[0][0]:
                    latest_update = result[0][0]
                    time_diff = datetime.now() - latest_update
                    timeliness = max(0, 1 - (time_diff.days / 30))  # Assume 30 days is the threshold
                    metrics['timeliness'] = timeliness
                else:
                    metrics['timeliness'] = 0
            except Exception as e:
                logger.warning(f"Error calculating timeliness: {e}")
                metrics['timeliness'] = 0
            
            # Determine status for each metric
            for metric_name, value in metrics.items():
                threshold = thresholds.get(metric_name, 0.9)
                if value >= threshold:
                    metrics[f'{metric_name}_status'] = 'GOOD'
                elif value >= threshold * 0.8:
                    metrics[f'{metric_name}_status'] = 'WARNING'
                else:
                    metrics[f'{metric_name}_status'] = 'CRITICAL'
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating data quality metrics: {e}")
            return {
                'completeness': 0,
                'accuracy': 0,
                'consistency': 0,
                'timeliness': 0,
                'error': str(e)
            }
    
    def _store_validation_results(self, results: Dict[str, Any]):
        """Store validation results in database"""
        try:
            # Store overall validation result
            overall_query = """
            INSERT INTO labor_analytics.validation_results 
            (rule_id, table_name, validation_status, error_message, validated_at)
            VALUES (%s, %s, %s, %s, %s)
            """
            
            # Get rule_id for overall validation (assuming rule_id = 1 for overall)
            self.db_connector.execute_query(
                overall_query,
                (1, 'overall', results['overall_status'], 
                 f"Validation completed with {len(results['errors'])} errors", 
                 datetime.now())
            )
            
            # Store individual table results
            for table_result in results['tables_validated']:
                for error in table_result.get('errors', []):
                    self.db_connector.execute_query(
                        overall_query,
                        (1, table_result['table_name'], 'FAIL', 
                         f"{error['field_name']}: {error['error_message']}", 
                         datetime.now())
                    )
            
            # Store data quality metrics
            for metric_name, value in results['data_quality_metrics'].items():
                if isinstance(value, (int, float)) and not metric_name.endswith('_status'):
                    status = results['data_quality_metrics'].get(f'{metric_name}_status', 'UNKNOWN')
                    self.db_connector.execute_query(
                        """
                        INSERT INTO labor_analytics.data_quality_metrics 
                        (table_name, metric_name, metric_value, status, measured_at)
                        VALUES (%s, %s, %s, %s, %s)
                        """,
                        ('overall', metric_name, value, status, datetime.now())
                    )
            
            logger.info("Validation results stored in database")
            
        except Exception as e:
            logger.error(f"Error storing validation results: {e}")
    
    def _send_notifications(self, results: Dict[str, Any]):
        """Send notifications based on validation results"""
        try:
            notification_config = self.config.get('notification_settings', {})
            
            if not notification_config.get('critical_errors', True):
                return
            
            # Check if there are critical errors
            has_critical_errors = results['overall_status'] == 'FAIL'
            
            # Check data quality thresholds
            quality_metrics = results.get('data_quality_metrics', {})
            for metric_name, value in quality_metrics.items():
                if isinstance(value, (int, float)) and not metric_name.endswith('_status'):
                    status = quality_metrics.get(f'{metric_name}_status', 'UNKNOWN')
                    if status == 'CRITICAL':
                        has_critical_errors = True
                        break
            
            if has_critical_errors:
                self._send_email_notification(results)
                
        except Exception as e:
            logger.error(f"Error sending notifications: {e}")
    
    def _send_email_notification(self, results: Dict[str, Any]):
        """Send email notification for critical errors"""
        try:
            # This would integrate with your email service
            # For now, just log the notification
            logger.warning("CRITICAL: Data validation failed - email notification would be sent here")
            logger.warning(f"Validation results: {results['overall_status']}")
            logger.warning(f"Total errors: {len(results['errors'])}")
            
        except Exception as e:
            logger.error(f"Error sending email notification: {e}")
    
    def start_scheduled_validation(self, interval_minutes: int = 60):
        """Start scheduled validation runs"""
        logger.info(f"Starting scheduled validation every {interval_minutes} minutes")
        
        schedule.every(interval_minutes).minutes.do(self.run_full_validation)
        
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
    
    def get_validation_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent validation history"""
        return self.validation_history[-limit:] if self.validation_history else []
    
    def export_validation_report(self, output_path: str = None) -> str:
        """Export comprehensive validation report"""
        if not self.validation_history:
            logger.warning("No validation history available")
            return ""
        
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"../excel/validation_engine_report_{timestamp}.xlsx"
        
        try:
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                # Overall summary
                latest_result = self.validation_history[-1]
                summary_data = {
                    'Metric': [
                        'Overall Status',
                        'Total Tables Validated',
                        'Tables with Errors',
                        'Total Errors',
                        'Data Quality Score',
                        'Validation Timestamp'
                    ],
                    'Value': [
                        latest_result['overall_status'],
                        len(latest_result['tables_validated']),
                        sum(1 for t in latest_result['tables_validated'] if not t['valid']),
                        len(latest_result['errors']),
                        f"{latest_result['data_quality_metrics'].get('accuracy', 0):.2%}",
                        latest_result['timestamp']
                    ]
                }
                
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
                
                # Table validation details
                table_details = []
                for table_result in latest_result['tables_validated']:
                    table_details.append({
                        'Table': table_result['table_name'],
                        'Status': 'PASS' if table_result['valid'] else 'FAIL',
                        'Total Records': table_result['total_records'],
                        'Valid Records': table_result['valid_records'],
                        'Invalid Records': table_result['invalid_records'],
                        'Error Count': len(table_result['errors'])
                    })
                
                if table_details:
                    table_df = pd.DataFrame(table_details)
                    table_df.to_excel(writer, sheet_name='Table_Details', index=False)
                
                # Data quality metrics
                quality_data = []
                for metric_name, value in latest_result['data_quality_metrics'].items():
                    if isinstance(value, (int, float)) and not metric_name.endswith('_status'):
                        quality_data.append({
                            'Metric': metric_name,
                            'Value': value,
                            'Status': latest_result['data_quality_metrics'].get(f'{metric_name}_status', 'UNKNOWN')
                        })
                
                if quality_data:
                    quality_df = pd.DataFrame(quality_data)
                    quality_df.to_excel(writer, sheet_name='Quality_Metrics', index=False)
            
            logger.info(f"Validation report exported: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error exporting validation report: {e}")
            return ""


if __name__ == "__main__":
    # Test the validation engine
    engine = ValidationEngine()
    
    # Run a single validation
    results = engine.run_full_validation()
    
    print("Validation Results:")
    print(f"Overall Status: {results['overall_status']}")
    print(f"Tables Validated: {len(results['tables_validated'])}")
    print(f"Total Errors: {len(results['errors'])}")
    
    # Export report
    report_path = engine.export_validation_report()
    print(f"Report exported to: {report_path}")
