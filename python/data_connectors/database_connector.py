"""
Database Connector for Labor Analytics
Handles PostgreSQL connections and data operations with pgAdmin integration
"""

import psycopg2
import pandas as pd
import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from contextlib import contextmanager
import structlog

logger = structlog.get_logger(__name__)


class DatabaseConnector:
    """PostgreSQL database connector with pgAdmin integration"""
    
    def __init__(self, config_path: str = "../config/database_config.json"):
        """Initialize database connector with configuration"""
        self.config = self._load_config(config_path)
        self.connection = None
        self._setup_logging()
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load database configuration from JSON file"""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            logger.info("Database configuration loaded successfully")
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
    
    @contextmanager
    def get_connection(self):
        """Context manager for database connections"""
        conn = None
        try:
            conn = psycopg2.connect(
                host=self.config['database']['host'],
                port=self.config['database']['port'],
                database=self.config['database']['database'],
                user=self.config['database']['username'],
                password=self.config['database']['password'],
                sslmode=self.config['database']['sslmode']
            )
            conn.autocommit = False
            yield conn
        except psycopg2.Error as e:
            logger.error(f"Database connection error: {e}")
            if conn:
                conn.rollback()
            raise
        finally:
            if conn:
                conn.close()
    
    def execute_query(self, query: str, params: Optional[Tuple] = None) -> List[Tuple]:
        """Execute a SQL query and return results"""
        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                try:
                    cursor.execute(query, params)
                    if query.strip().upper().startswith('SELECT'):
                        results = cursor.fetchall()
                        logger.info(f"Query executed successfully, returned {len(results)} rows")
                        return results
                    else:
                        conn.commit()
                        logger.info(f"Query executed successfully, affected {cursor.rowcount} rows")
                        return []
                except psycopg2.Error as e:
                    conn.rollback()
                    logger.error(f"Query execution error: {e}")
                    raise
    
    def execute_many(self, query: str, params_list: List[Tuple]) -> int:
        """Execute a query with multiple parameter sets"""
        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                try:
                    cursor.executemany(query, params_list)
                    conn.commit()
                    affected_rows = cursor.rowcount
                    logger.info(f"Batch query executed successfully, affected {affected_rows} rows")
                    return affected_rows
                except psycopg2.Error as e:
                    conn.rollback()
                    logger.error(f"Batch query execution error: {e}")
                    raise
    
    def get_dataframe(self, query: str, params: Optional[Tuple] = None) -> pd.DataFrame:
        """Execute a query and return results as a pandas DataFrame"""
        with self.get_connection() as conn:
            try:
                df = pd.read_sql_query(query, conn, params=params)
                logger.info(f"DataFrame created successfully with {len(df)} rows and {len(df.columns)} columns")
                return df
            except Exception as e:
                logger.error(f"DataFrame creation error: {e}")
                raise
    
    def insert_dataframe(self, df: pd.DataFrame, table_name: str, schema: str = "labor_analytics") -> int:
        """Insert a pandas DataFrame into a database table"""
        with self.get_connection() as conn:
            try:
                # Create column list for INSERT statement
                columns = list(df.columns)
                placeholders = ', '.join(['%s'] * len(columns))
                column_names = ', '.join(columns)
                
                query = f"INSERT INTO {schema}.{table_name} ({column_names}) VALUES ({placeholders})"
                
                # Convert DataFrame to list of tuples
                data_tuples = [tuple(row) for row in df.values]
                
                with conn.cursor() as cursor:
                    cursor.executemany(query, data_tuples)
                    conn.commit()
                    affected_rows = cursor.rowcount
                    logger.info(f"DataFrame inserted successfully into {schema}.{table_name}, affected {affected_rows} rows")
                    return affected_rows
            except Exception as e:
                conn.rollback()
                logger.error(f"DataFrame insertion error: {e}")
                raise
    
    def update_records(self, table_name: str, update_data: Dict[str, Any], 
                      where_conditions: Dict[str, Any], schema: str = "labor_analytics") -> int:
        """Update records in a table based on conditions"""
        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                try:
                    # Build SET clause
                    set_clause = ', '.join([f"{k} = %s" for k in update_data.keys()])
                    
                    # Build WHERE clause
                    where_clause = ' AND '.join([f"{k} = %s" for k in where_conditions.keys()])
                    
                    query = f"UPDATE {schema}.{table_name} SET {set_clause} WHERE {where_clause}"
                    
                    # Combine parameters
                    params = tuple(list(update_data.values()) + list(where_conditions.values()))
                    
                    cursor.execute(query, params)
                    conn.commit()
                    affected_rows = cursor.rowcount
                    logger.info(f"Update executed successfully, affected {affected_rows} rows")
                    return affected_rows
                except psycopg2.Error as e:
                    conn.rollback()
                    logger.error(f"Update error: {e}")
                    raise
    
    def delete_records(self, table_name: str, where_conditions: Dict[str, Any], 
                      schema: str = "labor_analytics") -> int:
        """Delete records from a table based on conditions"""
        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                try:
                    # Build WHERE clause
                    where_clause = ' AND '.join([f"{k} = %s" for k in where_conditions.keys()])
                    
                    query = f"DELETE FROM {schema}.{table_name} WHERE {where_clause}"
                    params = tuple(where_conditions.values())
                    
                    cursor.execute(query, params)
                    conn.commit()
                    affected_rows = cursor.rowcount
                    logger.info(f"Delete executed successfully, affected {affected_rows} rows")
                    return affected_rows
                except psycopg2.Error as e:
                    conn.rollback()
                    logger.error(f"Delete error: {e}")
                    raise
    
    def get_table_schema(self, table_name: str, schema: str = "labor_analytics") -> pd.DataFrame:
        """Get table schema information"""
        query = """
        SELECT 
            column_name,
            data_type,
            is_nullable,
            column_default,
            character_maximum_length
        FROM information_schema.columns 
        WHERE table_schema = %s AND table_name = %s
        ORDER BY ordinal_position
        """
        return self.get_dataframe(query, (schema, table_name))
    
    def get_table_row_count(self, table_name: str, schema: str = "labor_analytics") -> int:
        """Get the number of rows in a table"""
        query = f"SELECT COUNT(*) FROM {schema}.{table_name}"
        result = self.execute_query(query)
        return result[0][0] if result else 0
    
    def backup_table(self, table_name: str, backup_suffix: str = None, schema: str = "labor_analytics") -> str:
        """Create a backup of a table"""
        if not backup_suffix:
            backup_suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        backup_table_name = f"{table_name}_backup_{backup_suffix}"
        
        query = f"CREATE TABLE {schema}.{backup_table_name} AS SELECT * FROM {schema}.{table_name}"
        self.execute_query(query)
        
        logger.info(f"Table backup created: {schema}.{backup_table_name}")
        return backup_table_name
    
    def restore_table(self, backup_table_name: str, target_table_name: str, schema: str = "labor_analytics") -> bool:
        """Restore a table from backup"""
        try:
            # Drop target table if exists
            drop_query = f"DROP TABLE IF EXISTS {schema}.{target_table_name}"
            self.execute_query(drop_query)
            
            # Restore from backup
            restore_query = f"CREATE TABLE {schema}.{target_table_name} AS SELECT * FROM {schema}.{backup_table_name}"
            self.execute_query(restore_query)
            
            logger.info(f"Table restored successfully: {schema}.{target_table_name}")
            return True
        except Exception as e:
            logger.error(f"Table restore error: {e}")
            return False
    
    def get_connection_status(self) -> Dict[str, Any]:
        """Get database connection status and health metrics"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    # Get database size
                    cursor.execute("""
                        SELECT pg_size_pretty(pg_database_size(current_database())) as db_size
                    """)
                    db_size = cursor.fetchone()[0]
                    
                    # Get active connections
                    cursor.execute("""
                        SELECT count(*) as active_connections 
                        FROM pg_stat_activity 
                        WHERE state = 'active'
                    """)
                    active_connections = cursor.fetchone()[0]
                    
                    # Get database uptime
                    cursor.execute("""
                        SELECT date_trunc('second', current_timestamp - pg_postmaster_start_time()) as uptime
                    """)
                    uptime = cursor.fetchone()[0]
                    
                    return {
                        "status": "connected",
                        "database_size": db_size,
                        "active_connections": active_connections,
                        "uptime": str(uptime),
                        "timestamp": datetime.now().isoformat()
                    }
        except Exception as e:
            logger.error(f"Connection status check failed: {e}")
            return {
                "status": "disconnected",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def test_connection(self) -> bool:
        """Test database connection"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("SELECT 1")
                    result = cursor.fetchone()
                    return result[0] == 1
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False


if __name__ == "__main__":
    # Test the database connector
    connector = DatabaseConnector()
    
    if connector.test_connection():
        print("Database connection successful!")
        
        # Test getting employee data
        try:
            df = connector.get_dataframe("SELECT * FROM labor_analytics.employees LIMIT 5")
            print(f"Employee data sample:\n{df}")
        except Exception as e:
            print(f"Error getting employee data: {e}")
    else:
        print("Database connection failed!")
