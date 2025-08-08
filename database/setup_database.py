#!/usr/bin/env python3
"""
Database Setup Script for Labor Analytics
Initializes PostgreSQL database with schema and sample data
"""

import psycopg2
import json
import sys
import os
from pathlib import Path
import structlog

logger = structlog.get_logger(__name__)


def load_config(config_path: str = "../config/database_config.json") -> dict:
    """Load database configuration"""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {config_path}")
        sys.exit(1)
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON in configuration file: {config_path}")
        sys.exit(1)


def create_database(config: dict) -> bool:
    """Create the database if it doesn't exist"""
    try:
        # Connect to PostgreSQL server (not to specific database)
        conn = psycopg2.connect(
            host=config['database']['host'],
            port=config['database']['port'],
            user=config['database']['username'],
            password=config['database']['password'],
            database='postgres'  # Connect to default postgres database
        )
        
        conn.autocommit = True
        cursor = conn.cursor()
        
        # Check if database exists
        cursor.execute("SELECT 1 FROM pg_database WHERE datname = %s", (config['database']['database'],))
        exists = cursor.fetchone()
        
        if not exists:
            logger.info(f"Creating database: {config['database']['database']}")
            cursor.execute(f"CREATE DATABASE {config['database']['database']}")
            logger.info("Database created successfully")
        else:
            logger.info(f"Database {config['database']['database']} already exists")
        
        cursor.close()
        conn.close()
        return True
        
    except Exception as e:
        logger.error(f"Error creating database: {e}")
        return False


def create_user(config: dict) -> bool:
    """Create database user if it doesn't exist"""
    try:
        # Connect to PostgreSQL server
        conn = psycopg2.connect(
            host=config['database']['host'],
            port=config['database']['port'],
            user=config['database']['username'],
            password=config['database']['password'],
            database='postgres'
        )
        
        conn.autocommit = True
        cursor = conn.cursor()
        
        # Check if user exists
        cursor.execute("SELECT 1 FROM pg_user WHERE usename = %s", (config['database']['username'],))
        exists = cursor.fetchone()
        
        if not exists:
            logger.info(f"Creating user: {config['database']['username']}")
            cursor.execute(f"CREATE USER {config['database']['username']} WITH PASSWORD %s", 
                         (config['database']['password'],))
            logger.info("User created successfully")
        else:
            logger.info(f"User {config['database']['username']} already exists")
        
        # Grant privileges
        cursor.execute(f"GRANT ALL PRIVILEGES ON DATABASE {config['database']['database']} TO {config['database']['username']}")
        cursor.execute(f"ALTER USER {config['database']['username']} CREATEDB")
        
        cursor.close()
        conn.close()
        return True
        
    except Exception as e:
        logger.error(f"Error creating user: {e}")
        return False


def execute_schema_file(config: dict, schema_file: str) -> bool:
    """Execute SQL schema file"""
    try:
        # Connect to the target database
        conn = psycopg2.connect(
            host=config['database']['host'],
            port=config['database']['port'],
            database=config['database']['database'],
            user=config['database']['username'],
            password=config['database']['password']
        )
        
        conn.autocommit = True
        cursor = conn.cursor()
        
        # Read and execute schema file
        schema_path = Path(__file__).parent / "schema" / schema_file
        if not schema_path.exists():
            logger.error(f"Schema file not found: {schema_path}")
            return False
        
        with open(schema_path, 'r') as f:
            sql_content = f.read()
        
        logger.info(f"Executing schema file: {schema_file}")
        cursor.execute(sql_content)
        
        cursor.close()
        conn.close()
        
        logger.info(f"Schema file {schema_file} executed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error executing schema file {schema_file}: {e}")
        return False


def setup_pgadmin(config: dict) -> bool:
    """Setup pgAdmin configuration"""
    try:
        pgadmin_config_dir = Path(__file__).parent / "pgadmin_config"
        pgadmin_config_dir.mkdir(exist_ok=True)
        
        # Create pgAdmin server configuration
        server_config = {
            "name": "Labor Analytics Database",
            "host": config['database']['host'],
            "port": config['database']['port'],
            "maintenance_db": config['database']['database'],
            "username": config['database']['username'],
            "password": config['database']['password'],
            "sslmode": config['database']['sslmode']
        }
        
        server_config_path = pgadmin_config_dir / "server_config.json"
        with open(server_config_path, 'w') as f:
            json.dump(server_config, f, indent=2)
        
        # Create pgAdmin connection script
        connection_script = f"""#!/bin/bash
# pgAdmin Connection Script for Labor Analytics

echo "Setting up pgAdmin connection..."

# Start pgAdmin (if not already running)
docker run -d \\
    --name pgadmin \\
    -p {config['pgadmin']['port']}:80 \\
    -e PGADMIN_DEFAULT_EMAIL={config['pgadmin']['email']} \\
    -e PGADMIN_DEFAULT_PASSWORD={config['pgadmin']['password']} \\
    -e PGADMIN_CONFIG_SERVER_MODE=False \\
    dpage/pgadmin4

echo "pgAdmin started on http://localhost:{config['pgadmin']['port']}"
echo "Email: {config['pgadmin']['email']}"
echo "Password: {config['pgadmin']['password']}"
echo ""
echo "To connect to the database:"
echo "1. Open pgAdmin in your browser"
echo "2. Add new server with these details:"
echo "   - Name: Labor Analytics Database"
echo "   - Host: {config['database']['host']}"
echo "   - Port: {config['database']['port']}"
echo "   - Database: {config['database']['database']}"
echo "   - Username: {config['database']['username']}"
echo "   - Password: {config['database']['password']}"
"""
        
        script_path = pgadmin_config_dir / "start_pgadmin.sh"
        with open(script_path, 'w') as f:
            f.write(connection_script)
        
        # Make script executable
        os.chmod(script_path, 0o755)
        
        logger.info("pgAdmin configuration created successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error setting up pgAdmin: {e}")
        return False


def verify_setup(config: dict) -> bool:
    """Verify database setup"""
    try:
        # Test connection
        conn = psycopg2.connect(
            host=config['database']['host'],
            port=config['database']['port'],
            database=config['database']['database'],
            user=config['database']['username'],
            password=config['database']['password']
        )
        
        cursor = conn.cursor()
        
        # Check if tables exist
        cursor.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'labor_analytics'
            ORDER BY table_name
        """)
        
        tables = [row[0] for row in cursor.fetchall()]
        expected_tables = [
            'departments', 'employees', 'projects', 'time_tracking', 
            'payroll', 'validation_rules', 'validation_results', 
            'data_quality_metrics', 'audit_log'
        ]
        
        missing_tables = set(expected_tables) - set(tables)
        if missing_tables:
            logger.warning(f"Missing tables: {missing_tables}")
            return False
        
        # Check if sample data exists
        cursor.execute("SELECT COUNT(*) FROM labor_analytics.employees")
        employee_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM labor_analytics.departments")
        department_count = cursor.fetchone()[0]
        
        cursor.close()
        conn.close()
        
        logger.info(f"Database verification successful:")
        logger.info(f"  - Tables found: {len(tables)}")
        logger.info(f"  - Employees: {employee_count}")
        logger.info(f"  - Departments: {department_count}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error verifying setup: {e}")
        return False


def main():
    """Main setup function"""
    print("🚀 Labor Analytics Database Setup")
    print("=" * 50)
    
    # Setup logging
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.ConsoleRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # Load configuration
    config = load_config()
    
    print(f"📋 Database: {config['database']['database']}")
    print(f"🌐 Host: {config['database']['host']}:{config['database']['port']}")
    print(f"👤 User: {config['database']['username']}")
    print()
    
    # Step 1: Create user
    print("1️⃣  Creating database user...")
    if not create_user(config):
        print("❌ Failed to create user")
        sys.exit(1)
    print("✅ User created/verified")
    
    # Step 2: Create database
    print("2️⃣  Creating database...")
    if not create_database(config):
        print("❌ Failed to create database")
        sys.exit(1)
    print("✅ Database created/verified")
    
    # Step 3: Execute schema
    print("3️⃣  Setting up database schema...")
    if not execute_schema_file(config, "labor_analytics_schema.sql"):
        print("❌ Failed to execute schema")
        sys.exit(1)
    print("✅ Schema created successfully")
    
    # Step 4: Setup pgAdmin
    print("4️⃣  Setting up pgAdmin configuration...")
    if not setup_pgadmin(config):
        print("❌ Failed to setup pgAdmin")
        sys.exit(1)
    print("✅ pgAdmin configuration created")
    
    # Step 5: Verify setup
    print("5️⃣  Verifying setup...")
    if not verify_setup(config):
        print("❌ Setup verification failed")
        sys.exit(1)
    print("✅ Setup verified successfully")
    
    print("\n🎉 Database setup completed successfully!")
    print("\n📊 Next steps:")
    print("1. Start pgAdmin: ./database/pgadmin_config/start_pgadmin.sh")
    print("2. Run validation engine: python python/validation_engine/run_validation.py")
    print("3. Open Power BI and connect to the database")
    print("\n🔗 Connection details:")
    print(f"   Host: {config['database']['host']}")
    print(f"   Port: {config['database']['port']}")
    print(f"   Database: {config['database']['database']}")
    print(f"   Username: {config['database']['username']}")
    print(f"   Password: {config['database']['password']}")


if __name__ == "__main__":
    main()
