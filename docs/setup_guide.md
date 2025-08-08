# Labor Analytics Dashboard Setup Guide

This guide will walk you through setting up the complete Labor Analytics Dashboard with Power BI, PostgreSQL, and automated validation.

## 📋 Prerequisites

Before starting, ensure you have the following installed:

### Required Software
- **PostgreSQL 14+** - Database server
- **Python 3.8+** - For validation engine
- **Power BI Desktop** - For dashboard creation
- **pgAdmin 4** - Database management (optional, can use Docker)
- **Git** - Version control

### System Requirements
- **Operating System**: Windows 10+, macOS 10.15+, or Ubuntu 18.04+
- **RAM**: Minimum 8GB, Recommended 16GB+
- **Storage**: 10GB free space
- **Network**: Internet connection for package installation

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone <repository-url>
cd "Labor PowerBI Dashboard"
```

### 2. Install Python Dependencies

```bash
cd python
pip install -r requirements.txt
```

### 3. Configure Database

Edit the database configuration file:

```bash
# Edit config/database_config.json
{
  "database": {
    "host": "localhost",
    "port": 5432,
    "database": "labor_analytics",
    "username": "your_username",
    "password": "your_secure_password"
  }
}
```

### 4. Setup Database

```bash
cd database
python setup_database.py
```

### 5. Create Excel Validation Rules

```bash
cd excel
python create_validation_rules.py
```

### 6. Test Validation Engine

```bash
cd python/validation_engine
python run_validation.py --mode single
```

## 📊 Detailed Setup Instructions

### PostgreSQL Installation

#### Windows
1. Download PostgreSQL from https://www.postgresql.org/download/windows/
2. Run the installer with default settings
3. Note down the password for the `postgres` user
4. Add PostgreSQL to your PATH environment variable

#### macOS
```bash
# Using Homebrew
brew install postgresql
brew services start postgresql

# Or using the official installer
# Download from https://www.postgresql.org/download/macosx/
```

#### Ubuntu/Debian
```bash
sudo apt update
sudo apt install postgresql postgresql-contrib
sudo systemctl start postgresql
sudo systemctl enable postgresql
```

### pgAdmin Setup (Optional)

#### Using Docker (Recommended)
```bash
# Start pgAdmin container
docker run -d \
  --name pgadmin \
  -p 5050:80 \
  -e PGADMIN_DEFAULT_EMAIL=admin@company.com \
  -e PGADMIN_DEFAULT_PASSWORD=admin123 \
  dpage/pgadmin4

# Access at http://localhost:5050
```

#### Manual Installation
1. Download pgAdmin from https://www.pgadmin.org/download/
2. Install following the wizard
3. Launch pgAdmin and add your database server

### Python Environment Setup

#### Create Virtual Environment
```bash
# Windows
python -m venv labor_env
labor_env\Scripts\activate

# macOS/Linux
python3 -m venv labor_env
source labor_env/bin/activate
```

#### Install Dependencies
```bash
pip install --upgrade pip
pip install -r python/requirements.txt
```

### Power BI Setup

1. **Download Power BI Desktop**
   - Visit https://powerbi.microsoft.com/desktop/
   - Download and install Power BI Desktop

2. **Connect to Database**
   - Open Power BI Desktop
   - Click "Get Data" → "Database" → "PostgreSQL database"
   - Enter connection details:
     - Server: `localhost`
     - Database: `labor_analytics`
     - Username: `your_username`
     - Password: `your_password`

3. **Import Data**
   - Select all tables: `employees`, `departments`, `projects`, `time_tracking`, `payroll`
   - Click "Load"

## 🔧 Configuration

### Database Configuration

The main configuration file is `config/database_config.json`:

```json
{
  "database": {
    "host": "localhost",
    "port": 5432,
    "database": "labor_analytics",
    "username": "labor_user",
    "password": "secure_password_123",
    "sslmode": "prefer"
  },
  "pgadmin": {
    "host": "localhost",
    "port": 5050,
    "email": "admin@laboranalytics.com",
    "password": "admin_password_123"
  }
}
```

### Validation Configuration

Edit `config/validation_config.json` to customize validation rules:

```json
{
  "validation_rules": {
    "employee_data": {
      "required_fields": ["employee_id", "first_name", "last_name"],
      "ranges": {
        "salary": {"min": 20000, "max": 200000}
      }
    }
  },
  "data_quality_thresholds": {
    "completeness": 0.95,
    "accuracy": 0.98
  }
}
```

## 🧪 Testing the Setup

### 1. Test Database Connection

```bash
cd python
python -c "
from data_connectors.database_connector import DatabaseConnector
connector = DatabaseConnector()
print('Database connection:', '✅ Success' if connector.test_connection() else '❌ Failed')
"
```

### 2. Test Validation Engine

```bash
cd python/validation_engine
python run_validation.py --mode single --verbose
```

### 3. Test API Mode

```bash
python run_validation.py --mode api
# Open http://localhost:8000/docs in your browser
```

### 4. Test Scheduled Validation

```bash
python run_validation.py --mode scheduled --interval 30
# Runs validation every 30 minutes
```

## 📊 Power BI Dashboard Creation

### 1. Basic Dashboard Setup

1. **Create New Report**
   - Open Power BI Desktop
   - Create a new report

2. **Add Visualizations**
   - **Employee Overview**: Card showing total employees
   - **Department Distribution**: Pie chart of employees by department
   - **Salary Analysis**: Bar chart of average salary by department
   - **Time Tracking**: Line chart of hours worked over time
   - **Payroll Summary**: Table showing payroll data

3. **Create Relationships**
   - Link `employees.department_id` to `departments.department_id`
   - Link `time_tracking.employee_id` to `employees.employee_id`
   - Link `payroll.employee_id` to `employees.employee_id`

### 2. Advanced Visualizations

#### KPI Dashboard
- Employee count trends
- Average salary metrics
- Overtime hours analysis
- Project completion rates

#### Drill-down Reports
- Employee performance by department
- Time tracking by project
- Payroll analysis by period

#### Data Quality Dashboard
- Validation results summary
- Error rates by table
- Data completeness metrics

## 🔄 Automation Setup

### Scheduled Validation

#### Using Cron (Linux/macOS)
```bash
# Edit crontab
crontab -e

# Add line for hourly validation
0 * * * * cd /path/to/project/python/validation_engine && python run_validation.py --mode single
```

#### Using Task Scheduler (Windows)
1. Open Task Scheduler
2. Create Basic Task
3. Set trigger to run daily
4. Action: Start a program
5. Program: `python`
6. Arguments: `run_validation.py --mode single`

### Email Notifications

Configure email notifications in `config/database_config.json`:

```json
{
  "email_notifications": {
    "enabled": true,
    "smtp_server": "smtp.gmail.com",
    "smtp_port": 587,
    "sender_email": "alerts@company.com",
    "recipients": ["admin@company.com"]
  }
}
```

## 🛠️ Troubleshooting

### Common Issues

#### Database Connection Failed
```bash
# Check PostgreSQL service
sudo systemctl status postgresql

# Check connection
psql -h localhost -U your_username -d labor_analytics
```

#### Python Import Errors
```bash
# Ensure virtual environment is activated
source labor_env/bin/activate  # Linux/macOS
labor_env\Scripts\activate     # Windows

# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

#### Power BI Connection Issues
1. Check firewall settings
2. Verify PostgreSQL is listening on the correct port
3. Ensure user has proper permissions

#### Validation Engine Errors
```bash
# Check logs
python run_validation.py --mode single --verbose

# Test individual components
python -c "from data_connectors.excel_validator import ExcelValidator; print('Excel validator OK')"
```

### Performance Optimization

#### Database Optimization
```sql
-- Create indexes for better performance
CREATE INDEX idx_employees_department_id ON labor_analytics.employees(department_id);
CREATE INDEX idx_time_tracking_date ON labor_analytics.time_tracking(date);
CREATE INDEX idx_payroll_period ON labor_analytics.payroll(pay_period_start, pay_period_end);
```

#### Python Performance
```bash
# Use multiprocessing for large datasets
python run_validation.py --mode single --workers 4

# Optimize memory usage
export PYTHONOPTIMIZE=1
```

## 📈 Monitoring and Maintenance

### Regular Maintenance Tasks

1. **Database Maintenance**
   ```sql
   -- Weekly vacuum and analyze
   VACUUM ANALYZE labor_analytics.employees;
   VACUUM ANALYZE labor_analytics.time_tracking;
   ```

2. **Log Rotation**
   ```bash
   # Rotate validation logs
   logrotate /etc/logrotate.d/labor_analytics
   ```

3. **Backup Strategy**
   ```bash
   # Daily database backup
   pg_dump labor_analytics > backup_$(date +%Y%m%d).sql
   ```

### Health Checks

Create a health check script:

```bash
#!/bin/bash
# health_check.sh

echo "Checking Labor Analytics Dashboard..."

# Check database
python -c "
from data_connectors.database_connector import DatabaseConnector
connector = DatabaseConnector()
status = connector.get_connection_status()
print(f'Database: {status[\"status\"]}')
"

# Check validation engine
python validation_engine/run_validation.py --mode single

# Check Power BI refresh
# (Add Power BI REST API calls here)

echo "Health check completed"
```

## 🆘 Support

### Getting Help

1. **Check Documentation**
   - Review this setup guide
   - Check `docs/` folder for additional documentation

2. **Logs and Debugging**
   - Enable verbose logging: `--verbose` flag
   - Check Python logs in console output
   - Review PostgreSQL logs: `/var/log/postgresql/`

3. **Common Solutions**
   - Restart services if needed
   - Check network connectivity
   - Verify file permissions

### Contact Information

- **Technical Support**: Create an issue in the repository
- **Documentation**: Check the `docs/` folder
- **Configuration**: Review `config/` files

## 🎉 Next Steps

After successful setup:

1. **Customize Dashboard**
   - Modify Power BI visualizations
   - Add company-specific KPIs
   - Create custom reports

2. **Extend Validation Rules**
   - Add business-specific rules to Excel file
   - Create custom validation functions
   - Configure notification thresholds

3. **Scale the System**
   - Add more data sources
   - Implement data warehouse
   - Set up automated reporting

4. **User Training**
   - Train users on Power BI
   - Document business processes
   - Create user guides

---

**Congratulations!** Your Labor Analytics Dashboard is now ready for use. 🎉
