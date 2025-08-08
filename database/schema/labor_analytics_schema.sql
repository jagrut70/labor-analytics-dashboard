-- Labor Analytics Database Schema
-- PostgreSQL Database for Power BI Dashboard with Validation Rules

-- Create database (run as superuser)
-- CREATE DATABASE labor_analytics;

-- Connect to the database
-- \c labor_analytics;

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";

-- Create schema for better organization
CREATE SCHEMA IF NOT EXISTS labor_analytics;

-- Set search path
SET search_path TO labor_analytics, public;

-- =============================================
-- CORE TABLES
-- =============================================

-- Departments table
CREATE TABLE departments (
    department_id SERIAL PRIMARY KEY,
    department_name VARCHAR(100) NOT NULL UNIQUE,
    department_code VARCHAR(10) NOT NULL UNIQUE,
    manager_id INTEGER,
    budget_amount DECIMAL(12,2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Employees table
CREATE TABLE employees (
    employee_id VARCHAR(20) PRIMARY KEY,
    first_name VARCHAR(50) NOT NULL,
    last_name VARCHAR(50) NOT NULL,
    email VARCHAR(100) UNIQUE,
    phone VARCHAR(20),
    department_id INTEGER REFERENCES departments(department_id),
    position VARCHAR(100),
    hire_date DATE NOT NULL,
    termination_date DATE,
    salary DECIMAL(10,2),
    hourly_rate DECIMAL(8,2),
    employment_status VARCHAR(20) DEFAULT 'ACTIVE',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Projects table
CREATE TABLE projects (
    project_id SERIAL PRIMARY KEY,
    project_code VARCHAR(20) NOT NULL UNIQUE,
    project_name VARCHAR(200) NOT NULL,
    client_name VARCHAR(100),
    start_date DATE,
    end_date DATE,
    budget DECIMAL(12,2),
    status VARCHAR(20) DEFAULT 'ACTIVE',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Time tracking table
CREATE TABLE time_tracking (
    time_entry_id SERIAL PRIMARY KEY,
    employee_id VARCHAR(20) REFERENCES employees(employee_id),
    project_id INTEGER REFERENCES projects(project_id),
    date DATE NOT NULL,
    hours_worked DECIMAL(5,2) NOT NULL,
    overtime_hours DECIMAL(5,2) DEFAULT 0,
    task_description TEXT,
    billable BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Payroll table
CREATE TABLE payroll (
    payroll_id SERIAL PRIMARY KEY,
    employee_id VARCHAR(20) REFERENCES employees(employee_id),
    pay_period_start DATE NOT NULL,
    pay_period_end DATE NOT NULL,
    gross_pay DECIMAL(10,2) NOT NULL,
    net_pay DECIMAL(10,2) NOT NULL,
    taxes DECIMAL(10,2) DEFAULT 0,
    benefits DECIMAL(10,2) DEFAULT 0,
    overtime_pay DECIMAL(10,2) DEFAULT 0,
    bonus DECIMAL(10,2) DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- =============================================
-- VALIDATION TABLES
-- =============================================

-- Validation rules table
CREATE TABLE validation_rules (
    rule_id SERIAL PRIMARY KEY,
    rule_name VARCHAR(100) NOT NULL UNIQUE,
    table_name VARCHAR(50) NOT NULL,
    rule_type VARCHAR(50) NOT NULL, -- 'range', 'required', 'format', 'business_logic'
    rule_definition JSONB NOT NULL,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Validation results table
CREATE TABLE validation_results (
    result_id SERIAL PRIMARY KEY,
    rule_id INTEGER REFERENCES validation_rules(rule_id),
    table_name VARCHAR(50) NOT NULL,
    record_id VARCHAR(50),
    validation_status VARCHAR(20) NOT NULL, -- 'PASS', 'FAIL', 'WARNING'
    error_message TEXT,
    validated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Data quality metrics table
CREATE TABLE data_quality_metrics (
    metric_id SERIAL PRIMARY KEY,
    table_name VARCHAR(50) NOT NULL,
    metric_name VARCHAR(100) NOT NULL,
    metric_value DECIMAL(5,4) NOT NULL,
    threshold DECIMAL(5,4),
    status VARCHAR(20) NOT NULL, -- 'GOOD', 'WARNING', 'CRITICAL'
    measured_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- =============================================
-- AUDIT TABLES
-- =============================================

-- Audit log table
CREATE TABLE audit_log (
    audit_id SERIAL PRIMARY KEY,
    table_name VARCHAR(50) NOT NULL,
    operation VARCHAR(20) NOT NULL, -- 'INSERT', 'UPDATE', 'DELETE'
    record_id VARCHAR(50),
    old_values JSONB,
    new_values JSONB,
    user_id VARCHAR(50),
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- =============================================
-- INDEXES FOR PERFORMANCE
-- =============================================

-- Employees indexes
CREATE INDEX idx_employees_department_id ON employees(department_id);
CREATE INDEX idx_employees_hire_date ON employees(hire_date);
CREATE INDEX idx_employees_status ON employees(employment_status);
CREATE INDEX idx_employees_name ON employees(last_name, first_name);

-- Time tracking indexes
CREATE INDEX idx_time_tracking_employee_id ON time_tracking(employee_id);
CREATE INDEX idx_time_tracking_date ON time_tracking(date);
CREATE INDEX idx_time_tracking_project_id ON time_tracking(project_id);
CREATE INDEX idx_time_tracking_employee_date ON time_tracking(employee_id, date);

-- Payroll indexes
CREATE INDEX idx_payroll_employee_id ON payroll(employee_id);
CREATE INDEX idx_payroll_period ON payroll(pay_period_start, pay_period_end);
CREATE INDEX idx_payroll_employee_period ON payroll(employee_id, pay_period_start);

-- Validation indexes
CREATE INDEX idx_validation_results_rule_id ON validation_results(rule_id);
CREATE INDEX idx_validation_results_status ON validation_results(validation_status);
CREATE INDEX idx_validation_results_table ON validation_results(table_name);

-- =============================================
-- CONSTRAINTS
-- =============================================

-- Add foreign key constraint for departments manager
ALTER TABLE departments ADD CONSTRAINT fk_departments_manager 
    FOREIGN KEY (manager_id) REFERENCES employees(employee_id);

-- Add check constraints
ALTER TABLE employees ADD CONSTRAINT chk_salary_positive 
    CHECK (salary > 0);

ALTER TABLE employees ADD CONSTRAINT chk_hourly_rate_positive 
    CHECK (hourly_rate > 0);

ALTER TABLE time_tracking ADD CONSTRAINT chk_hours_worked_positive 
    CHECK (hours_worked >= 0);

ALTER TABLE time_tracking ADD CONSTRAINT chk_hours_per_day 
    CHECK (hours_worked <= 24);

ALTER TABLE payroll ADD CONSTRAINT chk_gross_pay_positive 
    CHECK (gross_pay > 0);

ALTER TABLE payroll ADD CONSTRAINT chk_net_pay_positive 
    CHECK (net_pay > 0);

ALTER TABLE payroll ADD CONSTRAINT chk_net_less_than_gross 
    CHECK (net_pay <= gross_pay);

-- =============================================
-- TRIGGERS FOR AUDIT
-- =============================================

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Apply triggers to all tables
CREATE TRIGGER update_employees_updated_at BEFORE UPDATE ON employees
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_departments_updated_at BEFORE UPDATE ON departments
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_projects_updated_at BEFORE UPDATE ON projects
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_time_tracking_updated_at BEFORE UPDATE ON time_tracking
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_payroll_updated_at BEFORE UPDATE ON payroll
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- =============================================
-- SAMPLE DATA INSERTION
-- =============================================

-- Insert sample departments
INSERT INTO departments (department_name, department_code, budget_amount) VALUES
('Human Resources', 'HR', 500000.00),
('Information Technology', 'IT', 750000.00),
('Finance', 'FIN', 400000.00),
('Operations', 'OPS', 600000.00),
('Sales', 'SALES', 800000.00);

-- Insert sample employees
INSERT INTO employees (employee_id, first_name, last_name, email, department_id, position, hire_date, salary, hourly_rate) VALUES
('EMP001', 'John', 'Smith', 'john.smith@company.com', 1, 'HR Manager', '2020-01-15', 75000.00, 36.06),
('EMP002', 'Jane', 'Doe', 'jane.doe@company.com', 2, 'Senior Developer', '2019-03-20', 85000.00, 40.87),
('EMP003', 'Mike', 'Johnson', 'mike.johnson@company.com', 3, 'Financial Analyst', '2021-06-10', 65000.00, 31.25),
('EMP004', 'Sarah', 'Wilson', 'sarah.wilson@company.com', 4, 'Operations Manager', '2020-11-05', 70000.00, 33.65),
('EMP005', 'David', 'Brown', 'david.brown@company.com', 5, 'Sales Representative', '2021-02-28', 60000.00, 28.85);

-- Insert sample projects
INSERT INTO projects (project_code, project_name, client_name, start_date, end_date, budget) VALUES
('PRJ001', 'Website Redesign', 'ABC Corp', '2023-01-01', '2023-06-30', 50000.00),
('PRJ002', 'Mobile App Development', 'XYZ Inc', '2023-03-01', '2023-12-31', 100000.00),
('PRJ003', 'Data Migration', 'Internal', '2023-02-15', '2023-05-15', 25000.00);

-- Update departments with managers
UPDATE departments SET manager_id = 'EMP001' WHERE department_code = 'HR';
UPDATE departments SET manager_id = 'EMP002' WHERE department_code = 'IT';
UPDATE departments SET manager_id = 'EMP003' WHERE department_code = 'FIN';
UPDATE departments SET manager_id = 'EMP004' WHERE department_code = 'OPS';
UPDATE departments SET manager_id = 'EMP005' WHERE department_code = 'SALES';

-- Insert sample time tracking data
INSERT INTO time_tracking (employee_id, project_id, date, hours_worked, overtime_hours, task_description) VALUES
('EMP001', 3, '2023-04-01', 8.0, 0, 'HR system setup'),
('EMP002', 1, '2023-04-01', 8.0, 2.0, 'Frontend development'),
('EMP003', 3, '2023-04-01', 8.0, 0, 'Financial data analysis'),
('EMP004', 2, '2023-04-01', 8.0, 1.5, 'Project planning'),
('EMP005', 1, '2023-04-01', 8.0, 0, 'Client requirements gathering');

-- Insert sample payroll data
INSERT INTO payroll (employee_id, pay_period_start, pay_period_end, gross_pay, net_pay, taxes, overtime_pay) VALUES
('EMP001', '2023-04-01', '2023-04-15', 2884.62, 2307.69, 576.93, 0),
('EMP002', '2023-04-01', '2023-04-15', 3269.23, 2615.38, 653.85, 163.46),
('EMP003', '2023-04-01', '2023-04-15', 2500.00, 2000.00, 500.00, 0),
('EMP004', '2023-04-01', '2023-04-15', 2692.31, 2153.85, 538.46, 50.48),
('EMP005', '2023-04-01', '2023-04-15', 2307.69, 1846.15, 461.54, 0);

-- Insert sample validation rules
INSERT INTO validation_rules (rule_name, table_name, rule_type, rule_definition) VALUES
('salary_range_check', 'employees', 'range', '{"field": "salary", "min": 20000, "max": 200000}'),
('hours_worked_range', 'time_tracking', 'range', '{"field": "hours_worked", "min": 0, "max": 24}'),
('required_employee_fields', 'employees', 'required', '{"fields": ["employee_id", "first_name", "last_name", "department_id", "hire_date"]}'),
('overtime_calculation', 'time_tracking', 'business_logic', '{"formula": "overtime_hours = max(0, hours_worked - 8)", "tolerance": 0.1}');

COMMIT;
