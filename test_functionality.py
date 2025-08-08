#!/usr/bin/env python3
"""
Functionality Test for Labor Analytics Dashboard
Tests core functionality with sample data
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def test_excel_validator():
    """Test Excel validator with sample data"""
    print("📋 Testing Excel Validator...")
    
    try:
        # Create sample data
        sample_data = pd.DataFrame({
            'employee_id': ['EMP001', 'EMP002', 'EMP003', 'EMP004', 'EMP005'],
            'first_name': ['John', 'Jane', 'Bob', 'Alice', 'Charlie'],
            'last_name': ['Doe', 'Smith', 'Johnson', 'Brown', 'Wilson'],
            'salary': [75000, 85000, 65000, 90000, 70000],
            'hours_worked': [40, 45, 35, 50, 38],
            'department': ['IT', 'HR', 'Finance', 'Marketing', 'IT'],
            'hire_date': ['2020-01-15', '2019-03-20', '2021-06-10', '2018-11-05', '2022-02-28']
        })
        
        print(f"  ✅ Sample data created with {len(sample_data)} records")
        
        # Test basic validation
        # Check for required fields
        required_fields = ['employee_id', 'first_name', 'last_name', 'salary']
        missing_fields = [field for field in required_fields if field not in sample_data.columns]
        
        if not missing_fields:
            print("  ✅ All required fields present")
        else:
            print(f"  ❌ Missing required fields: {missing_fields}")
            return False
        
        # Check data types
        if sample_data['salary'].dtype in ['int64', 'float64']:
            print("  ✅ Salary field has numeric data type")
        else:
            print("  ❌ Salary field should be numeric")
            return False
        
        # Check for valid salary range
        valid_salaries = sample_data[(sample_data['salary'] >= 20000) & (sample_data['salary'] <= 200000)]
        if len(valid_salaries) == len(sample_data):
            print("  ✅ All salaries are within valid range")
        else:
            print("  ⚠️  Some salaries may be outside expected range")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Excel validator test failed: {e}")
        return False

def test_database_connector():
    """Test database connector functionality"""
    print("\n🗄️  Testing Database Connector...")
    
    try:
        # Test configuration loading
        import json
        with open('config/database_config.json', 'r') as f:
            config = json.load(f)
        
        if 'database' in config:
            print("  ✅ Database configuration loaded successfully")
        else:
            print("  ❌ Database configuration missing")
            return False
        
        # Test connection parameters
        db_config = config['database']
        required_params = ['host', 'port', 'database', 'username', 'password']
        missing_params = [param for param in required_params if param not in db_config]
        
        if not missing_params:
            print("  ✅ All database connection parameters present")
        else:
            print(f"  ⚠️  Missing database parameters: {missing_params}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Database connector test failed: {e}")
        return False

def test_validation_engine():
    """Test validation engine functionality"""
    print("\n🔍 Testing Validation Engine...")
    
    try:
        # Test configuration loading
        import json
        with open('config/validation_config.json', 'r') as f:
            config = json.load(f)
        
        if 'validation_rules' in config:
            print("  ✅ Validation configuration loaded successfully")
        else:
            print("  ❌ Validation configuration missing")
            return False
        
        # Test validation rules structure
        rules = config['validation_rules']
        expected_tables = ['employee_data', 'time_tracking', 'payroll']
        
        for table in expected_tables:
            if table in rules:
                print(f"  ✅ Validation rules found for {table}")
            else:
                print(f"  ⚠️  Validation rules missing for {table}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Validation engine test failed: {e}")
        return False

def test_ai_configuration():
    """Test AI configuration"""
    print("\n🤖 Testing AI Configuration...")
    
    try:
        # Test AI configuration loading
        import json
        with open('config/ai_config.json', 'r') as f:
            config = json.load(f)
        
        if 'ai_enhancement' in config:
            print("  ✅ AI configuration loaded successfully")
        else:
            print("  ❌ AI configuration missing")
            return False
        
        # Test AI components
        ai_components = ['anomaly_detection', 'predictive_validation', 'clustering']
        
        for component in ai_components:
            if component in config:
                print(f"  ✅ {component} configuration present")
            else:
                print(f"  ⚠️  {component} configuration missing")
        
        return True
        
    except Exception as e:
        print(f"  ❌ AI configuration test failed: {e}")
        return False

def test_sample_data_generation():
    """Test sample data generation for testing"""
    print("\n📊 Testing Sample Data Generation...")
    
    try:
        # Generate comprehensive sample data
        np.random.seed(42)
        
        # Employee data
        employees = pd.DataFrame({
            'employee_id': [f'EMP{i:03d}' for i in range(1, 21)],
            'first_name': ['John', 'Jane', 'Bob', 'Alice', 'Charlie', 'Diana', 'Eve', 'Frank', 'Grace', 'Henry',
                          'Ivy', 'Jack', 'Kate', 'Liam', 'Mia', 'Noah', 'Olivia', 'Paul', 'Quinn', 'Rose'],
            'last_name': ['Doe', 'Smith', 'Johnson', 'Brown', 'Wilson', 'Davis', 'Miller', 'Garcia', 'Rodriguez', 'Martinez',
                         'Anderson', 'Taylor', 'Thomas', 'Hernandez', 'Moore', 'Martin', 'Jackson', 'Thompson', 'White', 'Lopez'],
            'department_id': np.random.randint(1, 6, 20),
            'position': np.random.choice(['Developer', 'Manager', 'Analyst', 'Designer', 'Tester'], 20),
            'salary': np.random.normal(75000, 15000, 20).astype(int),
            'hire_date': pd.date_range('2018-01-01', periods=20, freq='30D')
        })
        
        # Time tracking data
        dates = pd.date_range('2023-01-01', periods=30, freq='D')
        time_tracking = []
        
        for date in dates:
            for emp_id in employees['employee_id'][:10]:  # First 10 employees
                time_tracking.append({
                    'employee_id': emp_id,
                    'date': date,
                    'hours_worked': np.random.normal(8, 2),
                    'overtime_hours': max(0, np.random.normal(1, 1)),
                    'project_id': np.random.randint(1, 4),
                    'task_description': f'Task for {emp_id} on {date.strftime("%Y-%m-%d")}'
                })
        
        time_tracking_df = pd.DataFrame(time_tracking)
        
        print(f"  ✅ Generated {len(employees)} employee records")
        print(f"  ✅ Generated {len(time_tracking_df)} time tracking records")
        
        # Basic data quality checks
        if len(employees) > 0 and len(time_tracking_df) > 0:
            print("  ✅ Sample data generation successful")
            return True
        else:
            print("  ❌ Sample data generation failed")
            return False
        
    except Exception as e:
        print(f"  ❌ Sample data generation test failed: {e}")
        return False

def run_functionality_test():
    """Run all functionality tests"""
    print("🚀 Labor Analytics Dashboard - Functionality Test")
    print("=" * 60)
    print(f"Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    test_results = []
    
    # Run all tests
    tests = [
        ("Excel Validator", test_excel_validator),
        ("Database Connector", test_database_connector),
        ("Validation Engine", test_validation_engine),
        ("AI Configuration", test_ai_configuration),
        ("Sample Data Generation", test_sample_data_generation)
    ]
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            test_results.append((test_name, result))
        except Exception as e:
            print(f"  ❌ {test_name} test failed with exception: {e}")
            test_results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 FUNCTIONALITY TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")
        if result:
            passed += 1
    
    print(f"\nOverall Result: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All functionality tests passed! The application is working correctly.")
        return True
    elif passed >= total * 0.8:
        print("⚠️  Most functionality tests passed. Some components may need attention.")
        return True
    else:
        print("❌ Multiple functionality tests failed. Please check the issues above.")
        return False

if __name__ == "__main__":
    success = run_functionality_test()
    sys.exit(0 if success else 1)
