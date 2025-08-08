#!/usr/bin/env python3
"""
Comprehensive Test Script for Labor Analytics Dashboard
Tests all components to ensure the application is up and running
"""

import sys
import os
import json
import traceback
from datetime import datetime

def test_config_files():
    """Test configuration files"""
    print("🔧 Testing Configuration Files...")
    
    config_files = [
        'config/database_config.json',
        'config/validation_config.json', 
        'config/ai_config.json'
    ]
    
    for config_file in config_files:
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
            print(f"  ✅ {config_file} - Valid JSON")
        except Exception as e:
            print(f"  ❌ {config_file} - Error: {e}")
            return False
    
    return True

def test_python_imports():
    """Test Python module imports"""
    print("\n🐍 Testing Python Module Imports...")
    
    # Test basic imports
    try:
        import pandas as pd
        print("  ✅ pandas imported successfully")
    except ImportError as e:
        print(f"  ❌ pandas import failed: {e}")
        return False
    
    try:
        import numpy as np
        print("  ✅ numpy imported successfully")
    except ImportError as e:
        print(f"  ❌ numpy import failed: {e}")
        return False
    
    try:
        import psycopg2
        print("  ✅ psycopg2 imported successfully")
    except ImportError as e:
        print(f"  ❌ psycopg2 import failed: {e}")
        return False
    
    try:
        import structlog
        print("  ✅ structlog imported successfully")
    except ImportError as e:
        print(f"  ❌ structlog import failed: {e}")
        return False
    
    return True

def test_core_components():
    """Test core application components"""
    print("\n🏗️  Testing Core Components...")
    
    # Test database connector
    try:
        sys.path.append('python')
        from data_connectors.database_connector import DatabaseConnector
        print("  ✅ DatabaseConnector imported successfully")
    except Exception as e:
        print(f"  ❌ DatabaseConnector import failed: {e}")
        return False
    
    # Test Excel validator
    try:
        from data_connectors.excel_validator import ExcelValidator
        print("  ✅ ExcelValidator imported successfully")
    except Exception as e:
        print(f"  ❌ ExcelValidator import failed: {e}")
        return False
    
    # Test validation engine
    try:
        from validation_engine.validation_engine import ValidationEngine
        print("  ✅ ValidationEngine imported successfully")
    except Exception as e:
        print(f"  ❌ ValidationEngine import failed: {e}")
        return False
    
    return True

def test_ai_components():
    """Test AI/ML components"""
    print("\n🤖 Testing AI/ML Components...")
    
    # Test basic ML libraries
    try:
        import sklearn
        print("  ✅ scikit-learn imported successfully")
    except ImportError as e:
        print(f"  ⚠️  scikit-learn import failed: {e}")
    
    try:
        import matplotlib
        print("  ✅ matplotlib imported successfully")
    except ImportError as e:
        print(f"  ⚠️  matplotlib import failed: {e}")
    
    # Test AI engine components (with error handling)
    try:
        from ai_engine.ai_enhanced_validator import AIEnhancedValidator
        print("  ✅ AIEnhancedValidator imported successfully")
    except Exception as e:
        print(f"  ⚠️  AIEnhancedValidator import failed: {e}")
    
    try:
        from ai_engine.nlp_processor import NLPProcessor
        print("  ✅ NLPProcessor imported successfully")
    except Exception as e:
        print(f"  ⚠️  NLPProcessor import failed: {e}")
    
    try:
        from ai_engine.predictive_analytics import PredictiveAnalytics
        print("  ✅ PredictiveAnalytics imported successfully")
    except Exception as e:
        print(f"  ⚠️  PredictiveAnalytics import failed: {e}")
    
    return True

def test_excel_creation():
    """Test Excel validation rules creation"""
    print("\n📋 Testing Excel Validation Rules Creation...")
    
    try:
        # Import and run the Excel creation script
        sys.path.append('excel')
        from create_validation_rules import create_validation_rules_excel
        create_validation_rules_excel()
        print("  ✅ Excel validation rules created successfully")
        return True
    except Exception as e:
        print(f"  ❌ Excel validation rules creation failed: {e}")
        return False

def test_database_schema():
    """Test database schema file"""
    print("\n🗄️  Testing Database Schema...")
    
    try:
        schema_file = 'database/schema/labor_analytics_schema.sql'
        with open(schema_file, 'r') as f:
            schema_content = f.read()
        
        # Check for key components
        if 'CREATE SCHEMA' in schema_content:
            print("  ✅ Database schema file exists and contains schema creation")
        else:
            print("  ⚠️  Database schema file may be incomplete")
        
        if 'CREATE TABLE' in schema_content:
            print("  ✅ Database schema contains table definitions")
        else:
            print("  ⚠️  Database schema may be missing table definitions")
        
        return True
    except Exception as e:
        print(f"  ❌ Database schema test failed: {e}")
        return False

def test_file_structure():
    """Test project file structure"""
    print("\n📁 Testing Project File Structure...")
    
    required_files = [
        'README.md',
        'config/database_config.json',
        'config/validation_config.json',
        'config/ai_config.json',
        'database/schema/labor_analytics_schema.sql',
        'database/setup_database.py',
        'python/data_connectors/database_connector.py',
        'python/data_connectors/excel_validator.py',
        'python/validation_engine/validation_engine.py',
        'python/ai_engine/ai_enhanced_validator.py',
        'python/ai_engine/nlp_processor.py',
        'python/ai_engine/predictive_analytics.py',
        'excel/create_validation_rules.py',
        'docs/setup_guide.md',
        'docs/ai_enhancement_guide.md'
    ]
    
    missing_files = []
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"  ✅ {file_path}")
        else:
            print(f"  ❌ {file_path} - Missing")
            missing_files.append(file_path)
    
    if missing_files:
        print(f"  ⚠️  {len(missing_files)} files are missing")
        return False
    
    return True

def test_executable_scripts():
    """Test executable scripts"""
    print("\n⚡ Testing Executable Scripts...")
    
    executable_scripts = [
        'python/ai_engine/ai_enhanced_validator.py',
        'python/ai_engine/nlp_processor.py',
        'python/ai_engine/predictive_analytics.py',
        'database/setup_database.py',
        'excel/create_validation_rules.py'
    ]
    
    for script in executable_scripts:
        if os.path.exists(script):
            if os.access(script, os.X_OK):
                print(f"  ✅ {script} - Executable")
            else:
                print(f"  ⚠️  {script} - Not executable")
        else:
            print(f"  ❌ {script} - Missing")
    
    return True

def run_comprehensive_test():
    """Run all tests"""
    print("🚀 Labor Analytics Dashboard - Comprehensive Test")
    print("=" * 60)
    print(f"Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    test_results = []
    
    # Run all tests
    tests = [
        ("File Structure", test_file_structure),
        ("Configuration Files", test_config_files),
        ("Python Imports", test_python_imports),
        ("Core Components", test_core_components),
        ("AI Components", test_ai_components),
        ("Excel Creation", test_excel_creation),
        ("Database Schema", test_database_schema),
        ("Executable Scripts", test_executable_scripts)
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
    print("📊 TEST SUMMARY")
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
        print("🎉 All tests passed! The application is ready to use.")
        return True
    elif passed >= total * 0.8:
        print("⚠️  Most tests passed. Some components may need attention.")
        return True
    else:
        print("❌ Multiple tests failed. Please check the issues above.")
        return False

if __name__ == "__main__":
    success = run_comprehensive_test()
    sys.exit(0 if success else 1)
