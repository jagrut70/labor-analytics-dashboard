#!/usr/bin/env python3
"""
Main Runner Script for Labor Analytics Validation Engine
Executes validation with different modes: single run, scheduled, or API
"""

import argparse
import sys
import os
import signal
import time
from datetime import datetime
import structlog

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from validation_engine.validation_engine import ValidationEngine

logger = structlog.get_logger(__name__)


def setup_logging():
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


def signal_handler(signum, frame):
    """Handle shutdown signals gracefully"""
    logger.info(f"Received signal {signum}, shutting down gracefully...")
    sys.exit(0)


def run_single_validation():
    """Run a single validation cycle"""
    logger.info("Starting single validation run")
    
    try:
        engine = ValidationEngine()
        results = engine.run_full_validation()
        
        print("\n" + "="*60)
        print("VALIDATION RESULTS")
        print("="*60)
        print(f"Overall Status: {results['overall_status']}")
        print(f"Timestamp: {results['timestamp']}")
        print(f"Tables Validated: {len(results['tables_validated'])}")
        print(f"Total Errors: {len(results['errors'])}")
        print(f"Total Warnings: {len(results['warnings'])}")
        
        print("\n" + "-"*40)
        print("TABLE VALIDATION DETAILS")
        print("-"*40)
        for table_result in results['tables_validated']:
            status_icon = "✅" if table_result['valid'] else "❌"
            print(f"{status_icon} {table_result['table_name']}: "
                  f"{table_result['valid_records']}/{table_result['total_records']} valid records")
            
            if table_result['errors']:
                print(f"   Errors: {len(table_result['errors'])}")
        
        print("\n" + "-"*40)
        print("CROSS-VALIDATION RESULTS")
        print("-"*40)
        for cross_result in results['cross_validation_results']:
            status_icon = "✅" if cross_result['valid'] else "❌"
            print(f"{status_icon} {cross_result['rule_name']}: {cross_result['description']}")
            if not cross_result['valid']:
                print(f"   Invalid Records: {cross_result.get('invalid_records', 0)}")
        
        print("\n" + "-"*40)
        print("DATA QUALITY METRICS")
        print("-"*40)
        for metric_name, value in results['data_quality_metrics'].items():
            if isinstance(value, (int, float)) and not metric_name.endswith('_status'):
                status = results['data_quality_metrics'].get(f'{metric_name}_status', 'UNKNOWN')
                print(f"{metric_name}: {value:.2%} ({status})")
        
        # Export report
        report_path = engine.export_validation_report()
        print(f"\n📊 Validation report exported to: {report_path}")
        
        return results['overall_status'] == 'PASS'
        
    except Exception as e:
        logger.error(f"Error during single validation: {e}")
        print(f"❌ Validation failed with error: {e}")
        return False


def run_scheduled_validation(interval_minutes: int = 60):
    """Run scheduled validation"""
    logger.info(f"Starting scheduled validation every {interval_minutes} minutes")
    
    # Setup signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        engine = ValidationEngine()
        
        print(f"🔄 Starting scheduled validation every {interval_minutes} minutes")
        print("Press Ctrl+C to stop")
        
        # Run initial validation
        print("\n🚀 Running initial validation...")
        run_single_validation()
        
        # Start scheduled runs
        engine.start_scheduled_validation(interval_minutes)
        
    except KeyboardInterrupt:
        logger.info("Scheduled validation stopped by user")
        print("\n⏹️  Scheduled validation stopped")
    except Exception as e:
        logger.error(f"Error during scheduled validation: {e}")
        print(f"❌ Scheduled validation failed: {e}")


def run_api_mode():
    """Run validation engine in API mode"""
    logger.info("Starting validation engine in API mode")
    
    try:
        from fastapi import FastAPI, BackgroundTasks
        from fastapi.responses import JSONResponse
        import uvicorn
        
        app = FastAPI(title="Labor Analytics Validation API", version="1.0.0")
        engine = ValidationEngine()
        
        @app.get("/")
        async def root():
            return {"message": "Labor Analytics Validation API", "status": "running"}
        
        @app.get("/health")
        async def health_check():
            try:
                # Test database connection
                db_status = engine.db_connector.get_connection_status()
                return {
                    "status": "healthy",
                    "database": db_status,
                    "timestamp": datetime.now().isoformat()
                }
            except Exception as e:
                return JSONResponse(
                    status_code=500,
                    content={"status": "unhealthy", "error": str(e)}
                )
        
        @app.post("/validate")
        async def validate_data(background_tasks: BackgroundTasks):
            """Trigger validation run"""
            try:
                results = engine.run_full_validation()
                return {
                    "status": "success",
                    "validation_results": results,
                    "timestamp": datetime.now().isoformat()
                }
            except Exception as e:
                return JSONResponse(
                    status_code=500,
                    content={"status": "error", "error": str(e)}
                )
        
        @app.get("/history")
        async def get_validation_history(limit: int = 10):
            """Get validation history"""
            try:
                history = engine.get_validation_history(limit)
                return {
                    "status": "success",
                    "history": history,
                    "count": len(history)
                }
            except Exception as e:
                return JSONResponse(
                    status_code=500,
                    content={"status": "error", "error": str(e)}
                )
        
        @app.post("/export-report")
        async def export_report():
            """Export validation report"""
            try:
                report_path = engine.export_validation_report()
                return {
                    "status": "success",
                    "report_path": report_path,
                    "timestamp": datetime.now().isoformat()
                }
            except Exception as e:
                return JSONResponse(
                    status_code=500,
                    content={"status": "error", "error": str(e)}
                )
        
        print("🌐 Starting Validation API server...")
        print("📖 API Documentation available at: http://localhost:8000/docs")
        print("🔍 Health check available at: http://localhost:8000/health")
        
        uvicorn.run(app, host="0.0.0.0", port=8000)
        
    except ImportError:
        print("❌ FastAPI not installed. Install with: pip install fastapi uvicorn")
        return False
    except Exception as e:
        logger.error(f"Error starting API mode: {e}")
        print(f"❌ API mode failed: {e}")
        return False


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Labor Analytics Validation Engine")
    parser.add_argument(
        "--mode", 
        choices=["single", "scheduled", "api"], 
        default="single",
        help="Validation mode (default: single)"
    )
    parser.add_argument(
        "--interval", 
        type=int, 
        default=60,
        help="Interval in minutes for scheduled mode (default: 60)"
    )
    parser.add_argument(
        "--config", 
        type=str, 
        default="../config/validation_config.json",
        help="Path to validation configuration file"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    
    if args.verbose:
        structlog.configure(processors=[structlog.dev.ConsoleRenderer()])
    
    logger.info(f"Starting validation engine in {args.mode} mode")
    
    # Set environment variable for config path
    os.environ['VALIDATION_CONFIG_PATH'] = args.config
    
    try:
        if args.mode == "single":
            success = run_single_validation()
            sys.exit(0 if success else 1)
        elif args.mode == "scheduled":
            run_scheduled_validation(args.interval)
        elif args.mode == "api":
            run_api_mode()
        else:
            print(f"❌ Unknown mode: {args.mode}")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        print(f"❌ Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
