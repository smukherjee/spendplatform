#!/usr/bin/env python3
"""
Production Deployment Script for Hybrid Classification System v0.6
Automated setup and validation for production deployment
"""

import os
import sys
import json
import time
from pathlib import Path
from hybrid_classifier import HybridClassificationSystem
from production_api import ProductionClassificationAPI

def check_prerequisites():
    """Check system prerequisites"""
    print("🔍 Checking Prerequisites...")
    
    checks = {
        'python_version': sys.version_info >= (3, 7),
        'models_directory': Path('models').exists(),
        'data_directory': Path('data').exists(),
        'l2_model': Path('models/production_svm_enhanced.pkl').exists(),
        'l4_model': Path('models/l4_simplified_model.pkl').exists(),
        'l5_model': Path('models/l5_simplified_model.pkl').exists(),
        'taxonomy_file': Path('data/Categorization File.xlsx').exists(),
    }
    
    # Check required Python packages
    try:
        import pandas
        import numpy
        import sklearn
        import openpyxl
        checks['required_packages'] = True
    except ImportError as e:
        print(f"❌ Missing required package: {e}")
        checks['required_packages'] = False
    
    # Report results
    all_passed = True
    for check, passed in checks.items():
        status = "✅" if passed else "❌"
        print(f"   {status} {check.replace('_', ' ').title()}: {'PASS' if passed else 'FAIL'}")
        all_passed &= passed
    
    return all_passed

def validate_system():
    """Validate system functionality"""
    print("\n🧪 Validating System Functionality...")
    
    try:
        # Initialize system
        classifier = HybridClassificationSystem()
        if not classifier.initialize():
            print("❌ System initialization failed")
            return False
        
        # Test predictions
        test_cases = [
            "LED HAND LAMP 24VDC",
            "HYDRAULIC OIL BULK", 
            "WELDING ELECTRODE",
            "STEEL PIPE 100MM",
            "DIGITAL MULTIMETER"
        ]
        
        print("   Testing sample predictions...")
        for i, test_case in enumerate(test_cases, 1):
            result = classifier.predict_single(test_case)
            
            if 'error' in result:
                print(f"   ❌ Test {i} failed: {result['error']}")
                return False
            
            print(f"   ✅ Test {i}: {test_case[:20]}... → {result['adjusted_l2_category']}")
        
        # Test taxonomy mappings
        stats = classifier.get_system_stats()
        expected_mappings = {'l5_to_l4': 837, 'l5_to_l2': 837, 'l4_to_l2': 409}
        
        for mapping, expected_count in expected_mappings.items():
            actual_count = stats['taxonomy_mappings'].get(mapping, 0)
            if actual_count != expected_count:
                print(f"   ❌ Taxonomy mapping {mapping}: expected {expected_count}, got {actual_count}")
                return False
        
        print("   ✅ Taxonomy mappings validated")
        print("   ✅ All system tests passed")
        return True
        
    except Exception as e:
        print(f"❌ System validation failed: {e}")
        return False

def performance_benchmark():
    """Run performance benchmark"""
    print("\n⚡ Running Performance Benchmark...")
    
    try:
        api = ProductionClassificationAPI()
        if not api.initialize():
            print("❌ API initialization failed")
            return False
        
        # Single prediction benchmark
        test_item = "HYDRAULIC PUMP 50HP INDUSTRIAL"
        start_time = time.time()
        
        # Run multiple predictions
        num_predictions = 100
        for _ in range(num_predictions):
            result = api.classify_single(test_item)
            if result['status'] != 'success':
                print(f"❌ Prediction failed during benchmark")
                return False
        
        total_time = time.time() - start_time
        predictions_per_second = num_predictions / total_time
        
        print(f"   ✅ Single prediction performance: {predictions_per_second:.1f} predictions/second")
        print(f"   ✅ Average prediction time: {(total_time/num_predictions)*1000:.1f}ms")
        
        # Batch prediction benchmark
        batch_items = [test_item] * 50
        start_time = time.time()
        batch_results = api.classify_batch(batch_items)
        batch_time = time.time() - start_time
        batch_rate = len(batch_items) / batch_time
        
        print(f"   ✅ Batch prediction performance: {batch_rate:.1f} predictions/second")
        
        return True
        
    except Exception as e:
        print(f"❌ Performance benchmark failed: {e}")
        return False

def create_production_config():
    """Create production configuration"""
    print("\n⚙️ Creating Production Configuration...")
    
    config = {
        "deployment": {
            "environment": "production",
            "version": "0.6",
            "deployment_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "models": {
            "l2_model_path": "models/production_svm_enhanced.pkl",
            "l4_model_path": "models/l4_simplified_model.pkl", 
            "l5_model_path": "models/l5_simplified_model.pkl",
            "taxonomy_path": "data/Categorization File.xlsx"
        },
        "thresholds": {
            "high_confidence": 0.75,
            "medium_confidence": 0.50,
            "l5_override": 0.65,
            "minimum_improvement": 0.05
        },
        "performance": {
            "max_batch_size": 1000,
            "prediction_timeout": 30,
            "memory_limit": "1GB"
        },
        "logging": {
            "level": "INFO",
            "file": "logs/production.log",
            "max_size": "100MB"
        }
    }
    
    # Create logs directory
    Path('logs').mkdir(exist_ok=True)
    
    # Save configuration
    with open('config/production.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print("   ✅ Production configuration saved to config/production.json")
    return True

def deployment_summary():
    """Generate deployment summary"""
    print("\n📊 DEPLOYMENT SUMMARY")
    print("=" * 60)
    
    # System information
    api = ProductionClassificationAPI()
    api.initialize()
    stats = api.get_system_stats()
    
    print(f"🎯 System Status: {stats['status'].upper()}")
    print(f"📦 Version: v{stats['system']['version']}")
    print(f"🤖 Models Loaded: {len(stats['system']['models_loaded'])}")
    print(f"🗺️ Taxonomy Mappings: {sum(stats['system']['taxonomy_mappings'].values())}")
    
    # Capabilities
    print(f"\n🚀 Production Capabilities:")
    print(f"   • Multi-level classification (L2, L4, L5)")
    print(f"   • L5-based category override (threshold: 65%)")
    print(f"   • Taxonomy-aware hierarchical mapping")
    print(f"   • Batch processing support")
    print(f"   • Command-line interface")
    print(f"   • Health monitoring")
    
    # Usage examples
    print(f"\n💡 Usage Examples:")
    print(f"   # Classify single item")
    print(f"   python cli.py classify-text 'LED LAMP 24VDC'")
    print(f"   ")
    print(f"   # Classify file")
    print(f"   python cli.py classify-file data.xlsx --output results.xlsx")
    print(f"   ")
    print(f"   # Health check")
    print(f"   python cli.py health-check")
    
    print(f"\n✅ System ready for production use!")

def main():
    """Main deployment function"""
    print("🚀 HYBRID CLASSIFICATION SYSTEM v0.6 - PRODUCTION DEPLOYMENT")
    print("=" * 70)
    
    # Run deployment checks
    steps = [
        ("Prerequisites", check_prerequisites),
        ("System Validation", validate_system),
        ("Performance Benchmark", performance_benchmark),
        ("Production Config", create_production_config)
    ]
    
    for step_name, step_func in steps:
        print(f"\n{'='*20} {step_name} {'='*20}")
        
        if not step_func():
            print(f"\n❌ DEPLOYMENT FAILED at step: {step_name}")
            print("Please address the issues above and run deployment again.")
            return 1
    
    # Success - show deployment summary
    deployment_summary()
    
    print(f"\n{'='*70}")
    print("🎉 PRODUCTION DEPLOYMENT SUCCESSFUL!")
    print("The Hybrid Classification System v0.6 is ready for production use.")
    print("=" * 70)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
