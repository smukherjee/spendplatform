#!/usr/bin/env python3
"""
Production Classifier Command Line Interface v0.6
Command-line interface for the hybrid classification system
"""

import argparse
import json
import sys
import time
from pathlib import Path
from production_api import ProductionClassificationAPI

def classify_file_command(args):
    """Handle file classification command"""
    print(f"📁 Classifying file: {args.input_file}")
    
    # Initialize API
    api = ProductionClassificationAPI()
    if not api.initialize():
        print("❌ Failed to initialize classification API")
        return 1
    
    try:
        # Set confidence thresholds if provided
        thresholds = {}
        if args.high_threshold:
            thresholds['high'] = args.high_threshold
        if args.medium_threshold:
            thresholds['medium'] = args.medium_threshold
        if args.l5_override_threshold:
            thresholds['l5_override'] = args.l5_override_threshold
        
        # Classify file
        output_path = api.classify_file(
            file_path=args.input_file,
            description_column=args.description_column,
            output_path=args.output_file,
            confidence_thresholds=thresholds if thresholds else None
        )
        
        print(f"✅ Classification completed successfully")
        print(f"📊 Results saved to: {output_path}")
        
        # Show stats
        stats = api.get_api_stats()
        print(f"📈 Processed {stats['successful_predictions']} items")
        print(f"🔄 Category overrides: {stats['category_overrides']} ({stats['override_rate']:.1f}%)")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error during classification: {e}")
        return 1

def classify_text_command(args):
    """Handle single text classification command"""
    print(f"📝 Classifying text: {args.text}")
    
    # Initialize API
    api = ProductionClassificationAPI()
    if not api.initialize():
        print("❌ Failed to initialize classification API")
        return 1
    
    try:
        # Set confidence thresholds if provided
        thresholds = {}
        if args.high_threshold:
            thresholds['high'] = args.high_threshold
        if args.medium_threshold:
            thresholds['medium'] = args.medium_threshold
        if args.l5_override_threshold:
            thresholds['l5_override'] = args.l5_override_threshold
        
        # Classify text
        result = api.classify_single(
            description=args.text,
            confidence_thresholds=thresholds if thresholds else None
        )
        
        if result['status'] == 'success':
            print(f"✅ Classification successful")
            print(f"🎯 L2 Category: {result['prediction']['l2_category']}")
            print(f"🎯 L4 Category: {result['prediction']['l4_category']}")
            print(f"🎯 L5 Category: {result['prediction']['l5_category']}")
            print(f"📊 Confidence: {result['prediction']['confidence']:.3f}")
            print(f"🔧 Strategy: {result['prediction']['strategy']}")
            
            if result['prediction']['category_overridden']:
                print(f"🔄 L2 Category overridden by L5 (confidence ≥ 65%)")
            
            if args.verbose:
                print(f"\n📋 Detailed Results:")
                print(json.dumps(result, indent=2))
        else:
            print(f"❌ Classification failed: {result.get('error', 'Unknown error')}")
            return 1
        
        return 0
        
    except Exception as e:
        print(f"❌ Error during classification: {e}")
        return 1

def health_check_command(args):
    """Handle health check command"""
    print("🏥 Performing system health check...")
    
    # Initialize API
    api = ProductionClassificationAPI()
    if not api.initialize():
        print("❌ System initialization failed")
        return 1
    
    try:
        health = api.health_check()
        
        print(f"Status: {health['status']}")
        print(f"Timestamp: {time.ctime(health['timestamp'])}")
        
        print("\n🧪 Health Checks:")
        for check, status in health.get('checks', {}).items():
            if check != 'error':
                status_icon = "✅" if status == "passed" else "❌"
                print(f"   {status_icon} {check}: {status}")
        
        if 'error' in health.get('checks', {}):
            print(f"❌ Error: {health['checks']['error']}")
        
        # Show system stats
        stats = api.get_api_stats()
        print(f"\n📊 System Statistics:")
        print(f"   • Total Predictions: {stats['total_predictions']}")
        print(f"   • Success Rate: {stats['success_rate']:.1f}%")
        print(f"   • Category Override Rate: {stats['override_rate']:.1f}%")
        print(f"   • Uptime: {stats['uptime_seconds']:.1f}s")
        
        return 0 if health['status'] == 'healthy' else 1
        
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return 1

def main():
    """Main CLI function"""
    parser = argparse.ArgumentParser(
        description="Hybrid Classification System v0.6 - Command Line Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Classify a file
  python cli.py classify-file input.xlsx --output output.xlsx
  
  # Classify single text
  python cli.py classify-text "LED HAND LAMP 24VDC" --verbose
  
  # Health check
  python cli.py health-check
  
  # Custom thresholds
  python cli.py classify-file data.xlsx --l5-override-threshold 0.70
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # File classification command
    file_parser = subparsers.add_parser('classify-file', help='Classify descriptions from a file')
    file_parser.add_argument('input_file', help='Input file path (Excel or CSV)')
    file_parser.add_argument('--output', '-o', dest='output_file', help='Output file path')
    file_parser.add_argument('--description-column', '-d', default='Description', 
                           help='Name of description column (default: Description)')
    file_parser.add_argument('--high-threshold', type=float, 
                           help='High confidence threshold (default: 0.75)')
    file_parser.add_argument('--medium-threshold', type=float,
                           help='Medium confidence threshold (default: 0.50)')
    file_parser.add_argument('--l5-override-threshold', type=float,
                           help='L5 override threshold (default: 0.65)')
    
    # Text classification command
    text_parser = subparsers.add_parser('classify-text', help='Classify a single text description')
    text_parser.add_argument('text', help='Text description to classify')
    text_parser.add_argument('--verbose', '-v', action='store_true',
                           help='Show detailed results')
    text_parser.add_argument('--high-threshold', type=float,
                           help='High confidence threshold (default: 0.75)')
    text_parser.add_argument('--medium-threshold', type=float,
                           help='Medium confidence threshold (default: 0.50)')
    text_parser.add_argument('--l5-override-threshold', type=float,
                           help='L5 override threshold (default: 0.65)')
    
    # Health check command
    health_parser = subparsers.add_parser('health-check', help='Perform system health check')
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Execute command
    if args.command == 'classify-file':
        return classify_file_command(args)
    elif args.command == 'classify-text':
        return classify_text_command(args)
    elif args.command == 'health-check':
        return health_check_command(args)
    else:
        print(f"❌ Unknown command: {args.command}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
