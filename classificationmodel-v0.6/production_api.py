#!/usr/bin/env python3
"""
Production API for Hybrid Classification System v0.6
Fast, reliable classification service with L5-based L2 category override
"""

import pandas as pd
import numpy as np
import json
from typing import List, Dict, Any, Optional
from pathlib import Path
import time
from hybrid_classifier import HybridClassificationSystem

class ProductionClassificationAPI:
    """
    Production-ready API wrapper for the hybrid classification system
    Optimized for performance and reliability
    """
    
    def __init__(self, model_dir="models", data_dir="data"):
        """Initialize the production API"""
        self.classifier = HybridClassificationSystem(model_dir, data_dir)
        self.is_initialized = False
        self.stats = {
            'total_predictions': 0,
            'successful_predictions': 0,
            'failed_predictions': 0,
            'category_overrides': 0,
            'start_time': None
        }
        
    def initialize(self) -> bool:
        """Initialize the classification system"""
        try:
            start_time = time.time()
            success = self.classifier.initialize()
            
            if success:
                self.is_initialized = True
                self.stats['start_time'] = time.time()
                init_time = time.time() - start_time
                print(f"✅ Production API initialized in {init_time:.2f}s")
            
            return success
            
        except Exception as e:
            print(f"❌ API initialization failed: {e}")
            return False
    
    def classify_single(self, description: str, 
                       confidence_thresholds: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Classify a single item description
        
        Args:
            description: Item description text
            confidence_thresholds: Optional thresholds for classification
            
        Returns:
            dict: Classification result with metadata
        """
        if not self.is_initialized:
            return {
                'error': 'API not initialized',
                'status': 'error'
            }
        
        try:
            # Set default thresholds
            thresholds = confidence_thresholds or {
                'high': 0.75,
                'medium': 0.50,
                'l5_override': 0.65
            }
            
            # Get prediction
            start_time = time.time()
            result = self.classifier.predict_single(
                description,
                confidence_threshold_high=thresholds['high'],
                confidence_threshold_medium=thresholds['medium'],
                l5_override_threshold=thresholds['l5_override']
            )
            
            prediction_time = time.time() - start_time
            
            # Update stats
            self.stats['total_predictions'] += 1
            self.stats['successful_predictions'] += 1
            if result['category_adjusted']:
                self.stats['category_overrides'] += 1
            
            # Format response
            response = {
                'status': 'success',
                'prediction': {
                    'l2_category': result['adjusted_l2_category'],
                    'l4_category': result['adjusted_l4_category'],
                    'l5_category': result['l5_prediction'],
                    'confidence': result['final_confidence'],
                    'strategy': result['strategy'],
                    'category_overridden': result['category_adjusted'],
                    'improvement': result['improvement']
                },
                'metadata': {
                    'prediction_time_ms': round(prediction_time * 1000, 2),
                    'model_predictions': {
                        'l2': {'category': result['l2_prediction'], 'confidence': result['l2_confidence']},
                        'l4': {'category': result['l4_prediction'], 'confidence': result['l4_confidence']},
                        'l5': {'category': result['l5_prediction'], 'confidence': result['l5_confidence']}
                    }
                }
            }
            
            return response
            
        except Exception as e:
            self.stats['failed_predictions'] += 1
            return {
                'error': str(e),
                'status': 'error',
                'description': description
            }
    
    def classify_batch(self, descriptions: List[str], 
                      confidence_thresholds: Optional[Dict[str, float]] = None,
                      include_metadata: bool = False) -> List[Dict[str, Any]]:
        """
        Classify a batch of descriptions
        
        Args:
            descriptions: List of description strings
            confidence_thresholds: Optional thresholds for classification
            include_metadata: Whether to include detailed metadata
            
        Returns:
            list: List of classification results
        """
        if not self.is_initialized:
            return [{'error': 'API not initialized', 'status': 'error'} for _ in descriptions]
        
        results = []
        start_time = time.time()
        
        for desc in descriptions:
            if include_metadata:
                result = self.classify_single(desc, confidence_thresholds)
            else:
                # Lightweight prediction for batch processing
                try:
                    pred = self.classifier.predict_single(desc)
                    result = {
                        'status': 'success',
                        'l2_category': pred['adjusted_l2_category'],
                        'l4_category': pred['adjusted_l4_category'],
                        'l5_category': pred['l5_prediction'],
                        'confidence': pred['final_confidence'],
                        'category_overridden': pred['category_adjusted']
                    }
                    
                    if pred['category_adjusted']:
                        self.stats['category_overrides'] += 1
                        
                except Exception as e:
                    result = {'error': str(e), 'status': 'error'}
                    self.stats['failed_predictions'] += 1
                    
                self.stats['total_predictions'] += 1
                if result['status'] == 'success':
                    self.stats['successful_predictions'] += 1
            
            results.append(result)
        
        batch_time = time.time() - start_time
        print(f"⚡ Batch processed: {len(descriptions)} items in {batch_time:.2f}s "
              f"({len(descriptions)/batch_time:.1f} items/sec)")
        
        return results
    
    def classify_file(self, file_path: str, description_column: str = 'Description',
                     output_path: Optional[str] = None,
                     confidence_thresholds: Optional[Dict[str, float]] = None) -> str:
        """
        Classify descriptions from a file
        
        Args:
            file_path: Path to input file (Excel or CSV)
            description_column: Name of description column
            output_path: Optional output file path
            confidence_thresholds: Optional thresholds for classification
            
        Returns:
            str: Path to output file
        """
        if not self.is_initialized:
            raise RuntimeError("API not initialized")
        
        # Read input file
        file_path = Path(file_path)
        if file_path.suffix.lower() in ['.xlsx', '.xls']:
            df = pd.read_excel(file_path)
        elif file_path.suffix.lower() == '.csv':
            df = pd.read_csv(file_path)
        else:
            raise ValueError("Supported file formats: Excel (.xlsx, .xls) or CSV (.csv)")
        
        if description_column not in df.columns:
            raise ValueError(f"Column '{description_column}' not found in file")
        
        print(f"📁 Processing file: {file_path}")
        print(f"📊 Total items: {len(df)}")
        
        # Get predictions
        start_time = time.time()
        results_df = self.classifier.predict_dataframe(df, description_column, 
                                                      **(confidence_thresholds or {}))
        
        processing_time = time.time() - start_time
        print(f"⚡ Processing completed in {processing_time:.2f}s")
        
        # Generate output path
        if output_path is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_path = file_path.parent / f"classified_output_{timestamp}.xlsx"
        
        # Save results
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # Main results
            results_df.to_excel(writer, sheet_name='Classification_Results', index=False)
            
            # Summary statistics
            summary_data = {
                'Metric': [
                    'Total Items',
                    'Successfully Classified',
                    'Category Overrides (L5 > 65%)',
                    'High Confidence L2',
                    'Medium Enhanced',
                    'Low Enhanced',
                    'Processing Time (seconds)',
                    'Items per Second'
                ],
                'Value': [
                    len(results_df),
                    len(results_df[results_df['final_confidence'] > 0]),
                    len(results_df[results_df['category_adjusted'] == True]),
                    len(results_df[results_df['strategy'] == 'high_confidence_l2']),
                    len(results_df[results_df['strategy'].str.contains('medium', na=False)]),
                    len(results_df[results_df['strategy'].str.contains('low', na=False)]),
                    round(processing_time, 2),
                    round(len(df) / processing_time, 1)
                ]
            }
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
        
        print(f"💾 Results saved to: {output_path}")
        return str(output_path)
    
    def get_api_stats(self) -> Dict[str, Any]:
        """Get API performance statistics"""
        if not self.is_initialized:
            return {'status': 'not_initialized'}
        
        uptime = time.time() - self.stats['start_time'] if self.stats['start_time'] else 0
        
        stats = {
            'status': 'operational',
            'uptime_seconds': round(uptime, 2),
            'total_predictions': self.stats['total_predictions'],
            'successful_predictions': self.stats['successful_predictions'],
            'failed_predictions': self.stats['failed_predictions'],
            'success_rate': round(
                self.stats['successful_predictions'] / max(1, self.stats['total_predictions']) * 100, 2
            ),
            'category_overrides': self.stats['category_overrides'],
            'override_rate': round(
                self.stats['category_overrides'] / max(1, self.stats['successful_predictions']) * 100, 2
            ),
            'predictions_per_second': round(
                self.stats['total_predictions'] / max(1, uptime), 2
            ) if uptime > 0 else 0
        }
        
        # Add system stats
        system_stats = self.classifier.get_system_stats()
        stats.update({'system': system_stats})
        
        return stats
    
    def health_check(self) -> Dict[str, Any]:
        """Perform system health check"""
        health = {
            'status': 'healthy' if self.is_initialized else 'unhealthy',
            'timestamp': time.time(),
            'checks': {}
        }
        
        if self.is_initialized:
            try:
                # Test prediction
                test_result = self.classify_single("TEST ITEM FOR HEALTH CHECK")
                health['checks']['prediction'] = 'passed' if test_result['status'] == 'success' else 'failed'
                
                # Check models
                health['checks']['models'] = 'passed' if len(self.classifier.models) == 3 else 'failed'
                
                # Check taxonomy
                health['checks']['taxonomy'] = 'passed' if len(self.classifier.taxonomy_mappings) == 3 else 'failed'
                
            except Exception as e:
                health['checks']['error'] = str(e)
                health['status'] = 'unhealthy'
        
        return health


def main():
    """Demo of the production API"""
    print("🚀 PRODUCTION CLASSIFICATION API v0.6 - DEMO")
    print("=" * 60)
    
    # Initialize API
    api = ProductionClassificationAPI()
    
    if not api.initialize():
        print("❌ Failed to initialize API")
        return
    
    # Test single classification
    print("\n🧪 Testing single classification:")
    result = api.classify_single("LED HAND LAMP, 24VDC, 7W COMET MAKE")
    print(f"Result: {json.dumps(result, indent=2)}")
    
    # Test batch classification
    print("\n🧪 Testing batch classification:")
    test_items = [
        "HYDRAULIC OIL BULK IDEMITSU",
        "WELDING ELECTRODE 3.15MM",
        "DIGITAL MULTIMETER FLUKE"
    ]
    
    batch_results = api.classify_batch(test_items)
    for i, result in enumerate(batch_results):
        print(f"Item {i+1}: {result['l2_category']} (conf: {result['confidence']:.3f})")
    
    # Show API stats
    print("\n📊 API Statistics:")
    stats = api.get_api_stats()
    for key, value in stats.items():
        if key != 'system':
            print(f"   • {key}: {value}")
    
    # Health check
    print("\n🏥 Health Check:")
    health = api.health_check()
    print(f"Status: {health['status']}")
    for check, status in health.get('checks', {}).items():
        print(f"   • {check}: {status}")


if __name__ == "__main__":
    main()
