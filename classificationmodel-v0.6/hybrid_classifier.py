#!/usr/bin/env python3
"""
Production Hybrid Classification System v0.6
Enhanced multi-level classification with L5-based L2 category overwrite
"""

import pandas as pd
import numpy as np
import pickle
import warnings
from sklearn.metrics import accuracy_score, classification_report
import os
from pathlib import Path

warnings.filterwarnings('ignore')

class HybridClassificationSystem:
    """
    Production-ready hybrid classification system that combines L2, L4, and L5 models
    with taxonomy-based category adjustment when L5 confidence > 65%
    """
    
    def __init__(self, model_dir="models", data_dir="data"):
        """Initialize the hybrid classification system"""
        self.model_dir = Path(model_dir)
        self.data_dir = Path(data_dir)
        self.models = {}
        self.taxonomy_mappings = {}
        self.is_loaded = False
        
        # Model accuracy for confidence calibration
        self.model_accuracy = {
            'l2': 0.8028,
            'l4': 0.5469,
            'l5': 0.5030
        }
        
    def load_models(self):
        """Load all classification models"""
        try:
            # Load L2 model (production SVM enhanced)
            l2_path = self.model_dir / "production_svm_enhanced.pkl"
            with open(l2_path, 'rb') as f:
                self.models['l2'] = pickle.load(f)
            
            # Load L4 model
            l4_path = self.model_dir / "l4_simplified_model.pkl"
            with open(l4_path, 'rb') as f:
                self.models['l4'] = pickle.load(f)
                
            # Load L5 model
            l5_path = self.model_dir / "l5_simplified_model.pkl"
            with open(l5_path, 'rb') as f:
                self.models['l5'] = pickle.load(f)
                
            print("✅ All models loaded successfully")
            return True
            
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            return False
    
    def load_taxonomy_mappings(self):
        """Load taxonomy mappings from categorization file"""
        try:
            categorization_file = self.data_dir / "Categorization File.xlsx"
            taxonomy_df = pd.read_excel(categorization_file, sheet_name='Taxonomy')
            
            # Create mappings: Category 2=L2, Category 4=L4, Category 5=L5
            self.taxonomy_mappings = {
                'l5_to_l4': dict(zip(taxonomy_df['Category 5'], taxonomy_df['Category 4'])),
                'l5_to_l2': dict(zip(taxonomy_df['Category 5'], taxonomy_df['Category 2'])),
                'l4_to_l2': dict(zip(taxonomy_df['Category 4'], taxonomy_df['Category 2']))
            }
            
            print(f"✅ Taxonomy mappings loaded:")
            print(f"   • L5 → L4: {len(self.taxonomy_mappings['l5_to_l4'])} mappings")
            print(f"   • L5 → L2: {len(self.taxonomy_mappings['l5_to_l2'])} mappings")
            print(f"   • L4 → L2: {len(self.taxonomy_mappings['l4_to_l2'])} mappings")
            return True
            
        except Exception as e:
            print(f"❌ Error loading taxonomy: {e}")
            return False
    
    def initialize(self):
        """Initialize the complete system"""
        print("🚀 Initializing Hybrid Classification System v0.6")
        print("=" * 60)
        
        success = True
        success &= self.load_models()
        success &= self.load_taxonomy_mappings()
        
        if success:
            self.is_loaded = True
            print("✅ System initialization complete!")
        else:
            print("❌ System initialization failed!")
            
        return success
    
    def _get_l2_prediction(self, description):
        """Get L2 prediction and confidence"""
        text_vectorized = self.models['l2']['vectorizer'].transform([description])
        
        if hasattr(self.models['l2']['model'], 'predict_proba'):
            proba = self.models['l2']['model'].predict_proba(text_vectorized)[0]
            pred_idx = np.argmax(proba)
            prediction = self.models['l2']['label_encoder'].inverse_transform([pred_idx])[0]
            confidence = float(np.max(proba) * self.model_accuracy['l2'])
        else:
            prediction = self.models['l2']['model'].predict(text_vectorized)[0]
            confidence = self.model_accuracy['l2']  # Default confidence
            
        return prediction, confidence
    
    def _get_l4_prediction(self, description):
        """Get L4 prediction and confidence"""
        text_vectorized = self.models['l4']['vectorizer'].transform([description])
        
        if hasattr(self.models['l4']['model'], 'predict_proba'):
            proba = self.models['l4']['model'].predict_proba(text_vectorized)[0]
            pred_idx = np.argmax(proba)
            prediction = self.models['l4']['model'].classes_[pred_idx]
            confidence = float(np.max(proba) * self.model_accuracy['l4'])
        else:
            # Use decision function for SVM models
            decision = self.models['l4']['model'].decision_function(text_vectorized)[0]
            pred_idx = np.argmax(decision)
            prediction = self.models['l4']['model'].classes_[pred_idx]
            # Normalize decision function scores to 0-1 range
            decision_scores = np.array(decision)
            exp_scores = np.exp(decision_scores - np.max(decision_scores))
            softmax_proba = exp_scores / np.sum(exp_scores)
            confidence = float(np.max(softmax_proba) * self.model_accuracy['l4'])
            
        return prediction, confidence
    
    def _get_l5_prediction(self, description):
        """Get L5 prediction and confidence"""
        text_vectorized = self.models['l5']['vectorizer'].transform([description])
        
        if hasattr(self.models['l5']['model'], 'predict_proba'):
            proba = self.models['l5']['model'].predict_proba(text_vectorized)[0]
            pred_idx = np.argmax(proba)
            prediction = self.models['l5']['model'].classes_[pred_idx]
            confidence = float(np.max(proba) * self.model_accuracy['l5'])
        else:
            # Use decision function for SVM models
            decision = self.models['l5']['model'].decision_function(text_vectorized)[0]
            pred_idx = np.argmax(decision)
            prediction = self.models['l5']['model'].classes_[pred_idx]
            # Normalize decision function scores to 0-1 range
            decision_scores = np.array(decision)
            exp_scores = np.exp(decision_scores - np.max(decision_scores))
            softmax_proba = exp_scores / np.sum(exp_scores)
            confidence = float(np.max(softmax_proba) * self.model_accuracy['l5'])
            
        return prediction, confidence
    
    def predict_single(self, description, confidence_threshold_high=0.75, 
                      confidence_threshold_medium=0.50, l5_override_threshold=0.65):
        """
        Predict classification for a single item with hybrid enhancement
        
        Args:
            description: Item description text
            confidence_threshold_high: Threshold for high confidence (use L2 directly)
            confidence_threshold_medium: Threshold for medium confidence (try enhancement)
            l5_override_threshold: L5 confidence threshold for category override
            
        Returns:
            dict: Classification results with all predictions and confidence scores
        """
        if not self.is_loaded:
            raise RuntimeError("System not initialized. Call initialize() first.")
        
        # Get predictions from all models
        l2_prediction, l2_confidence = self._get_l2_prediction(description)
        l4_prediction, l4_confidence = self._get_l4_prediction(description)
        l5_prediction, l5_confidence = self._get_l5_prediction(description)
        
        # Initialize result
        result = {
            'description': description,
            'l2_prediction': l2_prediction,
            'l2_confidence': l2_confidence,
            'l4_prediction': l4_prediction,
            'l4_confidence': l4_confidence,
            'l5_prediction': l5_prediction,
            'l5_confidence': l5_confidence,
            'final_prediction': l2_prediction,
            'final_confidence': l2_confidence,
            'strategy': 'high_confidence_l2',
            'improvement': 0.0,
            'category_adjusted': False,
            'adjusted_l2_category': l2_prediction,
            'adjusted_l4_category': l4_prediction
        }
        
        # Hybrid enhancement logic
        if l2_confidence >= confidence_threshold_high:
            # High confidence L2 - use as is
            result['strategy'] = 'high_confidence_l2'
            
        else:
            # Medium or low confidence - try enhancement
            l4_improvement = max(0, l4_confidence - l2_confidence)
            l5_improvement = max(0, l5_confidence - l2_confidence)
            
            # Choose best enhancement (prefer L5 if both available)
            if l5_improvement > l4_improvement and l5_improvement > 0.05:
                result['final_prediction'] = l5_prediction
                result['final_confidence'] = l5_confidence
                result['improvement'] = l5_improvement
                
                if l2_confidence < confidence_threshold_medium:
                    result['strategy'] = 'low_confidence_replaced_by_l5'
                else:
                    result['strategy'] = 'medium_confidence_enhanced_by_l5'
                    
            elif l4_improvement > 0.05:
                result['final_prediction'] = l4_prediction
                result['final_confidence'] = l4_confidence
                result['improvement'] = l4_improvement
                
                if l2_confidence < confidence_threshold_medium:
                    result['strategy'] = 'low_confidence_enhanced_by_l4'
                else:
                    result['strategy'] = 'medium_confidence_enhanced_by_l4'
        
        # CRITICAL: L5-based category override when L5 confidence > 65%
        if l5_confidence >= l5_override_threshold:
            result['category_adjusted'] = True
            
            # Override L2 category using L5→L2 taxonomy mapping
            if l5_prediction in self.taxonomy_mappings['l5_to_l2']:
                result['adjusted_l2_category'] = self.taxonomy_mappings['l5_to_l2'][l5_prediction]
            
            # Override L4 category using L5→L4 taxonomy mapping
            if l5_prediction in self.taxonomy_mappings['l5_to_l4']:
                result['adjusted_l4_category'] = self.taxonomy_mappings['l5_to_l4'][l5_prediction]
        
        # If using L4 enhancement, also adjust L2 using L4→L2 mapping
        elif 'l4' in result['strategy'] and l4_prediction in self.taxonomy_mappings['l4_to_l2']:
            result['adjusted_l2_category'] = self.taxonomy_mappings['l4_to_l2'][l4_prediction]
        
        return result
    
    def predict_batch(self, descriptions, **kwargs):
        """
        Predict classifications for a batch of descriptions
        
        Args:
            descriptions: List of description strings or pandas Series
            **kwargs: Additional arguments for predict_single
            
        Returns:
            list: List of prediction results
        """
        if not self.is_loaded:
            raise RuntimeError("System not initialized. Call initialize() first.")
            
        results = []
        for desc in descriptions:
            try:
                result = self.predict_single(desc, **kwargs)
                results.append(result)
            except Exception as e:
                print(f"Warning: Error predicting for '{desc[:50]}...': {e}")
                # Add error result
                error_result = {
                    'description': desc,
                    'error': str(e),
                    'final_prediction': 'ERROR',
                    'final_confidence': 0.0
                }
                results.append(error_result)
        
        return results
    
    def predict_dataframe(self, df, description_column='Description', **kwargs):
        """
        Predict classifications for a pandas DataFrame
        
        Args:
            df: Input DataFrame
            description_column: Name of the column containing descriptions
            **kwargs: Additional arguments for predict_single
            
        Returns:
            pandas.DataFrame: DataFrame with prediction results
        """
        if not self.is_loaded:
            raise RuntimeError("System not initialized. Call initialize() first.")
            
        if description_column not in df.columns:
            raise ValueError(f"Column '{description_column}' not found in DataFrame")
        
        # Get predictions for all descriptions
        descriptions = df[description_column].tolist()
        results = self.predict_batch(descriptions, **kwargs)
        
        # Convert to DataFrame
        results_df = pd.DataFrame(results)
        
        # Add original DataFrame columns
        for col in df.columns:
            if col not in results_df.columns:
                results_df[col] = df[col].values
        
        return results_df
    
    def get_system_stats(self):
        """Get system statistics and configuration"""
        if not self.is_loaded:
            return {"status": "not_initialized"}
            
        return {
            "status": "initialized",
            "models_loaded": list(self.models.keys()),
            "taxonomy_mappings": {k: len(v) for k, v in self.taxonomy_mappings.items()},
            "model_accuracy": self.model_accuracy,
            "version": "0.6"
        }


def main():
    """Demo function showing how to use the system"""
    print("🧪 HYBRID CLASSIFICATION SYSTEM v0.6 - DEMO")
    print("=" * 60)
    
    # Initialize system
    classifier = HybridClassificationSystem()
    
    if not classifier.initialize():
        print("❌ Failed to initialize system")
        return
    
    # Test with sample descriptions
    test_descriptions = [
        "LED HAND LAMP, 24VDC, 7W COMET MAKE",
        "HYDRAULIC OIL BULK IDEMITSU",
        "WELDING ELECTRODE 3.15MM",
        "STEEL PIPE 100MM DIAMETER",
        "DIGITAL MULTIMETER FLUKE"
    ]
    
    print("\n🧪 Testing with sample descriptions:")
    print("-" * 40)
    
    for desc in test_descriptions:
        result = classifier.predict_single(desc)
        print(f"\n📝 Description: {desc}")
        print(f"🎯 Final Category: {result['adjusted_l2_category']}")
        print(f"📊 Confidence: {result['final_confidence']:.3f}")
        print(f"🔧 Strategy: {result['strategy']}")
        if result['category_adjusted']:
            print(f"🔄 Category Adjusted: L5 confidence {result['l5_confidence']:.3f} ≥ 0.65")
    
    print(f"\n📊 System Statistics:")
    stats = classifier.get_system_stats()
    for key, value in stats.items():
        print(f"   • {key}: {value}")


if __name__ == "__main__":
    main()
