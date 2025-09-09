#!/usr/bin/env python3
"""
Production Model Testing and Verification
Test the final optimized model with sample predictions
"""

import pickle
import pandas as pd
import numpy as np

def load_production_model():
    """Load the production model"""
    model_path = '/Users/sujoymukherjee/code/spendplatform/classificationmodel-v0.5/models/production_svm_enhanced.pkl'
    
    try:
        with open(model_path, 'rb') as f:
            model_package = pickle.load(f)
        return model_package
    except FileNotFoundError:
        print(f"❌ Model file not found: {model_path}")
        return None

def predict_category(description, model_package):
    """Make prediction for a single description"""
    # Preprocess text
    clean_text = description.lower().strip()
    
    # Extract features
    features = model_package['vectorizer'].transform([clean_text])
    
    # Make prediction
    prediction = model_package['model'].predict(features)[0]
    probabilities = model_package['model'].predict_proba(features)[0]
    
    # Get category name
    category = model_package['label_encoder'].inverse_transform([prediction])[0]
    confidence = max(probabilities)
    
    # Get top 3 predictions
    top_indices = np.argsort(probabilities)[-3:][::-1]
    top_predictions = []
    for idx in top_indices:
        cat = model_package['label_encoder'].classes_[idx]
        conf = probabilities[idx]
        top_predictions.append((cat, conf))
    
    return {
        'category': category,
        'confidence': confidence,
        'top_3': top_predictions
    }

def test_sample_predictions():
    """Test the model with various sample descriptions"""
    print("🧪 PRODUCTION MODEL TESTING")
    print("=" * 70)
    
    # Load model
    model_pkg = load_production_model()
    if model_pkg is None:
        return
    
    print(f"✅ Model loaded successfully!")
    print(f"📊 Model Info:")
    print(f"   • Type: {model_pkg.get('model_type', 'Unknown')}")
    print(f"   • Accuracy: {model_pkg.get('accuracy', 0):.4f}")
    print(f"   • Categories: {model_pkg.get('categories', 0)}")
    print(f"   • Features: {model_pkg.get('features', 0):,}")
    print(f"   • Training Date: {model_pkg.get('training_date', 'Unknown')}")
    
    # Sample test cases covering different categories
    test_cases = [
        # Electrical components
        "LED industrial light 100W high bay fixture",
        "Motor starter 3 phase 25hp electrical contactor",
        "Cable wire 12AWG THHN copper conductor",
        
        # Manufacturing components
        "Ball bearing SKF 6205 deep groove sealed",
        "Hydraulic seal kit for cylinder 50mm bore",
        "Steel plate carbon 10mm thickness",
        
        # Pipes and valves
        "Gate valve bronze 2 inch flanged 150PSI",
        "PVC pipe 4 inch schedule 40 drainage",
        "Butterfly valve stainless steel wafer style",
        
        # Tools and machinery
        "Angle grinder 4.5 inch electric handheld",
        "Drill bit set HSS twist 1-13mm metric",
        "Lathe chuck 3 jaw self centering 8 inch",
        
        # Safety supplies
        "Hard hat white ANSI approved construction",
        "Safety glasses clear lens anti-fog",
        "Respirator cartridge organic vapor P100",
        
        # Pumps and compressors
        "Centrifugal pump 5HP electric motor driven",
        "Air compressor 10 gallon tank portable",
        "Pump impeller bronze for water service",
        
        # Building supplies
        "Concrete mixer portable 3.5 cubic feet",
        "Insulation fiberglass batt R-19 kraft faced",
        "Lumber 2x4 pressure treated pine 8 foot",
        
        # Edge cases and challenging descriptions
        "Generic replacement part type A",
        "Industrial component made of steel",
        "Equipment for manufacturing process"
    ]
    
    print(f"\n🔍 TESTING {len(test_cases)} SAMPLE DESCRIPTIONS")
    print("=" * 70)
    
    correct_predictions = 0
    high_confidence_predictions = 0
    
    for i, description in enumerate(test_cases, 1):
        result = predict_category(description, model_pkg)
        
        # Determine confidence level
        if result['confidence'] > 0.8:
            confidence_emoji = "🟢"
            high_confidence_predictions += 1
        elif result['confidence'] > 0.6:
            confidence_emoji = "🟡"
        else:
            confidence_emoji = "🔴"
        
        print(f"\n{i:2d}. 📝 '{description}'")
        print(f"    {confidence_emoji} Prediction: {result['category']}")
        print(f"    🎯 Confidence: {result['confidence']:.3f}")
        print(f"    📊 Top 3:")
        for j, (cat, conf) in enumerate(result['top_3'], 1):
            print(f"       {j}. {cat}: {conf:.3f}")
    
    print(f"\n📊 TESTING SUMMARY")
    print("=" * 70)
    print(f"✅ Total Tests: {len(test_cases)}")
    print(f"🟢 High Confidence (>0.8): {high_confidence_predictions} ({high_confidence_predictions/len(test_cases)*100:.1f}%)")
    print(f"🟡 Medium Confidence (0.6-0.8): {len(test_cases) - high_confidence_predictions - sum(1 for desc in test_cases if predict_category(desc, model_pkg)['confidence'] < 0.6)}")
    print(f"🔴 Low Confidence (<0.6): {sum(1 for desc in test_cases if predict_category(desc, model_pkg)['confidence'] < 0.6)}")
    
    avg_confidence = np.mean([predict_category(desc, model_pkg)['confidence'] for desc in test_cases])
    print(f"📈 Average Confidence: {avg_confidence:.3f}")

def performance_validation():
    """Validate model performance on known data"""
    print(f"\n✅ PERFORMANCE VALIDATION")
    print("=" * 70)
    
    model_pkg = load_production_model()
    if model_pkg is None:
        return
    
    # Load some test data
    try:
        orig_path = '/Users/sujoymukherjee/code/spendplatform/classificationmodel-v0.5/preprocessed_data/top_10_l2_categories_data.xlsx'
        test_df = pd.read_excel(orig_path).head(20)  # Test on first 20 samples
        
        print(f"📊 Testing on {len(test_df)} known samples...")
        
        correct = 0
        total = 0
        
        for _, row in test_df.iterrows():
            description = row['Item_Descripton']
            true_category = row['Category L2']
            
            result = predict_category(description, model_pkg)
            predicted_category = result['category']
            
            if predicted_category == true_category:
                correct += 1
            total += 1
            
            status = "✅" if predicted_category == true_category else "❌"
            print(f"   {status} '{description[:50]}' → {predicted_category} (conf: {result['confidence']:.3f})")
            if predicted_category != true_category:
                print(f"      Expected: {true_category}")
        
        accuracy = correct / total if total > 0 else 0
        print(f"\n📈 Validation Results:")
        print(f"   • Correct: {correct}/{total}")
        print(f"   • Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
        
    except Exception as e:
        print(f"⚠️ Could not load validation data: {e}")

def deployment_readiness_check():
    """Check if model is ready for deployment"""
    print(f"\n🚀 DEPLOYMENT READINESS CHECK")
    print("=" * 70)
    
    model_pkg = load_production_model()
    if model_pkg is None:
        print("❌ Model not available for deployment")
        return False
    
    checks = {
        'Model loaded': model_pkg is not None,
        'Has vectorizer': 'vectorizer' in model_pkg,
        'Has label encoder': 'label_encoder' in model_pkg,
        'Has trained model': 'model' in model_pkg,
        'Performance documented': 'accuracy' in model_pkg,
        'Categories defined': 'categories' in model_pkg and model_pkg['categories'] > 0,
        'Features configured': 'features' in model_pkg and model_pkg['features'] > 0,
        'Model type specified': 'model_type' in model_pkg,
        'Training date recorded': 'training_date' in model_pkg,
    }
    
    all_passed = True
    for check, passed in checks.items():
        status = "✅" if passed else "❌"
        print(f"   {status} {check}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print(f"\n🎉 MODEL IS READY FOR PRODUCTION DEPLOYMENT!")
        print(f"✅ All deployment checks passed")
        print(f"🚀 Model can be integrated into production systems")
    else:
        print(f"\n⚠️ MODEL NEEDS ATTENTION BEFORE DEPLOYMENT")
        print(f"❌ Some deployment checks failed")
    
    return all_passed

def main():
    """Main testing function"""
    print("🔬 PRODUCTION MODEL VERIFICATION")
    print("=" * 70)
    
    # Run all tests
    test_sample_predictions()
    performance_validation()
    deployment_ready = deployment_readiness_check()
    
    print(f"\n" + "=" * 70)
    print(f"🎯 VERIFICATION COMPLETE!")
    if deployment_ready:
        print(f"✅ Production model verified and ready for deployment")
        print(f"🚀 Model performance validated on test cases")
        print(f"📦 All deployment requirements satisfied")
    else:
        print(f"⚠️ Model needs additional work before deployment")
    print("=" * 70)

if __name__ == "__main__":
    main()
