#!/usr/bin/env python3
"""
Enhanced Feature Engineering for Logistic Regression
Optimized feature extraction for linear models in text classification
"""

import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from textstat import flesch_reading_ease, syllable_count
import string
from collections import Counter
import warnings

warnings.filterwarnings('ignore')

class EnhancedFeatureExtractor:
    """Enhanced feature extractor optimized for Logistic Regression"""
    
    def __init__(self, config=None):
        self.config = config
        self.tfidf_vectorizer = None
        self.char_vectorizer = None
        self.scaler = None
        self.feature_names = None
        
    def extract_text_statistics(self, texts):
        """Extract advanced text statistics features"""
        features = []
        
        for text in texts:
            text_str = str(text) if text else ""
            
            # Basic length features
            char_count = len(text_str)
            word_count = len(text_str.split())
            sentence_count = len(re.split(r'[.!?]+', text_str))
            
            # Character-level features
            upper_count = sum(1 for c in text_str if c.isupper())
            lower_count = sum(1 for c in text_str if c.islower())
            digit_count = sum(1 for c in text_str if c.isdigit())
            punct_count = sum(1 for c in text_str if c in string.punctuation)
            space_count = sum(1 for c in text_str if c.isspace())
            
            # Ratios (normalized features work well with Logistic Regression)
            upper_ratio = upper_count / max(char_count, 1)
            digit_ratio = digit_count / max(char_count, 1)
            punct_ratio = punct_count / max(char_count, 1)
            
            # Word-level features
            avg_word_length = np.mean([len(word) for word in text_str.split()]) if word_count > 0 else 0
            unique_words = len(set(text_str.lower().split()))
            unique_word_ratio = unique_words / max(word_count, 1)
            
            # Readability features (works well for technical descriptions)
            try:
                readability = flesch_reading_ease(text_str) if text_str.strip() else 0
                syllables = syllable_count(text_str) if text_str.strip() else 0
                avg_syllables = syllables / max(word_count, 1)
            except:
                readability = 0
                avg_syllables = 0
            
            # Technical/domain-specific features
            has_model_number = int(bool(re.search(r'\b[A-Z]{2,}\d+', text_str)))
            has_dimension = int(bool(re.search(r'\d+\s*[x×]\s*\d+', text_str)))
            has_measurement = int(bool(re.search(r'\d+\s*(mm|cm|m|inch|ft)', text_str, re.IGNORECASE)))
            has_capacity = int(bool(re.search(r'capacity|cap', text_str, re.IGNORECASE)))
            has_material = int(bool(re.search(r'steel|plastic|rubber|metal|wood', text_str, re.IGNORECASE)))
            has_make_brand = int(bool(re.search(r'make|brand', text_str, re.IGNORECASE)))
            
            # Pattern features
            has_parentheses = int('(' in text_str or ')' in text_str)
            has_colon = int(':' in text_str)
            has_dash = int('-' in text_str or '–' in text_str)
            has_ampersand = int('&' in text_str)
            
            # Numerical features
            number_count = len(re.findall(r'\d+', text_str))
            decimal_count = len(re.findall(r'\d+\.\d+', text_str))
            
            features.append([
                # Length features
                char_count, word_count, sentence_count, avg_word_length,
                
                # Character composition
                upper_ratio, digit_ratio, punct_ratio, unique_word_ratio,
                
                # Readability
                readability, avg_syllables,
                
                # Technical features
                has_model_number, has_dimension, has_measurement,
                has_capacity, has_material, has_make_brand,
                
                # Pattern features
                has_parentheses, has_colon, has_dash, has_ampersand,
                
                # Numerical features
                number_count, decimal_count
            ])
        
        feature_names = [
            'char_count', 'word_count', 'sentence_count', 'avg_word_length',
            'upper_ratio', 'digit_ratio', 'punct_ratio', 'unique_word_ratio',
            'readability', 'avg_syllables',
            'has_model_number', 'has_dimension', 'has_measurement',
            'has_capacity', 'has_material', 'has_make_brand',
            'has_parentheses', 'has_colon', 'has_dash', 'has_ampersand',
            'number_count', 'decimal_count'
        ]
        
        return pd.DataFrame(features, columns=feature_names)
    
    def extract_ngram_features(self, texts, is_training=True):
        """Extract enhanced n-gram features optimized for LR"""
        if is_training:
            # Optimized TF-IDF parameters for Logistic Regression
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=2000,  # Increased for better representation
                ngram_range=(1, 3),  # Include trigrams for better context
                min_df=2,
                max_df=0.95,
                stop_words='english',
                sublinear_tf=True,  # Important for LR
                use_idf=True,
                smooth_idf=True,
                lowercase=True,
                strip_accents='unicode',
                analyzer='word',
                token_pattern=r'\b[a-zA-Z][a-zA-Z0-9]*\b'  # Better tokenization
            )
            
            tfidf_features = self.tfidf_vectorizer.fit_transform(texts)
        else:
            if self.tfidf_vectorizer is None:
                raise ValueError("TF-IDF vectorizer not fitted")
            tfidf_features = self.tfidf_vectorizer.transform(texts)
        
        return tfidf_features
    
    def extract_character_features(self, texts, is_training=True):
        """Extract character-level n-gram features"""
        if is_training:
            # Character-level features for capturing morphological patterns
            self.char_vectorizer = CountVectorizer(
                analyzer='char',
                ngram_range=(2, 4),
                max_features=500,
                min_df=3,
                max_df=0.9,
                lowercase=True
            )
            
            char_features = self.char_vectorizer.fit_transform(texts)
        else:
            if self.char_vectorizer is None:
                raise ValueError("Character vectorizer not fitted")
            char_features = self.char_vectorizer.transform(texts)
        
        return char_features
    
    def extract_domain_features(self, texts):
        """Extract domain-specific features for industrial spend data"""
        features = []
        
        # Domain-specific keywords
        electrical_keywords = ['cable', 'wire', 'electrical', 'switch', 'circuit', 'voltage', 'amp']
        filtration_keywords = ['filter', 'filtration', 'membrane', 'cartridge', 'strainer']
        machinery_keywords = ['machine', 'motor', 'gear', 'bearing', 'shaft', 'pump', 'valve']
        tool_keywords = ['tool', 'spanner', 'wrench', 'screwdriver', 'hammer', 'plier']
        chemical_keywords = ['chemical', 'oil', 'grease', 'lubricant', 'solvent', 'acid']
        
        keyword_groups = {
            'electrical': electrical_keywords,
            'filtration': filtration_keywords,
            'machinery': machinery_keywords,
            'tools': tool_keywords,
            'chemicals': chemical_keywords
        }
        
        for text in texts:
            text_lower = str(text).lower()
            
            domain_features = {}
            
            # Count keywords from each domain
            for domain, keywords in keyword_groups.items():
                count = sum(1 for keyword in keywords if keyword in text_lower)
                domain_features[f'{domain}_keywords'] = count
                domain_features[f'has_{domain}'] = int(count > 0)
            
            # Technical patterns
            domain_features['has_part_number'] = int(bool(re.search(r'\b[A-Z0-9]{3,}-[A-Z0-9]{2,}', text)))
            domain_features['has_specification'] = int(bool(re.search(r'\d+\s*(V|A|W|HP|RPM)', text, re.IGNORECASE)))
            domain_features['has_size_spec'] = int(bool(re.search(r'\d+\s*(mm|cm|inch|")', text)))
            domain_features['has_grade_spec'] = int(bool(re.search(r'grade|class|type', text, re.IGNORECASE)))
            
            features.append(list(domain_features.values()))
        
        # Create feature names
        feature_names = []
        for domain in keyword_groups.keys():
            feature_names.extend([f'{domain}_keywords', f'has_{domain}'])
        feature_names.extend(['has_part_number', 'has_specification', 'has_size_spec', 'has_grade_spec'])
        
        return pd.DataFrame(features, columns=feature_names)
    
    def extract_all_features(self, texts, is_training=True):
        """Extract all enhanced features optimized for Logistic Regression"""
        print("Extracting enhanced features for Logistic Regression...")
        
        # Convert to list if pandas Series
        if hasattr(texts, 'tolist'):
            texts = texts.tolist()
        
        # Preprocess texts
        processed_texts = [str(text) if text else "" for text in texts]
        
        # 1. Text statistics features
        print("  • Extracting text statistics...")
        text_stats = self.extract_text_statistics(processed_texts)
        
        # 2. TF-IDF features
        print("  • Extracting TF-IDF features...")
        tfidf_features = self.extract_ngram_features(processed_texts, is_training)
        
        # 3. Character-level features
        print("  • Extracting character-level features...")
        char_features = self.extract_character_features(processed_texts, is_training)
        
        # 4. Domain-specific features
        print("  • Extracting domain-specific features...")
        domain_features = self.extract_domain_features(processed_texts)
        
        # Combine sparse and dense features
        from scipy.sparse import hstack, csr_matrix
        
        # Convert dense features to sparse
        text_stats_sparse = csr_matrix(text_stats.values)
        domain_features_sparse = csr_matrix(domain_features.values)
        
        # Combine all features
        all_features = hstack([
            tfidf_features,
            char_features,
            text_stats_sparse,
            domain_features_sparse
        ])
        
        # Store feature names for training
        if is_training:
            tfidf_names = [f'tfidf_{i}' for i in range(tfidf_features.shape[1])]
            char_names = [f'char_{i}' for i in range(char_features.shape[1])]
            
            self.feature_names = (
                tfidf_names + 
                char_names + 
                [f'stat_{name}' for name in text_stats.columns] +
                [f'domain_{name}' for name in domain_features.columns]
            )
        
        print(f"✅ Extracted {all_features.shape[1]} enhanced features")
        print(f"   • TF-IDF: {tfidf_features.shape[1]} features")
        print(f"   • Character n-grams: {char_features.shape[1]} features")  
        print(f"   • Text statistics: {text_stats.shape[1]} features")
        print(f"   • Domain features: {domain_features.shape[1]} features")
        
        return all_features


def enhanced_logistic_regression_pipeline(train_data, test_data, target_column='Category L2'):
    """Complete pipeline with enhanced features for Logistic Regression"""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import classification_report, accuracy_score
    from sklearn.model_selection import cross_val_score
    
    print("🚀 Enhanced Logistic Regression Pipeline")
    print("="*50)
    
    # Initialize feature extractor
    feature_extractor = EnhancedFeatureExtractor()
    
    # Extract features
    print("\n📊 Feature Extraction")
    print("-" * 30)
    
    X_train = feature_extractor.extract_all_features(train_data['Item_Descripton'], is_training=True)
    X_test = feature_extractor.extract_all_features(test_data['Item_Descripton'], is_training=False)
    
    # Prepare labels
    print("\n🏷️  Label Preparation")
    print("-" * 30)
    
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train_data[target_column])
    y_test = label_encoder.transform(test_data[target_column])
    
    print(f"✅ Classes: {len(label_encoder.classes_)}")
    
    # Train enhanced Logistic Regression
    print("\n🎯 Model Training")
    print("-" * 30)
    
    # Optimized parameters for enhanced features
    lr_model = LogisticRegression(
        C=1.0,  # Regularization strength
        max_iter=1000,
        solver='liblinear',  # Good for smaller datasets
        class_weight='balanced',
        random_state=42,
        penalty='l2'  # L2 regularization works well with many features
    )
    
    import time
    start_time = time.time()
    lr_model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    # Evaluate
    print("\n📈 Model Evaluation")
    print("-" * 30)
    
    y_pred = lr_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Cross-validation
    cv_scores = cross_val_score(lr_model, X_train, y_train, cv=5, scoring='accuracy')
    
    print(f"✅ Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"✅ CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"✅ Training Time: {training_time:.2f}s")
    
    # Detailed classification report
    print(f"\n📋 Classification Report")
    print("-" * 50)
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
    
    return {
        'model': lr_model,
        'feature_extractor': feature_extractor,
        'label_encoder': label_encoder,
        'accuracy': accuracy,
        'cv_scores': cv_scores,
        'training_time': training_time
    }


if __name__ == "__main__":
    # Test the enhanced feature extraction
    print("Testing Enhanced Feature Extractor...")
    
    # Load data
    train_data = pd.read_excel('preprocessed_data/train_data.xlsx')
    test_data = pd.read_excel('preprocessed_data/test_data.xlsx')
    
    # Clean column names
    train_data.columns = train_data.columns.str.strip()
    test_data.columns = test_data.columns.str.strip()
    
    # Fill missing values
    train_data['Item_Descripton'] = train_data['Item_Descripton'].fillna('').astype(str)
    test_data['Item_Descripton'] = test_data['Item_Descripton'].fillna('').astype(str)
    
    # Run enhanced pipeline
    results = enhanced_logistic_regression_pipeline(train_data, test_data)
    
    print(f"\n🎯 Enhanced LR Performance: {results['accuracy']*100:.2f}% accuracy")
