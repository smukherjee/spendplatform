#!/usr/bin/env python3
"""
Multi-Level Supervised Classification Model for Spend Platform Categorization
Trains and evaluates machine learning models for transaction categorization at L2, L3, and L5 levels
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt
import seaborn as sns
import re
from collections import Counter
import warnings
import pickle
import os
from pathlib import Path
from typing import Dict, List, Optional, Any, Union

warnings.filterwarnings('ignore')

class MultiLevelSpendCategorizationModel:
    """Multi-level supervised learning model for spend transaction categorization"""

    def __init__(self, target_levels: Optional[List[str]] = None):
        if target_levels is None:
            target_levels = ['Category L1', 'Category L2', 'Category L3']
        self.target_levels: List[str] = target_levels
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.scaler: StandardScaler = StandardScaler()
        self.models: Dict[str, Any] = {}
        self.feature_names: Optional[List[str]] = None
        self.training_data: Optional[pd.DataFrame] = None

    def load_training_data(self, excel_file_path):
        """Load and preprocess training data"""
        print("Loading training data...")

        # Read Excel file - try different header detection
        df_raw = pd.read_excel(excel_file_path, header=None)

        # Check if first row contains headers or if headers are in row 2
        if len(df_raw) > 2 and pd.notna(df_raw.iloc[0, 1]) and str(df_raw.iloc[0, 1]).strip() == 'Item Code':
            # Headers are in row 0
            self.training_data = pd.read_excel(excel_file_path, header=0)
            print("Detected headers in row 0")
        elif len(df_raw) > 2 and pd.notna(df_raw.iloc[2, 1]) and str(df_raw.iloc[2, 1]).strip() == 'Item Code':
            # Headers are in row 2 (original format)
            self.training_data = pd.read_excel(excel_file_path, header=2)
            print("Detected headers in row 2")
        else:
            # Default reading
            self.training_data = pd.read_excel(excel_file_path)
            print("Using default header detection")

        # Clean column names
        self.training_data.columns = self.training_data.columns.str.strip()

        # Remove the unnamed first column if it exists
        if 'Unnamed: 0' in self.training_data.columns:
            self.training_data = self.training_data.drop('Unnamed: 0', axis=1)

        # Clean text data
        self.training_data['Item_Descripton'] = self.training_data['Item_Descripton'].fillna('').astype(str)

        print(f"✅ Loaded {len(self.training_data)} training samples")
        print(f"✅ Target levels: {self.target_levels}")

        # Show statistics for each level
        for level in self.target_levels:
            if level in self.training_data.columns:
                unique_count = self.training_data[level].nunique()
                print(f"✅ {level}: {unique_count} unique categories")

        return self.training_data

    def preprocess_text(self, text):
        """Preprocess text data for feature extraction"""
        if not isinstance(text, str) or text == '':
            return ''

        # Convert to lowercase
        text = text.lower()

        # Remove special characters but keep spaces and alphanumeric
        text = re.sub(r'[^\w\s]', ' ', text)

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)

        return text.strip()

    def extract_features(self, df, is_training=True):
        """Extract features from the dataset"""
        print("Extracting features...")

        df_processed = df.copy()
        df_processed = df_processed.reset_index(drop=True)  # Reset index to ensure alignment

        # Apply text preprocessing
        df_processed['processed_description'] = df_processed['Item_Descripton'].apply(self.preprocess_text)

        # Text features using TF-IDF
        if is_training:
            self.vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.9
            )
            text_features = self.vectorizer.fit_transform(df_processed['processed_description'])
        else:
            if self.vectorizer is None:
                raise ValueError("Vectorizer not fitted. Call with is_training=True first.")
            text_features = self.vectorizer.transform(df_processed['processed_description'])

        # Convert to DataFrame
        text_feature_names = [f'text_{i}' for i in range(text_features.shape[1])]
        text_df = pd.DataFrame(text_features.toarray(), columns=text_feature_names, index=df_processed.index)  # type: ignore

        # Length-based features
        length_features = pd.DataFrame({
            'desc_length': df_processed['Item_Descripton'].str.len(),
            'word_count': df_processed['Item_Descripton'].str.split().str.len(),
            'unique_words': df_processed['processed_description'].apply(lambda x: len(set(x.split()))),
            'has_numbers': df_processed['Item_Descripton'].str.contains(r'\d').astype(int),
            'has_symbols': df_processed['Item_Descripton'].str.contains(r'[^\w\s]').astype(int)
        }, index=df_processed.index)

        # Combine all features
        X = pd.concat([text_df, length_features], axis=1)

        # Store feature names
        if is_training:
            self.feature_names = X.columns.tolist()

        print(f"✅ Extracted {X.shape[1]} features from {X.shape[0]} samples")
        return X

    def prepare_labels(self, df, target_level, is_training=True):
        """Prepare target labels for a specific level"""
        if target_level not in df.columns:
            raise ValueError(f"Target column '{target_level}' not found in data")

        y_raw = df[target_level]

        if is_training:
            self.label_encoders[target_level] = LabelEncoder()
            y_encoded = self.label_encoders[target_level].fit_transform(y_raw)
        else:
            if target_level not in self.label_encoders:
                raise ValueError(f"Label encoder for '{target_level}' not found")
            y_encoded = self.label_encoders[target_level].transform(y_raw)

        print(f"✅ Prepared labels for {target_level}: {len(y_encoded)} samples, {len(np.unique(y_encoded))} classes")  # type: ignore
        return y_encoded

    def train_model(self, X_train, y_train, model_type='random_forest'):
        """Train the classification model"""
        print(f"Training {model_type} model...")

        if model_type == 'random_forest':
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
        elif model_type == 'logistic_regression':
            model = LogisticRegression(
                random_state=42,
                max_iter=1000,
                multi_class='ovr'
            )
        elif model_type == 'svm':
            model = SVC(
                kernel='linear',
                probability=True,
                random_state=42
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        model.fit(X_train, y_train)
        print("✅ Model trained successfully")
        return model

    def train_multi_level_models(self, model_type='random_forest'):
        """Train separate models for each target level"""
        if self.training_data is None:
            raise ValueError("Training data not loaded. Call load_training_data() first.")

        print("="*80)
        print("TRAINING MULTI-LEVEL CATEGORIZATION MODELS")
        print("="*80)

        # Extract features once (shared across all levels)
        X = self.extract_features(self.training_data, is_training=True)

        # Split data for training and validation
        X_train, X_test, train_indices, test_indices = train_test_split(
            X, self.training_data.index, test_size=0.2, random_state=42
        )

        # Train a model for each level
        for level in self.target_levels:
            print(f"\n{'='*60}")
            print(f"TRAINING MODEL FOR {level}")
            print('='*60)

            # Prepare labels for this level
            y_full = self.prepare_labels(self.training_data, level, is_training=True)
            y_train = y_full[train_indices]  # type: ignore
            y_test = y_full[test_indices]  # type: ignore

            # Train model
            model = self.train_model(X_train, y_train, model_type)

            # Evaluate model
            self.evaluate_model(model, X_test, y_test, level)

            # Store the trained model
            self.models[level] = model

        print(f"\n✅ All models trained successfully for levels: {self.target_levels}")

    def evaluate_model(self, model, X_test, y_test, level):
        """Evaluate the trained model"""
        print(f"Evaluating {level} model...")

        # Make predictions
        y_pred = model.predict(X_test)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        print(".4f")

        # Get unique classes in test set
        unique_test_classes = np.unique(y_test)
        target_names = self.label_encoders[level].classes_[unique_test_classes]

        # Classification report with correct labels
        report = classification_report(y_test, y_pred, target_names=target_names, output_dict=True, labels=unique_test_classes)

        print("Top 5 categories by precision:")
        category_metrics = []
        for category, metrics in report.items():  # type: ignore
            if category not in ['accuracy', 'macro avg', 'weighted avg']:
                category_metrics.append((category, metrics['precision'], metrics['recall'], metrics['f1-score']))

        category_metrics.sort(key=lambda x: x[1], reverse=True)
        for category, precision, recall, f1 in category_metrics[:5]:
            print("30")

        return accuracy, report

    def save_models(self, output_path='multi_level_spend_categorization_model.pkl'):
        """Save the trained models and preprocessing objects"""
        print(f"Saving models to {output_path}...")

        model_data = {
            'models': self.models,
            'vectorizer': self.vectorizer,
            'label_encoders': self.label_encoders,
            'feature_names': self.feature_names,
            'target_levels': self.target_levels,
            'scaler': self.scaler
        }

        with open(output_path, 'wb') as f:
            pickle.dump(model_data, f)

        print("✅ Models saved successfully")

    def load_models(self, model_path):
        """Load trained models"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        print(f"Loading models from {model_path}...")
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)

        self.models = model_data['models']
        self.vectorizer = model_data['vectorizer']
        self.label_encoders = model_data['label_encoders']
        self.feature_names = model_data['feature_names']
        self.target_levels = model_data['target_levels']
        self.scaler = model_data['scaler']

        print("✅ Models loaded successfully")

def main():
    """Main function to train multi-level categorization models"""
    print("="*80)
    print("MULTI-LEVEL SPEND PLATFORM CATEGORIZATION MODEL TRAINING")
    print("="*80)

    # Training data path
    training_file = '/Users/sujoymukherjee/code/spendplatform/context/Copy of Catergorization Working Sheet_Labelled Data for Training.xlsx'

    # Initialize multi-level model
    model = MultiLevelSpendCategorizationModel(
        target_levels=[ 'Category L2']
    )

    # Load training data
    training_data = model.load_training_data(training_file)

    # Train models for all levels
    model.train_multi_level_models(model_type='random_forest')

    # Save the trained models
    model.save_models('multi_level_spend_categorization_model.pkl')

    print("\n" + "="*80)
    print("MULTI-LEVEL MODEL TRAINING COMPLETED!")
    print("="*80)

if __name__ == "__main__":
    main()
