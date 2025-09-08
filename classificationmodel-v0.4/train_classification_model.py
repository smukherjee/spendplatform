#!/usr/bin/env python3
"""
Supervised Classification Model for Spend Platform Categorization
Trains and evaluates machine learning models for transaction categorization at multiple levels
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
from catboost import CatBoostClassifier
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

# Import text parsers and configuration
from text_parsers import BasicTextParser, NLTKTextParser
from config import ModelConfig, get_config_from_args

warnings.filterwarnings('ignore')

class MultiLevelSpendCategorizationModel:
    """Multi-level supervised learning model for spend transaction categorization"""

    def __init__(self, target_levels=['Category L2'], text_parser=None, config=None):
        self.target_levels = target_levels
        self.label_encoders = {}
        self.vectorizer = None
        self.scaler = StandardScaler()
        self.models = {}
        self.feature_names = None
        self.training_data = None
        self.config = config  # Store config for TF-IDF settings

        # Initialize text parser
        if text_parser is None:
            self.text_parser = NLTKTextParser()  # Default to NLTK parser
        else:
            self.text_parser = text_parser

        print(f"✅ Using text parser: {self.text_parser.get_parser_info()['name']}")

    def load_preprocessed_data(self, train_file_path, test_file_path):
        """Load preprocessed train and test data"""
        print("Loading preprocessed training and test data...")

        # Load train and test data
        self.train_data = pd.read_excel(train_file_path)
        self.test_data = pd.read_excel(test_file_path)

        # Clean column names
        self.train_data.columns = self.train_data.columns.str.strip()
        self.test_data.columns = self.test_data.columns.str.strip()

        # Remove unnamed columns if they exist
        for df_name, df in [('train', self.train_data), ('test', self.test_data)]:
            unnamed_cols = [col for col in df.columns if 'Unnamed' in col]
            if unnamed_cols:
                df = df.drop(unnamed_cols, axis=1)
                if df_name == 'train':
                    self.train_data = df
                else:
                    self.test_data = df

        # Clean text data
        for df_name, df in [('train', self.train_data), ('test', self.test_data)]:
            df['Item_Descripton'] = df['Item_Descripton'].fillna('').astype(str)
            if df_name == 'train':
                self.train_data = df
            else:
                self.test_data = df

        print(f"✅ Loaded {len(self.train_data)} training samples")
        print(f"✅ Loaded {len(self.test_data)} test samples")
        print(f"✅ Target levels: {self.target_levels}")

        for level in self.target_levels:
            if level in self.train_data.columns:
                train_unique = self.train_data[level].nunique()
                test_unique = self.test_data[level].nunique()
                print(f"✅ {level} - Train: {train_unique} categories, Test: {test_unique} categories")

        return self.train_data, self.test_data

    def preprocess_text(self, text):
        """Text preprocessing using the configured parser"""
        return self.text_parser.preprocess_text(text)

    def extract_pos_features(self, df, use_pos=True):
        """Extract POS-based features using the configured parser"""
        print("Extracting POS-based features...")

        # Get processed descriptions
        processed_texts = df['processed_description'].tolist()

        # Extract POS features using the parser
        pos_features_list = self.text_parser.extract_pos_features(processed_texts)

        # Convert to DataFrame
        pos_df = pd.DataFrame(pos_features_list)
        print(f"✅ Extracted POS features: {pos_df.shape[1]} features")
        return pos_df

    def extract_features(self, df, is_training=True):
        """Extract features from the dataset"""
        print("Extracting features...")

        df_processed = df.copy()
        df_processed = df_processed.reset_index(drop=True)  # Reset index to ensure alignment

        # Apply text preprocessing
        df_processed['processed_description'] = df_processed['Item_Descripton'].apply(self.preprocess_text)

        # Text features using TF-IDF with enhanced parameters
        if is_training:
            # Use config values if available, otherwise use defaults
            max_features = self.config.tfidf_max_features if self.config else 1500
            ngram_range = self.config.tfidf_ngram_range if self.config else (1, 3)
            min_df = self.config.tfidf_min_df if self.config else 2
            max_df = self.config.tfidf_max_df if self.config else 0.95

            self.vectorizer = TfidfVectorizer(
                max_features=max_features,
                stop_words=None,    # We already removed stopwords in preprocessing
                ngram_range=ngram_range,
                min_df=min_df,
                max_df=max_df,
                sublinear_tf=True, # Apply sublinear scaling
                use_idf=True,
                smooth_idf=True
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

        # POS-based features (using simplified version for speed)
        pos_features = self.extract_pos_features(df_processed, use_pos=True)

        # Combine all features
        X = pd.concat([text_df, length_features, pos_features], axis=1)

        # Store feature names
        if is_training:
            self.feature_names = X.columns.tolist()

        print(f"✅ Extracted {X.shape[1]} features from {X.shape[0]} samples")
        return X

    def prepare_labels(self, df, is_training=True):
        """Prepare target labels"""
        missing_levels = [level for level in self.target_levels if level not in df.columns]
        if missing_levels:
            raise ValueError(f"Target columns {missing_levels} not found in data")

        y_raw = df[self.target_levels]

        if is_training:
            for level in self.target_levels:
                self.label_encoders[level] = LabelEncoder()
                y_encoded = self.label_encoders[level].fit_transform(y_raw[level])
        else:
            for level in self.target_levels:
                if level not in self.label_encoders:
                    raise ValueError(f"Label encoder for '{level}' not found")
                y_encoded = self.label_encoders[level].transform(y_raw[level])

        print(f"✅ Prepared labels: {len(y_encoded)} samples, {len(np.unique(y_encoded))} classes")
        return y_encoded

    def train_model(self, X_train, y_train, model_type='random_forest'):
        """Train the classification model"""
        print(f"Training {model_type} model...")

        if model_type == 'random_forest':
            # Use config values if available, otherwise use defaults
            n_estimators = self.config.rf_n_estimators if self.config else 200
            max_depth = self.config.rf_max_depth if self.config else 15
            min_samples_split = self.config.rf_min_samples_split if self.config else 5
            min_samples_leaf = self.config.rf_min_samples_leaf if self.config else 2

            self.model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                max_features='sqrt',   # Added feature selection
                random_state=42,
                n_jobs=-1,
                class_weight='balanced'  # Handle imbalanced classes
            )
        elif model_type == 'logistic_regression':
            self.model = LogisticRegression(
                random_state=42,
                max_iter=1000,
                C=1.0
            )
        elif model_type == 'svm':
            self.model = SVC(
                kernel='linear',
                C=1.0,
                random_state=42
            )
        elif model_type == 'catboost':
            self.model = CatBoostClassifier(
                iterations=500,
                learning_rate=0.1,
                depth=6,
                random_state=42,
                verbose=False,  # Suppress training output
                auto_class_weights='Balanced',  # Handle imbalanced classes
                eval_metric='Accuracy'
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Train the model
        self.model.fit(X_train, y_train)

        print("✅ Model training completed")
        return self.model

    def evaluate_model(self, X_test, y_test):
        """Evaluate model performance"""
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")

        print("Evaluating model...")

        # Make predictions
        y_pred = self.model.predict(X_test)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)

        print(".4f")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        # Cross-validation score
        cv_scores = cross_val_score(self.model, X_test, y_test, cv=5, scoring='accuracy')
        print(".4f")

        return {
            'accuracy': accuracy,
            'cv_accuracy': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'classification_report': report,
            'predictions': y_pred
        }

    def predict(self, df):
        """Make predictions on new data"""
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")

        print("Making predictions...")

        # Extract features
        X = self.extract_features(df, is_training=False)

        # Make predictions
        predictions_encoded = self.model.predict(X)

        # Decode predictions (use first target level since this is single-level model)
        target_level = self.target_levels[0] if self.target_levels else None
        if target_level is None:
            raise ValueError("No target levels specified")

        predictions = self.label_encoders[target_level].inverse_transform(predictions_encoded)

        # Get prediction probabilities
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(X)
            confidence_scores = np.max(probabilities, axis=1)
        else:
            confidence_scores = np.ones(len(predictions)) * 0.5  # Default confidence

        print(f"✅ Generated predictions for {len(predictions)} samples")

        return predictions, confidence_scores

    def save_model(self, filepath):
        """Save the trained model"""
        model_data = {
            'model': self.model,
            'vectorizer': self.vectorizer,
            'label_encoders': self.label_encoders,
            'feature_names': self.feature_names,
            'target_levels': self.target_levels,
            'scaler': self.scaler,
            'text_parser_config': self.text_parser.get_parser_info(),
            'config': self.config.__dict__ if self.config else None
        }

        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

        print(f"✅ Model saved to {filepath}")

    def load_model(self, filepath):
        """Load a trained model"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        self.model = model_data['model']
        self.vectorizer = model_data['vectorizer']
        self.label_encoders = model_data['label_encoders']
        self.feature_names = model_data['feature_names']
        self.target_levels = model_data['target_levels']
        self.scaler = model_data['scaler']

        # Recreate config if available
        if 'config' in model_data and model_data['config']:
            self.config = ModelConfig()
            for key, value in model_data['config'].items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
        else:
            self.config = None

        # Recreate text parser from config
        if 'text_parser_config' in model_data:
            parser_config = model_data['text_parser_config']
            parser_type = parser_config.get('type', 'NLTKTextParser')
            if parser_type == 'NLTKTextParser':
                self.text_parser = NLTKTextParser(parser_config.get('config', {}))
            elif parser_type == 'BasicTextParser':
                self.text_parser = BasicTextParser(parser_config.get('config', {}))
            elif parser_type == 'SpacyTextParser':
                try:
                    from text_parsers import SpacyTextParser
                    self.text_parser = SpacyTextParser(parser_config.get('config', {}))
                except ImportError:
                    print("⚠️ spaCy not available, falling back to NLTK parser")
                    self.text_parser = NLTKTextParser()
            else:
                # Fallback to NLTK parser
                self.text_parser = NLTKTextParser()
        else:
            # Backward compatibility - use NLTK parser
            self.text_parser = NLTKTextParser()

        print(f"✅ Model loaded from {filepath}")
        print(f"✅ Using text parser: {self.text_parser.get_parser_info()['name']}")

    def create_evaluation_plots(self, y_true, y_pred, save_path='model_evaluation.png'):
        """Create evaluation plots"""
        print("Creating evaluation plots...")

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Model Evaluation Results', fontsize=16)

        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0])
        axes[0, 0].set_title('Confusion Matrix')
        axes[0, 0].set_xlabel('Predicted')
        axes[0, 0].set_ylabel('Actual')

        # Feature Importance (if available)
        if self.model is not None and hasattr(self.model, 'feature_importances_') and self.feature_names is not None:
            feature_importance = self.model.feature_importances_  # type: ignore
            top_features = np.argsort(feature_importance)[-20:]  # Top 20 features

            if len(self.feature_names) > 20:
                feature_names = [self.feature_names[i] for i in top_features]
                importance_values = feature_importance[top_features]
            else:
                feature_names = self.feature_names
                importance_values = feature_importance

            axes[0, 1].barh(range(len(feature_names)), importance_values)
            axes[0, 1].set_yticks(range(len(feature_names)))
            axes[0, 1].set_yticklabels(feature_names)
            axes[0, 1].set_title('Top Feature Importances')
            axes[0, 1].set_xlabel('Importance')
        else:
            axes[0, 1].text(0.5, 0.5, 'Feature importance\nnot available\nfor this model type',
                           ha='center', va='center', transform=axes[0, 1].transAxes)
            axes[0, 1].set_title('Feature Importances (N/A)')

        # Prediction distribution
        y_pred_flat = np.array(y_pred).flatten()  # Ensure 1D array
        pred_counts = pd.Series(y_pred_flat).value_counts()
        pred_counts.plot(kind='bar', ax=axes[1, 0])
        axes[1, 0].set_title('Prediction Distribution')
        axes[1, 0].set_xlabel('Category')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].tick_params(axis='x', rotation=45)

        # True vs Predicted comparison
        comparison_df = pd.DataFrame({'True': y_true, 'Predicted': y_pred_flat})
        comparison_counts = comparison_df.groupby(['True', 'Predicted']).size().unstack(fill_value=0)
        comparison_counts.plot(kind='bar', stacked=True, ax=axes[1, 1])
        axes[1, 1].set_title('True vs Predicted Categories')
        axes[1, 1].set_xlabel('True Category')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].legend(title='Predicted', bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Evaluation plots saved to {save_path}")

def main():
    """Main function to train and evaluate the model"""
    print("="*80)
    print("SPEND PLATFORM CATEGORIZATION MODEL TRAINING")
    print("="*80)

    # Get configuration
    config = get_config_from_args()
    config.print_config()

    # Initialize text parser from config
    text_parser = config.get_text_parser()

    # Initialize model
    print(f"Initializing model for target level: {config.target_level}")
    model = MultiLevelSpendCategorizationModel(
        target_levels=[config.target_level],
        text_parser=text_parser,
        config=config
    )

    # Load preprocessed train and test data
    train_data, test_data = model.load_preprocessed_data(config.train_file, config.test_file)

    # Extract features and prepare labels
    print("\nPreparing training data...")
    X_train = model.extract_features(train_data, is_training=True)
    y_train = model.prepare_labels(train_data, is_training=True)

    print("\nPreparing test data...")
    X_test = model.extract_features(test_data, is_training=False)
    y_test = model.prepare_labels(test_data, is_training=False)

    # Train model
    model.train_model(X_train, y_train, model_type=config.model_type)

    # Evaluate model
    evaluation_results = model.evaluate_model(X_test, y_test)

    # Create evaluation plots
    model.create_evaluation_plots(y_test, evaluation_results['predictions'])

    # Save model
    model.save_model(config.model_save_path)

    # Make predictions on test data to demonstrate
    print("\n" + "="*60)
    print("DEMONSTRATION PREDICTIONS")
    print("="*60)

    predictions, confidences = model.predict(test_data)

    # Show sample predictions
    print("\nSample Predictions:")
    sample_size = min(10, len(test_data))
    for i in range(sample_size):
        true_cat = test_data.iloc[i][config.target_level]
        pred_cat = predictions[i]
        conf = confidences[i]
        desc = test_data.iloc[i]['Item_Descripton'][:50] + "..."

        status = "✅" if true_cat == pred_cat else "❌"
        print(f"{i+1:2d}. {desc:<50} | True: {true_cat} | Pred: {pred_cat} | Conf: {conf:.2f} {status}")

    # Show detailed performance by category
    print("\n" + "="*40)
    print("PERFORMANCE BY CATEGORY")
    print("="*40)

    # Get category names from label encoder
    category_names = model.label_encoders[config.target_level].classes_

    print("Category Performance:")
    for i, category in enumerate(category_names):
        # Get test samples for this category
        category_mask = (y_test == i)
        if category_mask.sum() > 0:
            category_predictions = evaluation_results['predictions'][category_mask]
            category_correct = (category_predictions == i).sum()
            category_total = category_mask.sum()
            category_accuracy = category_correct / category_total

            print(f"  • {category}: {category_accuracy:.2%} ({category_correct}/{category_total})")

    print("\n" + "="*80)
    print("MODEL TRAINING COMPLETED SUCCESSFULLY!")
    print("="*80)
    print("\n📊 Performance Summary:")
    print(f"  • Accuracy: {evaluation_results['accuracy']:.2%}")
    print(f"  • CV Accuracy: {evaluation_results['cv_accuracy']:.2%} ± {evaluation_results['cv_std']:.2%}")
    print("\n📁 Files Generated:")
    print(f"  • Model: {config.model_save_path}")
    print("  • Evaluation plots: model_evaluation.png")
    print("\n🎯 Ready for production use!")

if __name__ == "__main__":
    main()
