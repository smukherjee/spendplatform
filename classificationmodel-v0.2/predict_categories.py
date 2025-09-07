#!/usr/bin/env python3
"""
Spend Platform Categorization Model Predictor
Load trained model and make predictions on new data
"""

import pandas as pd
import numpy as np
import pickle
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class MultiLevelModelPredictor:
    """Predictor class for the trained multi-level categorization model"""

    def __init__(self, model_path=None, parser_type=None):
        """
        Initialize predictor with either model_path or parser_type

        Args:
            model_path: Direct path to model file (optional)
            parser_type: Parser type to automatically find model file (optional)
        """
        from config import ModelConfig

        if model_path and parser_type:
            raise ValueError("Cannot specify both model_path and parser_type")
        elif not model_path and not parser_type:
            # Default to basic parser
            parser_type = 'basic'

        if parser_type:
            config = ModelConfig()
            self.model_path = config.get_model_path_for_parser(parser_type)
            self.parser_type = parser_type
        else:
            self.model_path = model_path
            self.parser_type = None

        if not self.model_path:
            raise ValueError("Either model_path or parser_type must be provided")

        self.model = None
        self.parser = None  # Will be set when model is loaded
        self.load_model()

    def load_model(self):
        """Load the trained multi-level model"""
        if not self.model_path or not os.path.exists(self.model_path):
            error_msg = f"Model file not found: {self.model_path}"
            if self.parser_type:
                error_msg += f"\nMake sure to train a model with this parser first using:\n"
                error_msg += f"python train_classification_model.py --parser {self.parser_type}"
            raise FileNotFoundError(error_msg)

        print(f"Loading multi-level model from {self.model_path}...")
        with open(self.model_path, 'rb') as f:
            model_data = pickle.load(f)

        # Extract parser configuration from the model
        text_parser_config = model_data.get('text_parser_config', {})
        parser_type = text_parser_config.get('type', 'BasicTextParser')

        # Create the appropriate parser based on the stored configuration
        self.parser = self._create_parser_from_config(text_parser_config)

        # Create a simple object to hold model components
        class MultiLevelModelContainer:
            def __init__(self, model_data):
                self.models = model_data['model']  # Note: saved as 'model' not 'models'
                self.vectorizer = model_data['vectorizer']
                self.label_encoders = model_data['label_encoders']
                self.feature_names = model_data['feature_names']
                self.target_levels = model_data['target_levels']
                self.scaler = model_data['scaler']

        self.model = MultiLevelModelContainer(model_data)
        print("✅ Multi-level model loaded successfully")

    def _create_parser_from_config(self, config):
        """Create parser instance from stored configuration"""
        from config import ModelConfig
        from text_parsers import BasicTextParser, NLTKTextParser

        parser_type = config.get('type', 'BasicTextParser')

        if parser_type == 'BasicTextParser':
            return BasicTextParser()
        elif parser_type == 'NLTKTextParser':
            return NLTKTextParser()
        elif parser_type == 'SpacyTextParser':
            try:
                from text_parsers import SpacyTextParser
                spacy_config = config.get('config', {})
                return SpacyTextParser(spacy_config)
            except ImportError:
                print("⚠️ spaCy not available, falling back to NLTK")
                return NLTKTextParser()
        elif parser_type in ['BERTTextParser', 'RoBERTaTextParser', 'DistilBERTTextParser', 'LayoutLMv2TextParser', 'DONUTTextParser']:
            try:
                from text_parsers import BERTTextParser, RoBERTaTextParser, DistilBERTTextParser, LayoutLMv2TextParser, DONUTTextParser
                transformer_config = config.get('config', {})
                if parser_type == 'BERTTextParser':
                    return BERTTextParser(transformer_config)
                elif parser_type == 'RoBERTaTextParser':
                    return RoBERTaTextParser(transformer_config)
                elif parser_type == 'DistilBERTTextParser':
                    return DistilBERTTextParser(transformer_config)
                elif parser_type == 'LayoutLMv2TextParser':
                    return LayoutLMv2TextParser(transformer_config)
                elif parser_type == 'DONUTTextParser':
                    return DONUTTextParser(transformer_config)
            except ImportError:
                print("⚠️ Transformers not available, falling back to NLTK")
                return NLTKTextParser()
        else:
            print(f"⚠️ Unknown parser type '{parser_type}', using Basic parser")
            return BasicTextParser()

    def preprocess_text(self, text):
        """Preprocess text data for feature extraction"""
        if not isinstance(text, str) or text == '':
            return ''

        # Convert to lowercase
        text = text.lower()

        # Remove special characters but keep spaces and alphanumeric
        import re
        text = re.sub(r'[^\w\s]', ' ', text)

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)

        return text.strip()

    def extract_features(self, df):
        """Extract features from new data matching the model's training features"""
        if self.model is None:
            raise ValueError("Model not loaded")

        print("Extracting features from new data...")

        df_processed = df.copy()
        df_processed = df_processed.reset_index(drop=True)

        # Ensure required columns exist - try multiple possible column names
        description_col = None
        possible_cols = ['Item_Descripton', 'Description', 'Descripton', 'Order description',
                        'INV_ITEM_DESC', 'Material Item Name', 'TABLE_DESC']

        for col in possible_cols:
            if col in df_processed.columns:
                description_col = col
                break

        if description_col is None:
            available_cols = [col for col in df_processed.columns if 'desc' in col.lower() or 'name' in col.lower()]
            if available_cols:
                description_col = available_cols[0]
                print(f"Using column '{description_col}' as description column")
            else:
                raise ValueError(f"Data must contain a description column. Available columns: {list(df_processed.columns)}")

        # Use the found description column
        df_processed['Item_Descripton'] = df_processed[description_col]

        # Apply text preprocessing
        df_processed['processed_description'] = df_processed['Item_Descripton'].apply(self.preprocess_text)

        # Text features using TF-IDF
        text_features = self.model.vectorizer.transform(df_processed['processed_description'])

        # Convert to DataFrame
        text_feature_names = [f'text_{i}' for i in range(text_features.shape[1])]
        text_df = pd.DataFrame(text_features.toarray(), columns=text_feature_names, index=df_processed.index)  # type: ignore

        # Get the expected feature names from the model
        expected_features = self.model.feature_names

        # Create a DataFrame with all expected features, initialized to 0
        X = pd.DataFrame(0.0, index=df_processed.index, columns=expected_features)

        # Fill in the text features
        for col in text_df.columns:
            if col in X.columns:
                X[col] = text_df[col]

        # Add length-based features if expected
        length_feature_cols = ['desc_length', 'word_count', 'unique_words', 'has_numbers', 'has_symbols']
        for col in length_feature_cols:
            if col in expected_features:
                if col == 'desc_length':
                    X[col] = df_processed['Item_Descripton'].str.len()
                elif col == 'word_count':
                    X[col] = df_processed['Item_Descripton'].str.split().str.len()
                elif col == 'unique_words':
                    X[col] = df_processed['processed_description'].apply(lambda x: len(set(x.split())))
                elif col == 'has_numbers':
                    X[col] = df_processed['Item_Descripton'].str.contains(r'\d').astype(int)
                elif col == 'has_symbols':
                    X[col] = df_processed['Item_Descripton'].str.contains(r'[^\w\s]').astype(int)

        # Add POS-based features if expected
        pos_feature_cols = ['noun_count', 'verb_count', 'adj_count', 'adv_count', 'pron_count', 'total_tokens',
                           'avg_token_length', 'has_capitals', 'sentence_count', 'entity_count']
        pos_cols_present = [col for col in pos_feature_cols if col in expected_features]

        if pos_cols_present:
            print("Extracting POS-based features...")
            if self.parser is None:
                raise ValueError("Parser not initialized")
            pos_features_list = self.parser.extract_pos_features(df_processed['processed_description'].tolist())
            pos_df = pd.DataFrame(pos_features_list, index=df_processed.index)

            # Fill in POS features that are expected
            for col in pos_cols_present:
                if col in pos_df.columns:
                    X[col] = pos_df[col]
            print(f"✅ Extracted POS features: {len(pos_cols_present)} features")

        print(f"✅ Extracted {X.shape[1]} features from {X.shape[0]} samples")
        return X

    def predict_multi_level(self, df):
        """Make predictions for all levels"""
        if self.model is None:
            raise ValueError("Model not loaded")

        print("Making multi-level predictions...")

        # Extract features once
        X = self.extract_features(df)

        # Make predictions for each level
        predictions = {}
        confidences = {}

        for level in self.model.target_levels:
            print(f"Predicting {level}...")

            # Make predictions - handle both single model and multi-model cases
            if isinstance(self.model.models, dict):
                # Multi-level model case
                model = self.model.models[level]
            else:
                # Single-level model case
                model = self.model.models

            predictions_encoded = model.predict(X)

            # Decode predictions
            predictions[level] = self.model.label_encoders[level].inverse_transform(predictions_encoded)

            # Get prediction probabilities
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(X)
                confidences[level] = np.max(probabilities, axis=1)
            else:
                confidences[level] = np.ones(len(predictions[level])) * 0.5  # Default confidence

        print(f"✅ Generated predictions for {len(df)} samples across {len(self.model.target_levels)} levels")

        return predictions, confidences

    def predict_with_details(self, df):
        """Make predictions with detailed output for all levels"""
        predictions, confidences = self.predict_multi_level(df)

        # Create results dataframe
        results_df = df.copy()

        # Add predictions and confidences for each level
        for level in self.model.target_levels:  # type: ignore
            results_df[f'Predicted_{level.replace(" ", "_")}'] = predictions[level]
            results_df[f'Confidence_{level.replace(" ", "_")}'] = confidences[level]
            results_df[f'Confidence_Level_{level.replace(" ", "_")}'] = pd.cut(
                confidences[level],
                bins=[0, 0.5, 0.8, 1.0],
                labels=['Low', 'Medium', 'High']
            )

        return results_df

def main():
    """Main function to demonstrate model predictions"""
    print("="*80)
    print("SPEND PLATFORM CATEGORIZATION MODEL PREDICTOR")
    print("="*80)

    # Parse command line arguments
    import sys
    parser_type = None
    model_path = None

    args = sys.argv[1:]
    for i, arg in enumerate(args):
        if arg == '--parser' and i + 1 < len(args):
            parser_type = args[i + 1]
        elif arg == '--model' and i + 1 < len(args):
            model_path = args[i + 1]

    # Initialize predictor
    try:
        if parser_type:
            print(f"Loading model for parser: {parser_type}")
            predictor = MultiLevelModelPredictor(parser_type=parser_type)
        elif model_path:
            print(f"Loading model from path: {model_path}")
            predictor = MultiLevelModelPredictor(model_path=model_path)
        else:
            # Default to NLTK parser
            print("Loading default NLTK model...")
            predictor = MultiLevelModelPredictor(parser_type='nltk')
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("Please ensure the model file exists or train a model first.")
        return

    # Create sample data for demonstration
    print("\nCreating sample data for demonstration...")

    sample_data = pd.DataFrame({
        'Item_Code': ['ITEM001', 'ITEM002', 'ITEM003', 'ITEM004', 'ITEM005'],
        'Item_Descripton': [
            'Industrial hydraulic oil filter replacement',
            'Office supplies paper and toner cartridges',
            'Heavy duty welding equipment and accessories',
            'IT server maintenance and software licenses',
            'Manufacturing plant safety equipment and PPE'
        ],
        'Supplier_Name': ['ABC Industrial', 'Office Depot', 'WeldTech', 'TechCorp', 'SafetyFirst']
    })

    print("Sample data:")
    for i, row in sample_data.iterrows():
        print(f"  Item: {row['Item_Descripton'][:50]}...")

    # Make predictions
    print("\n" + "="*60)
    print("MAKING PREDICTIONS")
    print("="*60)

    results_df = predictor.predict_with_details(sample_data)

    # Display results
    print("\n" + "="*60)
    print("PREDICTION RESULTS")
    print("="*60)

    for i, row in results_df.iterrows():
        print("2d")
        for level in predictor.model.target_levels:  # type: ignore
            pred_col = f'Predicted_{level.replace(" ", "_")}'
            conf_col = f'Confidence_{level.replace(" ", "_")}'
            print("30")

    # Summary statistics
    print("\n" + "="*60)
    print("PREDICTION SUMMARY")
    print("="*60)

    print(f"Total predictions: {len(results_df)}")
    print(f"Levels predicted: {predictor.model.target_levels}")  # type: ignore

    for level in predictor.model.target_levels:  # type: ignore
        pred_col = f'Predicted_{level.replace(" ", "_")}'
        conf_col = f'Confidence_{level.replace(" ", "_")}'
        unique_preds = results_df[pred_col].nunique()
        avg_conf = results_df[conf_col].mean()
        print("30")

    # Save results
    results_file = 'prediction_results.xlsx'
    results_df.to_excel(results_file, index=False)
    print(f"\n✅ Results saved to {results_file}")

    print("\n" + "="*80)
    print("PREDICTION DEMONSTRATION COMPLETED!")
    print("="*80)

def predict_from_file(input_file, output_file='predictions.xlsx', parser_type=None, model_path=None):
    """Predict categories from Excel file"""
    print(f"Predicting categories from file: {input_file}")

    # Initialize predictor
    try:
        if parser_type:
            predictor = MultiLevelModelPredictor(parser_type=parser_type)
        elif model_path:
            predictor = MultiLevelModelPredictor(model_path=model_path)
        else:
            predictor = MultiLevelModelPredictor(parser_type='nltk')  # Default
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        return pd.DataFrame()

    # Load input data - handle files with headers in row 2
    df_raw = pd.read_excel(input_file, header=None)

    # Check if headers are in row 2 (common in some Excel files)
    if len(df_raw) > 2 and pd.notna(df_raw.iloc[2, 1]) and str(df_raw.iloc[2, 1]).strip() == 'Item Code':
        # Use row 2 as headers
        df = df_raw.iloc[2:].reset_index(drop=True)
        df.columns = df_raw.iloc[2].values
        print("Detected headers in row 2, using them as column names")
    else:
        # Use default reading
        df = pd.read_excel(input_file)

    print(f"Loaded {len(df)} records from {input_file}")
    print(f"Columns: {list(df.columns)}")

    # Make predictions
    results_df = predictor.predict_with_details(df)

    # Save results
    results_df.to_excel(output_file, index=False)
    print(f"✅ Predictions saved to {output_file}")

    return results_df

def predict_unclear_categories(input_file, output_file='unclear_predictions.xlsx'):
    """Predict categories only for records where Category L1 is 'unclear'"""
    print(f"Predicting categories for unclear records from file: {input_file}")

    # Load model
    predictor = MultiLevelModelPredictor('multi_level_spend_categorization_model.pkl')

    # Load input data - handle files with headers in row 2
    df_raw = pd.read_excel(input_file, header=None)

    # Check if headers are in row 2 (common in some Excel files)
    if len(df_raw) > 2 and pd.notna(df_raw.iloc[2, 1]) and str(df_raw.iloc[2, 1]).strip() == 'Item Code':
        # Use row 2 as headers
        df = df_raw.iloc[2:].reset_index(drop=True)
        df.columns = df_raw.iloc[2].values
        print("Detected headers in row 2, using them as column names")
    else:
        # Use default reading
        df = pd.read_excel(input_file)

    print(f"Loaded {len(df)} records from {input_file}")

    # Filter for records where Category L1 is 'unclear' (case insensitive)
    unclear_mask = df['Category L1'].str.lower() == 'unclear'
    unclear_df = df[unclear_mask].copy()

    if len(unclear_df) == 0:
        print("❌ No records found with Category L1 = 'unclear'")
        return pd.DataFrame()

    print(f"Found {len(unclear_df)} records with Category L1 = 'unclear'")
    print(f"Columns: {list(unclear_df.columns)}")

    # Make predictions on unclear records
    results_df = predictor.predict_with_details(unclear_df)

    # Save results
    results_df.to_excel(output_file, index=False)
    print(f"✅ Predictions for unclear records saved to {output_file}")

    return results_df

if __name__ == "__main__":
    main()
