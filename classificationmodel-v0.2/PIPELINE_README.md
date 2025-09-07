# Complete Spend Platform Categorization Pipeline

This script (`run_complete_pipeline.py`) orchestrates the entire machine learning pipeline for spend transaction categorization, from data analysis to model deployment.

## 🚀 What It Does

The complete pipeline runs four main steps:

1. **📊 Data Analysis** - Analyzes the training dataset, generates statistics, and creates visualizations
2. **🔧 Data Preprocessing** - Filters data by top categories, creates train/test splits, and prepares data for training
3. **🤖 Model Training** - Runs comprehensive experiments with multiple parser configurations (Basic, NLTK, spaCy, BERT, RoBERTa, DistilBERT, LayoutLMv2, DONUT)
4. **🔮 Prediction Demo** - Demonstrates the trained model with sample predictions

## 📋 Prerequisites

- Python 3.7+
- Training data file: `context/Copy of Catergorization Working Sheet_Labelled Data for Training.xlsx`
- All required dependencies installed (see requirements.txt or individual setup scripts)

## 🏃‍♂️ How to Run

### Option 1: Run the Complete Pipeline
```bash
python run_complete_pipeline.py
```

### Option 2: Run Individual Steps (if needed)
```bash
# Step 1: Data Analysis Only
python -c "from run_complete_pipeline import SpendPlatformPipeline; p = SpendPlatformPipeline(); p.step_1_data_analysis()"

# Step 2: Preprocessing Only
python -c "from run_complete_pipeline import SpendPlatformPipeline; p = SpendPlatformPipeline(); p.step_2_preprocessing()"

# Step 3: Training Experiments Only
python run_experiments_controlled.py

# Step 4: Prediction Demo Only
python -c "from run_complete_pipeline import SpendPlatformPipeline; p = SpendPlatformPipeline(); p.step_4_prediction_demo()"
```

## 📁 Output Files

The pipeline generates several output files:

### Data Analysis
- `data_analysis_visualizations.png` - Data distribution charts
- Analysis reports in console output

### Preprocessing
- `preprocessed_data/top_10_l2_categories_data.xlsx` - Filtered data
- `preprocessed_data/train_data.xlsx` - Training dataset
- `preprocessed_data/test_data.xlsx` - Test dataset

### Training
- Multiple model files (`.pkl`) for different parser configurations
- `experiments_controlled_output_*.txt` - Experiment results
- `model_evaluation.png` - Model performance plots

### Prediction
- `pipeline_prediction_results.xlsx` - Sample predictions
- Console output with prediction results

### Summary
- `pipeline_output_*/pipeline_summary_report.txt` - Complete execution report

## 🔧 Configuration

The pipeline uses the following default configurations:

- **Data File**: `context/Copy of Catergorization Working Sheet_Labelled Data for Training.xlsx`
- **Target Level**: Category L2 (top-level categorization)
- **Train/Test Split**: Automatic stratified split
- **Top Categories**: Top 10 categories by frequency
- **Model Type**: Random Forest (configurable via config.py)

## 📊 Expected Performance

Based on previous experiments, expect:
- **spaCy Parser + Optimized**: ~60% accuracy (best performer)
- **NLTK Parser + High Accuracy**: ~57% accuracy
- **Transformer Models**: ~54% accuracy (BERT, RoBERTa, etc.)
- **Basic Parser**: ~51% accuracy

## ⏱️ Execution Time

Typical execution times:
- **Data Analysis**: 2-5 minutes
- **Preprocessing**: 1-3 minutes
- **Training Experiments**: 15-30 minutes (9 experiments)
- **Prediction Demo**: 1-2 minutes
- **Total Pipeline**: 20-45 minutes

## 🛠️ Troubleshooting

### Common Issues

1. **Data File Not Found**
   ```
   ❌ Data file not found: context/Copy of Catergorization Working Sheet_Labelled Data for Training.xlsx
   ```
   **Solution**: Ensure the training data file exists in the context directory

2. **Missing Dependencies**
   ```
   ImportError: No module named 'spacy'
   ```
   **Solution**: Run setup scripts:
   ```bash
   python setup_spacy.py
   python setup_transformers.py
   ```

3. **Memory Issues**
   ```
   MemoryError: Unable to allocate array
   ```
   **Solution**: Use smaller batch sizes or reduce feature dimensions in config.py

4. **Timeout Errors**
   ```
   TIMEOUT: Experiment exceeded X minutes
   ```
   **Solution**: Increase timeout values in `run_experiments_controlled.py`

### Recovery Options

If the pipeline fails at any step, you can:
1. Fix the issue and re-run the complete pipeline (it will skip completed steps)
2. Run individual steps manually using the commands above
3. Use existing preprocessed data or trained models

## 📈 Performance Optimization

To improve performance:

1. **Use GPU for Transformers**: Install PyTorch with CUDA support
2. **Reduce Feature Dimensions**: Modify `max_features` in config.py
3. **Use Faster Parsers**: Start with spaCy instead of transformers
4. **Parallel Processing**: Modify experiment runner for parallel execution

## 🎯 Next Steps

After running the pipeline:

1. **Evaluate Results**: Check `pipeline_summary_report.txt`
2. **Fine-tune Models**: Adjust hyperparameters in `config.py`
3. **Deploy Model**: Use `predict_categories.py` for production predictions
4. **Monitor Performance**: Track accuracy on new data over time

## 📞 Support

For issues or questions:
1. Check the troubleshooting section above
2. Review the individual component documentation
3. Examine the generated summary report for detailed error information
