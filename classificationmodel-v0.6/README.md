# Hybrid Classification System v0.6

A production-ready hybrid classification system that combines L2, L4, and L5 models with intelligent category override capabilities.

## 🎯 Key Features

- **Multi-Level Classification**: Combines L2, L4, and L5 classification models
- **Intelligent Category Override**: Automatically overrides L2 categories when L5 confidence > 65%
- **Taxonomy-Based Mapping**: Uses official taxonomy hierarchy from Categorization File.xlsx
- **Production-Ready**: Optimized for performance, reliability, and scalability
- **Multiple Interfaces**: Command-line, Python API, and batch processing support

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Initialize System

```python
from hybrid_classifier import HybridClassificationSystem

# Initialize the system
classifier = HybridClassificationSystem()
classifier.initialize()

# Classify a single item
result = classifier.predict_single("LED HAND LAMP, 24VDC, 7W")
print(f"L2 Category: {result['adjusted_l2_category']}")
print(f"Confidence: {result['final_confidence']:.3f}")
```

### 3. Production API

```python
from production_api import ProductionClassificationAPI

# Initialize API
api = ProductionClassificationAPI()
api.initialize()

# Classify single item
result = api.classify_single("HYDRAULIC OIL BULK")
print(result['prediction']['l2_category'])

# Classify file
api.classify_file("input.xlsx", output_path="classified_output.xlsx")
```

### 4. Command Line Interface

```bash
# Classify a file
python cli.py classify-file input.xlsx --output results.xlsx

# Classify single text
python cli.py classify-text "WELDING ELECTRODE 3.15MM" --verbose

# Health check
python cli.py health-check
```

## 📊 System Architecture

### Model Hierarchy
- **L2 Model**: Production SVM Enhanced (80.28% accuracy)
- **L4 Model**: Random Forest Simplified (54.69% accuracy)  
- **L5 Model**: SVM Simplified (50.30% accuracy)

### Classification Logic
1. **High Confidence L2** (≥75%): Use L2 prediction directly
2. **Medium Confidence L2** (50-75%): Try L4/L5 enhancement
3. **Low Confidence L2** (<50%): Replace with best L4/L5 prediction
4. **L5 Override** (L5 confidence ≥65%): Override L2/L4 categories using taxonomy

### Taxonomy Mapping
- L5 → L4: 837 mappings
- L5 → L2: 837 mappings  
- L4 → L2: 409 mappings

## 🔧 Configuration

Edit `config/settings.py` to customize:

```python
CONFIDENCE_THRESHOLDS = {
    'high': 0.75,       # High confidence threshold
    'medium': 0.50,     # Medium confidence threshold
    'l5_override': 0.65 # L5 override threshold
}
```

## 📁 Directory Structure

```
classificationmodel-v0.6/
├── hybrid_classifier.py      # Core classification system
├── production_api.py         # Production API wrapper
├── cli.py                    # Command-line interface
├── requirements.txt          # Dependencies
├── config/
│   └── settings.py          # Configuration settings
├── models/
│   ├── production_svm_enhanced.pkl    # L2 model
│   ├── l4_simplified_model.pkl        # L4 model
│   └── l5_simplified_model.pkl        # L5 model
└── data/
    └── Categorization File.xlsx       # Taxonomy mappings
```

## 🧪 Testing

### Basic Test
```python
# Test the system
from production_api import ProductionClassificationAPI

api = ProductionClassificationAPI()
api.initialize()

# Test items
test_items = [
    "LED HAND LAMP, 24VDC, 7W",
    "HYDRAULIC OIL BULK",
    "WELDING ELECTRODE 3.15MM"
]

for item in test_items:
    result = api.classify_single(item)
    print(f"{item} → {result['prediction']['l2_category']}")
```

### Health Check
```bash
python cli.py health-check
```

## 📈 Performance

- **Throughput**: ~50-100 classifications/second (single-threaded)
- **Memory Usage**: ~500MB for loaded models
- **Initialization Time**: ~2-3 seconds
- **Category Override Rate**: ~60-70% for medium/low confidence items

## 🎯 Production Usage

### Batch Processing
```python
# Process large files
api.classify_file(
    file_path="unclassified_transactions.xlsx",
    description_column="Description",
    output_path="classified_results.xlsx"
)
```

### Custom Thresholds
```python
# Custom confidence thresholds
result = api.classify_single(
    description="INDUSTRIAL PUMP",
    confidence_thresholds={
        'high': 0.80,
        'medium': 0.55,
        'l5_override': 0.70
    }
)
```

## 🔄 Category Override Logic

When L5 confidence ≥ 65%:
1. **L2 Category**: Derived from L5 using L5→L2 taxonomy mapping
2. **L4 Category**: Derived from L5 using L5→L4 taxonomy mapping
3. **Hierarchical Consistency**: Ensures proper taxonomy relationships

Example:
```
L5: "Lamps/Lights" (confidence: 0.68)
↓ (L5→L4 mapping)
L4: "Lighting Fixtures, Accessories & Lamp Components"
↓ (L5→L2 mapping)
L2: "Electrical"
```

## 📋 Output Format

Classification results include:
- **l2_category**: Final L2 category (potentially overridden)
- **l4_category**: Final L4 category (potentially overridden)
- **l5_category**: L5 prediction
- **confidence**: Final confidence score
- **category_overridden**: Boolean indicating if L5 override was applied
- **strategy**: Classification strategy used
- **improvement**: Confidence improvement achieved

## 🔧 Troubleshooting

### Common Issues

1. **Models not found**: Ensure all .pkl files are in the `models/` directory
2. **Taxonomy file missing**: Verify `Categorization File.xlsx` is in `data/` directory
3. **Column not found**: Check description column name matches your data
4. **Memory issues**: Consider processing files in smaller batches

### Debugging
```python
# Enable verbose logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Check system status
stats = api.get_api_stats()
print(f"System status: {stats}")
```

## 📊 Monitoring

The system provides comprehensive statistics:
- Total predictions processed
- Success/failure rates
- Category override rates
- Processing performance metrics
- Model-specific statistics

## 🚀 Deployment

For production deployment:
1. Install dependencies: `pip install -r requirements.txt`
2. Copy model files to `models/` directory
3. Copy taxonomy file to `data/` directory
4. Test with health check: `python cli.py health-check`
5. Start processing: Use API or CLI as needed

## 📝 Version History

**v0.6** (2025-09-09)
- Production-ready hybrid classification system
- L5-based L2 category override when confidence > 65%
- Taxonomy-based hierarchical mapping
- Complete API and CLI interfaces
- Comprehensive testing and monitoring

---

## 📞 Support

For issues or questions, check the health status and system logs:
```bash
python cli.py health-check
```
