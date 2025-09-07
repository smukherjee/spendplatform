#!/usr/bin/env python3
"""
Complete Data Pipeline for Spend Platform Categorization
Runs the full pipeline from data analysis to preprocessing to training to prediction
"""

import sys
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

class SpendPlatformPipeline:
    """Complete pipeline orchestrator for spend categorization"""

    def __init__(self, base_dir=None):
        self.base_dir = base_dir or os.path.dirname(os.path.abspath(__file__))
        self.context_dir = os.path.join(os.path.dirname(self.base_dir), 'context')
        self.data_file = os.path.join(self.context_dir, 'Copy of Catergorization Working Sheet_Labelled Data for Training.xlsx')
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_id = f"pipeline_run_{self.timestamp}"
        self.output_dir = os.path.join(self.base_dir, self.run_id)
        self.results = {}

        # Create the pipeline-specific output directory
        Path(self.output_dir).mkdir(exist_ok=True)
        print(f"📁 Created pipeline output directory: {self.run_id}")

    def run_command(self, command, description, cwd=None):
        """Run a command and capture output"""
        print(f"\n{'='*80}")
        print(f"🔄 {description}")
        print('='*80)

        start_time = time.time()

        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=cwd or self.base_dir,
                capture_output=True,
                text=True,
                check=True
            )

            end_time = time.time()
            duration = end_time - start_time

            print(f"✅ {description} completed in {duration:.2f} seconds")
            print(result.stdout)

            return True, result.stdout, result.stderr

        except subprocess.CalledProcessError as e:
            end_time = time.time()
            duration = end_time - start_time

            print(f"❌ {description} failed after {duration:.2f} seconds")
            print(f"Error: {e}")
            print(f"STDOUT: {e.stdout}")
            print(f"STDERR: {e.stderr}")

            return False, e.stdout, e.stderr

    def step_1_data_analysis(self):
        """Step 1: Run data analysis"""
        print(f"\n{'🚀'*20}")
        print("STEP 1: DATA ANALYSIS")
        print(f"{'🚀'*20}")

        # Create a temporary analysis script that saves to pipeline directory
        analysis_script = f"""
import sys
sys.path.append('{self.base_dir}')

from data_analysis import CategorizationDataAnalyzer
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# Run analysis
analyzer = CategorizationDataAnalyzer('{self.data_file}')
analyzer.run_complete_analysis()

# Move generated visualization to pipeline directory
import os
import shutil
viz_file = 'data_analysis_visualizations.png'
if os.path.exists(viz_file):
    shutil.move(viz_file, '{self.output_dir}/data_analysis_visualizations.png')
    print(f"✅ Moved visualization to: {self.run_id}/data_analysis_visualizations.png")
"""

        with open('temp_analysis.py', 'w') as f:
            f.write(analysis_script)

        success, stdout, stderr = self.run_command(
            "python temp_analysis.py",
            "Data Analysis"
        )

        # Clean up
        if os.path.exists('temp_analysis.py'):
            os.remove('temp_analysis.py')

        self.results['data_analysis'] = {'success': success, 'output': stdout, 'error': stderr}
        return success

    def step_2_preprocessing(self):
        """Step 2: Run data preprocessing"""
        print(f"\n{'📊'*20}")
        print("STEP 2: DATA PREPROCESSING")
        print(f"{'📊'*20}")

        # Create a temporary preprocessing script that saves to pipeline directory
        preprocess_script = f"""
import sys
sys.path.append('{self.base_dir}')

from preprocess_data import DataPreprocessor

# Run preprocessing with pipeline-specific output directory
preprocessor = DataPreprocessor('{self.data_file}', output_dir='{self.output_dir}/preprocessed_data')
success = preprocessor.run_preprocessing_pipeline()

if not success:
    sys.exit(1)
"""

        with open('temp_preprocess.py', 'w') as f:
            f.write(preprocess_script)

        success, stdout, stderr = self.run_command(
            "python temp_preprocess.py",
            "Data Preprocessing"
        )

        # Clean up
        if os.path.exists('temp_preprocess.py'):
            os.remove('temp_preprocess.py')

        self.results['preprocessing'] = {'success': success, 'output': stdout, 'error': stderr}
        return success

    def step_3_training_experiments(self):
        """Step 3: Run training experiments"""
        print(f"\n{'🤖'*20}")
        print("STEP 3: MODEL TRAINING EXPERIMENTS")
        print(f"{'🤖'*20}")

        # Create a temporary training script that saves to pipeline directory
        training_script = f"""
import sys
import os
sys.path.append('{self.base_dir}')

# Change to pipeline directory for output files
os.chdir('{self.output_dir}')

# Import and run experiments
from run_experiments_controlled import main
main()
"""

        with open('temp_training.py', 'w') as f:
            f.write(training_script)

        success, stdout, stderr = self.run_command(
            "python temp_training.py",
            "Training Experiments"
        )

        # Clean up
        if os.path.exists('temp_training.py'):
            os.remove('temp_training.py')

        # Move any model files that were created in the main directory to pipeline directory
        self._move_generated_files_to_pipeline_dir()

        self.results['training'] = {'success': success, 'output': stdout, 'error': stderr}
        return success

    def step_4_prediction_demo(self):
        """Step 4: Run prediction demonstration"""
        print(f"\n{'🔮'*20}")
        print("STEP 4: PREDICTION DEMONSTRATION")
        print(f"{'🔮'*20}")

        # Create a temporary prediction script
        prediction_script = f"""
import sys
import os
sys.path.append('{self.base_dir}')

# Change to pipeline directory for output files
os.chdir('{self.output_dir}')

from predict_categories import MultiLevelModelPredictor
import pandas as pd

# Load the best performing model (spaCy optimized based on experiments)
# First try to load from the pipeline directory
model_path = None
for file in os.listdir('.'):
    if file.startswith('spend_categorization_model_') and file.endswith('.pkl'):
        if 'spacy' in file:
            model_path = file
            break
        elif model_path is None:  # Take the first model as fallback
            model_path = file

if model_path:
    try:
        predictor = MultiLevelModelPredictor(model_path=model_path)
        print(f"✅ Loaded model from pipeline directory: {{model_path}}")
    except Exception as e:
        print(f"❌ Failed to load model {{model_path}}: {{e}}")
        sys.exit(1)
else:
    print("❌ No trained models found in pipeline directory")
    sys.exit(1)

# Create sample data for demonstration
sample_data = pd.DataFrame({{
    'Item_Code': ['DEMO001', 'DEMO002', 'DEMO003', 'DEMO004', 'DEMO005'],
    'Item_Descripton': [
        'Industrial hydraulic pump maintenance and repair services',
        'Office supplies including paper, pens, and printer cartridges',
        'Heavy duty industrial welding equipment and safety gear',
        'IT infrastructure server hardware and software licenses',
        'Manufacturing facility electrical system upgrade and installation'
    ],
    'Supplier_Name': ['Industrial Solutions Inc', 'Office Supply Co', 'WeldTech Pro', 'TechCorp Systems', 'ElectroTech Ltd']
}})

print("\\n📋 Sample Data for Prediction:")
for i, row in sample_data.iterrows():
    print(f"  {{i+1}}. {{row['Item_Descripton'][:60]}}...")

# Make predictions
results_df = predictor.predict_with_details(sample_data)

# Display results
print("\\n🎯 Prediction Results:")
print("="*80)
for i, row in results_df.iterrows():
    print(f"\\nItem {{i+1}}: {{row['Item_Descripton'][:50]}}...")
    for level in predictor.model.target_levels:
        pred_col = f'Predicted_{{level.replace(" ", "_")}}'
        conf_col = f'Confidence_{{level.replace(" ", "_")}}'
        if pred_col in row and conf_col in row:
            print(f"  {{level}}: {{row[pred_col]}} ({{row[conf_col]:.2f}})")

# Save results
results_df.to_excel('pipeline_prediction_results.xlsx', index=False)
print("\\n✅ Results saved to pipeline_prediction_results.xlsx")
"""

        with open('temp_predict.py', 'w') as f:
            f.write(prediction_script)

        success, stdout, stderr = self.run_command(
            "python temp_predict.py",
            "Prediction Demonstration"
        )

        # Clean up
        if os.path.exists('temp_predict.py'):
            os.remove('temp_predict.py')

        self.results['prediction'] = {'success': success, 'output': stdout, 'error': stderr}
        return success

    def _move_generated_files_to_pipeline_dir(self):
        """Move any generated files from main directory to pipeline directory"""
        import shutil

        # Files that might be generated in the main directory
        files_to_move = [
            '*.pkl',  # Model files
            'experiments_controlled_output_*.txt',  # Experiment results
            'model_evaluation.png',  # Evaluation plots
            'pipeline_prediction_results.xlsx'  # Prediction results
        ]

        for pattern in files_to_move:
            import glob
            matching_files = glob.glob(os.path.join(self.base_dir, pattern))
            for file_path in matching_files:
                if os.path.exists(file_path):
                    filename = os.path.basename(file_path)
                    dest_path = os.path.join(self.output_dir, filename)
                    shutil.move(file_path, dest_path)
                    print(f"✅ Moved {filename} to {self.run_id}/")

    def generate_summary_report(self):
        """Generate a comprehensive summary report"""
        print(f"\n{'📋'*20}")
        print("GENERATING SUMMARY REPORT")
        print(f"{'📋'*20}")

        # Create output directory
        Path(self.output_dir).mkdir(exist_ok=True)

        # Generate summary
        summary = f"""
SPEND PLATFORM CATEGORIZATION PIPELINE - EXECUTION REPORT
{'='*80}
Execution Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Pipeline ID: {self.timestamp}

PIPELINE STEPS SUMMARY:
{'='*80}

"""

        all_success = True
        for step, result in self.results.items():
            status = "✅ SUCCESS" if result['success'] else "❌ FAILED"
            summary += f"{step.upper()}: {status}\n"
            if not result['success']:
                all_success = False

        summary += f"\nOVERALL STATUS: {'✅ ALL STEPS COMPLETED SUCCESSFULLY' if all_success else '❌ SOME STEPS FAILED'}\n\n"

        # Add detailed results
        summary += "DETAILED RESULTS:\n"
        summary += "="*80 + "\n\n"

        for step, result in self.results.items():
            summary += f"{step.upper()}:\n"
            summary += "-"*40 + "\n"
            if result['success']:
                summary += "Status: SUCCESS\n"
            else:
                summary += "Status: FAILED\n"
                if result['error']:
                    summary += f"Error: {result['error'][:500]}...\n"
            summary += "\n"

        # Save summary
        summary_file = os.path.join(self.output_dir, "pipeline_summary_report.txt")
        with open(summary_file, 'w') as f:
            f.write(summary)

        print(summary)
        print(f"📄 Summary report saved to: {self.run_id}/pipeline_summary_report.txt")

        return all_success

    def run_complete_pipeline(self):
        """Run the complete pipeline"""
        print(f"🚀 STARTING COMPLETE SPEND PLATFORM PIPELINE")
        print(f"Timestamp: {self.timestamp}")
        print(f"Data File: {self.data_file}")
        print("="*80)

        start_time = time.time()

        # Step 1: Data Analysis
        if not self.step_1_data_analysis():
            print("❌ Pipeline failed at data analysis step")
            self.generate_summary_report()
            return False

        # Step 2: Preprocessing
        if not self.step_2_preprocessing():
            print("❌ Pipeline failed at preprocessing step")
            self.generate_summary_report()
            return False

        # Step 3: Training Experiments
        if not self.step_3_training_experiments():
            print("❌ Pipeline failed at training step")
            self.generate_summary_report()
            return False

        # Step 4: Prediction Demo
        if not self.step_4_prediction_demo():
            print("❌ Pipeline failed at prediction step")
            self.generate_summary_report()
            return False

        end_time = time.time()
        total_duration = end_time - start_time

        print(f"\n{'🎉'*20}")
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print(f"Total execution time: {total_duration:.2f} seconds")
        print(f"{'🎉'*20}")

        self.generate_summary_report()
        return True

def main():
    """Main function"""
    # Check if data file exists
    base_dir = os.path.dirname(os.path.abspath(__file__))
    context_dir = os.path.join(os.path.dirname(base_dir), 'context')
    data_file = os.path.join(context_dir, 'Copy of Catergorization Working Sheet_Labelled Data for Training.xlsx')

    if not os.path.exists(data_file):
        print(f"❌ Data file not found: {data_file}")
        print("Please ensure the training data file exists in the context directory.")
        return 1

    # Run pipeline
    pipeline = SpendPlatformPipeline()
    success = pipeline.run_complete_pipeline()

    return 0 if success else 1

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
