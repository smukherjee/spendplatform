import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

class SpendDataAnalyzer:
    """
    Comprehensive data analysis class for spend platform Excel files.
    Provides statistical analysis, visualization, and insights for transaction data.
    """

    def __init__(self):
        self.sample_data = None
        self.categorization_data = None
        self.training_data = None
        self.taxonomy_data = None

    def load_data(self):
        """Load all Excel files and prepare data for analysis."""
        try:
            # Load sample spend data
            self.sample_data = pd.read_excel('/Users/sujoymukherjee/code/spendplatform/context/sample spend data- filled.xlsx')

            # Load categorization file sheets
            xls = pd.ExcelFile('/Users/sujoymukherjee/code/spendplatform/context/Categorization File.xlsx')
            self.categorization_data = pd.read_excel(xls, 'Working File')
            self.taxonomy_data = pd.read_excel(xls, 'Taxonomy')

            # Load labelled training data
            training_df = pd.read_excel('/Users/sujoymukherjee/code/spendplatform/context/Copy of Catergorization Working Sheet_Labelled Data for Training.xlsx')
            training_df.columns = training_df.iloc[1]  # Use second row as headers
            self.training_data = training_df.iloc[2:].reset_index(drop=True)

            print("Data loaded successfully!")
            print(f"Sample data: {self.sample_data.shape[0]} rows")
            print(f"Categorization data: {self.categorization_data.shape[0]} rows")
            print(f"Training data: {self.training_data.shape[0]} rows")
            print(f"Taxonomy data: {self.taxonomy_data.shape[0]} rows")

        except Exception as e:
            print(f"Error loading data: {e}")

    def analyze_sample_data(self):
        """Analyze the sample spend data file."""
        if self.sample_data is None:
            print("Sample data not loaded!")
            return

        print("\n" + "="*60)
        print("SAMPLE SPEND DATA ANALYSIS")
        print("="*60)

        # Basic statistics
        print(f"\nBasic Statistics:")
        print(f"Total transactions: {len(self.sample_data)}")
        print(f"Total spend: ${self.sample_data['Item Invoice Value'].sum():,.2f}")
        print(f"Average transaction value: ${self.sample_data['Item Invoice Value'].mean():,.2f}")
        print(f"Median transaction value: ${self.sample_data['Item Invoice Value'].median():,.2f}")

        # Tower/Practice analysis
        if 'Tower (Practice)' in self.sample_data.columns:
            print(f"\nTower (Practice) Distribution:")
            tower_counts = self.sample_data['Tower (Practice)'].value_counts()
            for tower, count in tower_counts.items():
                print(f"  {tower}: {count} transactions ({count/len(self.sample_data)*100:.1f}%)")

        # Supplier analysis
        if 'SUPPLIER_NAME' in self.sample_data.columns:
            print(f"\nTop 10 Suppliers by Transaction Count:")
            supplier_counts = self.sample_data['SUPPLIER_NAME'].value_counts().head(10)
            for supplier, count in supplier_counts.items():
                spend = self.sample_data[self.sample_data['SUPPLIER_NAME'] == supplier]['Item Invoice Value'].sum()
                print(f"  {supplier}: {count} transactions, ${spend:,.2f} spend")

        # Material/Item analysis
        if 'Material Item Name' in self.sample_data.columns:
            print(f"\nTop 10 Items by Frequency:")
            item_counts = self.sample_data['Material Item Name'].value_counts().head(10)
            for item, count in item_counts.items():
                print(f"  {item}: {count} transactions")

    def analyze_categorization_data(self):
        """Analyze the categorization training data."""
        if self.training_data is None:
            print("Training data not loaded!")
            return

        print("\n" + "="*60)
        print("CATEGORIZATION TRAINING DATA ANALYSIS")
        print("="*60)

        # Category hierarchy analysis
        print(f"\nCategory L1 Distribution:")
        l1_counts = self.training_data['Category L1'].value_counts()
        for cat, count in l1_counts.items():
            print(f"  {cat}: {count} items ({count/len(self.training_data)*100:.1f}%)")

        print(f"\nCategory L5 Distribution (Top 20):")
        l5_counts = self.training_data['Category L5'].value_counts().head(20)
        for cat, count in l5_counts.items():
            print(f"  {cat}: {count} items")

        # Text analysis
        if 'Item_Descripton' in self.training_data.columns:
            print(f"\nText Analysis:")
            descriptions = self.training_data['Item_Descripton'].dropna().astype(str)
            avg_length = descriptions.str.len().mean()
            print(f"  Average description length: {avg_length:.1f} characters")

            # Most common words
            all_words = ' '.join(descriptions).lower().split()
            word_counts = Counter(all_words)
            print(f"  Most common words (excluding common terms):")
            common_words = [word for word, count in word_counts.most_common(20)
                          if word not in ['the', 'and', 'or', 'for', 'with', 'in', 'on', 'at', 'to', 'of']]
            for word in common_words[:10]:
                print(f"    {word}: {word_counts[word]}")

    def analyze_taxonomy_structure(self):
        """Analyze the taxonomy hierarchy."""
        if self.taxonomy_data is None:
            print("Taxonomy data not loaded!")
            return

        print("\n" + "="*60)
        print("TAXONOMY STRUCTURE ANALYSIS")
        print("="*60)

        for level in range(1, 6):
            col_name = f'Category {level}'
            if col_name in self.taxonomy_data.columns:
                unique_count = self.taxonomy_data[col_name].nunique()
                print(f"Category Level {level}: {unique_count} unique categories")

                if level == 1:
                    print(f"  Level 1 categories: {sorted(self.taxonomy_data[col_name].dropna().unique())}")

    def create_visualizations(self):
        """Create visualizations for the data analysis."""
        if self.sample_data is None or self.training_data is None:
            print("Data not loaded for visualization!")
            return

        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")

        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Spend Platform Data Analysis', fontsize=16, fontweight='bold')

        # 1. Tower distribution
        if 'Tower (Practice)' in self.sample_data.columns:
            tower_data = self.sample_data['Tower (Practice)'].value_counts()
            axes[0, 0].pie(tower_data.values, labels=tower_data.index, autopct='%1.1f%%')
            axes[0, 0].set_title('Distribution by Tower (Practice)')

        # 2. Category L1 distribution
        if 'Category L1' in self.training_data.columns:
            l1_data = self.training_data['Category L1'].value_counts()
            axes[0, 1].bar(l1_data.index, l1_data.values)
            axes[0, 1].set_title('Category L1 Distribution')
            axes[0, 1].tick_params(axis='x', rotation=45)

        # 3. Top Category L5
        if 'Category L5' in self.training_data.columns:
            l5_data = self.training_data['Category L5'].value_counts().head(10)
            axes[1, 0].barh(l5_data.index, l5_data.values)
            axes[1, 0].set_title('Top 10 Category L5')
            axes[1, 0].set_xlabel('Count')

        # 4. Transaction value distribution
        if 'Item Invoice Value' in self.sample_data.columns:
            spend_data = self.sample_data['Item Invoice Value'].dropna()
            spend_data = spend_data[spend_data > 0]  # Remove negative/zero values
            if len(spend_data) > 0:
                axes[1, 1].hist(np.log10(spend_data + 1), bins=50, alpha=0.7)
                axes[1, 1].set_title('Transaction Value Distribution (log scale)')
                axes[1, 1].set_xlabel('Log10(Transaction Value)')
                axes[1, 1].set_ylabel('Frequency')

        plt.tight_layout()
        plt.savefig('/Users/sujoymukherjee/code/spendplatform/context/data_analysis_visualizations.png', dpi=300, bbox_inches='tight')
        print("\nVisualization saved as 'data_analysis_visualizations.png'")

    def generate_data_quality_report(self):
        """Generate a comprehensive data quality report."""
        print("\n" + "="*60)
        print("DATA QUALITY REPORT")
        print("="*60)

        if self.sample_data is not None:
            print(f"\nSample Data Quality:")
            total_rows = len(self.sample_data)
            print(f"  Total rows: {total_rows}")

            # Check for missing values
            missing_data = self.sample_data.isnull().sum()
            missing_percent = (missing_data / total_rows * 100).round(2)
            print(f"  Columns with missing data (>5%):")
            for col in self.sample_data.columns:
                if missing_percent[col] > 5:
                    print(f"    {col}: {missing_percent[col]}% missing ({missing_data[col]} rows)")

            # Check for duplicates
            if 'Material Code' in self.sample_data.columns:
                duplicates = self.sample_data.duplicated(subset=['Material Code']).sum()
                print(f"  Duplicate material codes: {duplicates}")

        if self.training_data is not None:
            print(f"\nTraining Data Quality:")
            total_rows = len(self.training_data)
            print(f"  Total rows: {total_rows}")

            # Missing categories
            if 'Category L1' in self.training_data.columns:
                missing_l1 = self.training_data['Category L1'].isnull().sum()
                print(f"  Missing Category L1: {missing_l1} ({missing_l1/total_rows*100:.1f}%)")

            if 'Category L5' in self.training_data.columns:
                missing_l5 = self.training_data['Category L5'].isnull().sum()
                print(f"  Missing Category L5: {missing_l5} ({missing_l5/total_rows*100:.1f}%)")

    def create_categorization_file(self, output_path=None):
        """Create a comprehensive categorization file from the training data."""
        if self.training_data is None:
            print("Training data not loaded!")
            return

        print("\n" + "="*60)
        print("CREATING CATEGORIZATION FILE")
        print("="*60)

        # Create a clean categorization mapping
        required_columns = ['Item Code', 'Item_Descripton', 'Category L1', 'Category L2', 'Category L3', 'Category L4', 'Category L5']
        available_columns = [col for col in required_columns if col in self.training_data.columns]

        categorization_mapping = self.training_data[available_columns].copy()

        # Clean the data
        print("Cleaning data...")
        initial_count = len(categorization_mapping)

        # Remove rows with missing essential fields
        categorization_mapping = categorization_mapping.dropna(subset=['Item_Descripton', 'Category L1'])
        categorization_mapping = categorization_mapping[categorization_mapping['Category L1'].str.strip() != '']
        categorization_mapping = categorization_mapping[categorization_mapping['Item_Descripton'].str.strip() != '']

        # Remove unclear categories
        if 'Unclear' in categorization_mapping['Category L1'].values:
            categorization_mapping = categorization_mapping[categorization_mapping['Category L1'] != 'Unclear']

        cleaned_count = len(categorization_mapping)
        print(f"Removed {initial_count - cleaned_count} invalid rows")

        # Add additional useful columns
        print("Adding analysis columns...")

        # Text length analysis
        categorization_mapping['description_length'] = categorization_mapping['Item_Descripton'].str.len()
        categorization_mapping['word_count'] = categorization_mapping['Item_Descripton'].str.split().str.len()

        # Category hierarchy completeness
        hierarchy_levels = ['Category L1', 'Category L2', 'Category L3', 'Category L4', 'Category L5']
        categorization_mapping['hierarchy_completeness'] = categorization_mapping[hierarchy_levels].notna().sum(axis=1)

        # Create category path
        def create_category_path(row):
            path_parts = []
            for level in hierarchy_levels:
                if level in row.index and pd.notna(row[level]):
                    path_parts.append(str(row[level]).strip())
            return ' > '.join(path_parts)

        categorization_mapping['category_path'] = categorization_mapping.apply(create_category_path, axis=1)

        # Statistics
        print(f"\nCategorization File Statistics:")
        print(f"Total items: {len(categorization_mapping)}")
        print(f"Unique Category L1: {categorization_mapping['Category L1'].nunique()}")
        print(f"Unique Category L5: {categorization_mapping['Category L5'].nunique()}")
        print(f"Average description length: {categorization_mapping['description_length'].mean():.1f} characters")
        print(f"Average word count: {categorization_mapping['word_count'].mean():.1f} words")

        # Category distribution
        print(f"\nCategory L1 Distribution:")
        l1_dist = categorization_mapping['Category L1'].value_counts()
        for cat, count in l1_dist.items():
            print(f"  {cat}: {count} items ({count/len(categorization_mapping)*100:.1f}%)")

        # Top Category L5
        print(f"\nTop 10 Category L5:")
        l5_dist = categorization_mapping['Category L5'].value_counts().head(10)
        for cat, count in l5_dist.items():
            print(f"  {cat}: {count} items")

        # Save the file
        if output_path is None:
            output_path = '/Users/sujoymukherjee/code/spendplatform/context/comprehensive_categorization.xlsx'

        categorization_mapping.to_excel(output_path, index=False)
        print(f"\nCategorization file saved to: {output_path}")

        # Create a simplified version for easy reference
        simple_mapping = categorization_mapping[['Item_Descripton', 'Category L1', 'Category L5', 'category_path']].copy()
        simple_path = output_path.replace('.xlsx', '_simple.xlsx')
        simple_mapping.to_excel(simple_path, index=False)
        print(f"Simple categorization file saved to: {simple_path}")

        return categorization_mapping

    def run_complete_analysis(self):
        """Run the complete data analysis pipeline."""
        print("Starting comprehensive data analysis...")

        self.load_data()
        self.analyze_sample_data()
        self.analyze_categorization_data()
        self.analyze_taxonomy_structure()
        self.create_visualizations()
        self.generate_data_quality_report()
        self.create_categorization_file()

        print("\n" + "="*60)
        print("ANALYSIS COMPLETE")
        print("="*60)
        print("Summary of key findings:")
        print("1. Data loaded from all Excel files successfully")
        print("2. Generated statistical summaries and distributions")
        print("3. Created visualizations saved to PNG file")
        print("4. Identified data quality issues and missing values")
        print("5. Analyzed category hierarchies and text patterns")
        print("6. Created comprehensive categorization file")


if __name__ == "__main__":
    analyzer = SpendDataAnalyzer()
    analyzer.run_complete_analysis()
