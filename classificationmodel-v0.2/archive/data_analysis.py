#!/usr/bin/env python3
"""
Data Analysis for Spend Platform Categorization Training Data
Comprehensive analysis of the categorization training dataset
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")

class CategorizationDataAnalyzer:
    """Comprehensive analyzer for categorization training data"""

    def __init__(self, excel_file_path):
        self.excel_file_path = excel_file_path
        self.df = None
        self.category_columns = ['Category L2', 'Category L3', 'Category L4', 'Category L5']

    def load_data(self):
        """Load and preprocess the training data"""
        print("Loading training data...")

        # Read Excel file with proper header
        df_raw = pd.read_excel(self.excel_file_path, header=None)

        # Find the header row (row 2 contains column names)
        header_row = 2
        self.df = pd.read_excel(self.excel_file_path, header=header_row)

        # Clean column names
        self.df.columns = self.df.columns.str.strip()

        # Remove the unnamed first column if it exists
        if 'Unnamed: 0' in self.df.columns:
            self.df = self.df.drop('Unnamed: 0', axis=1)

        # Clean text data
        self.df['Item_Descripton'] = self.df['Item_Descripton'].fillna('').astype(str)

        print(f"✅ Loaded {len(self.df)} training samples")
        print(f"✅ Columns: {list(self.df.columns)}")
        return self.df

    def analyze_basic_statistics(self):
        """Analyze basic statistics of the dataset"""
        print("\n" + "="*60)
        print("BASIC DATA STATISTICS")
        print("="*60)

        print(f"Total samples: {len(self.df) if self.df is not None else 0}")
        print(f"Total features: {len(self.df.columns) if self.df is not None else 0}")

        # Text statistics
        if self.df is not None:
            descriptions = self.df['Item_Descripton']
            print("\n📝 Text Statistics:")
            print(f"  Average description length: {descriptions.str.len().mean():.1f} characters")
            print(f"  Median description length: {descriptions.str.len().median():.1f} characters")
            print(f"  Max description length: {descriptions.str.len().max()} characters")
            print(f"  Min description length: {descriptions.str.len().min()} characters")

            # Word count statistics
            word_counts = descriptions.str.split().str.len()
            print(f"  Average word count: {word_counts.mean():.1f} words")
            print(f"  Median word count: {word_counts.median():.1f} words")

        # Category hierarchy analysis
        print("\n🏷️  Category Hierarchy Analysis:")
        for col in self.category_columns:
            if self.df is not None and col in self.df.columns:
                unique_count = self.df[col].nunique()
                print(f"  {col}: {unique_count} unique categories")

    def analyze_category_distributions(self):
        """Analyze the distribution of categories at each level"""
        if self.df is None:
            print("❌ No data loaded")
            return

        print("\n" + "="*60)
        print("CATEGORY DISTRIBUTION ANALYSIS")
        print("="*60)

        for col in self.category_columns:
            if col in self.df.columns:
                print(f"\n📊 {col} Distribution:")
                value_counts = self.df[col].value_counts()

                print(f"  Total unique categories: {len(value_counts)}")
                print(f"  Most frequent category: '{value_counts.index[0]}' ({value_counts.iloc[0]} items)")

                # Show top 10 categories
                print("  Top 10 categories:")
                for i, (category, count) in enumerate(value_counts.head(10).items()):
                    percentage = (count / len(self.df)) * 100
                    print(f"{i+1:2d}. {category:<30} {count:5d} ({percentage:5.1f}%)")

                # Show distribution statistics
                print(f"  Distribution stats:")
                print(f"    Mean items per category: {value_counts.mean():.1f}")
                print(f"    Median items per category: {value_counts.median():.1f}")
                print(f"    Max items in a category: {value_counts.max()}")
                print(f"    Min items in a category: {value_counts.min()}")

    def analyze_text_patterns(self):
        """Analyze text patterns in item descriptions"""
        if self.df is None:
            print("❌ No data loaded")
            return

        print("\n" + "="*60)
        print("TEXT PATTERN ANALYSIS")
        print("="*60)

        descriptions = self.df['Item_Descripton']

        # Most common words
        print("\n🔤 Most Common Words:")
        all_words = []
        for desc in descriptions:
            words = re.findall(r'\b\w+\b', str(desc).lower())
            all_words.extend(words)

        word_freq = Counter(all_words)
        print("  Top 20 most common words:")
        for i, (word, freq) in enumerate(word_freq.most_common(20)):
            print(f"{i+1:2d}. {word:<15} {freq:5d}")

        # Word length distribution
        print("\n📏 Word Length Distribution:")
        word_lengths = [len(word) for word in all_words]
        print(f"  Average word length: {np.mean(word_lengths):.1f} characters")
        print(f"  Median word length: {np.median(word_lengths):.1f} characters")

        # Special characters and patterns
        print("\n🔍 Special Patterns:")
        has_numbers = descriptions.str.contains(r'\d').sum()
        has_uppercase = descriptions.str.contains(r'[A-Z]').sum()
        has_symbols = descriptions.str.contains(r'[^\w\s]').sum()

        print(f"  Descriptions with numbers: {has_numbers} ({has_numbers/len(descriptions)*100:.1f}%)")
        print(f"  Descriptions with uppercase: {has_uppercase} ({has_uppercase/len(descriptions)*100:.1f}%)")
        print(f"  Descriptions with symbols: {has_symbols} ({has_symbols/len(descriptions)*100:.1f}%)")

    def analyze_category_relationships(self):
        """Analyze relationships between category levels"""
        if self.df is None:
            print("❌ No data loaded")
            return

        print("\n" + "="*60)
        print("CATEGORY RELATIONSHIP ANALYSIS")
        print("="*60)

        # L2 to L3 relationships (skipping L1 as requested)
        if 'Category L2' in self.df.columns and 'Category L3' in self.df.columns:
            print("\n🔗 Category L2 → L3 Relationships:")
            l2_l3 = self.df.groupby(['Category L2', 'Category L3']).size().reset_index(name='count')
            l2_l3 = l2_l3.sort_values('count', ascending=False)

            print("  Top 10 L2→L3 combinations:")
            for idx, row in enumerate(l2_l3.head(10).itertuples(), 1):
                print(f"{idx:2d}. {row[1]:<20} → {row[2]:<20} {row[3]:5d}")

        # Category path analysis
        print("\n🛣️  Category Path Analysis:")
        self.df['category_path'] = self.df[self.category_columns].apply(
            lambda x: ' → '.join(x.astype(str)), axis=1
        )

        path_counts = self.df['category_path'].value_counts()
        print(f"  Total unique category paths: {len(path_counts)}")
        print("  Most common category paths:")
        for i, (path, count) in enumerate(path_counts.head(5).items()):
            print(f"{i+1:2d}. {path:<30} {count:5d}")

    def create_visualizations(self):
        """Create visualizations for the analysis"""
        if self.df is None:
            print("❌ No data loaded")
            return

        print("\n" + "="*60)
        print("CREATING VISUALIZATIONS")
        print("="*60)

        # Set up the plotting area
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Spend Platform Categorization Data Analysis', fontsize=16)

        # Category L2 distribution (skipping L1 as requested)
        if 'Category L2' in self.df.columns:
            l2_counts = self.df['Category L2'].value_counts()
            axes[0, 0].pie(l2_counts.values, labels=l2_counts.index, autopct='%1.1f%%')
            axes[0, 0].set_title('Category L2 Distribution')

        # Category L5 top 10
        if 'Category L5' in self.df.columns:
            l5_counts = self.df['Category L5'].value_counts().head(10)
            l5_counts.plot(kind='barh', ax=axes[0, 1])
            axes[0, 1].set_title('Top 10 Category L5')
            axes[0, 1].set_xlabel('Count')

        # Description length distribution
        desc_lengths = self.df['Item_Descripton'].str.len()
        axes[1, 0].hist(desc_lengths, bins=30, alpha=0.7)
        axes[1, 0].set_title('Description Length Distribution')
        axes[1, 0].set_xlabel('Length (characters)')
        axes[1, 0].set_ylabel('Frequency')

        # Word count distribution
        word_counts = self.df['Item_Descripton'].str.split().str.len()
        axes[1, 1].hist(word_counts, bins=20, alpha=0.7)
        axes[1, 1].set_title('Word Count Distribution')
        axes[1, 1].set_xlabel('Word Count')
        axes[1, 1].set_ylabel('Frequency')

        plt.tight_layout()
        plt.savefig('data_analysis_visualizations.png', dpi=300, bbox_inches='tight')
        print("✅ Visualizations saved as 'data_analysis_visualizations.png'")

    def generate_summary_report(self):
        """Generate a comprehensive summary report"""
        if self.df is None:
            print("❌ No data loaded")
            return {}

        print("\n" + "="*60)
        print("SUMMARY REPORT")
        print("="*60)

        report = {
            'total_samples': len(self.df),
            'category_hierarchy': {},
            'text_statistics': {},
            'data_quality': {}
        }

        # Category hierarchy info
        for col in self.category_columns:
            if col in self.df.columns:
                report['category_hierarchy'][col] = {
                    'unique_categories': self.df[col].nunique(),
                    'most_common': self.df[col].value_counts().index[0],
                    'most_common_count': self.df[col].value_counts().iloc[0]
                }

        # Text statistics
        descriptions = self.df['Item_Descripton']
        report['text_statistics'] = {
            'avg_length': descriptions.str.len().mean(),
            'avg_word_count': descriptions.str.split().str.len().mean(),
            'total_unique_words': len(set(' '.join(descriptions).split()))
        }

        # Data quality
        report['data_quality'] = {
            'missing_values': self.df.isnull().sum().sum(),
            'duplicate_descriptions': self.df['Item_Descripton'].duplicated().sum()
        }

        print("📊 Dataset Summary:")
        print(f"  • Total training samples: {report['total_samples']}")
        print(f"  • Missing values: {report['data_quality']['missing_values']}")
        print(f"  • Duplicate descriptions: {report['data_quality']['duplicate_descriptions']}")

        print("\n🏷️  Category Hierarchy:")
        for level, stats in report['category_hierarchy'].items():
            print(f"  • {level}: {stats['unique_categories']} categories")
            print(f"    Most common: {stats['most_common']} ({stats['most_common_count']} items)")

        print("\n📝 Text Statistics:")
        print(f"  • Average description length: {report['text_statistics']['avg_length']:.1f} characters")
        print(f"  • Average word count: {report['text_statistics']['avg_word_count']:.1f} words")
        print(f"  • Unique words: {report['text_statistics']['total_unique_words']}")

        return report

    def run_complete_analysis(self):
        """Run the complete analysis pipeline"""
        print("🚀 Starting comprehensive data analysis...")

        # Load data
        self.load_data()

        # Run all analyses
        self.analyze_basic_statistics()
        self.analyze_category_distributions()
        self.analyze_text_patterns()
        self.analyze_category_relationships()
        self.create_visualizations()
        self.generate_summary_report()

        print("\n✅ Analysis complete!")
        print("📁 Results saved to current directory")

def main():
    """Main function to run the data analysis"""
    excel_file = '/Users/sujoymukherjee/code/spendplatform/context/Copy of Catergorization Working Sheet_Labelled Data for Training.xlsx'

    analyzer = CategorizationDataAnalyzer(excel_file)
    analyzer.run_complete_analysis()

if __name__ == "__main__":
    main()
