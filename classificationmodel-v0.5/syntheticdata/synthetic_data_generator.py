#!/usr/bin/env python3
"""
Synthetic Data Generator for Spend Classification
Generates realistic item descriptions for each category hierarchy level (L2, L3, L4)
Ensures at least 200 distinct records per category
"""

import pandas as pd
import numpy as np
import random
from itertools import product
import re

class SyntheticDataGenerator:
    def __init__(self):
        self.existing_data = None
        self.taxonomy = None
        self.patterns = {}
        self.technical_terms = {}
        self.industry_keywords = {}
        
    def load_existing_data(self):
        """Load existing training data to understand patterns"""
        print("📊 Loading existing training data...")
        
        train_path = '/Users/sujoymukherjee/code/spendplatform/classificationmodel-v0.5/preprocessed_data/top_10_l2_categories_data.xlsx'
        self.existing_data = pd.read_excel(train_path)
        
        print(f"✅ Loaded {len(self.existing_data)} existing records")
        print(f"📋 Categories: {self.existing_data['Category L2'].nunique()}")
        
    def load_taxonomy(self):
        """Load taxonomy hierarchy from categorization file"""
        print("📁 Loading taxonomy hierarchy...")
        
        cat_path = '/Users/sujoymukherjee/code/spendplatform/context/archive/Categorization File.xlsx'
        self.taxonomy = pd.read_excel(cat_path, sheet_name='Taxonomy')
        
        print(f"✅ Loaded taxonomy with {len(self.taxonomy)} categories")
        print(f"📋 Hierarchy levels: L1({self.taxonomy['Category 1'].nunique()}) → L2({self.taxonomy['Category 2'].nunique()}) → L3({self.taxonomy['Category 3'].nunique()}) → L4({self.taxonomy['Category 4'].nunique()}) → L5({self.taxonomy['Category 5'].nunique()})")
        
    def analyze_patterns(self):
        """Analyze existing item description patterns"""
        print("🔍 Analyzing existing item description patterns...")
        
        if self.existing_data is None:
            print("❌ No existing data loaded")
            return
        
        # Analyze patterns by category
        for l2_category in self.existing_data['Category L2'].unique():
            category_data = self.existing_data[self.existing_data['Category L2'] == l2_category]
            descriptions = category_data['Item_Descripton'].tolist()
            
            # Extract common patterns
            patterns = {
                'prefixes': set(),
                'suffixes': set(),
                'keywords': set(),
                'technical_specs': set(),
                'brands': set(),
                'units': set()
            }
            
            for desc in descriptions:
                desc_upper = str(desc).upper()
                words = desc_upper.split()
                
                # Extract technical specifications (numbers with units)
                tech_specs = re.findall(r'\d+[\w]*(?:\.\d+)?(?:[A-Z]{1,4}|MM|CM|KG|LB|V|A|W|Hz|RPM|PSI|BAR)', desc_upper)
                patterns['technical_specs'].update(tech_specs)
                
                # Extract potential brands (capitalized words)
                brands = [word for word in words if word.isupper() and len(word) > 2 and not any(char.isdigit() for char in word)]
                patterns['brands'].update(brands[:3])  # Limit to avoid noise
                
                # Extract units
                units = re.findall(r'(?:MM|CM|M|KG|G|LB|V|A|W|Hz|RPM|PSI|BAR|L|ML|GAL)', desc_upper)
                patterns['units'].update(units)
                
                # Extract keywords (common meaningful words)
                keywords = [word for word in words if len(word) > 3 and not word.isdigit()]
                patterns['keywords'].update(keywords[:5])  # Limit to avoid noise
            
            self.patterns[l2_category] = patterns
            
        print(f"✅ Analyzed patterns for {len(self.patterns)} L2 categories")
        
    def define_industry_terms(self):
        """Define industry-specific terms for different categories"""
        self.industry_keywords = {
            'Electrical': {
                'components': ['CABLE', 'WIRE', 'SWITCH', 'RELAY', 'TRANSFORMER', 'CAPACITOR', 'RESISTOR', 'FUSE', 'BREAKER', 'CONTACTOR'],
                'specifications': ['240V', '110V', '415V', '50Hz', '60Hz', '16A', '32A', '63A', '100A'],
                'materials': ['COPPER', 'ALUMINUM', 'PVC', 'XLPE', 'RUBBER'],
                'types': ['SINGLE CORE', 'MULTI CORE', 'ARMORED', 'FLEXIBLE', 'RIGID'],
                'brands': ['ABB', 'SCHNEIDER', 'SIEMENS', 'LEGRAND', 'HAVELLS', 'L&T']
            },
            'Manufacturing Components & Supplies': {
                'components': ['BOLT', 'NUT', 'WASHER', 'SCREW', 'GASKET', 'SEAL', 'BEARING', 'BUSHING', 'SPRING'],
                'specifications': ['M6', 'M8', 'M10', 'M12', 'M16', 'M20', 'SS304', 'SS316', 'MS'],
                'materials': ['STAINLESS STEEL', 'MILD STEEL', 'BRASS', 'BRONZE', 'NYLON', 'RUBBER'],
                'types': ['HEX HEAD', 'SOCKET HEAD', 'COUNTERSUNK', 'PAN HEAD', 'ROUND HEAD'],
                'brands': ['UNBRAKO', 'AGRAWAL', 'SUNDRAM', 'TVS', 'MAHINDRA']
            },
            'Industrial Manufacturing & Processing Machinery & Accessories': {
                'components': ['CUTTING TOOL', 'DRILL BIT', 'END MILL', 'GRINDING WHEEL', 'SAW BLADE', 'INSERT', 'HOLDER'],
                'specifications': ['HSS', 'CARBIDE', 'TIN COATED', 'D6', 'D8', 'D10', 'D12', 'D16', 'D20'],
                'materials': ['HIGH SPEED STEEL', 'TUNGSTEN CARBIDE', 'CERAMIC', 'CBN', 'DIAMOND'],
                'types': ['TWIST DRILL', 'CORE DRILL', 'STEP DRILL', 'COUNTERSINK', 'REAMER'],
                'brands': ['SANDVIK', 'KENNAMETAL', 'ISCAR', 'MITSUBISHI', 'WALTER', 'WIDIA']
            },
            'Tools & General Machinery': {
                'components': ['WRENCH', 'SPANNER', 'SCREWDRIVER', 'PLIERS', 'HAMMER', 'CHISEL', 'FILE'],
                'specifications': ['8MM', '10MM', '12MM', '14MM', '17MM', '19MM', '22MM', '24MM'],
                'materials': ['CHROME VANADIUM', 'CARBON STEEL', 'ALLOY STEEL', 'HARDENED'],
                'types': ['COMBINATION', 'OPEN END', 'RING', 'ADJUSTABLE', 'PIPE', 'CHAIN'],
                'brands': ['STANLEY', 'TAPARIA', 'JHALANI', 'VENUS', 'FREEMAN', 'TRUMAX']
            },
            'Power Generation & Distribution Machinery & Accessories': {
                'components': ['GENERATOR', 'ALTERNATOR', 'ROTOR', 'STATOR', 'EXCITER', 'GOVERNOR', 'REGULATOR'],
                'specifications': ['1KVA', '5KVA', '10KVA', '25KVA', '50KVA', '100KVA', '415V', '11KV'],
                'materials': ['SILICON STEEL', 'COPPER WINDING', 'ALUMINUM WINDING'],
                'types': ['SYNCHRONOUS', 'ASYNCHRONOUS', 'BRUSHLESS', 'WOUND ROTOR'],
                'brands': ['CUMMINS', 'KIRLOSKAR', 'MAHINDRA', 'BAJAJ', 'ASHOK LEYLAND']
            },
            'Chemicals & Lubes': {
                'components': ['OIL', 'GREASE', 'LUBRICANT', 'FLUID', 'CLEANER', 'SOLVENT', 'ADDITIVE'],
                'specifications': ['ISO46', 'ISO68', 'ISO100', 'SAE20', 'SAE30', 'SAE40', 'EP2', 'EP3'],
                'materials': ['MINERAL', 'SYNTHETIC', 'SEMI-SYNTHETIC', 'BIODEGRADABLE'],
                'types': ['HYDRAULIC', 'ENGINE', 'GEAR', 'TRANSMISSION', 'CUTTING', 'COOLANT'],
                'brands': ['CASTROL', 'MOBIL', 'SHELL', 'BP', 'TOTAL', 'IDEMITSU', 'SERVO']
            },
            'Filtration': {
                'components': ['FILTER', 'CARTRIDGE', 'ELEMENT', 'HOUSING', 'STRAINER', 'SEPARATOR'],
                'specifications': ['5MICRON', '10MICRON', '25MICRON', '50MICRON', '100MICRON'],
                'materials': ['CELLULOSE', 'SYNTHETIC', 'GLASS FIBER', 'STAINLESS STEEL', 'NYLON'],
                'types': ['OIL', 'AIR', 'FUEL', 'HYDRAULIC', 'COOLANT', 'WATER', 'DUST'],
                'brands': ['MANN', 'MAHLE', 'FLEETGUARD', 'DONALDSON', 'PARKER', 'HYDAC']
            },
            'Pipes, Valves & Fittings': {
                'components': ['PIPE', 'VALVE', 'FITTING', 'ELBOW', 'TEE', 'REDUCER', 'FLANGE', 'COUPLING'],
                'specifications': ['1/2"', '3/4"', '1"', '1.5"', '2"', '3"', '4"', '6"', 'NB15', 'NB20', 'NB25'],
                'materials': ['STAINLESS STEEL', 'CARBON STEEL', 'BRASS', 'BRONZE', 'PVC', 'HDPE'],
                'types': ['BALL VALVE', 'GATE VALVE', 'GLOBE VALVE', 'CHECK VALVE', 'BUTTERFLY VALVE'],
                'brands': ['L&T', 'TOSHIBA', 'KIRLOSKAR', 'THERMAX', 'FORBES MARSHALL']
            },
            'Office Equipment, Furniture, & Supplies': {
                'components': ['DESK', 'CHAIR', 'CABINET', 'PRINTER', 'SCANNER', 'COMPUTER', 'MONITOR'],
                'specifications': ['A4', 'A3', 'LETTER', '80GSM', '70GSM', '24"', '27"', 'FULL HD'],
                'materials': ['WOOD', 'METAL', 'PLASTIC', 'FABRIC', 'LEATHER'],
                'types': ['EXECUTIVE', 'WORKSTATION', 'ERGONOMIC', 'ADJUSTABLE', 'STACKABLE'],
                'brands': ['HP', 'CANON', 'EPSON', 'DELL', 'LENOVO', 'GODREJ', 'STEELCASE']
            },
            'Material H&ling, Storage & Packaging': {
                'components': ['PALLET', 'CONTAINER', 'BOX', 'CRATE', 'RACK', 'SHELF', 'TROLLEY', 'CONVEYOR'],
                'specifications': ['1200X1000', '1200X800', '1000X800', '50KG', '100KG', '500KG', '1000KG'],
                'materials': ['WOOD', 'PLASTIC', 'METAL', 'CORRUGATED', 'ALUMINUM'],
                'types': ['EURO PALLET', 'ASIA PALLET', 'STACKABLE', 'NESTABLE', 'COLLAPSIBLE'],
                'brands': ['NILKAMAL', 'SUPREME', 'GODREJ', 'SCHAEFER', 'KRONES']
            }
        }
        
    def generate_item_description(self, l2_category, l3_category, l4_category):
        """Generate realistic item description based on category"""
        
        # Get industry terms for this L2 category
        terms = self.industry_keywords.get(l2_category, {
            'components': ['COMPONENT', 'PART', 'ITEM'],
            'specifications': ['STANDARD', 'PREMIUM'],
            'materials': ['STEEL', 'PLASTIC'],
            'types': ['TYPE A', 'TYPE B'],
            'brands': ['GENERIC', 'STANDARD']
        })
        
        # Generate description components
        components = []
        
        # Add technical specification (30% chance)
        if random.random() < 0.3:
            if terms['specifications']:
                components.append(random.choice(terms['specifications']))
        
        # Add material (25% chance)
        if random.random() < 0.25:
            if terms['materials']:
                components.append(random.choice(terms['materials']))
        
        # Add main component (always)
        if terms['components']:
            components.append(random.choice(terms['components']))
        
        # Add type/variant (40% chance)
        if random.random() < 0.4:
            if terms['types']:
                components.append(random.choice(terms['types']))
        
        # Add brand (20% chance)
        if random.random() < 0.2:
            if terms['brands']:
                components.append(f"MAKE: {random.choice(terms['brands'])}")
        
        # Add category-specific details from L3/L4
        if l3_category and l3_category != l2_category:
            # Extract meaningful words from L3 category
            l3_words = [word for word in l3_category.split() if len(word) > 3 and word not in ['&', 'and', 'the']]
            if l3_words and random.random() < 0.3:
                components.append(random.choice(l3_words).upper())
        
        if l4_category and l4_category != l3_category:
            # Extract meaningful words from L4 category
            l4_words = [word for word in l4_category.split() if len(word) > 3 and word not in ['&', 'and', 'the']]
            if l4_words and random.random() < 0.2:
                components.append(random.choice(l4_words).upper())
        
        # Add random technical details (10% chance)
        if random.random() < 0.1:
            tech_details = [
                f"SIZE: {random.choice(['SMALL', 'MEDIUM', 'LARGE'])}",
                f"COLOR: {random.choice(['BLACK', 'WHITE', 'BLUE', 'RED', 'YELLOW'])}",
                f"LENGTH: {random.randint(10, 500)}MM",
                f"DIA: {random.randint(5, 100)}MM",
                f"CAPACITY: {random.randint(1, 1000)}L",
                f"PRESSURE: {random.randint(1, 100)}BAR"
            ]
            components.append(random.choice(tech_details))
        
        # Combine components
        description = ' '.join(components)
        
        # Add item code prefix (15% chance)
        if random.random() < 0.15:
            prefix = f"{random.choice(['X', 'A', 'B', 'C'])}{random.randint(10, 99)}{random.choice(['EQ', 'NE', 'MP', 'ST'])}{random.randint(100000, 999999)}"
            description = f"{prefix} {description}"
        
        return description.strip()
    
    def generate_synthetic_data(self, min_records_per_category=200):
        """Generate synthetic data ensuring minimum records per category"""
        print(f"🏭 Generating synthetic data with min {min_records_per_category} records per category...")
        
        if self.taxonomy is None:
            print("❌ No taxonomy data loaded")
            return pd.DataFrame()
        
        synthetic_data = []
        
        # Get unique category combinations
        unique_combinations = self.taxonomy[['Category 2', 'Category 3', 'Category 4']].drop_duplicates()
        
        print(f"📊 Found {len(unique_combinations)} unique L2-L3-L4 combinations")
        
        # Generate data for each L2 category
        l2_categories = unique_combinations['Category 2'].unique()
        
        for l2_cat in l2_categories:
            print(f"\n🔧 Generating data for L2: {l2_cat}")
            
            # Get L3-L4 combinations for this L2
            l2_combinations = unique_combinations[unique_combinations['Category 2'] == l2_cat]
            
            # Ensure at least min_records_per_category for L2
            records_needed = max(min_records_per_category, len(l2_combinations) * 50)
            
            for i in range(records_needed):
                # Select random L3-L4 combination for this L2
                combo = l2_combinations.sample(1).iloc[0]
                l3_cat = combo['Category 3']
                l4_cat = combo['Category 4']
                
                # Generate item description
                description = self.generate_item_description(l2_cat, l3_cat, l4_cat)
                
                # Create record
                record = {
                    'Item Code': f"SYN{i+1:06d}_{l2_cat[:3].upper()}",
                    'Item_Descripton': description,
                    'Category L1': 'Mro Supplies',  # All are MRO supplies
                    'Category L2': l2_cat,
                    'Category L3': l3_cat,
                    'Category L4': l4_cat,
                    'Category L5': l4_cat  # Use L4 as L5 for simplicity
                }
                
                synthetic_data.append(record)
            
            print(f"   ✅ Generated {records_needed} records for {l2_cat}")
        
        # Convert to DataFrame
        synthetic_df = pd.DataFrame(synthetic_data)
        
        print(f"\n✅ Generated {len(synthetic_df)} total synthetic records")
        print(f"📊 L2 categories: {synthetic_df['Category L2'].nunique()}")
        print(f"📊 L3 categories: {synthetic_df['Category L3'].nunique()}")
        print(f"📊 L4 categories: {synthetic_df['Category L4'].nunique()}")
        
        # Show distribution
        print(f"\n📋 Records per L2 category:")
        l2_counts = synthetic_df['Category L2'].value_counts()
        for cat, count in l2_counts.items():
            print(f"   {cat}: {count} records")
        
        return synthetic_df
    
    def save_synthetic_data(self, synthetic_df, filename='synthetic_spend_data.xlsx'):
        """Save synthetic data to Excel file"""
        output_path = f'/Users/sujoymukherjee/code/spendplatform/classificationmodel-v0.5/syntheticdata/{filename}'
        
        print(f"\n💾 Saving synthetic data to {filename}...")
        
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # Save main data
            synthetic_df.to_excel(writer, sheet_name='Synthetic_Data', index=False)
            
            # Save summary statistics
            summary_data = {
                'Category': synthetic_df['Category L2'].value_counts().index.tolist(),
                'Record_Count': synthetic_df['Category L2'].value_counts().values.tolist()
            }
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            # Save sample data for each category
            for l2_cat in synthetic_df['Category L2'].unique()[:10]:  # Limit to first 10
                cat_data = synthetic_df[synthetic_df['Category L2'] == l2_cat].head(20)
                sheet_name = l2_cat[:31].replace('/', '_').replace('&', 'and')  # Excel sheet name limits
                cat_data.to_excel(writer, sheet_name=sheet_name, index=False)
        
        print(f"✅ Saved synthetic data with {len(synthetic_df)} records")
        print(f"📁 File: {output_path}")
        
        return output_path

def main():
    """Main function to generate synthetic data"""
    print("🚀 SYNTHETIC DATA GENERATION FOR SPEND CLASSIFICATION")
    print("=" * 70)
    
    generator = SyntheticDataGenerator()
    
    # Load data and analyze patterns
    generator.load_existing_data()
    generator.load_taxonomy()
    generator.analyze_patterns()
    generator.define_industry_terms()
    
    # Generate synthetic data
    synthetic_df = generator.generate_synthetic_data(min_records_per_category=200)
    
    # Save data
    output_file = generator.save_synthetic_data(synthetic_df)
    
    print(f"\n🎉 SYNTHETIC DATA GENERATION COMPLETE!")
    print(f"📁 Output file: {output_file}")
    print(f"📊 Total records: {len(synthetic_df)}")
    print(f"🏷️ Categories covered: L2({synthetic_df['Category L2'].nunique()}) | L3({synthetic_df['Category L3'].nunique()}) | L4({synthetic_df['Category L4'].nunique()})")

if __name__ == "__main__":
    main()
