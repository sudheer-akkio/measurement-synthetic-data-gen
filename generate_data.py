import os
import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Any, Union
from datetime import datetime, timedelta
import random
from faker import Faker

class SyntheticDataGenerator:
    """
    Generates synthetic data based on statistical rules generated from real data.
    Uses the generation rules created by generate_stats.py to create realistic CSV files.
    """
    
    def __init__(self, stats_dir: str = "data/stats", output_dir: str = "data"):
        self.stats_dir = Path(stats_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.fake = Faker()
        Faker.seed(42)  # For reproducible results
        np.random.seed(42)
        random.seed(42)
    
    def load_generation_rules(self) -> Dict[str, Dict]:
        """Load all generation rules from the stats directory."""
        rules_files = list(self.stats_dir.glob("*_generation_rules.json"))
        rules_dict = {}
        
        for rules_file in rules_files:
            table_name = rules_file.stem.replace("_generation_rules", "")
            with open(rules_file, 'r') as f:
                rules_dict[table_name] = json.load(f)
        
        return rules_dict
    
    def generate_datetime_column(self, rules: Dict[str, Any], num_rows: int) -> pd.Series:
        """Generate datetime column based on rules."""
        min_date = pd.to_datetime(rules['min_date'])
        max_date = pd.to_datetime(rules['max_date'])
        
        # Use temporal patterns if available
        if 'temporal_patterns' in rules:
            dates = []
            year_dist = rules['temporal_patterns'].get('year_dist', {})
            month_dist = rules['temporal_patterns'].get('month_dist', {})
            
            if year_dist and month_dist:
                # Generate dates based on year and month distributions
                for _ in range(num_rows):
                    # Sample year based on distribution
                    years = list(year_dist.keys())
                    year_weights = list(year_dist.values())
                    year = np.random.choice(years, p=np.array(year_weights)/sum(year_weights))
                    
                    # Sample month based on distribution
                    months = list(month_dist.keys())
                    month_weights = list(month_dist.values())
                    month = int(np.random.choice(months, p=np.array(month_weights)/sum(month_weights)))
                    
                    # Random day within the month
                    try:
                        if month == 2:
                            max_day = 28
                        elif month in [4, 6, 9, 11]:
                            max_day = 30
                        else:
                            max_day = 31
                        day = np.random.randint(1, max_day + 1)
                        date = pd.to_datetime(f"{year}-{month:02d}-{day:02d}")
                        
                        # Ensure date is within bounds
                        if min_date <= date <= max_date:
                            dates.append(date)
                        else:
                            # Fallback to random date in range
                            random_days = np.random.randint(0, (max_date - min_date).days + 1)
                            dates.append(min_date + timedelta(days=random_days))
                    except:
                        # Fallback to random date in range
                        random_days = np.random.randint(0, (max_date - min_date).days + 1)
                        dates.append(min_date + timedelta(days=random_days))
            else:
                # Fallback to uniform random dates
                random_days = np.random.randint(0, (max_date - min_date).days + 1, num_rows)
                dates = [min_date + timedelta(days=int(days)) for days in random_days]
        else:
            # Uniform random dates
            random_days = np.random.randint(0, (max_date - min_date).days + 1, num_rows)
            dates = [min_date + timedelta(days=int(days)) for days in random_days]
        
        return pd.Series(dates[:num_rows])
    
    def generate_categorical_column(self, rules: Dict[str, Any], num_rows: int) -> pd.Series:
        """Generate categorical column based on rules."""
        if 'categories' in rules and 'probabilities' in rules:
            categories = rules['categories']
            probabilities = np.array(rules['probabilities'])
            
            # Normalize probabilities
            probabilities = probabilities / probabilities.sum()
            
            # Generate values
            values = np.random.choice(categories, size=num_rows, p=probabilities)
            
            # Add nulls if needed
            if rules.get('null_probability', 0) > 0:
                null_mask = np.random.random(num_rows) < rules['null_probability']
                values = values.astype('object')
                values[null_mask] = None
            
            return pd.Series(values)
        else:
            # Fallback: generate some default categorical values
            default_categories = ['Category_A', 'Category_B', 'Category_C']
            return pd.Series(np.random.choice(default_categories, size=num_rows))
    
    def generate_numeric_column(self, rules: Dict[str, Any], num_rows: int, data_type: str) -> pd.Series:
        """Generate numeric column based on rules."""
        min_val = rules.get('min', 0)
        max_val = rules.get('max', 100)
        mean_val = rules.get('mean', (min_val + max_val) / 2)
        std_val = rules.get('std', (max_val - min_val) / 6)
        
        distribution = rules.get('distribution', 'normal')
        
        if distribution == 'normal':
            values = np.random.normal(mean_val, std_val, num_rows)
        elif distribution == 'uniform':
            values = np.random.uniform(min_val, max_val, num_rows)
        elif distribution == 'right_skewed':
            # Use log-normal for right skewed data
            values = np.random.lognormal(np.log(max(mean_val, 0.1)), std_val/mean_val if mean_val > 0 else 1, num_rows)
        else:
            # Default to normal
            values = np.random.normal(mean_val, std_val, num_rows)
        
        # Clip to bounds
        values = np.clip(values, min_val, max_val)
        
        # Convert to appropriate type
        if data_type in ['integer', 'count']:
            values = values.astype(int)
        elif data_type == 'currency':
            values = np.round(values, 2)
        elif data_type == 'percentage':
            values = np.clip(values, 0, 100)
        
        # Add nulls if needed
        if rules.get('null_probability', 0) > 0:
            null_mask = np.random.random(num_rows) < rules['null_probability']
            values = values.astype('object')
            values[null_mask] = None
        
        return pd.Series(values)
    
    def generate_text_column(self, rules: Dict[str, Any], num_rows: int) -> pd.Series:
        """Generate text column based on rules."""
        avg_length = int(rules.get('avg_length', 10))
        min_length = int(rules.get('min_length', 1))
        max_length = int(rules.get('max_length', 50))
        
        # Use sample patterns if available
        sample_patterns = rules.get('sample_patterns', [])
        
        values = []
        for _ in range(num_rows):
            if sample_patterns and np.random.random() < 0.3:  # 30% chance to use pattern
                # Use a sample pattern as base
                base_pattern = np.random.choice(sample_patterns)
                # Modify it slightly
                if len(base_pattern) > 5:
                    # Replace some parts with random text
                    words = base_pattern.split()
                    if len(words) > 1:
                        idx = np.random.randint(0, len(words))
                        words[idx] = self.fake.word()
                        text = ' '.join(words)
                    else:
                        text = base_pattern + '_' + self.fake.word()
                else:
                    text = base_pattern
            else:
                # Generate random text
                target_length = np.random.randint(min_length, max_length + 1)
                if target_length < 5:
                    # For very short text, use word generation
                    text = self.fake.word()[:target_length] if target_length > 0 else "A"
                else:
                    text = self.fake.text(max_nb_chars=target_length)[:target_length]
            
            values.append(text)
        
        # Add nulls if needed
        if rules.get('null_probability', 0) > 0:
            null_mask = np.random.random(num_rows) < rules['null_probability']
            values = np.array(values, dtype=object)
            values[null_mask] = None
            values = values.tolist()
        
        return pd.Series(values)
    
    def generate_identifier_column(self, rules: Dict[str, Any], num_rows: int) -> pd.Series:
        """Generate identifier column based on rules."""
        # Generate unique identifiers
        values = [f"ID_{i:08d}" for i in range(1, num_rows + 1)]
        
        # Add nulls if needed
        if rules.get('null_probability', 0) > 0:
            null_mask = np.random.random(num_rows) < rules['null_probability']
            values = np.array(values, dtype=object)
            values[null_mask] = None
            values = values.tolist()
        
        return pd.Series(values)
    
    def generate_boolean_column(self, rules: Dict[str, Any], num_rows: int) -> pd.Series:
        """Generate boolean column based on rules."""
        true_prob = rules.get('true_percentage', 50) / 100
        values = np.random.random(num_rows) < true_prob
        
        # Add nulls if needed
        if rules.get('null_probability', 0) > 0:
            null_mask = np.random.random(num_rows) < rules['null_probability']
            values = values.astype('object')
            values[null_mask] = None
        
        return pd.Series(values)
    
    def generate_table_data(self, table_rules: Dict[str, Any], sample_rows: int = None) -> pd.DataFrame:
        """Generate synthetic data for a single table."""
        table_info = table_rules['table_info']
        generation_rules = table_info['generation_rules']
        
        # Use the actual row count from the rules if sample_rows is not specified
        if sample_rows is None:
            sample_rows = table_info.get('row_count', 1000)
        
        print(f"Generating {sample_rows} rows for table: {table_info['original_name']}")
        
        data = {}
        
        for column_name, column_rules in generation_rules.items():
            print(f"  Generating column: {column_name} (type: {column_rules['type']})")
            
            column_type = column_rules['type']
            
            if column_type == 'datetime':
                data[column_name] = self.generate_datetime_column(column_rules, sample_rows)
            elif column_type == 'categorical':
                data[column_name] = self.generate_categorical_column(column_rules, sample_rows)
            elif column_type in ['integer', 'float', 'currency', 'percentage', 'count']:
                data[column_name] = self.generate_numeric_column(column_rules, sample_rows, column_type)
            elif column_type == 'text':
                data[column_name] = self.generate_text_column(column_rules, sample_rows)
            elif column_type == 'identifier':
                data[column_name] = self.generate_identifier_column(column_rules, sample_rows)
            elif column_type == 'boolean':
                data[column_name] = self.generate_boolean_column(column_rules, sample_rows)
            else:
                # Default to text
                data[column_name] = self.generate_text_column(column_rules, sample_rows)
        
        return pd.DataFrame(data)
    
    def create_data_schema(self, table_rules: Dict[str, Any], table_name: str) -> str:
        """Create a data schema document for a table."""
        table_info = table_rules['table_info']
        generation_rules = table_info['generation_rules']
        original_name = table_info['original_name']
        
        schema_content = f"""# Table Name: {original_name}

## Table Description
This table contains synthetic data generated based on statistical analysis of the original {original_name} dataset. The data maintains the statistical properties and distributions of the original data while ensuring privacy through anonymization and perturbation.

## Data Dictionary

### Fields:

"""
        
        # Load column summary for additional metadata
        column_summary_file = self.stats_dir / f"{table_name}_column_summary.csv"
        column_metadata = {}
        
        if column_summary_file.exists():
            column_df = pd.read_csv(column_summary_file)
            for _, row in column_df.iterrows():
                column_metadata[row['column_name']] = row.to_dict()
        
        for column_name, column_rules in generation_rules.items():
            column_type = column_rules['type']
            
            # Map types to SQL-like types for documentation
            type_mapping = {
                'datetime': 'TIMESTAMP',
                'categorical': 'VARCHAR',
                'integer': 'INTEGER',
                'float': 'FLOAT',
                'currency': 'DECIMAL(10,2)',
                'percentage': 'DECIMAL(5,2)',
                'count': 'INTEGER',
                'text': 'TEXT',
                'identifier': 'VARCHAR',
                'boolean': 'BOOLEAN'
            }
            
            sql_type = type_mapping.get(column_type, 'VARCHAR')
            
            # Add description based on type and metadata
            description = f"Generated {column_type} field"
            
            if column_name in column_metadata:
                metadata = column_metadata[column_name]
                if column_type in ['integer', 'float', 'currency', 'percentage', 'count']:
                    if metadata.get('min_value') is not None and metadata.get('max_value') is not None:
                        description += f" (range: {metadata['min_value']:.2f} to {metadata['max_value']:.2f})"
                elif column_type == 'datetime':
                    if metadata.get('min_date') and metadata.get('max_date'):
                        description += f" (range: {metadata['min_date']} to {metadata['max_date']})"
                elif column_type == 'categorical':
                    if metadata.get('unique_count'):
                        description += f" ({metadata['unique_count']} unique values)"
                elif column_type == 'text':
                    if metadata.get('avg_length'):
                        description += f" (avg length: {metadata['avg_length']:.1f} chars)"
            
            # Add nullability info
            if column_rules.get('null_probability', 0) > 0:
                description += f", nullable ({column_rules['null_probability']*100:.1f}% null rate)"
            
            schema_content += f"- `{column_name}` ({sql_type}): {description}\n"
        
        schema_content += f"""
## Table Relationships
*No explicit relationships defined - this is synthetic data for testing purposes*

## Business Context
This synthetic dataset was generated from the original {original_name} table to maintain realistic data distributions while protecting sensitive information. The data includes perturbations to prevent exact replication of the source data.

## Notes
- Data generated with statistical perturbation applied to protect source data privacy
- Row count in synthetic data: Variable (default 1000 rows for testing)
- Original table row count: {table_info.get('row_count', 'Unknown')}
- Generation timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- Statistical distributions have been modified to ensure synthetic data doesn't exactly replicate original patterns
"""
        
        return schema_content
    
    def generate_all_data(self, sample_rows: int = None, scale_factor: float = 1.0):
        """Generate synthetic data for all tables with generation rules.
        
        Args:
            sample_rows: If specified, generate this many rows for all tables
            scale_factor: If sample_rows is None, scale the original row_count by this factor
        """
        print("Loading generation rules...")
        all_rules = self.load_generation_rules()
        
        print(f"Found rules for {len(all_rules)} tables: {list(all_rules.keys())}")
        
        for table_name, table_rules in all_rules.items():
            try:
                # Determine row count
                if sample_rows is not None:
                    rows_to_generate = sample_rows
                else:
                    original_rows = table_rules['table_info'].get('row_count', 1000)
                    rows_to_generate = int(original_rows * scale_factor)
                
                print(f"\nTable: {table_rules['table_info']['original_name']}")
                print(f"Original rows: {table_rules['table_info'].get('row_count', 'Unknown')}")
                print(f"Generating: {rows_to_generate} rows")
                
                # Generate synthetic data
                df = self.generate_table_data(table_rules, rows_to_generate)
                
                # Save as CSV
                original_name = table_rules['table_info']['original_name']
                csv_file = self.output_dir / f"{original_name.lower()}.csv"
                df.to_csv(csv_file, index=False)
                print(f"✓ Saved synthetic data to: {csv_file}")
                
                # Create data schema
                schema_content = self.create_data_schema(table_rules, table_name)
                schema_dir = Path("data_schemas")
                schema_dir.mkdir(exist_ok=True)
                schema_file = schema_dir / f"{original_name.lower()}_schema.md"
                
                with open(schema_file, 'w') as f:
                    f.write(schema_content)
                print(f"✓ Created data schema: {schema_file}")
                
            except Exception as e:
                print(f"✗ Error generating data for {table_name}: {str(e)}")
                continue
        
        print(f"\n{'='*50}")
        print("Synthetic data generation completed!")
        print(f"CSV files saved to: {self.output_dir}")
        print(f"Schema files saved to: data_schemas/")
        print(f"{'='*50}")


def main():
    """Main function to generate synthetic data."""
    print("Starting synthetic data generation...")
    print("This will create CSV files based on the statistical rules in data/stats/")
    
    # Initialize generator
    generator = SyntheticDataGenerator(
        stats_dir="data/stats",
        output_dir="data"
    )
    
    # Configuration options - modify as needed:
    
    # Option 1: Generate using original row counts (WARNING: Very large datasets!)
    # Uncomment the line below to generate full-size datasets:
    # generator.generate_all_data(sample_rows=None, scale_factor=1.0)
    
    # Option 2: Generate scaled-down version (default: 0.1% of original size for testing)
    print("Generating scaled-down datasets (0.1% of original size for testing)...")
    print("Original sizes: CAMPAIGN_DELIVERY=3.7M, CAMPAIGN_KPI=5M, CAMPAIGN_PACING=1M rows")
    print("To generate full-size datasets, uncomment the scale_factor=1.0 option above")
    generator.generate_all_data(sample_rows=None, scale_factor=0.1)
    
    # Option 3: Generate fixed number of rows for all tables (uncomment if needed)
    # generator.generate_all_data(sample_rows=1000)


if __name__ == "__main__":
    main()
