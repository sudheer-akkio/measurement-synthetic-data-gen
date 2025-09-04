import os
import pandas as pd
import numpy as np
import json
import re
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
    
    def __init__(self, stats_dir: str = "data/stats", output_dir: str = "data", schema_dir: str = "data_schemas"):
        self.stats_dir = Path(stats_dir)
        self.output_dir = Path(output_dir)
        self.schema_dir = Path(schema_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.fake = Faker()
        Faker.seed(42)  # For reproducible results
        np.random.seed(42)
        random.seed(42)
        
        # Cache for schema-based column types
        self._schema_column_types = {}
    
    def load_generation_rules(self) -> Dict[str, Dict]:
        """Load all generation rules from the stats directory."""
        rules_files = list(self.stats_dir.glob("*_generation_rules.json"))
        rules_dict = {}
        
        for rules_file in rules_files:
            table_name = rules_file.stem.replace("_generation_rules", "")
            with open(rules_file, 'r') as f:
                rules_dict[table_name] = json.load(f)
        
        return rules_dict
    
    def _parse_schema_column_types(self, table_name: str) -> Dict[str, str]:
        """Parse schema file to extract column data types.
        
        Args:
            table_name: Name of the table (e.g., 'campaign_delivery')
            
        Returns:
            Dict mapping column names to their schema data types (INTEGER, FLOAT, VARCHAR, etc.)
        """
        if table_name in self._schema_column_types:
            return self._schema_column_types[table_name]
        
        # Look for schema file
        schema_file = self.schema_dir / f"{table_name}_schema.md"
        if not schema_file.exists():
            print(f"Warning: Schema file not found: {schema_file}")
            return {}
        
        column_types = {}
        
        try:
            with open(schema_file, 'r') as f:
                content = f.read()
            
            # Parse markdown lines that define columns
            # Pattern: - `COLUMN_NAME` (DATA_TYPE): Description
            pattern = r'^-\s*`([^`]+)`\s*\(([^)]+)\):'
            
            for line in content.split('\n'):
                match = re.match(pattern, line.strip())
                if match:
                    column_name = match.group(1).strip()
                    data_type = match.group(2).strip().upper()
                    
                    # Normalize data type names
                    if data_type in ['INTEGER', 'INT']:
                        normalized_type = 'INTEGER'
                    elif data_type in ['FLOAT', 'DOUBLE', 'DECIMAL', 'NUMERIC']:
                        normalized_type = 'FLOAT'
                    elif data_type in ['NUMBER']:
                        # NUMBER could be either integer or float - default to FLOAT for safety
                        normalized_type = 'FLOAT'
                    elif data_type in ['VARCHAR', 'TEXT', 'STRING']:
                        normalized_type = 'VARCHAR'
                    elif data_type in ['DATE', 'TIMESTAMP', 'DATETIME']:
                        normalized_type = 'DATE'
                    elif data_type in ['BOOLEAN', 'BOOL']:
                        normalized_type = 'BOOLEAN'
                    else:
                        normalized_type = data_type
                    
                    column_types[column_name] = normalized_type
            
            print(f"Parsed {len(column_types)} column types from {schema_file}")
            
        except Exception as e:
            print(f"Warning: Failed to parse schema file {schema_file}: {e}")
            return {}
        
        # Cache the results
        self._schema_column_types[table_name] = column_types
        return column_types
    
    def _get_schema_data_type(self, table_name: str, column_name: str) -> str:
        """Get the schema data type for a specific column.
        
        Returns:
            Schema data type (INTEGER, FLOAT, VARCHAR, etc.) or None if not found
        """
        column_types = self._parse_schema_column_types(table_name)
        return column_types.get(column_name)
    
    def _is_rate_column_by_schema(self, table_name: str, column_name: str) -> bool:
        """Determine if a column represents a rate/percentage based on schema and context.
        
        Only columns that are:
        1. FLOAT type in schema
        2. Have names indicating rates/percentages (like CTR, CVR, *_RATE, etc.)
        
        Should be treated as rate columns for Beta distribution sampling.
        R_* columns are NOT automatically rate columns - they need explicit rate naming.
        """
        schema_type = self._get_schema_data_type(table_name, column_name)
        if schema_type != 'FLOAT':
            return False
        
        name_upper = column_name.upper()
        
        # Only specific rate/percentage column patterns
        rate_patterns = [
            'CTR',  # Click-through rate
            'CVR',  # Conversion rate  
            'CONVERSION_RATE',
            'VTR',  # View-through rate
            'VIDEO_COMPLETION_RATE',
            'COMPLETION_RATE',
            'ENGAGEMENT_RATE'
        ]
        
        # Check for exact matches or *_RATE pattern (but not R_* prefix)
        if name_upper in rate_patterns:
            return True
        if name_upper.endswith('_RATE') and not name_upper.startswith('R_'):
            return True
            
        return False
    
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
        """Generate numeric column based on rules (supports 'integer' and 'float')."""
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
        if data_type == 'integer':
            values = values.astype(int)
        
        # Enforce non-negative values unless explicitly allowed by rules
        if not rules.get('allow_negative', False):
            values = np.maximum(values, 0)

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
        """Generate identifier column based on rules.
        
        Behavior:
        - If categories/probabilities are provided (from stats), sample from them to
          preserve real-world patterns and formatting. This avoids fabricating values
          like "ID_00000001" and keeps consistency with related NAME fields.
        - Otherwise, fall back to simple string identifiers that mirror source style if
          a sample pattern was provided, else a sequential numeric-like id.
        """
        # Prefer using categorical-style rules when available
        categories = rules.get('categories')
        probabilities = rules.get('probabilities')
        if categories and probabilities and len(categories) == len(probabilities):
            probs = np.array(probabilities, dtype=float)
            probs = probs / probs.sum() if probs.sum() > 0 else None
            if probs is not None:
                values = np.random.choice(categories, size=num_rows, p=probs)
            else:
                values = np.random.choice(categories, size=num_rows)
            values = values.astype(object)
        else:
            # Fallback: sequential ids without synthetic prefixing
            values = [str(i) for i in range(1, num_rows + 1)]
        
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
    
    def _is_constant_numeric_rules(self, column_rules: Dict[str, Any]) -> bool:
        """Check if numeric rules indicate a constant-valued column (min==max or std==0)."""
        try:
            has_min_max = 'min' in column_rules and 'max' in column_rules
            if has_min_max and column_rules['min'] == column_rules['max']:
                return True
            if 'std' in column_rules and (column_rules['std'] is None or float(column_rules['std']) == 0.0):
                return True
        except Exception:
            pass
        return False

    def _sample_rate_like_values(self, n: int, name_upper: str, const_value: float, null_probability: float) -> pd.Series:
        """Sample plausible marketing rates using Beta priors without relying on other columns.

        The Beta distribution is chosen due to [0,1] support. Parameters are selected
        using name-based heuristics with a concentration that yields light variability.
        If the column had a non-zero constant, center the prior near that value.
        """
        # Default targets by metric type
        defaults = {
            'CTR': (0.01, 200.0),                 # ~1%
            'CVR': (0.03, 150.0),                 # ~3%
            'CONVERSION_RATE': (0.03, 150.0),     # alias
            'VTR': (0.60, 40.0),                  # ~60%
            'VIDEO_COMPLETION_RATE': (0.60, 40.0)
        }

        # Generic defaults
        if name_upper.startswith('R_ENGAGEMENT'):
            target_mean, k = 0.05, 120.0         # ~5%
        elif name_upper.startswith('R_') or name_upper.endswith('_RATE'):
            target_mean, k = 0.02, 150.0         # ~2%
        else:
            target_mean, k = defaults.get(name_upper, (0.02, 150.0))

        # If the constant had a positive value, center near it (bounded away from 0/1)
        if const_value is not None:
            try:
                c = float(const_value)
                if np.isfinite(c) and 0.0 <= c <= 1.0:
                    # Blend toward default if c is extreme 0/1
                    eps_low, eps_high = 0.002, 0.98
                    base = np.clip(c, eps_low, eps_high)
                    # If c is 0, bump slightly; if 1, reduce slightly
                    if c <= eps_low:
                        base = max(target_mean, 0.005)
                    elif c >= eps_high:
                        base = min(target_mean if target_mean < 0.9 else 0.9, 0.98)
                    target_mean = base
            except Exception:
                pass

        alpha = max(1e-3, target_mean * k)
        beta = max(1e-3, (1.0 - target_mean) * k)

        samples = np.random.beta(alpha, beta, size=n)

        if null_probability and null_probability > 0:
            null_mask = np.random.random(n) < null_probability
            samples = samples.astype(object)
            samples[null_mask] = None
        return pd.Series(samples)

    def _postprocess_numeric_columns(self, df: pd.DataFrame, generation_rules: Dict[str, Any], table_name: str = None) -> pd.DataFrame:
        """Simplified post-processing: perturb constant numeric columns using realistic rules.

        - Detect constant via rules (min==max or std==0) or generated data (nunique<=1).
        - For actual rate columns (determined by schema + name patterns), sample from Beta distribution.
        - For other numerics, add appropriate noise while respecting bounds and types.
        - CRITICAL: Use schema data types to determine INTEGER vs FLOAT processing.
        """
        adjusted_df = df.copy()

        for column_name, column_rules in generation_rules.items():
            col_type = column_rules.get('type')
            if col_type not in ['integer', 'float']:
                continue

            series = adjusted_df[column_name]

            # Identify constants
            try:
                is_constant_rules = self._is_constant_numeric_rules(column_rules)
            except Exception:
                is_constant_rules = False

            is_constant_series = False
            try:
                is_constant_series = series.nunique(dropna=True) <= 1
            except Exception:
                pass

            if not (is_constant_rules or is_constant_series):
                continue

            # Use schema-based determination for rate columns
            is_rate_column = False
            if table_name:
                is_rate_column = self._is_rate_column_by_schema(table_name, column_name)

            null_prob = column_rules.get('null_probability', 0)
            non_null = series.dropna()
            const_val = float(non_null.iloc[0]) if len(non_null) > 0 else None

            # Apply rate processing only to actual rate columns (schema FLOAT + rate naming)
            if is_rate_column:
                adjusted = self._sample_rate_like_values(len(adjusted_df), column_name.upper(), const_val, null_prob)
            else:
                # Non-rate numeric: add small noise around constant while respecting bounds
                min_val = column_rules.get('min', None)
                max_val = column_rules.get('max', None)
                center = const_val if const_val is not None else column_rules.get('mean', 0)

                # For integer columns, use integer-based noise and operations
                if col_type == 'integer':
                    # Use integer center value
                    center = int(center) if center is not None else 0
                    
                    # Scale: small integer range for noise
                    if min_val is not None and max_val is not None and np.isfinite(min_val) and np.isfinite(max_val):
                        range_span = int(max_val) - int(min_val)
                        # For integers, use at least 1 for scale, but keep it reasonable
                        scale = max(int(range_span * 0.02), 1)
                    else:
                        scale = max(int(abs(center) * 0.05), 1) if center is not None else 1

                    # Generate integer noise using discrete distribution
                    # Use Poisson for non-negative integers or normal rounded to int
                    if center >= 0 and not column_rules.get('allow_negative', False):
                        # For non-negative integers, use Poisson-like distribution
                        noise = np.random.poisson(lam=scale, size=len(adjusted_df)) - scale
                    else:
                        # For potentially negative integers, use rounded normal
                        noise = np.round(np.random.normal(loc=0.0, scale=scale, size=len(adjusted_df))).astype(int)
                    
                    samples = center + noise

                    # Clip to bounds if provided - keep as integers
                    if min_val is not None:
                        samples = np.maximum(samples, int(min_val))
                    if max_val is not None:
                        samples = np.minimum(samples, int(max_val))

                    # Ensure we have integers
                    samples = samples.astype(int)
                    
                    # If still constant after operations, force minimal integer jitter within bounds
                    if len(pd.Series(samples).dropna().unique()) <= 1:
                        tweak = np.random.choice([-1, 0, 1], size=len(samples))
                        samples = samples + tweak
                        if min_val is not None:
                            samples = np.maximum(samples, int(min_val))
                        if max_val is not None:
                            samples = np.minimum(samples, int(max_val))
                        # Ensure still integers after tweaking
                        samples = samples.astype(int)
                
                else:  # col_type == 'float'
                    # Scale: small fraction of range or of center magnitude
                    if min_val is not None and max_val is not None and np.isfinite(min_val) and np.isfinite(max_val):
                        range_span = float(max_val) - float(min_val)
                        scale = max(range_span * 0.02, 1e-6)  # Small scale for floats
                    else:
                        scale = max(abs(center) * 0.05, 1e-6) if center is not None else 1e-6

                    noise = np.random.normal(loc=0.0, scale=scale, size=len(adjusted_df))
                    samples = (center if center is not None else 0.0) + noise

                    # Clip to bounds if provided
                    if min_val is not None:
                        samples = np.maximum(samples, float(min_val))
                    if max_val is not None:
                        samples = np.minimum(samples, float(max_val))

                    samples = samples.astype(float)

                # Preserve original nulls or apply null probability
                if series.isnull().any():
                    samples = pd.Series(samples, dtype=object)
                    samples[series.isnull().values] = None
                    adjusted = samples
                else:
                    if null_prob and null_prob > 0:
                        m = np.random.random(len(samples)) < null_prob
                        samples = samples.astype(object)
                        samples[m] = None
                    adjusted = pd.Series(samples)

            adjusted_df[column_name] = adjusted

            # Guarantee non-constant outcome; add tiny jitter if needed
            try:
                if adjusted_df[column_name].nunique(dropna=True) <= 1:
                    if is_rate_column:
                        # Only apply rate-like processing to actual rate columns
                        base = adjusted_df[column_name].fillna(const_val if const_val is not None else 0.0).astype(float).values
                        tiny = np.random.beta(2, 200, size=len(adjusted_df)) * 0.01
                        adjusted_df[column_name] = base + tiny
                    elif col_type == 'integer':
                        # For integer columns, add small integer jitter
                        base_val = int(const_val) if const_val is not None else 0
                        base = adjusted_df[column_name].fillna(base_val).astype(int).values
                        tiny = np.random.choice([-1, 0, 1], size=len(adjusted_df))
                        adjusted_df[column_name] = (base + tiny).astype(int)
                    else:
                        # For float columns that aren't rate-like
                        base = adjusted_df[column_name].fillna(const_val if const_val is not None else 0.0).astype(float).values
                        tiny = np.random.normal(0.0, 1e-6, size=len(adjusted_df))
                        adjusted_df[column_name] = base + tiny
            except Exception:
                pass

            # Final enforcement: prevent negative values unless explicitly allowed
            try:
                if not column_rules.get('allow_negative', False):
                    col = adjusted_df[column_name]
                    if col_type == 'integer':
                        # For integer columns, ensure we maintain integer type
                        if col.dtype == 'O':  # object type (with nulls)
                            mask = col.notnull()
                            adjusted_df.loc[mask, column_name] = pd.to_numeric(col[mask], errors='coerce').clip(lower=0).astype(int)
                        else:
                            adjusted_df[column_name] = col.clip(lower=0).astype(int)
                    else:  # float column
                        if col.dtype == 'O':
                            mask = col.notnull()
                            adjusted_df.loc[mask, column_name] = pd.to_numeric(col[mask], errors='coerce').clip(lower=0)
                        else:
                            adjusted_df[column_name] = col.clip(lower=0)
            except Exception:
                pass

        return adjusted_df

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
            elif column_type in ['integer', 'float']:
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
        
        df = pd.DataFrame(data)

        # Simplified numeric post-processing: perturb constant numeric columns
        try:
            # Extract table name from table_info for schema lookups
            original_name = table_info.get('original_name', '')
            table_name = original_name.lower() if original_name else None
            df = self._postprocess_numeric_columns(df, generation_rules, table_name)
        except Exception as e:
            print(f"  Warning: numeric post-processing failed: {e}")
        
        return df
    
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
                'text': 'TEXT',
                'identifier': 'VARCHAR',
                'boolean': 'BOOLEAN'
            }
            
            sql_type = type_mapping.get(column_type, 'VARCHAR')
            
            # Add description based on type and metadata
            description = f"Generated {column_type} field"
            
            if column_name in column_metadata:
                metadata = column_metadata[column_name]
                if column_type in ['integer', 'float']:
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
    
    def generate_all_data(self, sample_rows: int = None, scale_factor: float = 1.0, create_schema: bool = False):
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
                
                # If id->name mappings exist, ensure consistency across paired columns
                try:
                    mappings = table_rules['table_info'].get('id_name_mappings', [])
                    for m in mappings:
                        id_col = m.get('id_column')
                        name_col = m.get('name_column')
                        mapping = m.get('mapping', {})
                        if id_col in df.columns and name_col in df.columns and mapping:
                            # Regenerate the name column based on the id values for row-wise consistency
                            df[name_col] = df[id_col].map(lambda x: mapping.get(str(x), x))
                except Exception:
                    pass
                
                # Save as CSV
                original_name = table_rules['table_info']['original_name']
                csv_file = self.output_dir / f"{original_name.lower()}.csv"
                df.to_csv(csv_file, index=False)
                print(f"✓ Saved synthetic data to: {csv_file}")
                
                # Create data schema
                if create_schema:
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
    generator.generate_all_data(sample_rows=None, scale_factor=0.1, create_schema=False)
    
    # Option 3: Generate fixed number of rows for all tables (uncomment if needed)
    # generator.generate_all_data(sample_rows=1000)


if __name__ == "__main__":
    main()
