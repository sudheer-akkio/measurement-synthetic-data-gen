import os
import pandas as pd
import numpy as np
import sys
from pathlib import Path
import json
import re
from typing import Dict, List, Any, Union
from datetime import datetime
from collections import Counter

project_root = Path(__file__).parent
sys.path.append(str(project_root))

sys.path.append(str(Path(__file__).parent / 'data-warehouse-sdk'))

from src.connectors.snowflake import SnowflakeConnector as snow # type: ignore
from src.utils import get_snowflake_connection # type: ignore

class DataStatisticsGenerator:
    """
    Generates comprehensive statistics for synthetic data generation.
    Handles data anonymization and intelligent statistical analysis.
    """
    
    def __init__(self, snowflake_obj, output_dir: str = "data"):
        self.sf_obj = snowflake_obj
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Anonymization mappings
        self.brand_mapping = {
            'bobs discount furniture': 'io_tech',
            'bob\'s discount furniture': 'io_tech',
            'bobs furniture': 'io_tech',
            'bob\'s furniture': 'io_tech',
            'bobs': 'io_tech',
            'bob': 'io_tech', 
            'bdf': 'io_tech',
            'discount furniture': 'furniture_retailer',
            'hmi': 'akkio',
            'horizon': 'akkio',
            'blu': 'akkio',
            'blushift': 'akkio'
        }
        
        # Generic partner/tactic replacements
        self.partner_tactics_mapping = {
            'vizio': 'partner_tv',
            'samsung': 'partner_tv',
            'roku': 'partner_streaming',
            'placeiq': 'location_provider',
            'taboola': 'content_network',
            'outbrain': 'content_network',
            'facebook': 'social_platform',
            'google': 'search_platform',
            'amazon': 'ecommerce_platform',
            'netflix': 'streaming_service',
            'hulu': 'streaming_service',
            'disney': 'media_company'
        }
    
    def anonymize_text(self, text: str) -> str:
        """Anonymize text by replacing brand names and partner/tactic names."""
        if not isinstance(text, str):
            return text
            
        # Sort brand mappings by length (longest first) to avoid partial replacements
        sorted_brands = sorted(self.brand_mapping.items(), key=lambda x: len(x[0]), reverse=True)
        
        # Replace brand names (case-insensitive)
        for old_brand, new_brand in sorted_brands:
            pattern = re.compile(re.escape(old_brand), re.IGNORECASE)
            text = pattern.sub(new_brand, text)
        
        # Replace partner/tactic names (case-insensitive)
        for old_name, new_name in self.partner_tactics_mapping.items():
            pattern = re.compile(re.escape(old_name), re.IGNORECASE)
            text = pattern.sub(new_name, text)
        
        return text
    
    def perturb_distribution_parameters(self, stats: Dict[str, Any], data_type: str) -> Dict[str, Any]:
        """
        *** DISTRIBUTION PERTURBATION METHOD ***
        Slightly modify statistical parameters to ensure synthetic data doesn't exactly 
        mimic the original client data distributions. This adds variation while maintaining
        realistic data characteristics.
        """
        perturbed_stats = stats.copy()
        
        if data_type in ['integer', 'float', 'currency', 'percentage', 'count']:
            # Add small random variation to numeric statistics (±5-15%)
            variation_factor = np.random.uniform(0.85, 1.15)  # ±15% variation
            small_variation = np.random.uniform(0.95, 1.05)   # ±5% variation
            
            if 'mean' in perturbed_stats:
                perturbed_stats['mean'] *= variation_factor
            if 'std' in perturbed_stats:
                perturbed_stats['std'] *= small_variation
            if 'median' in perturbed_stats:
                perturbed_stats['median'] *= variation_factor
                
            # Adjust min/max to maintain realistic bounds but add some variation
            if 'min' in perturbed_stats and 'max' in perturbed_stats:
                range_adjustment = (perturbed_stats['max'] - perturbed_stats['min']) * 0.1
                perturbed_stats['min'] -= range_adjustment * np.random.uniform(0, 0.5)
                perturbed_stats['max'] += range_adjustment * np.random.uniform(0, 0.5)
                
        elif data_type == 'categorical':
            # Slightly adjust categorical distributions
            if 'value_distribution' in perturbed_stats:
                total_count = sum(perturbed_stats['value_distribution'].values())
                adjusted_dist = {}
                
                for value, count in perturbed_stats['value_distribution'].items():
                    # Add ±10% variation to each category count
                    variation = np.random.uniform(0.9, 1.1)
                    adjusted_count = max(1, int(count * variation))
                    adjusted_dist[value] = adjusted_count
                
                perturbed_stats['value_distribution'] = adjusted_dist
                perturbed_stats['top_10_values'] = dict(list(adjusted_dist.items())[:10])
                
        elif data_type == 'datetime':
            # Slightly shift date ranges
            if 'min_date' in perturbed_stats and 'max_date' in perturbed_stats:
                if perturbed_stats['min_date'] and perturbed_stats['max_date']:
                    # Add ±30 days variation to date ranges
                    date_shift_days = np.random.randint(-30, 31)
                    try:
                        min_date = datetime.fromisoformat(perturbed_stats['min_date'])
                        max_date = datetime.fromisoformat(perturbed_stats['max_date'])
                        
                        min_date += pd.Timedelta(days=date_shift_days)
                        max_date += pd.Timedelta(days=date_shift_days)
                        
                        perturbed_stats['min_date'] = min_date.isoformat()
                        perturbed_stats['max_date'] = max_date.isoformat()
                    except:
                        pass  # Keep original dates if parsing fails
                        
        elif data_type == 'text':
            # Slightly adjust text length statistics
            if 'avg_length' in perturbed_stats:
                perturbed_stats['avg_length'] *= np.random.uniform(0.9, 1.1)
            if 'min_length' in perturbed_stats:
                perturbed_stats['min_length'] = max(0, int(perturbed_stats['min_length'] * np.random.uniform(0.8, 1.0)))
            if 'max_length' in perturbed_stats:
                perturbed_stats['max_length'] = int(perturbed_stats['max_length'] * np.random.uniform(1.0, 1.2))
                
        elif data_type == 'boolean':
            # Slightly adjust boolean distributions (±5% variation)
            if 'true_percentage' in perturbed_stats and 'false_percentage' in perturbed_stats:
                variation = np.random.uniform(0.95, 1.05)
                perturbed_stats['true_percentage'] *= variation
                perturbed_stats['false_percentage'] = 100 - perturbed_stats['true_percentage']
                
                # Ensure percentages are within valid bounds
                perturbed_stats['true_percentage'] = max(0, min(100, perturbed_stats['true_percentage']))
                perturbed_stats['false_percentage'] = 100 - perturbed_stats['true_percentage']
        
        # Add perturbation metadata
        perturbed_stats['_distribution_perturbed'] = True
        perturbed_stats['_perturbation_timestamp'] = datetime.now().isoformat()
        
        return perturbed_stats
    
    def get_column_data_type(self, series: pd.Series) -> str:
        """Determine data type using pandas dtypes and simple cardinality heuristics.

        Goal: Avoid relying on column-name patterns (which misclassify fields like
        R_VIDEOVIEWS). We infer types strictly from the underlying pandas dtype
        and basic characteristics of the values.
        """
        # Prefer explicit dtype checks
        if pd.api.types.is_bool_dtype(series):
            return 'boolean'
        if pd.api.types.is_datetime64_any_dtype(series):
            return 'datetime'
        if pd.api.types.is_integer_dtype(series):
            return 'integer'
        if pd.api.types.is_float_dtype(series) or pd.api.types.is_numeric_dtype(series):
            # Covers floats and numeric objects coerced as numeric
            return 'float'

        # If dtype is object, attempt to detect datetime-like strings
        # We sample to keep performance acceptable on very large columns
        if pd.api.types.is_object_dtype(series):
            try:
                non_null = series.dropna()
                if not non_null.empty:
                    sample = non_null.astype(str).sample(
                        n=min(1000, len(non_null)), random_state=0
                    )
                    parsed = pd.to_datetime(sample, errors='coerce')
                    parse_success_ratio = float(parsed.notna().mean())
                    # Basic sanity check on year range to avoid accidental parsing
                    valid_year_ratio = 1.0
                    try:
                        years = parsed.dt.year
                        valid_year_ratio = float(years.between(1900, 2100).mean())
                    except Exception:
                        pass
                    if parse_success_ratio >= 0.8 and valid_year_ratio >= 0.8:
                        return 'datetime'
            except Exception:
                # If parsing raises, fall back to categorical/text logic below
                pass

        # For object/string columns: decide between categorical vs text
        try:
            total = len(series)
            unique = series.nunique(dropna=True)
            cardinality_ratio = unique / total if total > 0 else 0
        except Exception:
            cardinality_ratio = 1.0

        # Treat low-to-moderate cardinality as categorical; otherwise text
        return 'categorical' if cardinality_ratio <= 0.3 else 'text'
    
    def calculate_numeric_stats(self, series: pd.Series) -> Dict[str, Any]:
        """Calculate comprehensive statistics for numeric columns."""
        stats = {
            'min': float(series.min()),
            'max': float(series.max()),
            'mean': float(series.mean()),
            'median': float(series.median()),
            'std': float(series.std()),
            'q25': float(series.quantile(0.25)),
            'q75': float(series.quantile(0.75)),
            'null_count': int(series.isnull().sum()),
            'null_percentage': float(series.isnull().sum() / len(series) * 100),
            'unique_count': int(series.nunique()),
            'zero_count': int((series == 0).sum()),
            'negative_count': int((series < 0).sum()),
            'outliers_iqr': self.detect_outliers_iqr(series)
        }
        
        # Add distribution information
        stats['distribution_type'] = self.detect_distribution_type(series)
        
        return stats
    
    def calculate_categorical_stats(self, series: pd.Series) -> Dict[str, Any]:
        """Calculate statistics for categorical columns."""
        value_counts = series.value_counts()
        
        # Anonymize categorical values
        anonymized_values = {}
        for value, count in value_counts.items():
            if pd.isna(value):
                anonymized_values['NULL'] = int(count)
            else:
                anonymized_value = self.anonymize_text(str(value))
                anonymized_values[anonymized_value] = int(count)
        
        stats = {
            'unique_count': int(series.nunique()),
            'null_count': int(series.isnull().sum()),
            'null_percentage': float(series.isnull().sum() / len(series) * 100),
            'mode': self.anonymize_text(str(series.mode().iloc[0])) if not series.mode().empty else None,
            'value_distribution': anonymized_values,
            'top_10_values': dict(list(anonymized_values.items())[:10]),
            'cardinality_ratio': float(series.nunique() / len(series))
        }
        
        return stats
    
    def calculate_datetime_stats(self, series: pd.Series) -> Dict[str, Any]:
        """Calculate statistics for datetime columns."""
        # Convert to datetime if not already
        if not pd.api.types.is_datetime64_any_dtype(series):
            series = pd.to_datetime(series, errors='coerce')
        
        stats = {
            'min_date': series.min().isoformat() if not pd.isna(series.min()) else None,
            'max_date': series.max().isoformat() if not pd.isna(series.max()) else None,
            'null_count': int(series.isnull().sum()),
            'null_percentage': float(series.isnull().sum() / len(series) * 100),
            'unique_count': int(series.nunique()),
            'date_range_days': (series.max() - series.min()).days if not (pd.isna(series.min()) or pd.isna(series.max())) else None
        }
        
        # Add temporal patterns
        if not series.empty:
            stats['year_distribution'] = series.dt.year.value_counts().to_dict()
            stats['month_distribution'] = series.dt.month.value_counts().to_dict()
            stats['day_of_week_distribution'] = series.dt.dayofweek.value_counts().to_dict()
            stats['hour_distribution'] = series.dt.hour.value_counts().to_dict() if series.dt.hour.nunique() > 1 else None
        
        return stats
    
    def calculate_text_stats(self, series: pd.Series) -> Dict[str, Any]:
        """Calculate statistics for text columns."""
        # Convert to string and handle nulls
        text_series = series.astype(str)
        non_null_series = series.dropna()
        
        stats = {
            'null_count': int(series.isnull().sum()),
            'null_percentage': float(series.isnull().sum() / len(series) * 100),
            'unique_count': int(series.nunique()),
            'avg_length': float(text_series.str.len().mean()),
            'min_length': int(text_series.str.len().min()),
            'max_length': int(text_series.str.len().max()),
            'length_std': float(text_series.str.len().std())
        }
        
        # Character distribution
        if not non_null_series.empty:
            all_chars = ''.join(non_null_series.astype(str))
            char_counts = Counter(all_chars)
            stats['common_characters'] = dict(char_counts.most_common(20))
            
            # Pattern analysis
            stats['contains_numbers'] = float(text_series.str.contains(r'\d', na=False).sum() / len(text_series))
            stats['contains_special_chars'] = float(text_series.str.contains(r'[^a-zA-Z0-9\s]', na=False).sum() / len(text_series))
            stats['all_caps_ratio'] = float(text_series.str.isupper().sum() / len(text_series))
        
        # Sample anonymized values
        sample_values = non_null_series.sample(min(10, len(non_null_series))).tolist() if not non_null_series.empty else []
        stats['sample_values'] = [self.anonymize_text(str(val)) for val in sample_values]
        
        return stats
    
    def detect_outliers_iqr(self, series: pd.Series) -> Dict[str, Any]:
        """Detect outliers using IQR method."""
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        outliers = series[(series < lower_bound) | (series > upper_bound)]
        
        return {
            'count': len(outliers),
            'percentage': float(len(outliers) / len(series) * 100),
            'lower_bound': float(lower_bound),
            'upper_bound': float(upper_bound)
        }
    
    def detect_distribution_type(self, series: pd.Series) -> str:
        """Detect the likely distribution type of numeric data."""
        from scipy import stats
        
        # Remove nulls for analysis
        clean_series = series.dropna()
        
        if len(clean_series) < 10:
            return 'insufficient_data'
        
        try:
            # Test for normal distribution
            _, p_normal = stats.normaltest(clean_series)
            
            # Test for uniform distribution
            _, p_uniform = stats.kstest(clean_series, 'uniform')
            
            # Simple heuristics
            if p_normal > 0.05:
                return 'normal'
            elif clean_series.skew() > 1:
                return 'right_skewed'
            elif clean_series.skew() < -1:
                return 'left_skewed'
            elif p_uniform > 0.05:
                return 'uniform'
            else:
                return 'other'
        except:
            return 'unknown'
    
    def generate_table_statistics(self, table_name: str, limit=None) -> Dict[str, Any]:
        """Generate comprehensive statistics for a single table."""
        print(f"Generating statistics for table: {table_name}")
        
        # Load data from Snowflake
        df = self.sf_obj.load_table(table_name, limit=limit)
        
        print(f"Loaded {len(df)} rows and {len(df.columns)} columns")
        
        table_stats = {
            'table_name': self.anonymize_text(table_name),
            'original_table_name': table_name,
            'row_count': len(df),
            'column_count': len(df.columns),
            'total_cells': len(df) * len(df.columns),
            'generation_timestamp': datetime.now().isoformat(),
            'perturbation_applied': True,
            'perturbation_note': "Statistical distributions have been modified to prevent exact replication",
            'columns': {}
        }
        
        # Process each column
        for col in df.columns:
            print(f"  Processing column: {col}")
            
            series = df[col]
            data_type = self.get_column_data_type(series)
            
            col_stats = {
                'original_name': col,  # Keep original column name
                'data_type': data_type,
                'pandas_dtype': str(series.dtype),
                'row_count': len(series)
            }
            
            # Calculate type-specific statistics
            if data_type in ['integer', 'float', 'currency', 'percentage', 'count']:
                raw_stats = self.calculate_numeric_stats(series)
                # *** APPLY DISTRIBUTION PERTURBATION ***
                col_stats.update(self.perturb_distribution_parameters(raw_stats, data_type))
            elif data_type == 'datetime':
                raw_stats = self.calculate_datetime_stats(series)
                # *** APPLY DISTRIBUTION PERTURBATION ***
                col_stats.update(self.perturb_distribution_parameters(raw_stats, data_type))
            elif data_type in ['categorical']:
                raw_stats = self.calculate_categorical_stats(series)
                # *** APPLY DISTRIBUTION PERTURBATION ***
                col_stats.update(self.perturb_distribution_parameters(raw_stats, data_type))
            elif data_type == 'boolean':
                true_count = int(series.sum()) if series.dtype == bool else int((series == True).sum())
                false_count = len(series) - true_count - int(series.isnull().sum())
                raw_stats = {
                    'true_count': true_count,
                    'false_count': false_count,
                    'null_count': int(series.isnull().sum()),
                    'true_percentage': float(true_count / len(series) * 100),
                    'false_percentage': float(false_count / len(series) * 100)
                }
                # *** APPLY DISTRIBUTION PERTURBATION ***
                col_stats.update(self.perturb_distribution_parameters(raw_stats, data_type))
            else:  # text and other types
                raw_stats = self.calculate_text_stats(series)
                # *** APPLY DISTRIBUTION PERTURBATION ***
                col_stats.update(self.perturb_distribution_parameters(raw_stats, data_type))
            
            table_stats['columns'][col] = col_stats
        
        # Detect and capture stable ID->NAME mappings so we can preserve them during generation
        try:
            id_name_mappings: List[Dict[str, Any]] = []
            upper_cols = {c.upper(): c for c in df.columns}
            for uc in list(upper_cols.keys()):
                if uc.endswith('_ID'):
                    name_uc = uc[:-3] + '_NAME'
                    if name_uc in upper_cols:
                        id_col = upper_cols[uc]
                        name_col = upper_cols[name_uc]
                        subset = df[[id_col, name_col]].dropna()
                        if not subset.empty:
                            # Build mapping using the most common name for each id (in case of noise)
                            grp = subset.groupby(id_col)[name_col].agg(lambda s: s.value_counts().idxmax())
                            mapping = {str(k): str(v) for k, v in grp.to_dict().items()}
                            if len(mapping) > 0:
                                id_name_mappings.append({
                                    'id_column': id_col,
                                    'name_column': name_col,
                                    'mapping': mapping
                                })
                                # Also capture value distribution for the id column to drive generation probabilities
                                try:
                                    vc = df[id_col].astype(str).value_counts()
                                    table_stats['columns'].setdefault(id_col, {})
                                    table_stats['columns'][id_col]['value_distribution'] = {str(k): int(v) for k, v in vc.to_dict().items()}
                                except Exception:
                                    pass
            if id_name_mappings:
                table_stats['id_name_mappings'] = id_name_mappings
        except Exception:
            # Non-fatal; continue without relationship metadata
            pass
        
        return table_stats
    
    def export_statistics(self, stats: Dict[str, Any], table_name: str):
        """Export statistics to CSV and JSON files."""
        anonymized_table_name = self.anonymize_text(table_name.lower())
        
        # Export detailed JSON statistics
        json_file = self.output_dir / f"{anonymized_table_name}_detailed_stats.json"
        with open(json_file, 'w') as f:
            json.dump(stats, f, indent=2, default=str)
        print(f"Exported detailed statistics to: {json_file}")
        
        # Export column summary CSV
        column_data = []
        for col_name, col_stats in stats['columns'].items():
            row = {
                'column_name': col_name,  # Use original column name
                'data_type': col_stats['data_type'],
                'pandas_dtype': col_stats['pandas_dtype'],
                'null_count': col_stats.get('null_count', 0),
                'null_percentage': col_stats.get('null_percentage', 0),
                'unique_count': col_stats.get('unique_count', 0),
                'distribution_perturbed': col_stats.get('_distribution_perturbed', False)
            }
            
            # Add type-specific key metrics
            if col_stats['data_type'] in ['integer', 'float', 'currency', 'percentage', 'count']:
                row.update({
                    'min_value': col_stats.get('min'),
                    'max_value': col_stats.get('max'),
                    'mean_value': col_stats.get('mean'),
                    'std_value': col_stats.get('std'),
                    'distribution_type': col_stats.get('distribution_type')
                })
            elif col_stats['data_type'] == 'categorical':
                row.update({
                    'mode_value': col_stats.get('mode'),
                    'cardinality_ratio': col_stats.get('cardinality_ratio')
                })
            elif col_stats['data_type'] == 'datetime':
                row.update({
                    'min_date': col_stats.get('min_date'),
                    'max_date': col_stats.get('max_date'),
                    'date_range_days': col_stats.get('date_range_days')
                })
            elif col_stats['data_type'] == 'text':
                row.update({
                    'avg_length': col_stats.get('avg_length'),
                    'min_length': col_stats.get('min_length'),
                    'max_length': col_stats.get('max_length')
                })
            
            column_data.append(row)
        
        # Export column summary
        csv_file = self.output_dir / f"{anonymized_table_name}_column_summary.csv"
        pd.DataFrame(column_data).to_csv(csv_file, index=False)
        print(f"Exported column summary to: {csv_file}")
        
        # Export generation rules for synthetic data
        rules_data = {
            'table_info': {
                'original_name': table_name,
                'anonymized_name': anonymized_table_name,
                'row_count': stats['row_count'],
                'generation_rules': {},
                'perturbation_applied': True,  # Flag indicating distributions were modified
                'perturbation_note': "Statistical parameters have been perturbed to avoid exact replication of source data"
            }
        }
        
        # Include ID->NAME mappings if discovered
        if 'id_name_mappings' in stats:
            rules_data['table_info']['id_name_mappings'] = stats['id_name_mappings']
        
        # Determine identifier columns from discovered mappings
        id_cols = set()
        if 'id_name_mappings' in stats:
            for m in stats['id_name_mappings']:
                id_cols.add(m.get('id_column'))

        for col_name, col_stats in stats['columns'].items():
            rules = {
                'type': col_stats['data_type'],
                'nullable': col_stats.get('null_percentage', 0) > 0,
                'null_probability': col_stats.get('null_percentage', 0) / 100,
                'distribution_perturbed': col_stats.get('_distribution_perturbed', False)
            }
            
            if col_stats['data_type'] in ['integer', 'float', 'currency', 'percentage', 'count']:
                rules.update({
                    'distribution': col_stats.get('distribution_type', 'normal'),
                    'min': col_stats.get('min'),
                    'max': col_stats.get('max'),
                    'mean': col_stats.get('mean'),
                    'std': col_stats.get('std')
                })
            elif col_stats['data_type'] == 'categorical':
                rules.update({
                    'categories': list(col_stats.get('value_distribution', {}).keys()),
                    'probabilities': list(col_stats.get('value_distribution', {}).values())
                })
            elif col_stats['data_type'] == 'datetime':
                rules.update({
                    'min_date': col_stats.get('min_date'),
                    'max_date': col_stats.get('max_date'),
                    'temporal_patterns': {
                        'year_dist': col_stats.get('year_distribution', {}),
                        'month_dist': col_stats.get('month_distribution', {}),
                        'day_of_week_dist': col_stats.get('day_of_week_distribution', {})
                    }
                })
            elif col_stats['data_type'] == 'text':
                rules.update({
                    'avg_length': col_stats.get('avg_length'),
                    'length_range': [col_stats.get('min_length'), col_stats.get('max_length')],
                    'sample_patterns': col_stats.get('sample_values', [])
                })

            # Override: if this column is an identifier (paired with *_NAME), export as identifier with categories
            if col_name in id_cols:
                rules['type'] = 'identifier'
                value_dist = stats['columns'].get(col_name, {}).get('value_distribution', {})
                if value_dist:
                    rules['categories'] = list(value_dist.keys())
                    rules['probabilities'] = list(value_dist.values())
                else:
                    # Fall back to mapping keys if distribution is unavailable
                    try:
                        mapping = next(m['mapping'] for m in stats['id_name_mappings'] if m['id_column'] == col_name)
                        rules['categories'] = list(mapping.keys())
                        rules['probabilities'] = [1 for _ in rules['categories']]
                    except Exception:
                        pass
            
            # Use original column name as key in generation rules
            rules_data['table_info']['generation_rules'][col_name] = rules
        
        # Export generation rules
        rules_file = self.output_dir / f"{anonymized_table_name}_generation_rules.json"
        with open(rules_file, 'w') as f:
            json.dump(rules_data, f, indent=2, default=str)
        print(f"Exported generation rules to: {rules_file}")


def main():
    """Main function to generate statistics for all tables."""
    
    # Database connection parameters
    database = "BLUSHIFT_HMI_STAGING"
    schema = "NEXT_BOBS_CLIENTDATA"
    
    # Initialize Snowflake connection
    print("Connecting to Snowflake...")
    sf_obj = get_snowflake_connection('horizon-staging', database=database, schema=schema)
    
    # Initialize statistics generator
    stats_generator = DataStatisticsGenerator(sf_obj, output_dir="data/stats")
    
    # Tables to process
    table_names = ["CAMPAIGN_DELIVERY", "CAMPAIGN_KPI", "CAMPAIGN_PACING"]
    # table_names = ["CAMPAIGN_DELIVERY"]

    limit = 5000000
    
    print(f"Processing {len(table_names)} tables...")
    
    # Process each table
    for table_name in table_names:
        try:
            print(f"\n{'='*50}")
            print(f"Processing table: {table_name}")
            print(f"{'='*50}")
            
            # Generate statistics
            table_stats = stats_generator.generate_table_statistics(table_name, limit=limit)
            
            # Export results
            stats_generator.export_statistics(table_stats, table_name)
            
            print(f"✓ Completed processing {table_name}")
            
        except Exception as e:
            print(f"✗ Error processing {table_name}: {str(e)}")
            continue
    
    print(f"\n{'='*50}")
    print("Statistics generation completed!")
    print(f"Output files saved to: {stats_generator.output_dir}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()

