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

# Columns managed by correlated metric generation (skipped by independent generation)
CORRELATED_METRIC_COLUMNS = {
    'R_CLICKS', 'R_MEDIACOST', 'R_ENGAGEMENTS',
    'R_VIDEOSTARTS', 'R_VIDEO25', 'R_VIDEO50', 'R_VIDEO75',
    'R_VIDEOCOMPLETES', 'R_VIEWS',
    'KPI_N', 'KPI_D',
    'DAILY_PLANNED_SPEND',
}

AUDIENCE_MAP = {
    'AUD_001': 'Tech Enthusiasts - Early Adopters',
    'AUD_002': 'Smart Home Adopters',
    'AUD_003': 'Young Professionals 25-34',
    'AUD_004': 'Eco-Conscious Consumers',
    'AUD_005': 'Mobile Accessory In-Market Shoppers',
    'AUD_006': 'Device Upgraders - Active Researchers',
    'AUD_007': 'Business Tech Decision Makers',
    'AUD_008': 'Connected Home - Parents with Kids',
    'AUD_009': 'Site Visitors - Retargeting',
    'AUD_010': 'Cart Abandoners - Retargeting',
    'AUD_011': 'Lookalike - High-Value Customers',
    'AUD_012': 'Past Purchasers - Cross-Sell',
    'AUD_013': 'Competitor Brand Conquesting',
    'AUD_014': 'Frequent Online Shoppers',
    'AUD_015': 'High-Income HH 100K+',
    'AUD_016': 'College Students and Gen Z',
    'AUD_017': 'Outdoor and Active Lifestyle',
    'AUD_018': 'CTV Cord Cutters',
    'AUD_019': 'Gaming and Entertainment Enthusiasts',
    'AUD_020': 'Auto Enthusiasts - In-Vehicle Tech',
}

AUDIENCE_IDS = list(AUDIENCE_MAP.keys())

# Audience pools keyed by campaign context for correlated assignment
_AUD_RETARGETING = ['AUD_009', 'AUD_010', 'AUD_012']
_AUD_UPPER_FUNNEL = ['AUD_001', 'AUD_002', 'AUD_003', 'AUD_004', 'AUD_015', 'AUD_018']
_AUD_MID_FUNNEL = ['AUD_005', 'AUD_006', 'AUD_007', 'AUD_008', 'AUD_016']
_AUD_LOWER_FUNNEL = ['AUD_005', 'AUD_006', 'AUD_011', 'AUD_013', 'AUD_014']
_AUD_AUDIO = ['AUD_001', 'AUD_017', 'AUD_019', 'AUD_003']
_AUD_SOCIAL = ['AUD_003', 'AUD_016', 'AUD_014', 'AUD_019']
_AUD_VIDEO_CTV = ['AUD_018', 'AUD_019', 'AUD_001', 'AUD_015']

RETARGETING_TACTICS = {
    'display retargeting', 'video retargeting', 'remarketing', 'native retargeting',
}

# State-to-region mapping (US Census regions, aligned with CHANNEL_EXECUTION_NAME suffixes)
STATE_TO_REGION = {
    'CT': 'NORTHEAST', 'DC': 'NORTHEAST', 'DE': 'NORTHEAST', 'MA': 'NORTHEAST',
    'MD': 'NORTHEAST', 'ME': 'NORTHEAST', 'NH': 'NORTHEAST', 'NJ': 'NORTHEAST',
    'NY': 'NORTHEAST', 'PA': 'NORTHEAST', 'RI': 'NORTHEAST', 'VT': 'NORTHEAST',
    'AL': 'SOUTHEAST', 'AR': 'SOUTHEAST', 'FL': 'SOUTHEAST', 'GA': 'SOUTHEAST',
    'KY': 'SOUTHEAST', 'LA': 'SOUTHEAST', 'MS': 'SOUTHEAST', 'NC': 'SOUTHEAST',
    'SC': 'SOUTHEAST', 'TN': 'SOUTHEAST', 'VA': 'SOUTHEAST', 'WV': 'SOUTHEAST',
    'IA': 'MIDWEST', 'IL': 'MIDWEST', 'IN': 'MIDWEST', 'KS': 'MIDWEST',
    'MI': 'MIDWEST', 'MN': 'MIDWEST', 'MO': 'MIDWEST', 'ND': 'MIDWEST',
    'NE': 'MIDWEST', 'OH': 'MIDWEST', 'SD': 'MIDWEST', 'WI': 'MIDWEST',
    'AZ': 'SOUTHWEST', 'NM': 'SOUTHWEST', 'OK': 'SOUTHWEST', 'TX': 'SOUTHWEST',
    'AK': 'WEST', 'CA': 'WEST', 'CO': 'WEST', 'HI': 'WEST', 'ID': 'WEST',
    'MT': 'WEST', 'NV': 'WEST', 'OR': 'WEST', 'UT': 'WEST', 'WA': 'WEST',
    'WY': 'WEST',
}

# Maps CHANNEL_EXECUTION_NAME region suffix to the region values used in our column
_EXEC_NAME_REGION_MAP = {
    'Northeast': 'NORTHEAST',
    'Southeast': 'SOUTHEAST',
    'Midwest': 'MIDWEST',
    'Southwest': 'SOUTHWEST',
}


def _build_dma_centroid_lookup(location_csv: str = "data/location_data.csv") -> Dict[str, List[Dict]]:
    """Build a region-keyed lookup of DMA centroids from the zip-level location CSV.

    Returns a dict: { 'NORTHEAST': [ {dma_code, dma_name, state, region, lat, lon}, ... ], ... }
    """
    df = pd.read_csv(location_csv)

    # Parse lat/lon from the JSON LOCATION column
    lats, lons = [], []
    for loc_json in df['LOCATION']:
        parsed = json.loads(loc_json)
        lats.append(float(parsed['latitude']))
        lons.append(float(parsed['longitude']))
    df['_lat'] = lats
    df['_lon'] = lons

    # Compute centroids per DMA: avg lat/lon, mode state
    centroids = df.groupby('DMA_CODE').agg(
        dma_name=('DMA_NAME', 'first'),
        state=('STATE', lambda x: x.mode().iloc[0]),
        latitude=('_lat', 'mean'),
        longitude=('_lon', 'mean'),
    ).reset_index()

    centroids['region'] = centroids['state'].map(STATE_TO_REGION).fillna('WEST')
    centroids['latitude'] = centroids['latitude'].round(6)
    centroids['longitude'] = centroids['longitude'].round(6)

    region_lookup: Dict[str, List[Dict]] = {}
    for _, row in centroids.iterrows():
        entry = {
            'dma_code': str(int(row['DMA_CODE'])),
            'dma_name': row['dma_name'],
            'state': row['state'],
            'region': row['region'],
            'latitude': row['latitude'],
            'longitude': row['longitude'],
        }
        region_lookup.setdefault(row['region'], []).append(entry)

    return region_lookup


class SyntheticDataGenerator:
    """
    Generates synthetic data based on statistical rules generated from real data.
    Uses the generation rules created by generate_stats.py to create realistic CSV files.
    """

    SCHEMA_DIR = Path("data_schemas")

    def __init__(self, stats_dir: str = "data/stats", output_dir: str = "data"):
        self.stats_dir = Path(stats_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.fake = Faker()
        Faker.seed(42)
        np.random.seed(42)
        random.seed(42)
        self._schema_column_types = {}

        location_csv = Path(output_dir) / "location_data.csv"
        if location_csv.exists():
            self._dma_by_region = _build_dma_centroid_lookup(str(location_csv))
            self._all_dmas = [d for dmas in self._dma_by_region.values() for d in dmas]
            print(f"Loaded DMA centroid lookup: {len(self._all_dmas)} DMAs across {len(self._dma_by_region)} regions")
        else:
            self._dma_by_region = {}
            self._all_dmas = []
            print(f"Warning: location_data.csv not found at {location_csv.resolve()}, skipping location assignment")

    def load_generation_rules(self) -> Dict[str, Dict]:
        """Load all generation rules from the stats directory."""
        rules_files = list(self.stats_dir.glob("*_generation_rules.json"))
        rules_dict = {}
        for rules_file in rules_files:
            table_name = rules_file.stem.replace("_generation_rules", "")
            with open(rules_file, 'r') as f:
                rules_dict[table_name] = json.load(f)
        return rules_dict

    # ------------------------------------------------------------------
    # Schema helpers (used for type inference in post-processing)
    # ------------------------------------------------------------------

    def _parse_schema_column_types(self, table_name: str) -> Dict[str, str]:
        """Parse schema .md file to extract column data types."""
        if table_name in self._schema_column_types:
            return self._schema_column_types[table_name]

        schema_file = self.SCHEMA_DIR / f"{table_name}_schema.md"
        if not schema_file.exists():
            return {}

        column_types = {}
        try:
            with open(schema_file, 'r') as f:
                content = f.read()
            pattern = r'^-\s*`([^`]+)`\s*\(([^)]+)\):'
            for line in content.split('\n'):
                match = re.match(pattern, line.strip())
                if match:
                    col = match.group(1).strip()
                    dt = match.group(2).strip().upper()
                    if dt in ('INTEGER', 'INT'):
                        normalized = 'INTEGER'
                    elif dt in ('FLOAT', 'DOUBLE', 'DECIMAL', 'NUMERIC'):
                        normalized = 'FLOAT'
                    elif dt == 'NUMBER':
                        normalized = 'FLOAT'
                    elif dt in ('VARCHAR', 'TEXT', 'STRING'):
                        normalized = 'VARCHAR'
                    elif dt in ('DATE', 'TIMESTAMP', 'DATETIME'):
                        normalized = 'DATE'
                    elif dt in ('BOOLEAN', 'BOOL'):
                        normalized = 'BOOLEAN'
                    else:
                        normalized = dt
                    column_types[col] = normalized
        except Exception:
            return {}

        self._schema_column_types[table_name] = column_types
        return column_types

    def _get_schema_data_type(self, table_name: str, column_name: str) -> str:
        return self._parse_schema_column_types(table_name).get(column_name)

    # ------------------------------------------------------------------
    # Column generators
    # ------------------------------------------------------------------

    def generate_datetime_column(self, rules: Dict[str, Any], num_rows: int) -> pd.Series:
        """Generate datetime column based on rules."""
        min_date = pd.to_datetime(rules['min_date'])
        max_date = pd.to_datetime(rules['max_date'])

        if 'temporal_patterns' in rules:
            year_dist = rules['temporal_patterns'].get('year_dist', {})
            month_dist = rules['temporal_patterns'].get('month_dist', {})

            if year_dist and month_dist:
                dates = []
                years = list(year_dist.keys())
                year_weights = np.array(list(year_dist.values()), dtype=float)
                year_weights /= year_weights.sum()
                months = list(month_dist.keys())
                month_weights = np.array(list(month_dist.values()), dtype=float)
                month_weights /= month_weights.sum()

                sampled_years = np.random.choice(years, size=num_rows, p=year_weights)
                sampled_months = np.random.choice(months, size=num_rows, p=month_weights).astype(int)

                for yr, mo in zip(sampled_years, sampled_months):
                    try:
                        if mo == 2:
                            max_day = 28
                        elif mo in (4, 6, 9, 11):
                            max_day = 30
                        else:
                            max_day = 31
                        day = np.random.randint(1, max_day + 1)
                        date = pd.to_datetime(f"{yr}-{mo:02d}-{day:02d}")
                        if min_date <= date <= max_date:
                            dates.append(date)
                        else:
                            rd = np.random.randint(0, (max_date - min_date).days + 1)
                            dates.append(min_date + timedelta(days=rd))
                    except Exception:
                        rd = np.random.randint(0, (max_date - min_date).days + 1)
                        dates.append(min_date + timedelta(days=rd))
                return pd.Series(dates[:num_rows])

        random_days = np.random.randint(0, (max_date - min_date).days + 1, num_rows)
        return pd.Series([min_date + timedelta(days=int(d)) for d in random_days])

    def generate_categorical_column(self, rules: Dict[str, Any], num_rows: int) -> pd.Series:
        """Generate categorical column based on rules."""
        if 'categories' in rules and 'probabilities' in rules:
            categories = rules['categories']
            probs = np.array(rules['probabilities'], dtype=float)
            probs /= probs.sum()
            values = np.random.choice(categories, size=num_rows, p=probs)
            if rules.get('null_probability', 0) > 0:
                mask = np.random.random(num_rows) < rules['null_probability']
                values = values.astype('object')
                values[mask] = None
            return pd.Series(values)
        return pd.Series(np.random.choice(['Category_A', 'Category_B', 'Category_C'], size=num_rows))

    def generate_numeric_column(self, rules: Dict[str, Any], num_rows: int, data_type: str) -> pd.Series:
        """Generate numeric column based on rules (supports 'integer' and 'float')."""
        min_val = rules.get('min', 0)
        max_val = rules.get('max', 100)
        mean_val = rules.get('mean', (min_val + max_val) / 2)
        std_val = rules.get('std', (max_val - min_val) / 6)

        distribution = rules.get('distribution', 'normal')

        if distribution == 'normal':
            values = np.random.normal(mean_val, max(std_val, 1e-9), num_rows)
        elif distribution == 'uniform':
            values = np.random.uniform(min_val, max_val, num_rows)
        elif distribution == 'right_skewed':
            safe_mean = max(mean_val, 0.1)
            safe_std = max(std_val, 1e-9)
            variance = safe_std ** 2
            sigma_sq = np.log(1 + variance / max(safe_mean ** 2, 1e-10))
            sigma = min(np.sqrt(sigma_sq), 2.0)
            mu = np.log(safe_mean) - sigma_sq / 2
            values = np.random.lognormal(mu, sigma, num_rows)
        else:
            values = np.random.normal(mean_val, max(std_val, 1e-9), num_rows)

        values = np.clip(values, min_val, max_val)

        if data_type == 'integer':
            values = np.round(values).astype(int)

        if not rules.get('allow_negative', False):
            values = np.maximum(values, 0)

        if rules.get('null_probability', 0) > 0:
            mask = np.random.random(num_rows) < rules['null_probability']
            values = values.astype('object')
            values[mask] = None

        return pd.Series(values)

    def generate_text_column(self, rules: Dict[str, Any], num_rows: int) -> pd.Series:
        """Generate text column based on rules."""
        min_length = int(rules.get('min_length', 1))
        max_length = int(rules.get('max_length', 50))
        sample_patterns = rules.get('sample_patterns', [])

        values = []
        for _ in range(num_rows):
            if sample_patterns and np.random.random() < 0.3:
                base = np.random.choice(sample_patterns)
                if len(base) > 5:
                    words = base.split()
                    if len(words) > 1:
                        idx = np.random.randint(0, len(words))
                        words[idx] = self.fake.word()
                        text = ' '.join(words)
                    else:
                        text = base + '_' + self.fake.word()
                else:
                    text = base
            else:
                target = np.random.randint(min_length, max_length + 1)
                text = self.fake.word()[:target] if target < 5 else self.fake.text(max_nb_chars=target)[:target]
            values.append(text)

        if rules.get('null_probability', 0) > 0:
            mask = np.random.random(num_rows) < rules['null_probability']
            values = np.array(values, dtype=object)
            values[mask] = None
            values = values.tolist()

        return pd.Series(values)

    def generate_identifier_column(self, rules: Dict[str, Any], num_rows: int) -> pd.Series:
        """Generate identifier column based on rules."""
        categories = rules.get('categories')
        probabilities = rules.get('probabilities')
        if categories and probabilities and len(categories) == len(probabilities):
            probs = np.array(probabilities, dtype=float)
            probs = probs / probs.sum() if probs.sum() > 0 else None
            values = np.random.choice(categories, size=num_rows, p=probs) if probs is not None else np.random.choice(categories, size=num_rows)
            values = values.astype(object)
        else:
            values = [str(i) for i in range(1, num_rows + 1)]

        if rules.get('null_probability', 0) > 0:
            mask = np.random.random(num_rows) < rules['null_probability']
            values = np.array(values, dtype=object)
            values[mask] = None
            values = values.tolist()

        return pd.Series(values)

    def generate_boolean_column(self, rules: Dict[str, Any], num_rows: int) -> pd.Series:
        """Generate boolean column based on rules."""
        true_prob = rules.get('true_percentage', 50) / 100
        values = np.random.random(num_rows) < true_prob
        if rules.get('null_probability', 0) > 0:
            mask = np.random.random(num_rows) < rules['null_probability']
            values = values.astype('object')
            values[mask] = None
        return pd.Series(values)

    # ------------------------------------------------------------------
    # Correlated metric generation
    # ------------------------------------------------------------------

    def _generate_correlated_metrics(self, df: pd.DataFrame, table_name: str) -> pd.DataFrame:
        """Override independently-generated metric columns with correlated values.

        Uses R_IMPRESSIONS as the anchor metric and derives all others from it
        with realistic ratios so CTR, CPC, CPM, ROAS etc. come out sane.
        """
        n = len(df)
        tn = table_name.upper()

        if tn in ('CAMPAIGN_DELIVERY', 'CAMPAIGN_KPI'):
            df = self._correlate_delivery_metrics(df, n)

        if tn == 'CAMPAIGN_KPI':
            df = self._correlate_kpi_metrics(df, n)

        if tn == 'CAMPAIGN_PACING':
            df = self._correlate_pacing_metrics(df, n)

        return df

    def _correlate_delivery_metrics(self, df: pd.DataFrame, n: int) -> pd.DataFrame:
        """Correlate R_IMPRESSIONS -> R_CLICKS, R_MEDIACOST, video funnel, etc."""
        if 'R_IMPRESSIONS' not in df.columns:
            return df

        impressions = pd.to_numeric(df['R_IMPRESSIONS'], errors='coerce').fillna(0).values.astype(float)
        impressions = np.maximum(impressions, 1.0)

        if 'R_CLICKS' in df.columns:
            ctr = np.random.beta(2, 200, size=n)
            df['R_CLICKS'] = np.round(impressions * ctr).astype(int)

        if 'R_MEDIACOST' in df.columns:
            cpm = np.random.lognormal(mean=2.3, sigma=0.5, size=n)
            cpm = np.clip(cpm, 2.0, 60.0)
            df['R_MEDIACOST'] = np.round(impressions * cpm / 1000, 2)

        if 'R_ENGAGEMENTS' in df.columns:
            eng_rate = np.random.beta(2, 60, size=n)
            df['R_ENGAGEMENTS'] = np.round(impressions * eng_rate).astype(int)

        if 'R_VIDEOSTARTS' in df.columns:
            vid_rate = np.random.beta(5, 10, size=n)
            video_starts = np.round(impressions * vid_rate).astype(int)
            df['R_VIDEOSTARTS'] = video_starts
            starts_f = video_starts.astype(float)

            if 'R_VIDEO25' in df.columns:
                r25 = np.random.beta(40, 6, size=n)
                v25 = np.round(starts_f * r25).astype(int)
                df['R_VIDEO25'] = v25

                if 'R_VIDEO50' in df.columns:
                    r50 = np.random.beta(30, 13, size=n)
                    v50 = np.minimum(np.round(v25 * r50).astype(int), v25)
                    df['R_VIDEO50'] = v50

                    if 'R_VIDEO75' in df.columns:
                        r75 = np.random.beta(22, 18, size=n)
                        v75 = np.minimum(np.round(v50 * r75).astype(int), v50)
                        df['R_VIDEO75'] = v75

                        if 'R_VIDEOCOMPLETES' in df.columns:
                            rc = np.random.beta(18, 22, size=n)
                            vc = np.minimum(np.round(v75 * rc).astype(int), v75)
                            df['R_VIDEOCOMPLETES'] = vc

        if 'R_VIEWS' in df.columns:
            view_rate = np.random.beta(3, 7, size=n)
            df['R_VIEWS'] = np.round(impressions * view_rate).astype(int)

        return df

    def _correlate_kpi_metrics(self, df: pd.DataFrame, n: int) -> pd.DataFrame:
        """Set KPI_N / KPI_D so their ratio matches the KPI type (CTR/CPC/CPM/ROAS)."""
        if 'KPI' not in df.columns or 'KPI_N' not in df.columns or 'KPI_D' not in df.columns:
            return df

        impressions = pd.to_numeric(df.get('R_IMPRESSIONS', pd.Series(np.zeros(n))), errors='coerce').fillna(100).values.astype(float)
        clicks = pd.to_numeric(df.get('R_CLICKS', pd.Series(np.zeros(n))), errors='coerce').fillna(1).values.astype(float)
        cost = pd.to_numeric(df.get('R_MEDIACOST', pd.Series(np.zeros(n))), errors='coerce').fillna(1).values.astype(float)

        impressions = np.maximum(impressions, 1.0)
        clicks = np.maximum(clicks, 1.0)
        cost = np.maximum(cost, 0.01)

        kpi_n = np.zeros(n, dtype=float)
        kpi_d = np.zeros(n, dtype=float)

        kpi_col = df['KPI'].values

        for kpi_type in ['CTR', 'CPC', 'CPM', 'ROAS']:
            mask = kpi_col == kpi_type
            count = mask.sum()
            if count == 0:
                continue

            if kpi_type == 'CTR':
                kpi_d[mask] = impressions[mask]
                ratio = np.random.beta(2, 200, size=count)
                kpi_n[mask] = impressions[mask] * ratio

            elif kpi_type == 'CPC':
                kpi_d[mask] = clicks[mask]
                cpc_vals = np.random.lognormal(mean=0.4, sigma=0.5, size=count)
                cpc_vals = np.clip(cpc_vals, 0.20, 8.0)
                kpi_n[mask] = clicks[mask] * cpc_vals

            elif kpi_type == 'CPM':
                kpi_d[mask] = impressions[mask] / 1000.0
                cpm_vals = np.random.lognormal(mean=2.3, sigma=0.5, size=count)
                cpm_vals = np.clip(cpm_vals, 2.0, 60.0)
                kpi_n[mask] = (impressions[mask] / 1000.0) * cpm_vals

            elif kpi_type == 'ROAS':
                kpi_d[mask] = cost[mask]
                roas_vals = np.random.lognormal(mean=1.1, sigma=0.6, size=count)
                roas_vals = np.clip(roas_vals, 0.5, 15.0)
                kpi_n[mask] = cost[mask] * roas_vals

        df['KPI_N'] = np.round(kpi_n, 4)
        df['KPI_D'] = np.round(kpi_d, 4)

        return df

    def _correlate_pacing_metrics(self, df: pd.DataFrame, n: int) -> pd.DataFrame:
        """Correlate pacing: R_IMPRESSIONS -> R_MEDIACOST -> DAILY_PLANNED_SPEND."""
        if 'R_IMPRESSIONS' not in df.columns:
            return df

        impressions = pd.to_numeric(df['R_IMPRESSIONS'], errors='coerce').fillna(0).values.astype(float)
        impressions = np.maximum(impressions, 1.0)

        if 'R_MEDIACOST' in df.columns:
            cpm = np.random.lognormal(mean=2.3, sigma=0.5, size=n)
            cpm = np.clip(cpm, 2.0, 60.0)
            media_cost = np.round(impressions * cpm / 1000, 2)
            df['R_MEDIACOST'] = media_cost

            if 'DAILY_PLANNED_SPEND' in df.columns:
                plan_ratio = np.random.beta(8, 2, size=n)
                df['DAILY_PLANNED_SPEND'] = np.round(media_cost * plan_ratio, 2)

        return df

    # ------------------------------------------------------------------
    # Audience assignment (correlated to objective/funnel/tactic/channel)
    # ------------------------------------------------------------------

    def _assign_audience_columns(self, df: pd.DataFrame, table_name: str) -> pd.DataFrame:
        """Assign AUDIENCE_ID and AUDIENCE_NAME based on existing dimensions.

        For DELIVERY and KPI tables: uses OBJECTIVE, FUNNEL, TACTIC, CHANNEL.
        For PACING table: uses CAMPAIGN name pattern and CHANNEL.
        """
        n = len(df)
        tn = table_name.upper()
        audience_ids = np.empty(n, dtype=object)

        has_tactic = 'TACTIC' in df.columns
        has_objective = 'OBJECTIVE' in df.columns
        has_funnel = 'FUNNEL' in df.columns
        has_channel = 'CHANNEL' in df.columns

        tactic_vals = df['TACTIC'].values if has_tactic else np.array([''] * n)
        obj_vals = df['OBJECTIVE'].values if has_objective else np.array([''] * n)
        funnel_vals = df['FUNNEL'].values if has_funnel else np.array([''] * n)
        channel_vals = df['CHANNEL'].values if has_channel else np.array([''] * n)

        for i in range(n):
            tactic = str(tactic_vals[i]).lower().strip() if tactic_vals[i] else ''
            obj = str(obj_vals[i]).upper().strip() if obj_vals[i] else ''
            funnel = str(funnel_vals[i]).upper().strip() if funnel_vals[i] else ''
            channel = str(channel_vals[i]).upper().strip() if channel_vals[i] else ''

            if tactic in RETARGETING_TACTICS:
                pool = _AUD_RETARGETING
            elif channel in ('DIGITAL AUDIO', 'AUDIO'):
                pool = _AUD_AUDIO
            elif channel in ('SOCIAL',):
                pool = _AUD_SOCIAL
            elif channel in ('VIDEO', 'PERFORMANCE VIDEO') or 'CTV' in channel:
                pool = _AUD_VIDEO_CTV
            elif funnel == 'UPPER FUNNEL' or obj == 'AWARENESS':
                pool = _AUD_UPPER_FUNNEL
            elif funnel == 'LOWER FUNNEL' or obj == 'CONVERSION':
                pool = _AUD_LOWER_FUNNEL
            elif funnel == 'MID FUNNEL' or obj == 'ENGAGEMENT':
                pool = _AUD_MID_FUNNEL
            else:
                pool = AUDIENCE_IDS

            audience_ids[i] = pool[np.random.randint(len(pool))]

        df['AUDIENCE_ID'] = audience_ids
        df['AUDIENCE_NAME'] = [AUDIENCE_MAP[aid] for aid in audience_ids]
        return df

    # ------------------------------------------------------------------
    # Location assignment (correlated to CHANNEL_EXECUTION_NAME region)
    # ------------------------------------------------------------------

    def _assign_location_columns(self, df: pd.DataFrame, table_name: str) -> pd.DataFrame:
        """Assign DMA_CODE, DMA_NAME, STATE, REGION, LATITUDE, LONGITUDE.

        For DELIVERY and KPI: correlates to the region suffix in CHANNEL_EXECUTION_NAME.
        For PACING (no execution name): weighted random from full DMA pool.
        """
        if not self._all_dmas:
            return df

        n = len(df)
        tn = table_name.upper()
        has_exec_name = 'CHANNEL_EXECUTION_NAME' in df.columns

        dma_codes = np.empty(n, dtype=object)
        dma_names = np.empty(n, dtype=object)
        states = np.empty(n, dtype=object)
        regions = np.empty(n, dtype=object)
        latitudes = np.zeros(n, dtype=float)
        longitudes = np.zeros(n, dtype=float)

        if has_exec_name and tn in ('CAMPAIGN_DELIVERY', 'CAMPAIGN_KPI'):
            exec_names = df['CHANNEL_EXECUTION_NAME'].values
            for i in range(n):
                name = str(exec_names[i]) if exec_names[i] else ''
                suffix = name.rsplit('_', 1)[-1] if '_' in name else ''
                mapped_region = _EXEC_NAME_REGION_MAP.get(suffix)

                if mapped_region and mapped_region in self._dma_by_region:
                    pool = self._dma_by_region[mapped_region]
                else:
                    pool = self._all_dmas

                entry = pool[np.random.randint(len(pool))]
                dma_codes[i] = entry['dma_code']
                dma_names[i] = entry['dma_name']
                states[i] = entry['state']
                regions[i] = entry['region']
                latitudes[i] = entry['latitude']
                longitudes[i] = entry['longitude']
        else:
            indices = np.random.randint(0, len(self._all_dmas), size=n)
            for i in range(n):
                entry = self._all_dmas[indices[i]]
                dma_codes[i] = entry['dma_code']
                dma_names[i] = entry['dma_name']
                states[i] = entry['state']
                regions[i] = entry['region']
                latitudes[i] = entry['latitude']
                longitudes[i] = entry['longitude']

        df['DMA_CODE'] = dma_codes
        df['DMA_NAME'] = dma_names
        df['STATE'] = states
        df['REGION'] = regions
        df['LATITUDE'] = np.round(latitudes, 6)
        df['LONGITUDE'] = np.round(longitudes, 6)

        unique_regions = pd.Series(regions).value_counts().to_dict()
        correlated = has_exec_name and tn in ('CAMPAIGN_DELIVERY', 'CAMPAIGN_KPI')
        print(f"  Location assignment: correlated={correlated}, regions={unique_regions}")

        return df

    # ------------------------------------------------------------------
    # Post-processing for constant-value columns not handled by correlation
    # ------------------------------------------------------------------

    def _postprocess_constant_columns(self, df: pd.DataFrame, generation_rules: Dict[str, Any], table_name: str = None) -> pd.DataFrame:
        """Add jitter to constant-valued numeric columns where min=max."""
        for column_name, column_rules in generation_rules.items():
            col_type = column_rules.get('type')
            if col_type not in ('integer', 'float'):
                continue
            if column_name in CORRELATED_METRIC_COLUMNS:
                continue

            series = df[column_name]
            is_constant = False
            try:
                if series.nunique(dropna=True) <= 1:
                    is_constant = True
                elif 'min' in column_rules and 'max' in column_rules and column_rules['min'] == column_rules['max']:
                    is_constant = True
                elif 'std' in column_rules and (column_rules['std'] is None or float(column_rules['std']) == 0.0):
                    is_constant = True
            except Exception:
                pass

            if not is_constant:
                continue

            non_null = series.dropna()
            const_val = float(non_null.iloc[0]) if len(non_null) > 0 else 0

            if col_type == 'integer':
                center = int(const_val)
                jitter = np.random.choice([-1, 0, 0, 0, 1], size=len(df))
                vals = np.maximum(center + jitter, 0).astype(int)
            else:
                noise = np.random.normal(0, max(abs(const_val) * 0.01, 1e-6), size=len(df))
                vals = np.maximum(const_val + noise, 0.0)

            df[column_name] = vals

        return df

    # ------------------------------------------------------------------
    # Table generation orchestration
    # ------------------------------------------------------------------

    def generate_table_data(self, table_rules: Dict[str, Any], sample_rows: int = None) -> pd.DataFrame:
        """Generate synthetic data for a single table."""
        table_info = table_rules['table_info']
        generation_rules = table_info['generation_rules']

        if sample_rows is None:
            sample_rows = table_info.get('row_count', 1000)

        print(f"Generating {sample_rows} rows for table: {table_info['original_name']}")

        data = {}
        for column_name, column_rules in generation_rules.items():
            col_type = column_rules['type']
            print(f"  Generating column: {column_name} (type: {col_type})")

            if col_type == 'datetime':
                data[column_name] = self.generate_datetime_column(column_rules, sample_rows)
            elif col_type == 'categorical':
                data[column_name] = self.generate_categorical_column(column_rules, sample_rows)
            elif col_type in ('integer', 'float'):
                data[column_name] = self.generate_numeric_column(column_rules, sample_rows, col_type)
            elif col_type == 'text':
                data[column_name] = self.generate_text_column(column_rules, sample_rows)
            elif col_type == 'identifier':
                data[column_name] = self.generate_identifier_column(column_rules, sample_rows)
            elif col_type == 'boolean':
                data[column_name] = self.generate_boolean_column(column_rules, sample_rows)
            else:
                data[column_name] = self.generate_text_column(column_rules, sample_rows)

        df = pd.DataFrame(data)

        original_name = table_info.get('original_name', '')
        table_name = original_name.upper() if original_name else ''

        try:
            df = self._generate_correlated_metrics(df, table_name)
        except Exception as e:
            print(f"  Warning: correlated metric generation failed: {e}")

        try:
            df = self._assign_audience_columns(df, table_name)
        except Exception as e:
            print(f"  Warning: audience assignment failed: {e}")

        try:
            df = self._assign_location_columns(df, table_name)
        except Exception as e:
            print(f"  Warning: location assignment failed: {e}")

        try:
            df = self._postprocess_constant_columns(df, generation_rules, table_name.lower())
        except Exception as e:
            print(f"  Warning: constant-column post-processing failed: {e}")

        return df

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
                if sample_rows is not None:
                    rows_to_generate = sample_rows
                else:
                    original_rows = table_rules['table_info'].get('row_count', 1000)
                    rows_to_generate = int(original_rows * scale_factor)

                print(f"\nTable: {table_rules['table_info']['original_name']}")
                print(f"Original rows: {table_rules['table_info'].get('row_count', 'Unknown')}")
                print(f"Generating: {rows_to_generate} rows")

                df = self.generate_table_data(table_rules, rows_to_generate)

                # Enforce ID->NAME consistency
                try:
                    mappings = table_rules['table_info'].get('id_name_mappings', [])
                    for m in mappings:
                        id_col = m.get('id_column')
                        name_col = m.get('name_column')
                        mapping = m.get('mapping', {})
                        if id_col in df.columns and name_col in df.columns and mapping:
                            df[name_col] = df[id_col].map(lambda x, mp=mapping: mp.get(str(x), x))
                except Exception:
                    pass

                original_name = table_rules['table_info']['original_name']
                csv_file = self.output_dir / f"{original_name.lower()}.csv"
                df.to_csv(csv_file, index=False)
                print(f"Saved synthetic data to: {csv_file}")

            except Exception as e:
                print(f"Error generating data for {table_name}: {str(e)}")
                continue

        print(f"\n{'='*50}")
        print("Synthetic data generation completed!")
        print(f"CSV files saved to: {self.output_dir}")
        print(f"{'='*50}")


def main():
    """Main function to generate synthetic data."""
    print("Starting synthetic data generation...")
    print("This will create CSV files based on the statistical rules in data/stats/")

    generator = SyntheticDataGenerator(
        stats_dir="data/stats",
        output_dir="data"
    )

    print("Generating datasets (10% of original size)...")
    print("Original sizes: CAMPAIGN_DELIVERY=3.7M, CAMPAIGN_KPI=5M, CAMPAIGN_PACING=1M rows")
    generator.generate_all_data(sample_rows=None, scale_factor=0.1)


if __name__ == "__main__":
    main()
