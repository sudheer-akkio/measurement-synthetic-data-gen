"""
Weekly data refresh utility.

Slides the data window forward by adding new rows at the recent end and
removing the same number from the oldest end, keeping file sizes constant.

Usage:
    python refresh_data.py                  # refresh by 1 week
    python refresh_data.py --weeks 4        # catch up 4 weeks at once
    python refresh_data.py --dry-run        # preview without writing
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import timedelta
from generate_data import SyntheticDataGenerator


DATA_DIR = Path("data")
STATS_DIR = Path("data/stats")

TABLE_FILES = [
    "campaign_delivery.csv",
    "campaign_kpi.csv",
    "campaign_pacing.csv",
]


class DataRefresher:

    def __init__(self, data_dir: Path = DATA_DIR, stats_dir: Path = STATS_DIR):
        self.data_dir = data_dir
        self.stats_dir = stats_dir
        self.generator = SyntheticDataGenerator(
            stats_dir=str(stats_dir),
            output_dir=str(data_dir),
        )
        self.all_rules = self.generator.load_generation_rules()

    def _rules_for_file(self, filename: str) -> dict | None:
        key = filename.replace(".csv", "")
        return self.all_rules.get(key)

    def refresh_table(self, filename: str, weeks: int = 1, dry_run: bool = False):
        """Slide one table forward by *weeks* weeks."""
        csv_path = self.data_dir / filename
        if not csv_path.exists():
            print(f"SKIP {filename}: file not found")
            return

        table_rules = self._rules_for_file(filename)
        if table_rules is None:
            print(f"SKIP {filename}: no generation rules found")
            return

        df = pd.read_csv(csv_path)
        df['DATE'] = pd.to_datetime(df['DATE'])

        original_rows = len(df)
        date_min = df['DATE'].min()
        date_max = df['DATE'].max()

        print(f"\n{'='*60}")
        print(f"Table: {filename}")
        print(f"  Rows: {original_rows:,}")
        print(f"  Date range: {date_min.date()} .. {date_max.date()}")

        for week_i in range(1, weeks + 1):
            current_min = df['DATE'].min()
            current_max = df['DATE'].max()

            oldest_window_end = current_min + timedelta(days=6)
            oldest_mask = df['DATE'] <= oldest_window_end
            rows_to_remove = oldest_mask.sum()

            if rows_to_remove == 0:
                rows_to_remove = max(1, len(df) // 200)
                oldest_mask = df.index[:rows_to_remove]

            new_start = current_max + timedelta(days=1)
            new_end = current_max + timedelta(days=7)

            print(f"  Week {week_i}: drop {rows_to_remove:,} rows "
                  f"({current_min.date()}..{oldest_window_end.date()}), "
                  f"add {rows_to_remove:,} rows ({new_start.date()}..{new_end.date()})")

            if dry_run:
                continue

            df = df[~oldest_mask].reset_index(drop=True)

            rules_copy = _deep_copy_rules(table_rules)
            date_rule = rules_copy['table_info']['generation_rules'].get('DATE', {})
            date_rule['min_date'] = new_start.isoformat()
            date_rule['max_date'] = new_end.isoformat()
            if 'temporal_patterns' in date_rule:
                del date_rule['temporal_patterns']

            new_rows = self.generator.generate_table_data(rules_copy, rows_to_remove)
            new_rows['DATE'] = pd.to_datetime(new_rows['DATE'])

            df = pd.concat([df, new_rows], ignore_index=True)

        new_min = df['DATE'].min()
        new_max = df['DATE'].max()
        print(f"  Result: {len(df):,} rows, {new_min.date()} .. {new_max.date()}")

        if not dry_run:
            df.to_csv(csv_path, index=False)
            print(f"  Saved to {csv_path}")

    def refresh_all(self, weeks: int = 1, dry_run: bool = False):
        print(f"Refreshing all tables by {weeks} week(s)" +
              (" (DRY RUN)" if dry_run else ""))

        for filename in TABLE_FILES:
            self.refresh_table(filename, weeks=weeks, dry_run=dry_run)

        print(f"\n{'='*60}")
        print("Refresh complete!" + (" (dry run -- no files written)" if dry_run else ""))


def _deep_copy_rules(rules: dict) -> dict:
    """Cheap deep copy for JSON-serializable dicts."""
    import json
    return json.loads(json.dumps(rules, default=str))


def main():
    parser = argparse.ArgumentParser(description="Weekly data refresh utility")
    parser.add_argument("--weeks", type=int, default=1,
                        help="Number of weeks to advance (default: 1)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview changes without writing files")
    args = parser.parse_args()

    refresher = DataRefresher()
    refresher.refresh_all(weeks=args.weeks, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
