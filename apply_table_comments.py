import os
import pandas as pd
import sys
import re
from pathlib import Path

# sys.path.append(str(project_root))

sys.path.append(str(Path(__file__).parent / 'data-warehouse-sdk'))

from src.connectors.snowflake import SnowflakeConnector as snow # type: ignore
from src.utils import get_snowflake_connection, load_data_files, generate_table_name # type: ignore
from src.data_dictionary.DataDictionaryManager import DataDictionaryManager # type: ignore

database = "DEMO"
schema = "ANNALECT_DEMO"

obj = get_snowflake_connection('akkio', database=database, schema=schema)

root_path = Path(__file__).parent

schema_path = root_path / 'data_schemas'

files = [f for f in os.listdir(str(schema_path)) if f.startswith("masked_") and f.endswith('.md')]

print(files)

file_paths = [str(schema_path / f) for f in files]

def extract_table_name_from_markdown(file_path):
    """Extract table name from the first line of markdown file with format '# Table Name: TABLE_NAME'"""
    with open(file_path, 'r', encoding='utf-8') as f:
        first_line = f.readline().strip()
        # Use regex to extract table name from "# Table Name: TABLE_NAME"
        match = re.match(r'^#\s*Table Name:\s*(.+)$', first_line)
        if match:
            return match.group(1).strip()
        else:
            raise ValueError(f"Could not extract table name from file: {file_path}")

table_names = []
for file_path in file_paths:
    table_name = extract_table_name_from_markdown(file_path)
    table_names.append(table_name)

print(table_names)

manager = DataDictionaryManager(str(schema_path))

for i in range(len(table_names)):
    obj.table_name = table_names[i]

    schema_file = file_paths[i]

    schema = manager.load_data_dictionary(schema_file=schema_file, warehouse_type="snowflake")

    obj.apply_table_comments(schema.table_comment, schema.column_comments)