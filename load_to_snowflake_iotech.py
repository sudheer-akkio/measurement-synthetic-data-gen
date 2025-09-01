import os
import pandas as pd
import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.append(str(project_root))

sys.path.append(str(Path(__file__).parent / 'data-warehouse-sdk'))

from src.connectors.snowflake import SnowflakeConnector as snow # type: ignore
from src.utils import get_snowflake_connection, load_data_files, generate_table_name # type: ignore
from src.data_dictionary.DataDictionaryManager import DataDictionaryManager # type: ignore

database = "DEMO"
schema = "AKKIO_COMMON"

obj = get_snowflake_connection('akkio', database=database, schema=schema)

data_folder = project_root / "data"

# Load data dictionary information
dd_path = project_root / "data_schemas"

# Load data files using the utility function
for filename, table in load_data_files(
    data_folder=data_folder,
    file_suffix=".csv",
    read_chunk_size=10000000
):

    print(f"Loading {filename}...")
    
    # if there is data in obj.data, clear it
    if hasattr(obj, 'data') and len(obj.data) > 0:
        obj.data = pd.DataFrame()

    obj.data = table

     # Generate table name from filename first
    table_name = generate_table_name(filename, remove_suffix=".csv")

    manager = DataDictionaryManager(str(dd_path))

    schema_file = f"{table_name}_schema.md"

    schema_obj = manager.load_data_dictionary(schema_file=schema_file, warehouse_type="snowflake")

    # Use the table name from the schema if available, otherwise use the generated one
    if schema_obj and hasattr(schema_obj, 'table_name') and schema_obj.table_name:
        table_name = schema_obj.table_name
    obj.table_name = table_name
    
    # Check if table exists and truncate if it does, otherwise create it
    if obj.check_table_exists():
        print(f"Table {table_name} exists - continuing...")
        # obj.truncate_table()
        
        # # Apply comments to existing table
        # if schema and (schema.table_comment or schema.column_comments):
        #     obj.apply_table_comments(schema.table_comment, schema.column_comments)
    else:
        print(f"Table {table_name} does not exist - creating new table...")
        if schema_obj and schema_obj.column_datatypes:
            print(f"Using schema from {str(dd_path)} folder...")
            obj.create_table(schema_dict=schema_obj.column_datatypes)
        else:
            print("Warning: No schema file found, inferring types from data...")
            obj.create_table()
        
        # Apply comments to new table
        if schema_obj and (schema_obj.table_comment or schema_obj.column_comments):
            obj.apply_table_comments(schema_obj.table_comment, schema_obj.column_comments)

    # add rows to the table
    obj.add_rows(chunk_size=1000000, schema=schema_obj)