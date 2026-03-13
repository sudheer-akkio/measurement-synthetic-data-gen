# Akkio Post-LLM Code Generation: Tags, Validation, and AST Transformations

## Table of Contents

1. [Overview](#overview)
2. [Database Tags Reference](#database-tags-reference)
   - [Table-Level Tags](#table-level-tags)
   - [Column-Level Tags](#column-level-tags)
3. [How Tags Affect Code Generation](#how-tags-affect-code-generation)
4. [Post-LLM Code Validation Pipeline](#post-llm-code-validation-pipeline)
   - [SQL Validation Flow](#sql-validation-flow)
   - [Python Code Validation Flow](#python-code-validation-flow)
5. [AST (Abstract Syntax Tree) Fundamentals](#ast-abstract-syntax-tree-fundamentals)
   - [What is an AST?](#what-is-an-ast)
   - [Python AST Examples](#python-ast-examples)
   - [SQL AST with sqlglot](#sql-ast-with-sqlglot)
6. [AST Transformers in Akkio](#ast-transformers-in-akkio)
   - [Python AST Transformers](#python-ast-transformers)
   - [SQL AST Transformers](#sql-ast-transformers)
7. [Platform-Specific Support](#platform-specific-support)
   - [Snowflake](#snowflake)
   - [BigQuery](#bigquery)
   - [Databricks](#databricks)
8. [Key Files Reference](#key-files-reference)

---

## Overview

Akkio uses a sophisticated system of database metadata tags combined with AST (Abstract Syntax Tree) transformations to ensure LLM-generated code is correct, safe, and optimized for the target database platform.

**The flow:**
```
User Query → LLM generates code → Extract code → Repair (AST transforms) → Validate → Execute
```

Two types of code are generated:
- **Raw SQL** - For CreateAudience, CreateSegment, multi-table queries
- **Snowpark Python** - For AudienceExplore, ChatExplore (Insights)

Both use AST-based transformations, but with different libraries:
- Python: Built-in `ast` module
- SQL: `sqlglot` library

---

## Database Tags Reference

Tags can be specified in two ways:
1. **In the description field**: Using `:tag_name` syntax (e.g., `:upper`, `:lower`)
2. **In the tags dictionary**: As key-value pairs stored in the database

### Tag Detection Method

Both `DataSourceTableSupplementalInfoRecord` and `DataSourceColumnSupplementalInfoRecord` use:

```python
def is_tagged(self, tag: str) -> bool:
    tag = tag.lower()
    # Check description for ":tag" syntax
    if self.description and f":{tag}" in self.description.lower():
        return True
    # Check tags dictionary
    if self.tags is not None and tag in self.tags:
        return True
    return False
```

### Table-Level Tags

| Tag | Constant | Location | Meaning |
|-----|----------|----------|---------|
| `use_for_audience_gen` | `USE_FOR_CREATE_AUDIENCE_TAG` | `ml/src/dataset_parsing/datasource_info/constants.py` | Marks tables that can be used in audience creation SQL generation |
| `measurement` | `MEASUREMENT_TAG` | `ml/src/dataset_parsing/datasource_info/constants.py` | Marks tables for measurement/ad analytics features |
| `churnable-interaction` | `TABLE_TAGS.CHURNABLE` | `ml/src/db/models/datasource_supp_info.py` | Tables with interactions suitable for churn analysis |
| `akkio-context-ignore` | `TABLE_TAGS.CONTEXT_IGNORE` | `ml/src/db/models/datasource_supp_info.py` | Tables to exclude from LLM context/prompts |
| `akkio-ignore-for-context` | (in description) | - | Tables to skip when building context strings |

**Usage Example - Table Filtering:**
```python
# From ml/src/chat_explore/audience/feature_router.py
for table_supp_info in datasource_supp_info.tables:
    if table_supp_info.is_tagged(USE_FOR_CREATE_AUDIENCE_TAG):
        # Include this table in LLM context

# From ml/src/chat_explore/measurement/features/feature_handler.py
table_subset = [
    table.name for table in self._run_config.datasource_supp_info.tables 
    if table.is_tagged(MEASUREMENT_TAG)
]
```

### Column-Level Tags

| Tag | Syntax | Effect on Code Generation |
|-----|--------|---------------------------|
| `upper` | `:upper` or `{"upper": ""}` | Forces uppercase transformation on values in generated code |
| `lower` | `:lower` or `{"lower": ""}` | Forces lowercase transformation on values |
| `space-to-hyphen` | `:space-to-hyphen` or tags dict | Replaces spaces with hyphens in literal values |
| `space-to-underscore` | `:space-to-underscore` or tags dict | Replaces spaces with underscores in literal values |
| `str-list-col-pregen` | `:str-list-col-pregen` or tags dict | Array columns (`_ARRAY`) get converted to `_STR_LIST` columns in LIKE operations |
| `ethnicity` | `:ethnicity` | Triggers cultural disclaimer when used in audience SQL |
| `akkio-context-ignore` | Description or tags | Columns to exclude from LLM context |
| `csv` | `CSV_VALS_TAG` | Indicates comma-separated values; adds hint in context |
| `all-unique-values` | `ALL_UNIQUE_VALS_TAG` | All unique values are included in context |
| `gen-unique-values-embeddings` | Module constant | Controls embedding generation for values |
| `hierarchy` | `{"hierarchy": "1"}`, `{"hierarchy": "2"}`, etc. | Defines drill-down hierarchy for measurement charts |
| `audience_preview_data` | Tags dict | Includes column in audience preview results |

**Column Tag Constants:**
```python
# ml/src/dataset_parsing/datasource_info/constants.py
USE_FOR_CREATE_AUDIENCE_TAG = "use_for_audience_gen"
MEASUREMENT_TAG = "measurement"

# ml/src/dataset_parsing/datasource_info/modules/constants.py
ALL_UNIQUE_VALS_TAG = "all-unique-values"
CSV_VALS_TAG = "csv"
GEN_UNIQUE_VALS_EMBEDDINGS_TAG = "gen-unique-values-embeddings"
```

---

## How Tags Affect Code Generation

### 1. Context Filtering (Pre-LLM)

Before sending context to the LLM, tables and columns are filtered based on tags:

```python
# Tables with akkio-context-ignore are excluded
if table.tags is not None and "akkio-ignore-for-context" in table.tags:
    continue

# Columns with akkio-context-ignore are excluded  
if column.is_tagged("akkio-context-ignore"):
    continue
```

### 2. Value Transformation (Post-LLM)

After the LLM generates code, values are transformed based on column tags:

```python
# From ml/src/chat_explore/node_visitor.py
@staticmethod
def fix_col_value_by_tag(value: str, col_supp_info: DataSourceColumnSupplementalInfoRecord) -> str:
    if col_supp_info.is_tagged("upper"):
        value = value.upper()
    elif col_supp_info.is_tagged("lower"):
        value = value.lower()
    
    if col_supp_info.is_tagged("space-to-hyphen"):
        value = value.replace(" ", "-")
    elif col_supp_info.is_tagged("space-to-underscore"):
        value = value.replace(" ", "_")
    return value
```

### 3. Array Column Handling

Columns tagged with `str-list-col-pregen` trigger special handling:

```python
# KEYWORDS_ARRAY → KEYWORDS_STR_LIST
if col_supp_info.type == FieldType.LIST and col_supp_info.is_tagged("str-list-col-pregen"):
    new_col_name = col_name.replace("_ARRAY", "_STR_LIST")
```

### 4. Cultural Disclaimer

Columns tagged with `:ethnicity` trigger a disclaimer:

```python
# From ml/src/chat_explore/audience/features/create_audience.py
if col_info and col_info.is_tagged("ethnicity"):
    return CULTURAL_DISCLAIMER
```

---

## Post-LLM Code Validation Pipeline

### SQL Validation Flow

**Location:** `ml/src/code_validator/sql_validation_runner.py`

```
LLM Output → Extract SQL → Repair → Validate → Return
                 ↓              ↓          ↓
           _extract_code   SQLRepairer  SQLValidator
```

**Detailed Flow:**

```python
def validate(self, completion_data: list[str], messages: list[LLMMessage]) -> ValidationResult:
    for msg in completion_data:
        # 1. Extract SQL from markdown code blocks
        sql = self._extract_code(msg)
        
        # 2. Repair: Apply AST transformations based on column tags
        repairer_cls = SQL_REPAIRER_DIALECT_MAP.get(self._datasource.type)
        query = SQLQuery(query=sql, dialect=SQLDialect.from_datasource_type(self._datasource.type))
        repairer = repairer_cls(query, datasource_supp_info=self._datasource_supp_info)
        repairer.repair()
        sql = repairer.repaired_code
        
        # 3. Validate: Run all validation checks
        validator_cls = _SQL_VALIDATOR_DIALECT_MAP.get(self._datasource.type)
        validator = validator_cls(sql, datasource, datasource_supp_info, ...)
        validator.validate()
```

**SQL Validator Checks (`SQLValidator`):**

| Check Method | Description |
|--------------|-------------|
| `_check_dml()` | Rejects DML operations (INSERT, UPDATE, DELETE, DROP, ALTER, TRUNCATE, CREATE) |
| `_check_cte()` | Ensures CTEs only reference valid tables; outer query must be SELECT |
| `_check_unique_vals()` | Validates literal values against column's known unique values for IN, LIKE, ILIKE, EQ |
| `_check_array_contains()` | Forces use of `AKKIO_ARRAY_CONTAINS_LIKE` instead of `ARRAY_CONTAINS` (Snowflake) |
| `_check_primary_key_select()` | Ensures `PRIMARY_ID` is selected when required |
| `_check_intent()` | Optional LLM-based validation that SQL matches user intent |
| `explain_query_validation()` | Runs EXPLAIN on the query against actual database |
| `_dialect_specific_validation()` | Platform-specific additional checks |

### Python Code Validation Flow

**Location:** `ml/src/code_validator/validation_runner.py`

```python
def validate(self, completion_data: list[str], messages: list[LLMMessage]) -> ValidationResult:
    for msg in completion_data:
        # 1. Extract Python code
        code = self._extract_code(msg)
        
        # 2. Repair: Apply AST transformations
        repairer = RepairerGenerator(self._dialect, code, datasource_supp_info).generate()
        repairer.repair()
        code = repairer.repaired_code
        
        # 3. Validate: Run all validation checks
        validator = self._validator_cls(code, is_template=self._is_template, **kwargs)
        validator.validate()
        
        # 4. Check for dangerous imports
        if validator.has_external_imports():
            code = EXTERNAL_ACCESS_PYTHON_CODE  # Safe fallback
```

**Python Validator Checks:**

| Check | Description |
|-------|-------------|
| `_check_function()` | Entry function must be `perform_analysis(session)` or `perform_analysis(df)` |
| `_check_render()` | Validates render function usage (render_table, render_chart, etc.) |
| `_check_import()` | Blocks blacklisted imports and functions |
| `_check_sql()` | Local execution test using Snowpark local testing framework |
| `has_external_imports()` | Detects dangerous libraries (os, subprocess, sys, socket, requests) |

**Blacklisted Items:**

```python
# From ml/src/code_validator/constants.py
FUNCTION_BLACKLIST = ['exec', 'eval', 'compile', 'open', '__import__', ...]
IMPORT_WHITELIST = ['pandas', 'numpy', 'plotly', 'snowflake', ...]  # Only these allowed
```

---

## AST (Abstract Syntax Tree) Fundamentals

### What is an AST?

An **Abstract Syntax Tree (AST)** is a tree-structured representation of source code where:
- **Each node** represents a construct in the code (function, variable, operator, literal, etc.)
- **Parent-child relationships** represent how constructs are nested/composed
- **"Abstract"** means it ignores syntactic details like parentheses, semicolons, and whitespace

### Python AST Examples

#### Example 1: Simple Assignment
```python
x = 1 + 2
```

AST Structure:
```
Module
└── Assign
    ├── targets: [Name(id='x')]
    └── value: BinOp
              ├── left: Constant(value=1)
              ├── op: Add
              └── right: Constant(value=2)
```

#### Example 2: Method Chaining (Pandas/Snowpark Style)
```python
df['COLUMN'].str.upper().contains('VALUE')
```

AST Structure:
```
Call (contains)
└── func: Attribute
    └── value: Call (upper)
              └── func: Attribute
                  └── value: Attribute (str)
                      └── value: Subscript (df['COLUMN'])
```

#### Example 3: Function Call in Comparison
```python
upper(df['CATEGORY']) == 'HOME DECOR'
```

AST Structure:
```
Compare
├── left: Call
│   ├── func: Name(id='upper')
│   └── args: [Subscript]
│             ├── value: Name(id='df')
│             └── slice: Constant('CATEGORY')
├── ops: [Eq]
└── comparators: [Constant('HOME DECOR')]
```

#### Example 4: Snowpark Function
```python
def perform_analysis(session):
    df = session.table("CUSTOMERS")
    result = df.filter(df["STATUS"] == "ACTIVE").select(["NAME", "EMAIL"])
    render_table(result)
```

AST Structure:
```
Module
└── FunctionDef(name='perform_analysis')
    ├── args: [arg(arg='session')]
    └── body:
        ├── Assign: df = session.table("CUSTOMERS")
        │   └── Call(func=Attribute(value=Name('session'), attr='table'))
        ├── Assign: result = df.filter(...).select(...)
        │   └── Call(func=Attribute(attr='select'))
        │       └── value: Call(func=Attribute(attr='filter'))
        └── Expr: render_table(result)
            └── Call(func=Name('render_table'))
```

### Python AST Key Methods

```python
import ast

# Parse source code into AST
tree = ast.parse(code)

# Pretty-print AST structure
print(ast.dump(tree, indent=2))

# Convert AST back to source code
modified_code = ast.unparse(tree)

# Fill in line/column info after modifications
ast.fix_missing_locations(tree)
```

### Python AST Key Node Types

| Category | Node Type | Example |
|----------|-----------|---------|
| **Module Level** | `Module` | Root node |
| | `FunctionDef` | `def foo(): ...` |
| | `ClassDef` | `class Foo: ...` |
| | `Import` | `import x` |
| | `ImportFrom` | `from x import y` |
| **Statements** | `Assign` | `x = 1` |
| | `Return` | `return value` |
| | `If` | `if/elif/else` |
| | `For` | `for x in y: ...` |
| | `Expr` | Standalone expression |
| **Expressions** | `Name` | Variable reference: `x` |
| | `Constant` | Literal: `42`, `"hello"` |
| | `BinOp` | Binary: `x + y`, `x \| y` |
| | `Compare` | Comparison: `x == y` |
| | `Call` | Function call: `func(args)` |
| **Access** | `Attribute` | `obj.attr` |
| | `Subscript` | `obj[key]` or `df["COL"]` |
| **Collections** | `List` | `[1, 2, 3]` |
| | `Dict` | `{"a": 1}` |

### SQL AST with sqlglot

For SQL, Akkio uses the `sqlglot` library which provides similar AST capabilities:

| Aspect | Python AST | SQL AST (sqlglot) |
|--------|------------|-------------------|
| **Library** | `ast` (built-in) | `sqlglot` (third-party) |
| **Input** | Python source code | SQL query string |
| **Parse** | `ast.parse(code)` | `sqlglot.parse_one(sql, dialect="snowflake")` |
| **Unparse** | `ast.unparse(tree)` | `parsed.sql(dialect="snowflake")` |
| **Transform** | Custom `NodeTransformer` | `parsed.transform(fn)` |

**sqlglot Node Types:**

```python
from sqlglot.expressions import (
    EQ,              # column = 'value'
    ArrayContains,   # ARRAY_CONTAINS(col, val)
    Cast,            # CAST(x AS type)
    Column,          # table.column_name
    ILike,           # column ILIKE '%pattern%'
    In,              # column IN ('a', 'b', 'c')
    Like,            # column LIKE '%pattern%'
    Literal,         # 'string' or 123
    Lower,           # LOWER(column)
    Upper,           # UPPER(column)
    Placeholder,     # :param (for parameterized queries)
)
```

---

## AST Transformers in Akkio

### Python AST Transformers

**Location:** `ml/src/chat_explore/node_visitor.py`

Python's `ast` module provides two key classes:

**`ast.NodeVisitor`** - Read-only tree traversal:
```python
class FunctionFinder(ast.NodeVisitor):
    def visit_FunctionDef(self, node):
        print(f"Found function: {node.name}")
        self.generic_visit(node)  # Continue to child nodes
```

**`ast.NodeTransformer`** - Read-write tree traversal:
```python
class UppercaseStrings(ast.NodeTransformer):
    def visit_Constant(self, node):
        if isinstance(node.value, str):
            return ast.Constant(value=node.value.upper())  # Replace node
        return node  # Keep original
```

#### Key Transformers in Akkio

**1. `SnowflakeRemoveUpperLowerCallsBySchema`**

Removes redundant `upper()`/`lower()` calls and transforms string literals based on column tags.

```python
# Before (LLM generated):
upper(df['CATEGORY']) == 'home decor'

# After (if CATEGORY is tagged with :upper :space-to-hyphen):
df['CATEGORY'] == 'HOME-DECOR'
```

**2. `SnowflakeHandleArrayColsWithStrListPregenBySchema`**

Converts `_ARRAY` columns to `_STR_LIST` columns in `.like()` and `akkio_array_contains_like()` contexts for columns tagged with `str-list-col-pregen`.

```python
# Before:
df['KEYWORDS_ARRAY'].like('%sports%')

# After:
df['KEYWORDS_STR_LIST'].like('%sports%')
```

**3. `SnowflakeHandleAkkioArrayContainsLikeMatchValuesBySchema`**

Transforms `akkio_array_contains_like()` calls to LIKE expressions with proper value transformations.

```python
# Before:
akkio_array_contains_like(df["KEYWORDS_ARRAY"], ["sports", "fitness"])

# After:
df['KEYWORDS_STR_LIST'].like('%sports%') | df['KEYWORDS_STR_LIST'].like('%fitness%')
```

**4. `PandasNodeTransformer`**

Adds compatibility fixes for pandas operations:
- Adds `na=False` to `.str.contains()` calls
- Adds `observed=True` to `groupby()` calls
- Limits `.to_pandas()` result size

**5. `SnowflakeJoinSuffix`**

Adds `rsuffix` parameters to join operations to avoid column name collisions.

**6. `SnowflakeDateAddLit` / `SnowflakeArrayContainsAddLit`**

Wraps literal values in `lit()` for Snowflake compatibility.

**7. `SnowflakeTableFinderNodeVisitor`**

Read-only visitor that extracts all table names referenced in `session.table()` calls.

**8. `TryExceptInjector`**

Wraps `perform_analysis()` body in try/except for error handling.

### SQL AST Transformers

**Location:** `ml/src/code_repairer/dialects/sql.py`, `ml/src/code_repairer/dialects/snowflake_sql.py`

#### Base SQL Transformer: `_transform_handle_value_tags`

Transforms literal values based on column tags for these expression types:
- `EQ` (equality)
- `Like`
- `ILike`
- `In`
- `ArrayContains`

```python
def transform_value(val: str, col_description: str | None, col_tags: dict | None) -> str:
    if col_tags is not None:
        if "lower" in col_tags: val = val.lower()
        elif "upper" in col_tags: val = val.upper()
        if "space-to-underscore" in col_tags: val = val.replace(" ", "_")
        if "space-to-hyphen" in col_tags: val = val.replace(" ", "-")
    # Similar checks for description-based tags (:upper, :lower, etc.)
    return val
```

#### Snowflake-Specific: `_handle_array_columns`

- Removes alias columns from FLATTEN operations
- Populates mapping of flatten aliases to column supplemental info
- Replaces flattened array column references with `_STR_LIST` columns in LIKE expressions

#### Snowflake-Specific: `_transform_akkio_array_contains_like`

Transforms `AKKIO_ARRAY_CONTAINS_LIKE` function calls:
- Converts `_ARRAY` columns to `_STR_LIST` for tagged columns
- Transforms to LIKE expressions with proper value formatting
- Handles both single values and JSON array inputs

---

## Platform-Specific Support

### Repairer Map (`SQL_REPAIRER_DIALECT_MAP`)

```python
# From ml/src/code_validator/constants.py
SQL_REPAIRER_DIALECT_MAP: dict[DataSourceType, type[SQLCodeRepairer]] = {
    DataSourceType.SNOWFLAKE: SnowflakeSQLCodeRepairer,
    DataSourceType.BIGQUERY: SQLCodeRepairer,  # Base only
    DataSourceType.S3: SQLCodeRepairer,
    DataSourceType.DUCKDB: SQLCodeRepairer,
    DataSourceType.DATABRICKS: DatabricksSQLCodeRepairer,
}
```

### Validator Map (`_SQL_VALIDATOR_DIALECT_MAP`)

```python
# From ml/src/code_validator/_constants.py
_SQL_VALIDATOR_DIALECT_MAP: dict[DataSourceType, type[SQLValidator]] = {
    DataSourceType.SNOWFLAKE: SnowflakeSQLValidator,
    DataSourceType.BIGQUERY: SQLValidator,  # Base only
    DataSourceType.S3: SQLValidator,
    DataSourceType.DUCKDB: SQLValidator,
    DataSourceType.DATABRICKS: SQLValidator,
    DataSourceType.DATABRICKS_WAREHOUSE: SQLValidator,
    DataSourceType.DATABRICKS_VOLUME: SQLValidator,
}
```

### Snowflake

**Files:**
- `ml/src/code_repairer/dialects/snowflake_sql.py`
- `ml/src/code_repairer/dialects/snowflake.py` (Python)
- `ml/src/code_validator/dialects/snowflake_sql.py`
- `ml/src/code_validator/dialects/snowflake.py` (Python)

**Unique Features:**
- Full array handling (`_ARRAY` → `_STR_LIST`)
- `AKKIO_ARRAY_CONTAINS_LIKE` UDF support
- `LATERAL FLATTEN` handling
- Local testing validation with Snowpark mock

### BigQuery

**Files:**
- `ml/src/code_repairer/dialects/bigquery_sql.py`

**Implementation:**
```python
class BigQuerySQLCodeRepairer(SQLCodeRepairer):
    """Repairer for BigQuery SQL code."""
    ...  # No dialect-specific overrides - uses base class only
```

**Features:** Base tag transformations only

### Databricks

**Files:**
- `ml/src/code_repairer/dialects/databricks_sql.py`

**Implementation:**
```python
class DatabricksSQLCodeRepairer(SQLCodeRepairer):
    """Repairer for databricks SQL code."""

    def dialect_specific_repair(self) -> None:
        def remove_table_aliases(node: sqlglot.Expression) -> sqlglot.Expression:
            # Databricks doesn't like aliases in ORDER BY
            if isinstance(node, exp.Order):
                for child_node in node.find_all(exp.Column):
                    if child_node.table:
                        child_node.set("table", None)  # Remove alias
            return node

        if self._parsed:
            self._parsed = self._parsed.transform(remove_table_aliases)
```

**Unique Features:** Removes table aliases from ORDER BY clauses

### Feature Comparison Matrix

| Feature | Snowflake | BigQuery | Databricks |
|---------|-----------|----------|------------|
| Tag-based value transforms (`:upper`, etc.) | ✅ | ✅ | ✅ |
| DML blocking | ✅ | ✅ | ✅ |
| CTE validation | ✅ | ✅ | ✅ |
| Unique value validation | ✅ | ✅ | ✅ |
| EXPLAIN in database | ✅ | ✅ | ✅ |
| Intent scoring (LLM) | ✅ | ✅ | ✅ |
| Array → STR_LIST conversion | ✅ | ❌ | ❌ |
| `AKKIO_ARRAY_CONTAINS_LIKE` | ✅ | ❌ | ❌ |
| FLATTEN handling | ✅ | ❌ | ❌ |
| ORDER BY alias removal | ❌ | ❌ | ✅ |
| Local execution testing (Python) | ✅ | ❌ | ❌ |

---

## Key Files Reference

### Core Validation & Repair

| File | Purpose |
|------|---------|
| `ml/src/code_validator/sql_validation_runner.py` | Main SQL validation runner |
| `ml/src/code_validator/validation_runner.py` | Main Python validation runner |
| `ml/src/code_validator/dialects/sql.py` | Base SQL validator with all checks |
| `ml/src/code_validator/constants.py` | Dialect maps, blacklists, error types |
| `ml/src/code_validator/_constants.py` | Validator dialect mapping |

### SQL Repairers

| File | Purpose |
|------|---------|
| `ml/src/code_repairer/dialects/sql.py` | Base SQL repairer with tag transforms |
| `ml/src/code_repairer/dialects/snowflake_sql.py` | Snowflake-specific SQL repairs |
| `ml/src/code_repairer/dialects/bigquery_sql.py` | BigQuery SQL repairer (base only) |
| `ml/src/code_repairer/dialects/databricks_sql.py` | Databricks SQL repairer |

### Python Repairers & Transformers

| File | Purpose |
|------|---------|
| `ml/src/chat_explore/node_visitor.py` | All Python AST transformers |
| `ml/src/code_repairer/dialects/snowflake.py` | Snowflake Python repairer |
| `ml/src/code_validator/dialects/snowflake.py` | Snowflake Python validator |

### Tags & Supplemental Info

| File | Purpose |
|------|---------|
| `ml/src/db/models/datasource_supp_info.py` | Table/Column supplemental info models with `is_tagged()` |
| `ml/src/dataset_parsing/datasource_info/constants.py` | Tag constants |
| `ml/src/sql_parser/model.py` | TableInfo/ColumnInfo with `is_tagged()` |

### Feature Entry Points

| File | Purpose |
|------|---------|
| `ml/src/chat_explore/audience/features/create_audience.py` | CreateAudience feature with SQL generation |
| `ml/src/chat_explore/segment/features/create_segment.py` | CreateSegment feature |
| `ml/src/chat_explore/multi_table/features/feature_handler.py` | Multi-table query handler |
| `ml/src/chat_explore/measurement/features/feature_handler.py` | Measurement feature handler |

---

## Appendix: Complete Transformer Example

### Python AST Transformer: Removing Redundant upper()

```python
import ast

class RemoveUpperTransformer(ast.NodeTransformer):
    def __init__(self, datasource_supp_info):
        self._ds_info = datasource_supp_info
    
    def visit_Compare(self, node):
        self.generic_visit(node)  # Process children first
        
        # Check if left side is upper(df['COLUMN'])
        if isinstance(node.left, ast.Call):
            if isinstance(node.left.func, ast.Name) and node.left.func.id == 'upper':
                # Get the subscript: df['COLUMN']
                if isinstance(node.left.args[0], ast.Subscript):
                    subscript = node.left.args[0]
                    col_name = subscript.slice.value
                    
                    # Get column info
                    col_info = self._ds_info.get_column_by_name(col_name)
                    
                    if col_info and col_info.is_tagged("upper"):
                        # Remove the upper() call, keep just the subscript
                        node.left = subscript
                        
                        # Transform the comparison value
                        if isinstance(node.comparators[0], ast.Constant):
                            old_val = node.comparators[0].value
                            new_val = old_val.upper()
                            if col_info.is_tagged("space-to-hyphen"):
                                new_val = new_val.replace(" ", "-")
                            node.comparators[0].value = new_val
        
        return node

# Usage:
tree = ast.parse(llm_generated_code)
transformer = RemoveUpperTransformer(datasource_supp_info)
modified_tree = transformer.visit(tree)
ast.fix_missing_locations(modified_tree)
repaired_code = ast.unparse(modified_tree)
```

### SQL AST Transformer: Value Tag Handling

```python
import sqlglot
from sqlglot.expressions import Column, Like, Literal

def transform_handle_value_tags(node: sqlglot.Expression) -> sqlglot.Expression:
    if isinstance(node, Like) and isinstance(node.this, Column):
        col = node.this
        col_name = col.name.upper()
        table_name = col.table.upper() if col.table else ""
        
        col_info = datasource_supp_info.get_table_column_by_name(table_name, col_name)
        
        if col_info and isinstance(node.expression, Literal):
            val = node.expression.this
            
            # Apply tag-based transformations
            if col_info.is_tagged("upper"):
                val = val.upper()
            elif col_info.is_tagged("lower"):
                val = val.lower()
            
            if col_info.is_tagged("space-to-hyphen"):
                val = val.replace(" ", "-")
            elif col_info.is_tagged("space-to-underscore"):
                val = val.replace(" ", "_")
            
            node.set("expression", Literal(this=val, is_string=True))
    
    return node

# Usage:
parsed = sqlglot.parse_one(sql, dialect="snowflake")
parsed = parsed.transform(transform_handle_value_tags)
repaired_sql = parsed.sql(dialect="snowflake", pretty=True)
```

---

*Documentation generated from Akkio codebase analysis - January 2026*
