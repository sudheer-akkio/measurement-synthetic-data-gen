# Table Name: CAMPAIGN_PACING

## Table Description
IO Tech campaign measurement pacing data | :short-name:pacing: :measurement:pacing: |<

## Data Dictionary

### Fields:

- `DATE` (DATE): Generated datetime field
- `BRAND` (VARCHAR): Brand | :upper
- `LOB` (VARCHAR): Line of Business | :upper
- `CHANNEL` (VARCHAR): Delivery Channel | :upper
- `CAMPAIGN` (VARCHAR): Generated categorical field (2709 unique values)
- `PARTNER` (VARCHAR): Partner | :lower
- `DNU` (VARCHAR):
- `BUDGET_ID` (VARCHAR):
- `BUDGET_NAME` (VARCHAR):
- `REPORTING_CHANNEL` (VARCHAR):
- `BUYING_CHANNEL` (VARCHAR):
- `ESTIMATE_ID` (VARCHAR):
- `ESTIMATE_NAME` (VARCHAR):
- `DAILY_PLANNED_SPEND` (FLOAT):
- `SOURCE` (VARCHAR):
- `R_MEDIACOST` (FLOAT):
- `R_IMPRESSIONS` (FLOAT):

## Table Relationships
*No explicit relationships defined - this is synthetic data for testing purposes*

## Business Context
This synthetic dataset was generated from the original CAMPAIGN_PACING table to maintain realistic data distributions while protecting sensitive information. The data includes perturbations to prevent exact replication of the source data.

## Notes
- Data generated with statistical perturbation applied to protect source data privacy
- Row count in synthetic data: Variable (default 1000 rows for testing)
- Original table row count: 1048570
- Generation timestamp: 2025-09-01 13:24:40
- Statistical distributions have been modified to ensure synthetic data doesn't exactly replicate original patterns
