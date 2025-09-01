# Table Name: CAMPAIGN_KPI

## Table Description
IO Tech campaign measurement KPI data. This table contains campaign performance metrics and KPIs | :short-name:kpi: :measurement:kpi: |<

## Data Dictionary

### Fields:

- `DATE` (DATE): Date field for campaign data
- `BRAND` (VARCHAR): Brand | :upper
- `LOB` (VARCHAR): Line of Business | :upper
- `CHANNEL` (VARCHAR): Delivery Channel | :upper
- `SUB_CHANNEL` (VARCHAR): Delivery Sub-Channel | :upper
- `REPORTING_CHANNEL` (VARCHAR): Reporting channel field
- `OBJECTIVE` (VARCHAR): Campaign objective field
- `FUNNEL` (VARCHAR): Marketing funnel stage
- `CAMPAIGN` (VARCHAR): Campaign name or identifier
- `NETWORK` (VARCHAR): Advertising network
- `PARTNER` (VARCHAR): Partner or vendor information
- `AD_FORMAT` (VARCHAR): Advertisement format type
- `TACTIC` (VARCHAR): Marketing tactic used
- `HMI_AUDIENCE_NAME` (VARCHAR): HMI audience name identifier
- `HMI_BLU` (NUMBER): HMI BLU numeric identifier
- `HMI_CID` (VARCHAR): HMI campaign identifier
- `CREATIVE_NAME` (VARCHAR): Creative asset name
- `CREATIVE` (VARCHAR): Creative identifier or name
- `CREATIVE_GROUP` (VARCHAR): Creative group classification
- `CREATIVE_CONCEPT` (VARCHAR): Creative concept description
- `CHANNEL_EXECUTION_ID` (VARCHAR): Channel execution identifier
- `CHANNEL_EXECUTION_NAME` (VARCHAR): Channel execution name
- `KPI` (VARCHAR): Key Performance Indicator type
- `BENCHMARK_VALUE` (NUMBER): Benchmark value for comparison
- `KPI_CLASS` (VARCHAR): KPI classification category
- `KPI_N` (FLOAT): KPI numerator value
- `KPI_D` (FLOAT): KPI denominator value
- `BM_D` (NUMBER): Benchmark denominator value
- `BM_N` (NUMBER): Benchmark numerator value
- `R_MEDIACOST` (FLOAT): Media cost amount
- `R_IMPRESSIONS` (FLOAT): Number of impressions
- `R_CLICKS` (FLOAT): Number of clicks
- `R_ENGAGEMENTS` (NUMBER): Number of engagements
- `R_VIDEOSTARTS` (NUMBER): Number of video starts
- `R_VIDEO25` (NUMBER): Number of 25% video completions
- `R_VIDEO50` (NUMBER): Number of 50% video completions
- `R_VIDEO75` (NUMBER): Number of 75% video completions
- `R_VIDEOCOMPLETES` (NUMBER): Number of video completions
- `R_VIEWS` (NUMBER): Number of views

## Table Relationships
*No explicit relationships defined - this is synthetic data for testing purposes*

## Business Context
This table contains Bobs Discount Furniture campaign measurement KPI data with data type 'campaign' and owner 'akkio'. The table includes performance metrics and KPIs with transformation tags for data processing workflows. Fields marked with :upper tags should be processed with uppercase transformation.

## Notes
- Data generated with statistical perturbation applied to protect source data privacy
- Row count in synthetic data: Variable (default 1000 rows for testing)
- Original table row count: 5000000
- Generation timestamp: 2025-09-01 13:24:25
- Statistical distributions have been modified to ensure synthetic data doesn't exactly replicate original patterns
