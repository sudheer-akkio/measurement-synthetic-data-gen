# Table Name: CAMPAIGN_DELIVERY

## Table Description
This table contains IO Tech measurement data for campaign delivery. The data maintains campaign performance metrics across various channels and tactics for analysis and optimization | :short-name:delivery: :measurement:delivery: |<

## Data Dictionary

### Fields:

- `DATE` (DATE): Date field for campaign delivery data
- `BRAND` (VARCHAR): Brand | :upper
- `LOB` (VARCHAR): Line of Business | :upper
- `CHANNEL` (VARCHAR): Delivery Channel | :upper
- `SUB_CHANNEL` (VARCHAR): Delivery Sub-Channel | :upper
- `REPORTING_CHANNEL` (VARCHAR): Reporting Channel | :upper
- `OBJECTIVE` (VARCHAR): Objective | :upper
- `FUNNEL` (VARCHAR): Funnel | :upper
- `CAMPAIGN` (VARCHAR): Campaign identifier
- `NETWORK` (VARCHAR): Network | :lower
- `PARTNER` (VARCHAR): Partner | :lower
- `AD_FORMAT` (VARCHAR): Ad Format | :lower
- `TACTIC` (VARCHAR): Tactic | :lower
- `HMI_AUDIENCE_NAME` (VARCHAR): HMI Audience Name | :upper
- `HMI_BLU` (NUMBER): HMI Blu | :lower
- `CHANNEL_EXECUTION_TYPE` (VARCHAR): Channel Execution Type | :lower
- `CHANNEL_EXECUTION_ID` (VARCHAR): Channel Execution ID | :lower
- `CHANNEL_EXECUTION_NAME` (VARCHAR): Channel Execution Name | :lower
- `R_MEDIACOST` (FLOAT): Media cost metric
- `R_IMPRESSIONS` (FLOAT): Impressions metric
- `R_CLICKS` (FLOAT): Clicks metric
- `R_ENGAGEMENTS` (INTEGER): Engagements metric
- `R_VIDEOSTARTS` (INTEGER): Video starts metric
- `R_VIDEO25` (INTEGER): Video 25% completion metric
- `R_VIDEO50` (INTEGER): Video 50% completion metric
- `R_VIDEO75` (INTEGER): Video 75% completion metric
- `R_VIDEOCOMPLETES` (INTEGER): Video completion metric
- `R_VIEWS` (INTEGER): Views metric

## Table Relationships
*No explicit relationships defined - this is synthetic data for testing purposes*

## Business Context
This table contains IO Tech measurement data for campaign delivery performance tracking. The data supports campaign analysis, optimization, and reporting across various channels and tactics.

## Notes
- Data generated with statistical perturbation applied to protect source data privacy
- Row count in synthetic data: Variable (default 1000 rows for testing)
- Original table row count: 3736350
- Generation timestamp: 2025-09-01 13:25:31
- Statistical distributions have been modified to ensure synthetic data doesn't exactly replicate original patterns
