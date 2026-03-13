# Table Name: CAMPAIGN_PACING

## Table Description
Campaign pacing fact table containing daily planned spend budgets alongside actual media cost and impressions at the campaign-partner-budget-date grain. Serves as the primary source for budget pacing analysis, spend-vs-plan tracking, and budget utilization reporting. | :short-name:pacing :measurement:pacing: |<

## Data Dictionary

### Fields:

- `DATE` (DATE): Calendar date of the pacing record. Business purpose: primary time dimension for daily pacing analysis and spend-vs-plan trending.
- `BRAND` (VARCHAR): Brand identifier. Single value in dataset. | :lower :all-unique-values
- `LOB` (VARCHAR): Line of Business identifier. Single value in dataset. | :lower :all-unique-values
- `CHANNEL` (VARCHAR): Top-level media delivery channel (e.g., PROGRAMMATIC, SOCIAL, EMAIL, AFFILIATE). Business purpose: primary channel segmentation for budget pacing analysis. | :upper :all-unique-values
- `CAMPAIGN` (VARCHAR): Campaign name identifier encoding fiscal year, flight type, channel, and campaign type (e.g., IO_Tech_FY23_OF_PROG_BAU). Business purpose: primary campaign dimension for campaign-level pacing tracking.
- `PARTNER` (VARCHAR): Media partner or vendor (e.g., google, tiktok, snapchat, linkedin). Business purpose: identifies the buying partner for partner-level pacing analysis. | :lower :all-unique-values
- `DNU` (VARCHAR): Active/inactive status flag (ACTIVE). Business purpose: indicates whether the budget line is active for pacing. | :upper :all-unique-values
- `BUDGET_ID` (VARCHAR): Unique budget identifier (e.g., BUD_00001). Business purpose: links to the specific budget allocation for budget-level tracking. | :upper
- `BUDGET_NAME` (VARCHAR): Budget name with pipe-delimited segments (e.g., Search | BAU | WK01). Business purpose: human-readable budget label encoding channel, campaign type, and week for reporting.
- `REPORTING_CHANNEL` (VARCHAR): Channel used for reporting rollups in title case (e.g., Affiliate, Digital Audio, Performance Video). Business purpose: standardized channel grouping for executive reporting. Note: uses title case unlike CHANNEL column.
- `BUYING_CHANNEL` (VARCHAR): Channel used for media buying in title case (e.g., Affiliate, Programmatic, Social). Business purpose: identifies the buying channel for procurement-level analysis. Note: uses title case unlike CHANNEL column.
- `ESTIMATE_ID` (VARCHAR): Estimate/budget line identifier (e.g., BUD_00001). Business purpose: links to the specific budget estimate for estimate-level tracking. | :upper
- `ESTIMATE_NAME` (VARCHAR): Estimate name with pipe-delimited segments (e.g., Search | BAU | WK01). Business purpose: human-readable estimate label for reporting.
- `AUDIENCE_ID` (VARCHAR): Unique audience segment identifier (e.g., AUD_001). Business purpose: links to the targeted audience segment for audience-level pacing analysis and cross-table joins. | :upper
- `AUDIENCE_NAME` (VARCHAR): Descriptive audience segment name (e.g., Tech Enthusiasts - Early Adopters). Business purpose: human-readable audience label for reporting and segmentation analysis. Assigned based on campaign and channel context.
- `DMA_CODE` (VARCHAR): DMA market code (e.g., 501, 803). Business purpose: geographic market segmentation for market-level budget pacing analysis. | :upper
- `DMA_NAME` (VARCHAR): DMA market name (e.g., NEW YORK, CHICAGO, LOS ANGELES). Business purpose: human-readable market label for geographic pacing reporting. | :upper
- `STATE` (VARCHAR): US state 2-letter code (e.g., NY, CA, TX). Business purpose: state-level geographic segmentation for regional pacing analysis. | :upper
- `REGION` (VARCHAR): US census region (NORTHEAST, SOUTHEAST, MIDWEST, SOUTHWEST, WEST). Business purpose: high-level regional pacing comparison and budget allocation analysis. | :upper :all-unique-values
- `LATITUDE` (FLOAT): DMA centroid latitude coordinate. Business purpose: enables geographic map visualizations of pacing data.
- `LONGITUDE` (FLOAT): DMA centroid longitude coordinate. Business purpose: enables geographic map visualizations of pacing data.
- `DAILY_PLANNED_SPEND` (FLOAT): Daily planned/budgeted spend amount in dollars. Business purpose: the target daily spend used as the baseline for pacing analysis. Typical usage: compare to R_MEDIACOST for pacing ratio (actual/planned).
- `SOURCE` (VARCHAR): Data source identifier (CUSTOM). Single value in dataset. | :upper :all-unique-values
- `R_MEDIACOST` (FLOAT): Actual media cost in dollars. Business purpose: actual spend metric for pacing comparison against DAILY_PLANNED_SPEND.
- `R_IMPRESSIONS` (FLOAT): Number of ad impressions served. Business purpose: delivery volume metric for impression-based pacing.

## Table Relationships
- Shares dimensional columns (BRAND, LOB, CHANNEL, CAMPAIGN, PARTNER, AUDIENCE_ID) with CAMPAIGN_DELIVERY and CAMPAIGN_KPI tables for cross-table analysis.
- DAILY_PLANNED_SPEND vs R_MEDIACOST enables pacing ratio computation for budget utilization tracking.
- AUDIENCE_ID enables audience-level pacing analysis across all three campaign tables.
- DMA_CODE, STATE, and REGION are shared across all three campaign tables for geographic pacing analysis.

## Business Context
This is the primary pacing fact table for the IO Tech campaign measurement platform. It tracks daily planned spend budgets alongside actual media cost and impressions, enabling spend-vs-plan pacing analysis, budget utilization monitoring, and forecast-to-actual comparison. The table supports weekly budget tracking (via BUDGET_NAME week encoding), partner-level pacing, and channel-level budget allocation analysis. Data spans multiple fiscal years across 11 channels, approximately 160 campaigns, and 200+ budget lines.

## Notes
- Data generated with statistical perturbation applied to protect source data privacy
- Row count in synthetic data: Variable (default 1000 rows for testing)
- Original table row count: 1048570
- Generation timestamp: 2025-09-01 13:24:40
- Statistical distributions have been modified to ensure synthetic data doesn't exactly replicate original patterns
- BRAND and LOB columns contain lowercase underscore-separated values (e.g., io_tech)
- CAMPAIGN values use mixed case with underscores (e.g., IO_Tech_FY23_OF_PROG_BAU) — no case transformation tag applied
- REPORTING_CHANNEL and BUYING_CHANNEL use title case (e.g., Digital Audio, Performance Video) unlike the uppercase CHANNEL column — no case transformation tag applied since they are neither fully upper nor fully lower
- BUDGET_NAME and ESTIMATE_NAME use pipe-delimited mixed case (e.g., Search | BAU | WK01) — no case transformation tag applied
- BUDGET_ID and ESTIMATE_ID share the same value space (BUD_XXXXX format)
