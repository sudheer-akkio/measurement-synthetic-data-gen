# Table Name: CAMPAIGN_KPI

## Table Description
Campaign measurement KPI fact table containing performance metrics (CTR, CPC, CPM, ROAS) with numerator/denominator pairs, benchmark comparisons, media delivery metrics, and video engagement data at the campaign-creative-channel-date grain. Serves as the primary source for KPI trending, benchmark comparison, creative performance analysis, and channel-level optimization. | :short-name:kpi :measurement:kpi: |<

## Data Dictionary

### Fields:

- `DATE` (DATE): Calendar date of the campaign performance record. Business purpose: primary time dimension for trend analysis, pacing, and period-over-period comparisons.
- `BRAND` (VARCHAR): Brand identifier. Single value in dataset. | :lower :all-unique-values
- `LOB` (VARCHAR): Line of Business identifier. Single value in dataset. | :lower :all-unique-values
- `CHANNEL` (VARCHAR): Top-level media delivery channel (e.g., SEARCH, PROGRAMMATIC, DIGITAL AUDIO, SOCIAL). Business purpose: primary channel segmentation for budget allocation and performance comparison. | :upper :all-unique-values
- `SUB_CHANNEL` (VARCHAR): Sub-channel within the parent channel (e.g., DISPLAY, CONNECTED TV, SHOPPING LIA, IN-APP). Business purpose: granular channel breakout for tactic-level analysis. | :upper
- `REPORTING_CHANNEL` (VARCHAR): Channel used for reporting rollups (e.g., SEARCH, PROGRAMMATIC, DIGITAL AUDIO). Business purpose: standardized channel grouping for executive-level reporting. | :upper :all-unique-values
- `OBJECTIVE` (VARCHAR): Campaign objective (AWARENESS, CONVERSION, ENGAGEMENT, TRAFFIC). Business purpose: segments campaigns by marketing goal for objective-level performance analysis. | :upper :all-unique-values
- `FUNNEL` (VARCHAR): Marketing funnel stage (UPPER FUNNEL, MID FUNNEL, LOWER FUNNEL, FULL FUNNEL). Business purpose: maps campaigns to funnel position for full-funnel attribution and budget allocation analysis. | :upper :all-unique-values
- `CAMPAIGN` (VARCHAR): Campaign name identifier encoding fiscal year, flight type, channel, and campaign type (e.g., IO_Tech_FY24_IF_SEARCH_SEASONAL). Business purpose: primary campaign dimension for campaign-level performance tracking.
- `NETWORK` (VARCHAR): Advertising network or platform (e.g., google, bing, meta, linkedin, tiktok). Business purpose: identifies the ad platform for network-level performance benchmarking. | :lower :all-unique-values
- `PARTNER` (VARCHAR): Media partner or vendor (e.g., google, bing, dv360, meta). Business purpose: identifies the buying partner for partner-level spend and performance analysis. | :lower :all-unique-values
- `AD_FORMAT` (VARCHAR): Advertisement format type (e.g., static, native, rich media, podcast, carousel). Business purpose: segments performance by creative format for format optimization. | :lower :all-unique-values
- `TACTIC` (VARCHAR): Marketing tactic (e.g., display prospecting, audio streaming, native retargeting, video prospecting). Business purpose: identifies the targeting/delivery tactic for tactic-level optimization. | :lower
- `CREATIVE_NAME` (VARCHAR): Creative asset name in mixed case (e.g., Fall_Static_Collection_V1). Business purpose: human-readable creative identifier for creative performance reporting.
- `CREATIVE` (VARCHAR): Creative identifier in uppercase (e.g., FALL_STATIC_COLLECTION_V1). Business purpose: standardized creative key for joining and grouping. | :upper
- `CREATIVE_GROUP` (VARCHAR): Creative group classification in uppercase (e.g., SPRING_STATIC_PRODUCT_V2). Business purpose: groups related creatives for aggregate creative strategy analysis. | :upper
- `CREATIVE_CONCEPT` (VARCHAR): Creative concept description in uppercase (e.g., SPRING_ANIMATED_LIFESTYLE_V2). Business purpose: identifies the creative concept/theme for concept-level performance comparison. | :upper
- `CHANNEL_EXECUTION_ID` (VARCHAR): Unique channel execution identifier (e.g., CE_00092). Business purpose: links to the specific media execution/placement for execution-level reporting. | :upper
- `CHANNEL_EXECUTION_NAME` (VARCHAR): Channel execution name encoding tactic, targeting, product category, and region (e.g., PROG_Prospecting_In_Market_Bedroom_Southwest). Business purpose: descriptive execution label for reporting.
- `AUDIENCE_ID` (VARCHAR): Unique audience segment identifier (e.g., AUD_001). Business purpose: links to the targeted audience segment for audience-level KPI analysis and cross-table joins. | :upper
- `AUDIENCE_NAME` (VARCHAR): Descriptive audience segment name (e.g., Tech Enthusiasts - Early Adopters). Business purpose: human-readable audience label for reporting and segmentation analysis. Assigned based on campaign objective, funnel stage, tactic, and channel.
- `KPI` (VARCHAR): Key Performance Indicator type (CTR, CPC, CPM, ROAS). Business purpose: identifies which KPI the row's numerator/denominator represent; use to filter or pivot by KPI type. | :upper :all-unique-values
- `BENCHMARK_VALUE` (VARCHAR): Benchmark flag value (0 or 1). Business purpose: indicates whether a benchmark comparison value exists for this KPI row. | :all-unique-values
- `KPI_CLASS` (VARCHAR): KPI classification category (Primary). Business purpose: classifies the KPI tier for prioritized reporting. | :all-unique-values
- `KPI_N` (FLOAT): KPI numerator value. Business purpose: numerator for computing the KPI rate/ratio (e.g., clicks for CTR, revenue for ROAS). Typical usage: divide KPI_N by KPI_D to compute the KPI value.
- `KPI_D` (FLOAT): KPI denominator value. Business purpose: denominator for computing the KPI rate/ratio (e.g., impressions for CTR, spend for ROAS).
- `BM_D` (INTEGER): Benchmark denominator value. Business purpose: denominator for computing the benchmark KPI rate for comparison.
- `BM_N` (INTEGER): Benchmark numerator value. Business purpose: numerator for computing the benchmark KPI rate for comparison.
- `R_MEDIACOST` (FLOAT): Media cost in dollars. Business purpose: primary spend metric for budget tracking, pacing, and ROI calculation.
- `R_IMPRESSIONS` (FLOAT): Number of ad impressions served. Business purpose: reach metric and denominator for CPM and CTR calculations.
- `R_CLICKS` (FLOAT): Number of ad clicks. Business purpose: engagement metric; numerator for CTR; denominator for CPC.
- `R_ENGAGEMENTS` (INTEGER): Number of engagements (likes, shares, comments, etc.). Business purpose: social/interactive engagement metric.
- `R_VIDEOSTARTS` (INTEGER): Number of video play starts. Business purpose: video ad initiation metric; denominator for video completion rate. Only meaningful for video-capable channels.
- `R_VIDEO25` (INTEGER): Number of 25% video completions. Business purpose: early-funnel video engagement metric for quartile drop-off analysis.
- `R_VIDEO50` (INTEGER): Number of 50% video completions. Business purpose: mid-funnel video engagement metric.
- `R_VIDEO75` (INTEGER): Number of 75% video completions. Business purpose: late-funnel video engagement metric.
- `R_VIDEOCOMPLETES` (INTEGER): Number of 100% video completions. Business purpose: primary video performance KPI; divide by R_VIDEOSTARTS for Video Completion Rate (VCR).
- `R_VIEWS` (INTEGER): Number of qualified ad views. Business purpose: viewability/awareness metric.

## Table Relationships
- Shares dimensional columns (BRAND, LOB, CHANNEL, CAMPAIGN, PARTNER, AUDIENCE_ID) with CAMPAIGN_DELIVERY and CAMPAIGN_PACING tables for cross-table analysis.
- CHANNEL_EXECUTION_ID can be used to join with CAMPAIGN_DELIVERY for combining KPI and delivery-level metrics.
- AUDIENCE_ID enables audience-level KPI analysis across all three campaign tables.

## Business Context
This is the primary KPI fact table for the IO Tech campaign measurement platform. It stores pre-computed KPI numerator/denominator pairs (KPI_N, KPI_D) alongside benchmark values (BM_N, BM_D), enabling KPI computation, benchmark comparison, and trend analysis. The table also carries raw media delivery metrics (R_MEDIACOST, R_IMPRESSIONS, R_CLICKS, video quartile completions) for holistic campaign performance assessment. Data spans multiple fiscal years across 11 channels, 4 objectives, 4 funnel stages, and approximately 160 campaigns.

## Notes
- Data generated with statistical perturbation applied to protect source data privacy
- Row count in synthetic data: Variable (default 1000 rows for testing)
- Original table row count: 5000000
- Generation timestamp: 2025-09-01 13:24:25
- Statistical distributions have been modified to ensure synthetic data doesn't exactly replicate original patterns
- BRAND and LOB columns contain lowercase underscore-separated values (e.g., io_tech)
- CAMPAIGN values use mixed case with underscores (e.g., IO_Tech_FY24_IF_SEARCH_SEASONAL)
- CREATIVE_NAME uses mixed/title case; CREATIVE, CREATIVE_GROUP, and CREATIVE_CONCEPT use uppercase
- Video metrics (R_VIDEOSTARTS through R_VIDEOCOMPLETES) may be zero for non-video channels
