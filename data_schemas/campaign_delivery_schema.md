# Table Name: CAMPAIGN_DELIVERY

## Table Description
Campaign delivery fact table containing raw media performance metrics (spend, impressions, clicks, video engagement) at the campaign-channel-tactic-execution-date grain. Serves as the primary source for delivery tracking, spend analysis, channel performance, and video engagement reporting. | :short-name:delivery :measurement:delivery: |<

## Data Dictionary

### Fields:

- `DATE` (DATE): Calendar date of the delivery record. Business purpose: primary time dimension for daily delivery tracking, pacing analysis, and trend reporting.
- `BRAND` (VARCHAR): Brand identifier. Single value in dataset. | :lower :all-unique-values
- `LOB` (VARCHAR): Line of Business identifier. Single value in dataset. | :lower :all-unique-values
- `CHANNEL` (VARCHAR): Top-level media delivery channel (e.g., SEARCH, PROGRAMMATIC, DIGITAL AUDIO, SOCIAL, VIDEO). Business purpose: primary channel segmentation for delivery analysis and budget allocation. | :upper :all-unique-values
- `SUB_CHANNEL` (VARCHAR): Sub-channel within the parent channel (e.g., DISPLAY, CONNECTED TV, SHOPPING LIA, DEMAND GEN). Business purpose: granular channel breakout for tactic-level delivery analysis. | :upper
- `REPORTING_CHANNEL` (VARCHAR): Channel used for reporting rollups (e.g., SEARCH, PROGRAMMATIC, DIGITAL AUDIO). Business purpose: standardized channel grouping for executive reporting. | :upper :all-unique-values
- `OBJECTIVE` (VARCHAR): Campaign objective (AWARENESS, CONVERSION, ENGAGEMENT, TRAFFIC). Business purpose: segments delivery by marketing goal for objective-level analysis. | :upper :all-unique-values
- `FUNNEL` (VARCHAR): Marketing funnel stage (UPPER FUNNEL, MID FUNNEL, LOWER FUNNEL, FULL FUNNEL). Business purpose: maps campaigns to funnel position for full-funnel analysis. | :upper :all-unique-values
- `CAMPAIGN` (VARCHAR): Campaign name identifier encoding fiscal year, flight type, channel, and campaign type (e.g., IO_Tech_FY23_OF_AUDIO_SEASONAL). Business purpose: primary campaign dimension for campaign-level delivery tracking.
- `NETWORK` (VARCHAR): Advertising network or platform (e.g., google, bing, meta, programmatic). Business purpose: identifies the ad platform for network-level delivery analysis. | :lower :all-unique-values
- `PARTNER` (VARCHAR): Media partner or vendor (e.g., google, bing, dv360, spotify, pinterest). Business purpose: identifies the buying partner for partner-level spend and delivery analysis. | :lower :all-unique-values
- `AD_FORMAT` (VARCHAR): Advertisement format type (e.g., audio, carousel, native, static, yt video). Business purpose: segments delivery by creative format. | :lower :all-unique-values
- `TACTIC` (VARCHAR): Marketing tactic (e.g., display retargeting, audio streaming, nonbrand, brand). Business purpose: identifies the targeting/delivery tactic for tactic-level optimization. | :lower
- `CHANNEL_EXECUTION_TYPE` (VARCHAR): Type of channel execution identifier (e.g., ADSET_ID, CAMPAIGN_ID, akkio_PID). Business purpose: indicates the ID type used for the execution record. Values are mixed case.
- `CHANNEL_EXECUTION_ID` (VARCHAR): Unique channel execution identifier (e.g., CE_00001). Business purpose: links to the specific media execution/placement for execution-level reporting. | :upper
- `CHANNEL_EXECUTION_NAME` (VARCHAR): Channel execution name encoding tactic, targeting, product category, and region (e.g., PROG_Awareness_In_Market_Home_Northeast). Business purpose: descriptive execution label for reporting. Values are mixed case with underscores.
- `AUDIENCE_ID` (VARCHAR): Unique audience segment identifier (e.g., AUD_001). Business purpose: links to the targeted audience segment for audience-level performance analysis and cross-table joins. | :upper
- `AUDIENCE_NAME` (VARCHAR): Descriptive audience segment name (e.g., Tech Enthusiasts - Early Adopters). Business purpose: human-readable audience label for reporting and segmentation analysis. Assigned based on campaign objective, funnel stage, tactic, and channel.
- `R_MEDIACOST` (FLOAT): Media cost in dollars. Business purpose: primary spend metric for budget tracking, pacing, and ROI calculation.
- `R_IMPRESSIONS` (FLOAT): Number of ad impressions served. Business purpose: reach metric; denominator for CPM and CTR calculations.
- `R_CLICKS` (FLOAT): Number of ad clicks. Business purpose: engagement metric; numerator for CTR calculation.
- `R_ENGAGEMENTS` (INTEGER): Number of engagements (likes, shares, comments, etc.). Business purpose: social/interactive engagement metric.
- `R_VIDEOSTARTS` (INTEGER): Number of video play starts. Business purpose: video ad initiation metric; denominator for video completion rate. Only meaningful for video-capable channels.
- `R_VIDEO25` (INTEGER): Number of 25% video completions. Business purpose: early-funnel video engagement metric for quartile drop-off analysis.
- `R_VIDEO50` (INTEGER): Number of 50% video completions. Business purpose: mid-funnel video engagement metric.
- `R_VIDEO75` (INTEGER): Number of 75% video completions. Business purpose: late-funnel video engagement metric.
- `R_VIDEOCOMPLETES` (INTEGER): Number of 100% video completions. Business purpose: primary video performance KPI; divide by R_VIDEOSTARTS for Video Completion Rate (VCR).
- `R_VIEWS` (INTEGER): Number of qualified ad views. Business purpose: viewability/awareness metric.

## Table Relationships
- Shares dimensional columns (BRAND, LOB, CHANNEL, CAMPAIGN, PARTNER, CHANNEL_EXECUTION_ID, AUDIENCE_ID) with CAMPAIGN_KPI for combining delivery and KPI data.
- Shares dimensional columns (BRAND, LOB, CHANNEL, CAMPAIGN, PARTNER, AUDIENCE_ID) with CAMPAIGN_PACING for delivery-vs-planned-spend pacing analysis.
- AUDIENCE_ID enables audience-level performance analysis across all three campaign tables.

## Business Context
This is the primary delivery fact table for the IO Tech campaign measurement platform. It tracks raw media delivery metrics (spend, impressions, clicks, video quartile completions) at a granular execution level. The table supports delivery trend analysis, channel and tactic performance comparison, video engagement reporting, and spend monitoring. Data spans multiple fiscal years across 11 channels, 4 objectives, 4 funnel stages, and approximately 160 campaigns.

## Notes
- Data generated with statistical perturbation applied to protect source data privacy
- Row count in synthetic data: Variable (default 1000 rows for testing)
- Original table row count: 3736350
- Generation timestamp: 2025-09-01 13:25:31
- Statistical distributions have been modified to ensure synthetic data doesn't exactly replicate original patterns
- BRAND and LOB columns contain lowercase underscore-separated values (e.g., io_tech)
- CAMPAIGN values use mixed case with underscores (e.g., IO_Tech_FY23_OF_AUDIO_SEASONAL)
- CHANNEL_EXECUTION_TYPE contains mixed-case values (ADSET_ID, CAMPAIGN_ID, akkio_PID) — no case transformation tag applied
- CHANNEL_EXECUTION_NAME uses mixed case with underscores — no case transformation tag applied
- Video metrics (R_VIDEOSTARTS through R_VIDEOCOMPLETES) may be zero for non-video channels
