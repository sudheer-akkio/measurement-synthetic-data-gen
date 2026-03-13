# IO TECH MEASUREMENT - Client Context

**Last Updated:** 2026-03-12  
**Objective:** Marketing performance analysis across a multi-table campaign measurement schema — supporting delivery tracking, KPI trending, benchmark comparison, budget pacing, creative analysis, cross-channel optimization, and strategic media planning.

---

## 1. TABLES & PURPOSE

### Performance / Measurement Data

| Table | Answers | Key Metrics |
|-------|---------|-------------|
| **CAMPAIGN_DELIVERY** (~3.7M rows) | How is spend, impressions, clicks, and video engagement trending daily across channels, tactics, partners, and executions? | `R_MEDIACOST`, `R_IMPRESSIONS`, `R_CLICKS`, `R_ENGAGEMENTS`, `R_VIDEOSTARTS`, `R_VIDEOCOMPLETES` |
| **CAMPAIGN_KPI** (~5M rows) | How are KPIs (CTR, CPC, CPM, ROAS) performing vs benchmarks across campaigns, creatives, and channels? | `KPI_N`, `KPI_D`, `BM_N`, `BM_D`, `KPI` (type), plus raw delivery metrics |
| **CAMPAIGN_PACING** (~1M rows) | Are campaigns on track vs planned budgets? What is the pacing ratio by partner and channel? | `DAILY_PLANNED_SPEND`, `R_MEDIACOST`, `R_IMPRESSIONS` |

### First Party Data

| Table | Description |
|-------|-------------|
| **Customer_Churn_Data** (CHURN_DATA_IO_TECH_CHARGERS) | Customer engagement data including plan type, device type, monthly fee, charges, support tickets, and engagement scores for churn prediction and customer segmentation |
| **V_IO_TECH_SITE_DATA** | Site-level first-party data |
| **V_IO_TECH_SITE_ACTIVITY_DATA** | Site activity and behavioral data |

### Enrichment Data (Akkio Audience)

| Table | Description |
|-------|-------------|
| **V_AKKO_ATTRIBUTES_LATEST** | Consumer demographic and attribute enrichment data |
| **V_AKKO_AUTO_LATEST** | Auto/vehicle interest and ownership enrichment data |
| **V_AKKO_SMALL_BUSINESS_LATEST** | Small business interest and affinity enrichment data |
| **V_DAILY_AKKO_FACT_BROWSING_SUMMARY** | Daily aggregated browsing behavior summary |
| **V_DAILY_AKKO_FACT_CPG_DETAIL** | Daily CPG (consumer packaged goods) purchase detail |
| **V_DAILY_AKKO_FACT_CPG_SUMMARY** | Daily CPG purchase summary |
| **V_DAILY_AKKO_FACT_MEDIA_DETAIL** | Daily media consumption detail |
| **V_DAILY_AKKO_FACT_MEDIA_SUMMARY** | Daily media consumption summary |
| **V_DAILY_AKKO_FACT_PLACES_VISITED_DETAIL** | Daily location/places visited detail |
| **V_DAILY_AKKO_FACT_PLACES_VISITED_SUMMARY** | Daily location/places visited summary |
| **V_IO_TECH_MEDIA_IN_TAB** | Media interest/in-market signal data |

**Campaign data covers:** Multiple fiscal years (FY23–FY25+), daily granularity.  
**Channels:** ~11 channels including Search, Programmatic, Social, Video, Digital Audio, Email, Affiliate, and more.  
**Campaigns:** ~160 campaigns across 4 objectives and 4 funnel stages.

---

## 2. JOIN MODEL

The three campaign tables share dimensional columns and can be joined for cross-table analysis.

```sql
-- Delivery + KPI: combine raw delivery with KPI numerator/denominator pairs
FROM CAMPAIGN_DELIVERY d
JOIN CAMPAIGN_KPI k
  ON d.CHANNEL_EXECUTION_ID = k.CHANNEL_EXECUTION_ID
  AND d.DATE = k.DATE
  AND d.AUDIENCE_ID = k.AUDIENCE_ID

-- Delivery + Pacing: compare actual delivery against planned budgets
FROM CAMPAIGN_DELIVERY d
JOIN CAMPAIGN_PACING p
  ON d.BRAND = p.BRAND
  AND d.LOB = p.LOB
  AND d.CHANNEL = p.CHANNEL
  AND d.CAMPAIGN = p.CAMPAIGN
  AND d.PARTNER = p.PARTNER
  AND d.DATE = p.DATE
  AND d.AUDIENCE_ID = p.AUDIENCE_ID

-- KPI + Pacing: KPI performance in context of budget utilization
FROM CAMPAIGN_KPI k
JOIN CAMPAIGN_PACING p
  ON k.BRAND = p.BRAND
  AND k.LOB = p.LOB
  AND k.CHANNEL = p.CHANNEL
  AND k.CAMPAIGN = p.CAMPAIGN
  AND k.PARTNER = p.PARTNER
  AND k.DATE = p.DATE
  AND k.AUDIENCE_ID = p.AUDIENCE_ID
```

**Shared join keys across all three tables:** `BRAND`, `LOB`, `CHANNEL`, `CAMPAIGN`, `PARTNER`, `AUDIENCE_ID`, `DMA_CODE`, `STATE`, `REGION`.  
**Delivery ↔ KPI additionally share:** `CHANNEL_EXECUTION_ID` for execution-level joins.

---

## 3. CURRENCY, FORMATTING & DISPLAY RULES

- **`R_MEDIACOST` is in USD.** Always format as currency with `$` prefix and 2 decimal places (e.g., `$1,234.56`).
- **Round all calculated values to 2 decimal places** unless otherwise specified.
- **Add commas to all numeric outputs** in text, tables, and charts (e.g., `1,234,567` not `1234567`).
- **Percentages**: display with `%` suffix, 2 decimal places (e.g., `0.54%`, `3.21%`).
- **CTR / rate formatting**: Ratio formulas like `clicks / impressions` return a raw decimal (e.g., `0.0156`). **Always multiply by 100 before appending `%`** so the display reads `1.56%`, not `0.02%`. This applies to all rate metrics: CTR, CVR, VCR, VTR. Failing to multiply produces values that appear ~100x too low.
- **"Top" or "Best" CPA, CPC, CPM means the LOWEST value** (most cost-efficient), not the highest.

**MAKE SURE TO FOLLOW THIS: When creating a chart with a single trace, always set fig.update_layout(showlegend=True) to prevent Plotly from hiding the legend/tooltip label.**

---

## 4. CALCULATED METRICS & FORMULAS

These metrics are **not stored** in CAMPAIGN_DELIVERY — compute them in SQL or Snowpark. Both numerator and denominator must be non-zero; if the result is 0, Infinity, -Infinity, or NaN, treat it as **invalid** and display the underlying raw values to explain why the calculation failed.

### Column Mapping (CAMPAIGN_DELIVERY)

| Formula Term | Column(s) |
|---|---|
| Cost / Spend | `R_MEDIACOST` (USD) |
| Clicks | `R_CLICKS` |
| Impressions | `R_IMPRESSIONS` |
| Engagements | `R_ENGAGEMENTS` |
| Video Starts | `R_VIDEOSTARTS` (video channels only) |
| Video 25% | `R_VIDEO25` (video channels only) |
| Video 50% | `R_VIDEO50` (video channels only) |
| Video 75% | `R_VIDEO75` (video channels only) |
| Video Completions | `R_VIDEOCOMPLETES` (video channels only) |
| Views | `R_VIEWS` |
| Revenue | **NOT directly in CAMPAIGN_DELIVERY** — use CAMPAIGN_KPI with `KPI = 'ROAS'` |

### Row-Level Formulas (from CAMPAIGN_DELIVERY)

| Metric | Abbreviation | SQL Formula |
|--------|-------------|-------------|
| Click-Through Rate | CTR | `R_CLICKS / NULLIF(R_IMPRESSIONS, 0)` |
| Cost per Click | CPC | `R_MEDIACOST / NULLIF(R_CLICKS, 0)` |
| Cost per 1,000 Impressions | CPM | `(R_MEDIACOST / NULLIF(R_IMPRESSIONS, 0)) * 1000` |
| Cost per Engagement | CPE | `R_MEDIACOST / NULLIF(R_ENGAGEMENTS, 0)` |
| Cost per Video View | CPV | `R_MEDIACOST / NULLIF(R_VIEWS, 0)` |
| Cost per Completed Video View | CPCV | `R_MEDIACOST / NULLIF(R_VIDEOCOMPLETES, 0)` |
| Video Completion Rate | VCR | `R_VIDEOCOMPLETES / NULLIF(R_VIDEOSTARTS, 0)` |
| Video View-Through Rate | VTR | `R_VIDEOCOMPLETES / NULLIF(R_VIEWS, 0)` |

### KPI Table Calculations (CAMPAIGN_KPI)

The CAMPAIGN_KPI table stores pre-computed KPI numerator/denominator pairs. The `KPI` column identifies the metric type, and you compute the value as `KPI_N / NULLIF(KPI_D, 0)`.

| KPI Value | KPI_N represents | KPI_D represents | Computed As |
|-----------|-----------------|-----------------|-------------|
| `CTR` | Clicks | Impressions | `KPI_N / NULLIF(KPI_D, 0)` |
| `CPC` | Cost | Clicks | `KPI_N / NULLIF(KPI_D, 0)` |
| `CPM` | Cost | Impressions (÷1000) | `(KPI_N / NULLIF(KPI_D, 0)) * 1000` |
| `ROAS` | Revenue | Cost | `KPI_N / NULLIF(KPI_D, 0)` |

**Benchmark comparison:** Similarly compute `BM_N / NULLIF(BM_D, 0)` for the benchmark rate, then compare actual vs benchmark.

### Pacing Calculations (CAMPAIGN_PACING)

| Metric | Formula |
|--------|---------|
| Pacing Ratio | `R_MEDIACOST / NULLIF(DAILY_PLANNED_SPEND, 0)` |
| Budget Variance | `R_MEDIACOST - DAILY_PLANNED_SPEND` |
| Budget Utilization % | `(R_MEDIACOST / NULLIF(DAILY_PLANNED_SPEND, 0)) * 100` |
| Cumulative Pacing | `SUM(R_MEDIACOST) OVER (...) / NULLIF(SUM(DAILY_PLANNED_SPEND) OVER (...), 0)` |

### Aggregate / Average Formulas (CRITICAL)

**When computing averages across rows or groups, NEVER average pre-calculated ratios.** Instead, sum numerator and denominator separately, then divide:

| Metric | Correct Aggregate Formula |
|--------|--------------------------|
| Avg CTR | `SUM(R_CLICKS) / NULLIF(SUM(R_IMPRESSIONS), 0)` |
| Avg CPC | `SUM(R_MEDIACOST) / NULLIF(SUM(R_CLICKS), 0)` |
| Avg CPM | `(SUM(R_MEDIACOST) / NULLIF(SUM(R_IMPRESSIONS), 0)) * 1000` |
| Avg CPV | `SUM(R_MEDIACOST) / NULLIF(SUM(R_VIEWS), 0)` |
| Avg CPCV | `SUM(R_MEDIACOST) / NULLIF(SUM(R_VIDEOCOMPLETES), 0)` |
| Avg VCR | `SUM(R_VIDEOCOMPLETES) / NULLIF(SUM(R_VIDEOSTARTS), 0)` |
| Avg KPI (from KPI table) | `SUM(KPI_N) / NULLIF(SUM(KPI_D), 0)` (filter by `KPI` type first) |
| Avg Pacing Ratio | `SUM(R_MEDIACOST) / NULLIF(SUM(DAILY_PLANNED_SPEND), 0)` |

**Example — WRONG vs RIGHT:**
```sql
-- WRONG: averaging pre-computed ratios inflates small-volume rows
SELECT AVG(R_CLICKS / NULLIF(R_IMPRESSIONS, 0)) AS avg_ctr ...

-- RIGHT: sum numerator and denominator, then divide
SELECT SUM(R_CLICKS) / NULLIF(SUM(R_IMPRESSIONS), 0) AS avg_ctr ...
```

### Invalid Value Handling

All ratio/cost calculations require **both numerator and denominator to be non-zero**. In SQL, always wrap denominators with `NULLIF(..., 0)` so that division by zero returns NULL rather than an error. When displaying results:
- If the calculated value is NULL (from a zero denominator), display the raw components instead (e.g., show spend = $500, clicks = 0 rather than CPC = N/A)
- Replace any result that is `numpy.nan`, `numpy.inf`, or `-numpy.inf` with `numpy.nan`

---

## 5. CHANNEL-SPECIFIC METRIC RULES

Certain metric columns are **zero by design** for non-applicable channels. Never flag these as anomalies.

| Metric Group | Non-Zero Only For | Zero For |
|---|---|---|
| `R_VIDEOSTARTS`, `R_VIDEO25`, `R_VIDEO50`, `R_VIDEO75`, `R_VIDEOCOMPLETES` | VIDEO, SOCIAL (video formats), PROGRAMMATIC (CTV sub-channel) | SEARCH, DIGITAL AUDIO, EMAIL, AFFILIATE |
| `R_ENGAGEMENTS` | SOCIAL, channels with interactive formats | SEARCH (typically), DIGITAL AUDIO |
| `R_VIEWS` | Channels with viewable ad formats | Non-viewable channels |

**Video quartile funnel ordering is strict:**
- `R_VIDEOSTARTS >= R_VIDEO25 >= R_VIDEO50 >= R_VIDEO75 >= R_VIDEOCOMPLETES`

**Core funnel ordering:**
- `R_IMPRESSIONS > R_CLICKS > Conversions` (where conversions are available via KPI table)

---

## 6. BUSINESS OBJECTIVES & ANALYSIS PATTERNS

### 6a. Budget Pacing & Utilization
**Question:** "Are we on track with budget? Which campaigns are over/under-spending?"  
**Approach:** Use CAMPAIGN_PACING. Compute daily and cumulative pacing ratio (`R_MEDIACOST / DAILY_PLANNED_SPEND`). Group by `CHANNEL`, `CAMPAIGN`, or `PARTNER`. Flag pacing ratio < 0.85 as UNDERPACING and > 1.15 as OVERPACING. Use `BUDGET_NAME` week encoding (e.g., `WK01`) for weekly budget tracking.

### 6b. KPI Trending & Benchmark Comparison
**Question:** "How is CTR trending vs benchmark?"  
**Approach:** Use CAMPAIGN_KPI. Filter `KPI = 'CTR'`. Compute actual KPI as `KPI_N / NULLIF(KPI_D, 0)` and benchmark as `BM_N / NULLIF(BM_D, 0)`. Aggregate by month or week for trending. Use `BENCHMARK_VALUE` flag to confirm benchmark data exists. Compare actual vs benchmark: `(actual_kpi - benchmark_kpi) / NULLIF(benchmark_kpi, 0)` for % variance.

### 6c. Channel & Partner Performance
**Question:** "Which channels and partners are most efficient?"  
**Approach:** Use CAMPAIGN_DELIVERY. Aggregate `R_MEDIACOST`, `R_CLICKS`, `R_IMPRESSIONS` by `CHANNEL` or `PARTNER`. Compute CPC, CPM, CTR at aggregate level. Cross-reference with CAMPAIGN_KPI for ROAS by channel. Rank by efficiency (lowest CPC/CPM or highest CTR/ROAS).

### 6d. Creative Performance Analysis
**Question:** "Which creatives are performing best?"  
**Approach:** Use CAMPAIGN_KPI which has `CREATIVE_NAME`, `CREATIVE`, `CREATIVE_GROUP`, and `CREATIVE_CONCEPT` columns. Aggregate KPI_N/KPI_D by creative dimensions. Compare CTR, CPC across creative groups and concepts. Identify top-performing and underperforming creatives.

### 6e. Funnel & Objective Analysis
**Question:** "How does performance vary across funnel stages?"  
**Approach:** Group by `FUNNEL` (UPPER FUNNEL, MID FUNNEL, LOWER FUNNEL, FULL FUNNEL) or `OBJECTIVE` (AWARENESS, CONVERSION, ENGAGEMENT, TRAFFIC). Compare spend allocation, efficiency metrics (CPC, CPM), and engagement metrics across funnel stages. Upper funnel campaigns should index toward awareness metrics (impressions, CPM); lower funnel toward conversion metrics (CPA, ROAS).

### 6f. Video Engagement Analysis
**Question:** "How is video content performing?"  
**Approach:** Use CAMPAIGN_DELIVERY. Filter to video-capable channels. Compute video quartile drop-off: `R_VIDEOSTARTS → R_VIDEO25 → R_VIDEO50 → R_VIDEO75 → R_VIDEOCOMPLETES`. Step-to-step completion rates identify where viewers drop off. VCR (`R_VIDEOCOMPLETES / R_VIDEOSTARTS`) is the primary video KPI. Compare across `SUB_CHANNEL`, `AD_FORMAT`, and `TACTIC`.

### 6g. Tactic Optimization
**Question:** "Which tactics are most cost-effective?"  
**Approach:** Use CAMPAIGN_DELIVERY. Group by `TACTIC` (e.g., display retargeting, audio streaming, brand, nonbrand). Compare CPC, CPM, CTR across tactics. Retargeting tactics typically show higher CTR and lower CPA than prospecting tactics. Cross-reference with `OBJECTIVE` to ensure fair comparison (awareness tactics vs conversion tactics).

### 6h. Spend Trend & Anomaly Detection
**Question:** "Flag unusual spend or performance."  
**Approach:** Z-score per channel or campaign: `(value - AVG) / STDDEV`. Flag |z| > 2.5 as anomalous. Types: SPEND SPIKE, SPEND DROP, IMPRESSION SURGE, CTR COLLAPSE. Compare CAMPAIGN_PACING for budget deviation context.

### 6i. Forecasting
**Question:** "Where will performance trend next month?"  
**Approach:** Use trailing 4-week averages from CAMPAIGN_DELIVERY. Apply `REGR_SLOPE` for linear extrapolation by channel. Cross-reference with CAMPAIGN_PACING planned spend for budget-informed projections.

### 6j. Execution-Level Drill Down
**Question:** "What's happening at the execution/placement level?"  
**Approach:** Use `CHANNEL_EXECUTION_ID` and `CHANNEL_EXECUTION_NAME` from CAMPAIGN_DELIVERY or CAMPAIGN_KPI. Execution names encode tactic, targeting, product category, and region (e.g., `PROG_Awareness_In_Market_Home_Northeast`). Parse these for segment-level insights.

### 6k. Audience Performance Analysis
**Question:** "Which audiences are performing best? How does campaign performance vary by audience?"  
**Approach:** Use `AUDIENCE_ID` and `AUDIENCE_NAME` columns available in all three campaign tables. Group by `AUDIENCE_ID` or `AUDIENCE_NAME` to compare spend, efficiency (CPC, CPM, CTR), and KPI performance across audience segments. Cross-reference with CAMPAIGN_KPI for audience-level ROAS and benchmark comparison. Combine with CAMPAIGN_PACING for audience-level budget utilization. Audience segments are correlated to campaign objective, funnel stage, tactic, and channel — retargeting tactics map to retargeting audiences while upper-funnel awareness campaigns map to broad prospecting audiences. There are 20 audience segments spanning upper-funnel (e.g., Tech Enthusiasts, Smart Home Adopters), mid-funnel (e.g., In-Market Shoppers, Device Upgraders), lower-funnel (e.g., Site Visitors Retargeting, Lookalike), and cross-funnel (e.g., CTV Cord Cutters, College Students) segments.

### 6l. Geographic Performance Analysis
**Question:** "How does performance vary by region, state, or DMA market? Which markets are most efficient?"  
**Approach:** Use `REGION`, `STATE`, `DMA_CODE`, and `DMA_NAME` columns available in all three campaign tables. Group by `REGION` for high-level regional comparison (NORTHEAST, SOUTHEAST, MIDWEST, SOUTHWEST, WEST). Group by `STATE` for state-level analysis (50 US states). Group by `DMA_CODE, DMA_NAME` for market-level granularity (~186 DMAs). Compare spend allocation, efficiency metrics (CPC, CPM, CTR), and KPI performance across geographies. Use `LATITUDE` and `LONGITUDE` (DMA centroid coordinates) for map-based visualizations — scatter/bubble maps showing spend, impressions, or efficiency by market. Cross-dimensional analysis is supported: performance by region x channel, state x funnel, DMA x partner, etc. In CAMPAIGN_DELIVERY and CAMPAIGN_KPI, location is correlated to the region suffix in `CHANNEL_EXECUTION_NAME` (e.g., rows with `_Northeast` execution names are assigned to Northeast states/DMAs). Use CAMPAIGN_PACING for geographic budget pacing: compare `R_MEDIACOST` vs `DAILY_PLANNED_SPEND` by region or state.

---

## 7. DEFAULTS & INTENT MAPPING

**Defaults when unspecified:**
- Time range: full available date range
- Granularity: daily is available; aggregate to weekly or monthly as the question requires
- KPI table: when computing CTR, CPC, CPM, or ROAS and the user does not specify a source, prefer `CAMPAIGN_KPI` with the appropriate `KPI` filter for accuracy
- Pacing: use `CAMPAIGN_PACING` whenever budget or plan comparisons are mentioned
- Delivery metrics: use `CAMPAIGN_DELIVERY` for raw spend, impressions, clicks, video metrics

| User Says | Translate To |
|-----------|-------------|
| "by channel" / "by media type" | `GROUP BY CHANNEL` |
| "by sub-channel" | `GROUP BY SUB_CHANNEL` |
| "by reporting channel" | `GROUP BY REPORTING_CHANNEL` |
| "by partner" / "by platform" / "by vendor" | `GROUP BY PARTNER` |
| "by network" | `GROUP BY NETWORK` |
| "by campaign" | `GROUP BY CAMPAIGN` |
| "by tactic" | `GROUP BY TACTIC` |
| "by ad format" / "by format" | `GROUP BY AD_FORMAT` |
| "by objective" / "by goal" | `GROUP BY OBJECTIVE` |
| "by funnel" / "by funnel stage" | `GROUP BY FUNNEL` |
| "by audience" / "by audience segment" | `GROUP BY AUDIENCE_ID, AUDIENCE_NAME` |
| "by creative" / "by creative name" | `GROUP BY CREATIVE_NAME` (CAMPAIGN_KPI only) |
| "by creative group" | `GROUP BY CREATIVE_GROUP` (CAMPAIGN_KPI only) |
| "by creative concept" | `GROUP BY CREATIVE_CONCEPT` (CAMPAIGN_KPI only) |
| "by execution" / "by placement" | `GROUP BY CHANNEL_EXECUTION_ID, CHANNEL_EXECUTION_NAME` |
| "by region" / "by geography" | `GROUP BY REGION` |
| "by state" | `GROUP BY STATE` |
| "by DMA" / "by market" / "by metro" | `GROUP BY DMA_CODE, DMA_NAME` |
| "map" / "geographic map" / "show on map" | Use `LATITUDE`, `LONGITUDE` for scatter/bubble map plotting |
| "by budget" / "by budget line" | `GROUP BY BUDGET_ID, BUDGET_NAME` (CAMPAIGN_PACING only) |
| "trend" / "over time" / "weekly" | `GROUP BY DATE_TRUNC('week', DATE)` |
| "monthly" / "by month" | `GROUP BY DATE_TRUNC('month', DATE)` |
| "daily" / "by day" | `GROUP BY DATE` |
| "last week" / "this week" | Filter on `DATE` relative to max date in data |
| "YTD" / "year-to-date" | Cumulative window function over months |
| "vs benchmark" / "vs goal" | Use `BM_N / BM_D` in CAMPAIGN_KPI |
| "pacing" / "vs plan" / "vs budget" | Use CAMPAIGN_PACING: `R_MEDIACOST` vs `DAILY_PLANNED_SPEND` |
| "video" / "completion rate" / "VCR" | Video quartile metrics from CAMPAIGN_DELIVERY |
| "anomaly" / "spike" / "alert" | Z-score analysis on spend/impressions/CTR (see Section 6h) |
| "forecast" / "project" / "next month" | Trailing 4-week regression extrapolation (see Section 6i) |
| "spend" / "cost" / "media cost" | `R_MEDIACOST` |
| "medium" | Interpret as "platform" or "channel" |

---

## 8. DIMENSIONAL VALUES REFERENCE

### CAMPAIGN_DELIVERY & CAMPAIGN_KPI (shared dimensions)

| Dimension | Values | Case |
|-----------|--------|------|
| `BRAND` | io_tech | lowercase |
| `LOB` | io_tech | lowercase |
| `CHANNEL` | SEARCH, PROGRAMMATIC, DIGITAL AUDIO, SOCIAL, VIDEO, EMAIL, AFFILIATE, and more (~11 total) | UPPERCASE |
| `SUB_CHANNEL` | DISPLAY, CONNECTED TV, SHOPPING LIA, DEMAND GEN, IN-APP, and more | UPPERCASE |
| `REPORTING_CHANNEL` | SEARCH, PROGRAMMATIC, DIGITAL AUDIO, and more | UPPERCASE |
| `OBJECTIVE` | AWARENESS, CONVERSION, ENGAGEMENT, TRAFFIC | UPPERCASE |
| `FUNNEL` | UPPER FUNNEL, MID FUNNEL, LOWER FUNNEL, FULL FUNNEL | UPPERCASE |
| `NETWORK` | google, bing, meta, programmatic, linkedin, tiktok, and more | lowercase |
| `PARTNER` | google, bing, dv360, meta, spotify, pinterest, snapchat, tiktok, linkedin, and more | lowercase |
| `AD_FORMAT` | audio, carousel, native, static, yt video, rich media, podcast, and more | lowercase |
| `TACTIC` | display retargeting, display prospecting, audio streaming, nonbrand, brand, native retargeting, video prospecting, and more | lowercase |

### Audience Dimensions (shared across all three tables)

| Dimension | Values | Case |
|-----------|--------|------|
| `AUDIENCE_ID` | AUD_001 through AUD_020 (20 segments) | UPPERCASE |
| `AUDIENCE_NAME` | Tech Enthusiasts - Early Adopters, Smart Home Adopters, Young Professionals 25-34, Eco-Conscious Consumers, Mobile Accessory In-Market Shoppers, Device Upgraders - Active Researchers, Business Tech Decision Makers, Connected Home - Parents with Kids, Site Visitors - Retargeting, Cart Abandoners - Retargeting, Lookalike - High-Value Customers, Past Purchasers - Cross-Sell, Competitor Brand Conquesting, Frequent Online Shoppers, High-Income HH 100K+, College Students and Gen Z, Outdoor and Active Lifestyle, CTV Cord Cutters, Gaming and Entertainment Enthusiasts, Auto Enthusiasts - In-Vehicle Tech | Mixed case |

Audience segments are correlated to campaign dimensions: retargeting tactics map to retargeting audiences (AUD_009, AUD_010, AUD_012); upper-funnel/awareness campaigns map to broad audiences (AUD_001–AUD_004); lower-funnel/conversion campaigns map to intent audiences (AUD_005, AUD_006, AUD_011, AUD_013, AUD_014); audio channels favor AUD_017/AUD_019; social channels favor AUD_003/AUD_016; video/CTV channels favor AUD_018/AUD_019.

### Geographic Dimensions (shared across all three tables)

| Dimension | Values | Case |
|-----------|--------|------|
| `DMA_CODE` | ~186 DMA market codes (e.g., 501 = New York, 803 = Los Angeles) | Numeric string |
| `DMA_NAME` | NEW YORK, CHICAGO, LOS ANGELES, DALLAS-FT. WORTH, PHILADELPHIA, and ~181 more | UPPERCASE |
| `STATE` | All 50 US states as 2-letter codes (NY, CA, TX, FL, IL, etc.) | UPPERCASE |
| `REGION` | NORTHEAST, SOUTHEAST, MIDWEST, SOUTHWEST, WEST | UPPERCASE |
| `LATITUDE` | DMA centroid latitude (e.g., 40.7128 for New York area) | Numeric |
| `LONGITUDE` | DMA centroid longitude (e.g., -74.0060 for New York area) | Numeric |

Location is correlated to existing data: in CAMPAIGN_DELIVERY and CAMPAIGN_KPI, DMA/state/region are derived from the region suffix in `CHANNEL_EXECUTION_NAME` (e.g., rows with `_Northeast` are assigned to Northeast states and DMAs). In CAMPAIGN_PACING, location is assigned via weighted random from the full DMA pool.

### CAMPAIGN_KPI (additional creative dimensions)

| Dimension | Example Values | Case |
|-----------|---------------|------|
| `CREATIVE_NAME` | Fall_Static_Collection_V1 | Mixed case |
| `CREATIVE` | FALL_STATIC_COLLECTION_V1 | UPPERCASE |
| `CREATIVE_GROUP` | SPRING_STATIC_PRODUCT_V2 | UPPERCASE |
| `CREATIVE_CONCEPT` | SPRING_ANIMATED_LIFESTYLE_V2 | UPPERCASE |
| `KPI` | CTR, CPC, CPM, ROAS | UPPERCASE |
| `KPI_CLASS` | Primary | Title case |
| `BENCHMARK_VALUE` | 0, 1 (flag) | — |

### CAMPAIGN_PACING (additional dimensions)

| Dimension | Example Values | Case |
|-----------|---------------|------|
| `REPORTING_CHANNEL` | Affiliate, Digital Audio, Performance Video, Search | Title case |
| `BUYING_CHANNEL` | Affiliate, Programmatic, Social | Title case |
| `BUDGET_NAME` | Search \| BAU \| WK01 | Mixed, pipe-delimited |
| `ESTIMATE_NAME` | Search \| BAU \| WK01 | Mixed, pipe-delimited |
| `DNU` | ACTIVE | UPPERCASE |
| `SOURCE` | CUSTOM | UPPERCASE |

### CAMPAIGN naming convention
Campaign names encode fiscal year, flight type, channel, and campaign type:  
`IO_Tech_FY23_OF_AUDIO_SEASONAL` → Brand: IO Tech, Fiscal Year: FY23, Flight: OF (original flight), Channel: Audio, Type: Seasonal.

---

## 9. NON-CAMPAIGN DATA TABLES

The following tables are available for enrichment, first-party analysis, and cross-dataset exploration. They do not follow the same measurement schema as the campaign tables.

### First Party Data

- **Customer_Churn_Data** (`CHURN_DATA_IO_TECH_CHARGERS`): Customer-level data with `plan_type`, `device_type`, `monthly_fee`, `num_charges`, `support_tickets`, `engagement_score`, and `provider_name`. Used for churn prediction and customer segmentation.
- **V_IO_TECH_SITE_DATA**: Site-level first-party data.
- **V_IO_TECH_SITE_ACTIVITY_DATA**: Site activity and behavioral data.

### Enrichment Data (Akkio Audience Views)

**Static Audience Attributes:**
- **V_AKKO_ATTRIBUTES_LATEST**: Consumer demographic and attribute enrichment (latest snapshot).
- **V_AKKO_AUTO_LATEST**: Auto/vehicle interest and ownership data (latest snapshot).
- **V_AKKO_SMALL_BUSINESS_LATEST**: Small business interest and affinity data (latest snapshot).

**Daily Behavioral Fact Tables:**
- **V_DAILY_AKKO_FACT_BROWSING_SUMMARY**: Aggregated daily browsing behavior.
- **V_DAILY_AKKO_FACT_CPG_DETAIL / CPG_SUMMARY**: Daily consumer packaged goods purchase detail and summary.
- **V_DAILY_AKKO_FACT_MEDIA_DETAIL / MEDIA_SUMMARY**: Daily media consumption detail and summary.
- **V_DAILY_AKKO_FACT_PLACES_VISITED_DETAIL / PLACES_VISITED_SUMMARY**: Daily location visit detail and summary.

**Other:**
- **V_IO_TECH_MEDIA_IN_TAB**: Media interest and in-market signal data.

When listing available datasets to users, group them as:
- **Performance/Measurement Data**: CAMPAIGN_DELIVERY, CAMPAIGN_KPI, CAMPAIGN_PACING
- **First Party Data**: Customer_Churn_Data, V_IO_TECH_SITE_DATA, V_IO_TECH_SITE_ACTIVITY_DATA
- **Enrichment Data**: All V_AKKO_* and V_DAILY_AKKO_* tables, V_IO_TECH_MEDIA_IN_TAB

---

## 10. CONSTRAINTS & HARD BOUNDARIES

- **CAMPAIGN_PACING has different case conventions than DELIVERY/KPI** — `REPORTING_CHANNEL` and `BUYING_CHANNEL` use title case in PACING but uppercase in DELIVERY/KPI. Account for this when joining.
- **Revenue is only available via CAMPAIGN_KPI** — filter `KPI = 'ROAS'` where `KPI_N` = revenue and `KPI_D` = cost. CAMPAIGN_DELIVERY has no revenue column.
- **KPI table rows are per-KPI-type** — each row represents a single KPI (CTR, CPC, CPM, or ROAS). Always filter by `KPI` column before aggregating `KPI_N` / `KPI_D`.
- **Benchmark data may not exist for all rows** — check `BENCHMARK_VALUE` flag (1 = benchmark available) before computing benchmark comparisons.
- **Video metrics are zero for non-video channels** — `R_VIDEOSTARTS` through `R_VIDEOCOMPLETES` being zero for SEARCH, DIGITAL AUDIO, EMAIL, etc. is expected, not missing data.
- **BUDGET_ID and ESTIMATE_ID share the same value space** (`BUD_XXXXX` format) in CAMPAIGN_PACING.
- **CHANNEL_EXECUTION_TYPE values are mixed case** — `ADSET_ID`, `CAMPAIGN_ID`, `akkio_PID` — no uniform case transformation.
- **AUDIENCE_ID and AUDIENCE_NAME are available in all three campaign tables** — use `AUDIENCE_ID` as a join key alongside other shared dimensions. Audiences are correlated to campaign objective, funnel, tactic, and channel (not randomly assigned).
- **Geographic columns (DMA_CODE, DMA_NAME, STATE, REGION, LATITUDE, LONGITUDE) are available in all three campaign tables** — use for geographic performance analysis. `LATITUDE` and `LONGITUDE` represent DMA centroid coordinates (average of all zip codes within the DMA), suitable for map visualizations but not individual-address precision. In DELIVERY and KPI tables, location is correlated to the region suffix in `CHANNEL_EXECUTION_NAME`.
- **Non-campaign tables** (enrichment, first-party) have their own schemas and are not joinable to campaign tables via standard keys unless the user specifies a join strategy.
- Do not fabricate or estimate metrics not present in the data.
- Do not combine data across time periods unless specifically asked.
- For answers that return empty tables or charts, explain why in text being as specific as possible.
- Any reference to the term "medium" should be interpreted as "platform" or "channel."

---

## 11. DATA AVAILABILITY APPROACH

The analysis is strictly limited to the data contained within the schemas described above. If a question requires data outside of these schemas:

- Clearly identify which specific data points are not available in the current dataset
- Suggest alternative approaches using the existing data that might address the underlying need
- Outline what additional data would ideally be needed for a complete analysis
- Propose creative proxies or workarounds using available fields when possible
- Do not attempt to answer questions requiring unavailable data
- If asked about a metric that requires specific columns not present, DO NOT interpret the calculation automatically — ask for clarification

---

## 12. RESPONSE FORMAT GUIDELINES

1. **Default to Textual Analysis** — provide a written explanation or summary of insights when answering queries. If appropriate, output both a table and a chart in addition to the written explanation.
2. **Use Clear Metrics** — define metrics clearly when reporting. Report percentages to one decimal place for clarity.
3. **Handle Ambiguities** — if a query is incomplete or unclear (e.g., unspecified time period), seek clarification.
4. **Always use the most recent data available** at any reference point when looking back over a timeframe.
5. **Be precise about time periods.**
6. **Do not aggregate or combine data across time periods** unless specifically asked.

---

## 13. QUALITY CHECKLIST

Before finalizing any query or output:

**Query correctness:**
- [ ] Correct table used for the question (DELIVERY for raw metrics, KPI for KPI rates/benchmarks, PACING for budget tracking)
- [ ] `NULLIF(..., 0)` used in all division denominators
- [ ] `KPI` column filtered when using CAMPAIGN_KPI (e.g., `WHERE KPI = 'CTR'`)
- [ ] Channel-appropriate metrics only (no video metrics for SEARCH, etc.)
- [ ] Time aggregation matches the question (daily, weekly via `DATE_TRUNC`, monthly)
- [ ] Join keys correct when combining tables (`BRAND`, `LOB`, `CHANNEL`, `CAMPAIGN`, `PARTNER`, and optionally `CHANNEL_EXECUTION_ID`, `DMA_CODE`)
- [ ] Case conventions respected (CHANNEL is uppercase; PARTNER/NETWORK are lowercase; PACING's REPORTING_CHANNEL is title case)

**Calculation correctness:**
- [ ] Aggregate ratios use `SUM(numerator) / NULLIF(SUM(denominator), 0)` — never `AVG(ratio)`
- [ ] "Top" / "Best" CPA, CPC, CPM = lowest value (most efficient), not highest
- [ ] Invalid calculated values (NULL from zero denominator) show underlying raw values as explanation
- [ ] KPI_N / KPI_D computed correctly per KPI type; CPM requires `* 1000`

**Formatting:**
- [ ] All values rounded to 2 decimal places
- [ ] Commas in all numeric outputs (e.g., `1,234,567`)
- [ ] Dollar amounts prefixed with `$` (e.g., `$1,234.56`)
- [ ] Percentages suffixed with `%` (e.g., `0.54%`)
- [ ] Rate metrics multiplied by 100 before displaying as percentages
- [ ] Source tables and filters cited in response
