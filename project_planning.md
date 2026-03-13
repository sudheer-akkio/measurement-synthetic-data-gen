### 3 Data Generation Pieces 
- Campaign Data - DONE - create in client schema directly 
- 1p Data IO Tech - DONE 
- Source Tables -> create in common schema - Implement using Sudheer's data gen framework - Sudheer & Johnny
	- Spine - akkio_id generation
		- Auto - detail 
		- media - detail
		- cpg - detail
		- place - detail
	- minimal dbt pipeline - Christine 
		- to create the summary tables in common schema
		- create views for all the above common tables in client specific schema

### Functional Requirement 

- Refresh capabilities
	- campaign, media, cpg and place data
		1. Data-gen-script adds incremental rows - orchestration is harder 
		2. shift date values as a window on tables - less risk
			- this could be done via dbt
			- pick this strategy
	- spine & auto is time invariant - no refresh 
- Schema needs to match code/implementation 
- data size considerations 
	- measurement data should be more than 52 weeks? 

### Non-Functional Requirements
- Cost 
- Orchestration 