# HUD FHA Multifamily Housing Project

## Purpose
This repository holds all resources used in the attainment of the goals established for the HUD FHA Multifamily Housing Project.

## Goals


## Data
[HUD FHA MF Loan Production](https://www.hud.gov/program_offices/housing/mfh/mfdata/mfproduction)

### Data Dictionary
**FHA Number**: loan number

**Project Name**: name of project

**Project City**: name of city in which the project is located

**Project State**: name of state in which the project is located

**Basic FHA, Risk Share or Other**: categories of HUD programs

**Program Category**: more granular program categories of HUD programs

**Activity Description**: description of project activity

**Activity Group**: groupings of similar activities

**Facility Type**: the type of facility of the project

**Program Designation**: 

**Firm Commitment Activity**: 

**Lender at Firm Commitment Activity**: the firm providing financing

**Mortgage at time of Firm Commitment Issuance, Amendment or Reissuance**: the mortgage amount at the time of Firm Commitment Issuance, Amendment or Reissuance

**Unit or Bed Count**: number of units or beds for project

**Date of Firm Commitment Activity**: the date of the last firm commitment-related activity (could 
be an amendment, extension, or reissuance of the existing firm commitment).

**Fiscal Year of Firm Commitment Activity**: the fiscal year of the last firm commitment-related activity

**Mortgage at Firm Commitment Issuance**: mortgage amount at the firm commitment issuance

**Date of Firm Issue**: the date the firm commitment itself was issued (so this date does not 
change with subsequent amendments, extensions, etc).

**Fiscal Year of Firm Commitment**: the fiscal year the firm commitment was issued

**MAP or TAP**: types of application processing for FHA loans; Multifamily Accelerated Processing 
or Traditional Application Processing respectively

**LIHTC Designation**:

**Tax Exempt Bond Designation**:

**HOME Designation**:

**CDBG Designation**:

**Section 202 Refi Designation**:

**IRP Decoupling Designation**:

**HOPE VI Designation**: 

**Current Status**: current status is FHA loan

**Final Mortgage Amount**: the final mortgage amount of the FHA loan

## Audience
The audience for this project is the layperson.

## Hypotheses

### Exploration

#### Mortgage Amount and HUD Program
$H_0$ : The total final mortgage amount for a market is independent of HUD program (Basic FHA, Risk Share or Other)

$H_a$ : The total final mortgage amount for a market is dependent on HUD program

#### Mortgage Amount and Statue
$H_0$ : The mean final mortgage is the same across the current statuses

$H_a$ : The mean final mortgage is different depending on the current statuses

#### Mortgage Amount and Facility Type
$H_0$ : The mean final mortgage amount is the same across all facility types

$H_a$ : The mean final mortgage is different depending on the facility type

### Time Series Regression
$H_0$ : Mortgage lending and market development are independent of each other

$H_a$ : Mortgage lending and market development are dependent of each other

### Classification
$H_0$ : The mean final mortgage amount among all market are indistinguishable 

$H_a$ : The mean final mortgage amount among all market are distinguishable

## Deliverables

### Need to Haves:
1. Model
2. A well-documented jupyter notebook that contains our analysis
3. Presentation summarizing our findings

### Nice to Haves:
1. Map of markets
2. Web application

## Cloning
All files necessary for cloning and reproducing the work found in the final_project.ipynb file are contained within this repository.