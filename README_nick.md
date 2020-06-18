# Project Title 

Using publicly available data from the US Department of Housing and Urban Development (HUD), we try to predict surges in new construction within multifamily housing markets before they happen.     

# On Reproducibility

Our findings are reproducible for peer review.  To do so, however, requires a few things that must be done up front.

**1.) Installation of Python and its associated libraries**

All work for this project is performed using Python, specifically version 3.7 (or later).  To check which version you have on your computer, simply go to your command line and type ```python --version```.  If it's not installed on your computer, [Python](https://www.python.org) has an easy to follow menu for installation across Mac, Windows, and other various platforms.

Once python is situated on your machine, simply importing its associated libraries (matplotlib, sklearn, and others) into your working environment will allow you to see and manipulate the data in just about every way imaginable.  

**2.) Jupyter Notebook**

Our chosen platform for code devlepment is Jupyter Notebook, an open-source web application that allows you to create and share documents.  In it, the user can perform all phases of the Data Science Pipeline, from Data Acquisition through the modeling of discoveries made.  As with Python, you can check the version on your machine in the command line with ```jupyter notebook```; if it's not found, you can download [Anaconda](https://www.anaconda.com/products/individual) to install the complete line of offerings, of which Jupyter Notebook is one.  

**3.) Working with this Repo**

To work with the following information, you must first clone this repository.  In doing so, you are grabbing a full copy of our work and putting it on your computer for analysis.  Instructions for doing so can be found [here](https://help.github.com/en/github/creating-cloning-and-archiving-repositories/cloning-a-repository).


# Planning

Once teams were assigned, we immediately began working toward defining our MVP, our Minimum Viable Product.  A quick review of the customer demands and it was decided the MVP should be a high-performing classification model to determine when and where the client should act on deploying their sales assets.

To help with the mangagment of this project we employed the use of [Trello](https://trello.com/home), a web-based, Kanban-style accountability board.  It's essentially Post_It Notes on a board the rest of the team can see so each milestone is reached in an orderly fashion all the way through to project completion.

The initial exploration is into the historical multifamily construction booms of Houston, TX (2009), Seattle, WA (2010), and Dallas, TX (2012).  Using these as our targets, we try to examine the events leading up to each of them in hopes of finding commonalities and patterns they may share.  

## Time Series Regression Hypothesis

**Null Hypothesis** - Mortgage Lending and market development are independent from each other

**Alternate Hypothesis** - Mortgage Lending and market development are dependent on each other

## Classification (ANOVA)

**Null Hypothesis** - The mean final mortgage amount among all markets is indistinguishable

**Alternate Hypothesis** - The mean final mortgage amount among all markets is different

A single observation would be the total final mortgage amount for a city (market) and year of that mortgage

## Data Dictionary

Upon seeing the data for the first time, there was a good amount of industry jargon specific to dealing with HUD and various government funding programs.  In order to help us clear up what it was we were looking at, we developed the following dictionary of terms.

**Block Grant** - A block of money from the federal government given to state and local governments for use in social welfare programs like law enforcement, community development, and health services.  Unlike categorical grants, block grants are not earmarked for anything specific, so local governments can spend them as they see fit.  

**FHA** - 'Federal Housing Administration' - With regards to this project, we are talking about monetary loans that come from the Federal Housing Administration, also known as 'FHA Loans.'  These are federally-backed loans from approved money lenders made to people designated as low-to-moderate income, although there is no max or minimun income level listed for qualification.  They do, however, tend to be geared toward people with lower credit scores (as low as 500) to help get them into home ownership.

**HFA Risk Sharing** - 'Housing Finance Agency' - the regulatory body of the US Department of Housing and Urban Development (HUD), they took over financial affairs and daily operations of both Fannie Mae and Freddie Mac, who were selling mortgage-backed securities.  'HFA Risk Sharing' means that the HFA is sharing the risk of the loan with the FHA; if the loan defaults, both agencies split the loss.  Seeing this in our data means the multi-unit property / building has been purchased and the commitment is firm.

**Firm Commitment** - also known as 'firm commitment underwriting,' a firm commitment is a lender's promise to enter a loan agreement with a borrower.  As it pertains to this investigation, since the construction of new multifamily units is both expensive and intensive, we consider the value of 'Firm Issued' to mean that the money was allocated as promised and that the construction of new units took place.

**MAP** - 'Multifamily Accelerated Processing' - decentralized command from HUD, MAP allows underwriters (the people looking at your credit history) at qualified lenders to prepare FHA forms and conduct the preliminary underwriting for certain applications.  Not only does this speed up the process, but it reduces the risk HUD faces when allocating funds; MAP approved underwriters are their boots on the ground.

**TAP** - 'Traditional Application Processing' - In order for a lender to be part of the 'MAP' process, all of its underwriters must complete specialized HUD training where they learn all the parameters of HUD financing.  Without this training, applications must be processed by a HUD field staff member, which takes more time because only HUD-approved underwriters can decide whether or not a loan should be financed.

**LIHTC Designation** - 'Low Income Housing Tax Credit' - aimed at private equity, this is a federal tax credit to incentivize new construction of multi- and single-family homes for low-income individuals.  There are several parameters which must be met (e.g: building residents can't have a net income of over 50% of surrounding area net incomes), and the total allocation amount is based on the state's population size, at about $2.70 per person.

**'Tax Exempt Bond' Designation** - cities sell bonds to raise money for public improvements.  A tax-exempt bond (or, 'municipal bond')is desirable to investors because the interests they earn on their investment (the bond) is tax free.

**'HOME' Designation** - 'Home Investment Partnership Program' - a federal assistance program provided by HUD to provide decent and affordable housing, paticularly to low- and very-low-income Americans.  It is the single biggest block grant providing approximately $2B (2020 estimates) to states and municipalities looking to increase the amount of affordable housing within the US.

**'CDBG' Designation** - 'Community Development Block Grant' - a federal grant meant to develop viable urban communities by providing decent housing and expanding economic opportunities for people of low- to moderate-incomes.  Not only meant for housing and economic development, but for disaster recovery as well.

**'Section 202 Refi' Designation** - a federal program encouraging non-profit entitites to build housing with supportive services for severely low-income senior citizens.  The buildings must be specifically designed to service the needs of the elderly and either all residents must be over the age of 62 or have 80% of its occupants over the age of 55.

**'IRP Decoupling' Designation** - 'Interest Reduction Payments Decoupling' - some borrowers qualify for a government subsidy (financial assistance) that helps reduce their monthly interest payments.  This subsidy is paid directly to the lender to help cover the interest rate of the loan.  All well and good, but if the owner refinances, all the leftover interest the lender is expecting to make is lost.  An IRP Decopuling allows that remaining interest to move into the refinanced loan amount, so the lender can still plan on receiving the original interest payments despite the refinancing.  Wherever you see this designation, a refinance is likely to have taken place, though its specifics are not available 

**'HOME VI' Designation** - Title VI is funding for federally recognized tribes and Tribally Designated Housing Entities meant for refinancing and new home construction, as well as all the costs associated with community development.


## Summary Background on Stakeholder, TestFit.io

Based in Dallas, TX and originally named 'BuildingForge,' TestFit.io was founded by Clifton Harness (CEO) and Ryan Griege(CTO) in October of 2015.  They are a software company specifically meant to help architects craft buildings more quickly through their generative design tools that translate over to AutoCAD, a poplular tool used in all phases of the construction process.  Their enhanced visualizations help architects test parametric designs while streamlining the feasibility study process for commercial projects, both substantial roadblocks in any community investment project.

In January 2020, TestFit.io received $2,000,000 from Parkway Venture Capital, which they have earmarked for both personnel and market growth.  They are currently looking to expand their offerings into retail floorplan analysis ('adjacency studies') and hospital construction.  
