## Project Title 

## Team Members

- **Alec Hartman**

- **Daniel Guerrero**

- **Noah Melngailis**

- **Nick Joseph**

# Project Backstory

Our goal is to predict surges in new construction within multifamily housing markets before they happen.  Our stakeholder is Clifton Harness, CEO of TestFit.io, an architectural software company founded in 2015 (details below).  

The ideal end product of this project - its deliverable - is a market projection tool that can accurately predict whether or not our stakeholder should enter a particular market in the US.  We are basing our projections on the patterns we discovered leading to three substantial spikes in recent history: Houston (TX) in 2009, Seattle (WA) in 2010, and Dallas(TX) in 2012.  The factors that determine market booms are plenty; our tool is based only on the publicly available data from the US Department of Housing and Urban Development (HUD) found [here](https://www.hud.gov/sites/dfiles/Housing/documents/Initi_Endores_Firm%20Comm_DB_FY06_FY20_Q2.xlsx'). 

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

## How Do Cities Get Block Grants?

As the largest federal block grant, the 'Home Investment Partnership Program' ('Home Program') is meant for the creation / rehabilitation of affordable housing for low-income individuals.  It is a formulaic grant, meaning the amount of federal money going to the states and cities is based on statistical criteria and the amount of funds to be distributed (source: grants.gov).  There are several of these types of grants, and each has their own unique formula, but the money *must* be allocated and administered per the requirements of the law that gave birth to the grant.

The 'HOME Program' is not the only source of funding for new construction / rehabilitation; it is public money designed to work in conjunction with private money with the aim to bridge large-scale funds to small-scale needs.  Bob goes to City Council and says, 'I know you've got a hunk of cash for aid in housing.  Please use it to help me build / renovate this area down the street from me.'    

Here's how it works: a bill becomes a law, and the grant money goes to the States.  Participating jurisdictions (cities) put together - and approve - a 'Consolidated Plan' that outlines what they'll do with that money and how they'll allocate it to help improve affordable housing.  They present it to the State, and the State decides how much money they'll get based on the presented Consolidation Plan. 

## Files Used In Acquiring, Wrangling, Exploring, and Modeling the Data

**Wrangle.py**

In this project, wrangling data is a mulit-step effort that invovles acquring, cleaning, and making the data as uniform as possible so that it 'makes sense.'  In this particular dataset of government information, there were plenty of acronyms which had to be deciphered for better understanding.  

This function begins with the acquisition of Sheet 2 from the dataset (presented in XL format), titled 'Firm Cmtmts, Iss'd and Reiss'd' (long: 'Firm Commitments, Issued and Reissued').  From the term 'reissued,' it makes sense there would be duplicate loan numbers with which had to be dealt.  Within this file a function was created to both track and maintain the loan number counts; instead of counting a repeated loan number twice, it is only counted once.  

Other preparatory functions within this file make the features (the table column titles) lower case, create boolean values ('True' / 'False') where appropriate, remove outliers (like a $1 mortgage loan), and convert dates to datetimes for ease of data exploration.

The end result is a much more easy to understand Pandas DataFrame.

**Preprocessing.py**

With the assistance of the work done in the 'wrangle.py' file, 'preprocessing.py' is the file where we gain true value from our dataset.  Within this file are functions to group data by categorical (independent 'either / or') variables, aggregation of the data into a DataFrame dedicated only to New Construction, and data sorting by sums and counts of the 'final_mortgage_amount' category.

Also important within this file are functions to facilitate the implementation of year-over-year change in total mortgage amounts within the cities found in our dataset.

Lastly - and perhaps most importantly - this preprocessing file contains funcitons that split our data into two sets: training and test.  You want to see how well your model performs using the training data, and then test it against new, unknown testing data.  Splitting data like this allows you to make changes to the training set while keeping the test set (the stuff the model will face in the 'real world') untouched.  Our data is split 75% train, 25% test, which lies well within suggested practice parameters.

**Modeling**

Utilizing the vast and powerful Python library 'sklearn,' three functions were created to run three different types of models: Decision Trees, Random Forest, and K-Nearest Neighbors.  Decision Tree models are good for decision analysis, when you want to identify a strategy most likely to reach a goal.  Random Forest models are an extension of the Decision Tree model, but - just like in nature - there are a lot of trees in it.  The Decision Tree basically has a vote in its classification, and the Random Forest Model takes all the trees with the most votes - it combines all the trees to give more accurate results.  

Lastly, the K-Nearest Neighbors (KNN) model is one of the most used learning algorithms in the field of Data Science.  Considered 'lazy,' it takes a dataset with several points and tries to predict the classification of a new unknown point looking at the distance between previous points.  The Nearest Neighbor to the point in question gets that classification.

## Summary Background on Stakeholder, TestFit.io

Based in Dallas, TX and originally named 'BuildingForge,' TestFit.io was founded by Clifton Harness (CEO) and Ryan Griege (CTO) in October of 2015.  They are a software company specifically meant to help architects craft buildings more quickly through their generative design tools that translate easily over to AutoCAD, a poplular tool used in all phases of the construction process.  Their enhanced visualizations help architects test parametric designs while streamlining the feasibility study process for commercial projects, both substantial roadblocks in any community investment project.

In January 2020, TestFit.io received $2,000,000 from Parkway Venture Capital, which they have earmarked for both personnel and market growth.  They are currently looking to expand their offerings into retail floorplan analysis ('adjacency studies') and hospital construction.  

## About the Modeling Used for This Project


Before modeling, we had to come up with a way to label our data so that our models could reflect which markets were outperforming the rest.  In other words, for the results of our models to be understandable to stakeholders or anyone recreating our data, we had to figure out which outputs were good and which were not.  For the purposes of our MVP, it was decided to proceed using the 'Evolution Index' prevalent in pharmaceutical product analysis.

Suppose you've been selling an allergy medicine called 'Allergone.'  There is a load of competition you're facing - 'Allegra,' 'Claritin,' and 'Zyrtec' just to name a few.  Over time, you see your sales skyrocket - you can barely keep up with the demand.  That's great becauase you're moving product and staying busy, but how well are you actually doing against those other brands?  While *your* sales have increased, have they increased at the rate of the rest of the market?  Are you  *evolving* at the same rate as 'Allegra,' 'Claritin,' or 'Zyrtec?'  If not, you may be doomed to fail, despite your increase in sales.

The Evolution Index (EI) helps you get an objective view of your performance against that of your competitors in your market.  It is scored using the following formula:  EI = (100 + Product Growth %) / (100 + Market Growth %) X 100.   Any score above 100 means your hot dog sales are growing faster than the rest of the market, and any score under 100 means the other vendors are outpacing you.  

We used EI in the same manner, only with markets themselves instead of individual products.  
