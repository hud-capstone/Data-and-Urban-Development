## About 'Region of Boom'

'Region of Boom' is the [Codeup](https://codeup.com/) Data Science capstone project of Alec Hartman, Daniel Guerrero, Noah Melngilis, and Nick Joseph.  As a capstone, it is meant to showcase the cumulative learning that took place in both the Codeup classroom and - due to Covid-19 - instructor-led remote environments.

We worked with our stakeholder, TestFit.io (description follows) to find out which markets were best for them to enter using historical construction information.  At this time, they are still a smaller-sized company, and allocation of resources is more costly to them than to that of a larger company.

**Note:** All work in this repository - including stakeholder meetings, daily standups, and collaborative programming - took place in a remote environmnent using Zoom, an enterprise-level video communication platform utilizing cloud technology.  As many of those interested in our efforts would like to see how well we performed in a virtual team environment, we are happy to point to the product: a highly effective model against unknown, real world data

**___________________________________________________________________________________________________________________________________**

## How to Read Our Work

We have divided our work into three folders, to make it easier for readers to find the files they are looking for. 

**1.) final_project**

This folder houses our final report, with all of our findings, exploration and modeling. Here you will also find all of the py files needed to reproduce our work. 

**2.) mvp_folder**

This folder houses our "minimum viable product". Here you will find our initial round of exploration, analysis and modeling.

**3.) additiional_reference_notebooks**

Finally, this folder houses our individal notebooks, which were used to create the final notebook, as well as additional analysis, and a lot of testing for the functions we used throught our project.

**___________________________________________________________________________________________________________________________________**

## On Reproducibility

Our findings are reproducible for peer review.  To do so, however, requires a few things that must be done beforehand.

**1.) Installation of Python and Its Associated Libraries**

>All work for this project is performed using Python, specifically version 3.7 (or later).  To check which version you have on your computer, simply go to your command line and type ```python --version```.  If it's not installed on your computer, [Python](https://www.python.org) has an easy to follow menu for installation across Mac, Windows, and other various platforms.

>Once python is situated on your machine, simply importing its associated libraries (matplotlib, sklearn, and others) into your working environment will allow you to see and manipulate the data in just about every way imaginable.  

**2.) Jupyter Notebook**

>Our chosen code devlepment environment for this project is Jupyter Notebook, an open-source web application that allows you to create and share documents.  In it, the user can perform all phases of the Data Science Pipeline, from Data Acquisition through to Modeling and takeaways made.  As with Python, you can check the version on your machine in the command line with ```jupyter notebook```; if it's not found, you can download [Anaconda](https://www.anaconda.com/products/individual) to install their complete line of offerings, of which Jupyter Notebook is one. 

**3.) GitHub**

> [GitHub](https://github.com/) is an online version-control platform that provides hosting services for software development.  Each repository is a publicly accessible file from which users can pull information and see the development of a program at various points in its evolution.

To those unfamiliar with the jargon-heavy vocabulary of programming, think of baking a pie for friends.  

GitHub is the giant cookbook filled with recipes (called  'repositories,' or 'repos' for short) from all over the world.  Each recipe is detailed in both its ingredients list and the step-by-step processes you need to achieve a perfectly flaky - yet tender - pie crust.

But it's a cookbook written in real-time; as people try these recipes, they can make adjustments and write their changes into the recipe as they do so (for instance, how baking times differ between elevations).  Each change is as detailed as possible so that anyone finding it can browse through the contents of that recipe and see exactly what it is they're looking for.  

That is the essence of GitHub: a cloud platform where people can view software development at various stages and - with the proper permissions - volunteer their contributions in an effort to improve the performance of the original product.  

**4.) Working with This Repo**

>To work with the following information, you must first clone this repository.  In doing so, you are grabbing a full copy of our work and putting it on your computer for analysis.  Instructions for doing so can be found [here.](https://help.github.com/en/github/creating-cloning-and-archiving-repositories/cloning-a-repository)

**___________________________________________________________________________________________________________________________________**

## Moving from the Original Dataset and On To The New Dataset

By their very nature, projects evolve, and 'Region of Boom' was no different.

Analysis of the intial FHA Multifamily Loan Production data (made publicly available by the US Department of Housing and Urban Development) led to our first minimum viable product (MVP): a model that performed well with high rates of both recall and accuracy.  However, we as a team felt there was not enough data to accurately represent reality; because of the lack of data, we had to purposely overfit our model for it to work.  Doing so not only didn't sit well with us from an ethical standpoint, but from the quality of our end product (a high-performing model ready for deployment) as well.  Plus, it kept us from answering the very question that spawned this capstone project: 

How many high-density, multifamily structures are being built in the U.S. everyday?

That being said, we still pushed through with the project and reached an MVP which can be found [here](https://github.com/hud-capstone/capstone/blob/master/mvp_notebook.ipynb).

**___________________________________________________________________________________________________________________________________**

## Our New Data: Building Permit Surveys**

Using the United States Census Bureau Building Permit Survey data, we were able to gain more insight and create considerably more value for our stakeholder.  Not only is the data vast, but the Bureau has its own means of imputation - it's own way of filling missing data values - that we could employ while preserving the integrity of the original data.

Below are the links to the data we used, all of which is available to the public.

[United States Census Bureau Building Permits Survey](https://www.census.gov/construction/bps/): Main US Census webpage where our data was sourced

[ASCII files by State, Metropolitan Statistical Area (MSA), County or Place](https://www2.census.gov/econ/bps/): The comma-separated text files aggregated based on various geographical areas.  Basically, XL sheets.

[MSA Folder](https://www2.census.gov/econ/bps/Metro/): We used data on the [Metropolitan Statistical Area](#metropolitan-statistical-area-msa) level of granularity

[ASCII MSA Documentation](https://www.census.gov/construction/bps/sample.html): Documentation of the original features can be found in the resultant DataFrame from the 'acquire_building_permits' function

Though our source data did change, the project objective remained the same: use existing data to predict future market growth before it happens.

**___________________________________________________________________________________________________________________________________**

## Summary Background on Stakeholder, TestFit.io

Based in Dallas, TX and originally named 'BuildingForge,' TestFit.io was founded by Clifton Harness (CEO) and Ryan Griege(CTO) in October of 2015.  They are a software company specifically meant to help architects craft buildings more quickly through their generative design tools that translate over to AutoCAD, a poplular tool used in all phases of the construction process.  Their enhanced visualizations help architects test parametric designs while streamlining the feasibility study process for commercial projects, both substantial roadblocks in any community investment project.

In January 2020, TestFit.io received $2,000,000 from Parkway Venture Capital, which they have earmarked for both personnel and market growth.  They are currently looking to expand their offerings into retail floorplan analysis ('adjacency studies') and hospital construction.  

**___________________________________________________________________________________________________________________________________**

## Broad Terms To Know

#### Metropolitan Statistical Area
Using the US Census Bureau definition (as an actual definition seems to vary by source), a Metropolitan Statistical Area (MSA) is any urban center with a population of 50,000 or more people.  We did not pursue granularity down to to the micropolitan (between 10,000 and 50,000) level for several reasons, one of which was keeping stakeholder industry niche and profitability in mind.

#### US Department of Housing and Urban Development 
Founded in 1965 Under Lyndon Johnson's 'Great Society' program, the US Department of Housing and Urban Development ('HUD') is part of the Executive Branch of the United States.  Among its many activities since then, its primary objective is banning discrimination in housing development, and in 1992 sought out to actively revitalize public housing and flailing communities.

#### Federal Housing Administration
Actually older than HUD, the Federal Housing Administration (FHA) was originally founded as part of the National Housing Act of 1934.  It is through their efforts that standards of construction and loan underwriting (whether or not people get the money they need for home ownership) are upheld on a daily basis.  It was within the 'Great Society Program' that FHA was absorbed to become a branch of HUD.

#### Evolution Index

A key feature in the performance of our model is something called an 'Evolution Index,' or 'EI.'  While a more detailed explanation regarding the EI can be found in our [MVP README](https://github.com/hud-capstone/capstone/blob/master/README_nick.md) file, the EI is a way of finding out how one market is faring against rest of the markets out there.  Prevalent in pharmaceutical product analysis (how well, say, Advair sells in the San Antonio market as compared to every other market in the US), using this metric made sense because we were comparing *actual markets* to each other in order to determine which were going to perform best.

## How Do HUD Projects Get Funded?

As the largest federal block grant, the 'Home Investment Partnership Program' ('Home Program') is meant for the creation / rehabilitation of affordable housing for low-income individuals.  It is a formulaic grant, meaning the amount of federal money going to the states and cities is based on statistical criteria and the amount of funds to be distributed (source: grants.gov).  There are several of these types of grants, and each has their own unique formula, but the money *must* be allocated and administered per the requirements of the law that gave birth to the grant.

The 'HOME Program' is not the only source of funding for new construction / rehabilitation; it is public money designed to work in conjunction with private money with the aim of bridging large-scale funds to small-scale needs.  

Your Uncle Bob can go to the City Council and say, 'I know you've got a hunk of cash for aid in housing.  Please use it to help me build / renovate this area down the street from me.'    

Here's how it works: when a bill becomes a law, federal grant money goes to the States.  Participating jurisdictions (cities) put together - and approve - a 'Consolidated Plan' that outlines what they'll do with that money and how they'll allocate it to help improve affordable housing.  They present it to the State, and the State decides how much money they'll get based on that Consolidated Plan. 

## Data Dictionary

Upon receiving the new dataset, there were several columns of data ("features") that were codified, presumably for internal use by the Census Bureau.  

Within our 'wrangle.py' file is a function ('rename_building_permit_columns') used to rename those original columns into more easily interpretable names for our data exploration.  What follows is a brief data dictionary designed for easy reference when reviewing the final notebook.  

**cbsa_code** - 'core-based statistical area code' - the assigned code for a core-based statistical area.  For a better understanding of what a 'cbsa' is, think of a dot surrounded by a circle.  The dot is the city with a population between 10,000 and 50,000 people (just below the 'metropolitan' definition threshold), and the wider circle is the surrounding counties with a population that is socioeconomically tied to that dot.  At the time of this research (July, 2020), there are 929 cbsa's in the US (924) and Puerto Rico (5).  For San Antonio, our assigned CBSA code is 41700.  

**csa_code** -  'combined statistical area code' - a 'csa' is much like a 'cbsa,' but it's on a larger scale.  Using the above analogy, the 'csa' is where the wider circles meet.  By definition, at least 15% of the population must depend on the urban center in some way; today (again, July 2020) there are 172 csa's in the continental US and 3 in Puerto Rico

**bldgs_units_value_est** - our buildings permit dataset was divided into the number of buildings for each permit, the number of units in each building (living space or offices), and the estimated number of structures covered in the allotted permit.  For instance, in looking at our columns, you will see a title like '**three_to_four_units_value_est**.'  That can be read as the estimated number of three-to-four unit structures assigned to the permit.  With our stakeholder in mind, our main concern for this project was high-density unit housing, the **five_or_more_units** feature in our dataset.

**bldgs_units_value_rep** - just like the above, but the reported (not estimated) number of units assigned to that permit.

**___________________________________________________________________________________________________________________________________**

## About the Process

Akin to most projects, 'Region of Boom' followed the standard Data Science pipeline of Project Planning, Data Acquisition, Preparation, Exploration, and Preprocessing, and then moved forward with the Modeling and Delivery of our findings.

That said, anyone who has undertaken a project knows that a 'pipeline' is the wrong literal cue.  'Region of Boom' did not develop in a straight line from beginning to end.  In fact, when the project started, the entire effort was dedicated to a completely different dataset - one that we pined over for days only to come to the conclusion that it didn't meet our standards.  So much of the data was foggy and open to interpretation that for our model to work, we had to intentionally overfit its performance, and that did not sit well with us.  Ethics, baby - Data is an immensely powerful tool, and to quote Spiderman's Uncle Ben (okay, Voltaire): "With great power comes great responsibility."

Imagine your yard.  It's green and lush, but right in the middle of it there's this patch of yellow that just won't go away.  You've tried giving it extra water, some Ironite, and maybe even a little shade umbrella during the hottest part of the day.  But the extra love is all for naught, and that patch of yellow never goes away.

If we intentionally overfit our model, it'd be like you going to Home Depot, matching the color of the good grass, coming home, and just spray painting that yellow patch to make it look like the rest of the yard.  It's a temporary solution that never addresses the real problem and leads to a false representation of what you want the world to see: a beautiful, well-kept yard. 

In our case, the 'yard' is our final prediction model.  We want the business world to see it because we believe it could be of great value.  What good would it do if our 'predictive' model couldn't really predict anything?  (We'd call it the 'Skip Bayless Model,' but that's a completely different discussion.)

Because of this need to 'pretend' the model worked, we scrapped everything and started over using a different, more populated dataset that better lent itself to deep analysis: data from the US Census Bureau (mentioned above).  With this new data, we restarted the 'Data Pipeline,' and as we made our discoveries, realized it's more of a 'Data Curly Fry' than any sort of 'pipeline.'  As the project progessed, visualizations and modeling revealed to us new information that forced us to restart from some earlier point in the process. 

Data Science is not a Coen Brothers movie.  It's a Tarantino film where everything is out of sequence until *you* put the story together at the end. 

### A Note About Functions and Pandas Dataframes

Throughout the length of this repository, you will see the term 'function' as it pertains to the Python programming language only (other programming languages like C, PHP, and Javascript allow functions as well, but vary widely in their syntax).  

Functions help make program writing easier because they can store a whole block of code and then execute that code whenever the programmer needs it.  For instance, Python has a built-in function called 'len()' that tells you the length of the whatever object you put into the parentheses.  All one has to do is type 'len()' and they'll be able to see, for instance, how many characters are in a word.  No further coding, no deep diving... just 'len(word).'

Now that you know what a function is, it helps to learn a little about how to read them (in Python, at least).  Though they vary considerably in scope, Python functions read pretty much the same way; the best way to to explain this is to take a look at one:

```python
def function_name(parameters):
    """Easy-to-read description of what the function does"""
    block of code 
    you want your
    function to run

    value you want the function to return
```

The first line is the 'function header' - it's what you want to name the function.  Inside its parentheses are the parameters that give the function the information it needs to run.  The description in triple quotes describes the function to other code readers, and the code block underneath it is the actual code you want the function to execute.  Finally, the last line (with 'return' in red) is the value you get back after the function does its job.

Contained within our '.py' files are the unique and powerful functions that carry out tasks like cleaning and preparing the data.  Even though they are only mentioned in the Final Notebook as imports, it is recommended you visit the actual files themselves to gain a true appreciation for the labor involved.

As for Pandas DataFrames...

Pandas is a Python library (think 'group of Python-based capabilities') that lets Data Scientists look at data in a clean format from which they can compare features and do visualizations that actually *show* how the data is behaving.  One of the reasons the field of Data Science loves Pandas is because it can take in several different types of file formats and return a singular, familiar-looking DataFrame.

This DataFrame is the standard output from Pandas - it's a table of rows and columns like you'd see on an XL spreadsheet, but with superpowers.  Where XL has considerable size limitations that make some calculations measurable only by sun dial, Pandas can handle millions of datapoints with ease, and is thus the preferred software choice for Data Science involving Python.

Plus: DataFrames just look nice.  If you're going to be staring at data all day, why not make it easy on the eyes?

### What Is A Model, and Why Did We Choose Classification?

A Classification Model was chosen for this project because we are trying to predict *future* market growth based on the *observed value* of historical market growth (the data we got from the Census Bureau). 

Of all the ways to explain what we mean when we say 'model' in Data Science, the easiest way is to leave all the 'Data' and 'Science' out of it, and take, for instance, a simple trip to the grocery store. 

Most people aren't even aware of what it takes to get to the grocery store.  They just get in their car, crank on the ignition, and get their 'tunes' going.  Next thing they know, they're wondering why one brand of toilet paper is so much more expensive than the other.

But everything they did before comparing those two brands is where the idea of modeling lies.  

As soon as they opened their car door, an automated process took over.  They subconsciously knew that before they could go anywhere, they had to put in the keys, make sure the engine turned over, and that the car started.  Once engaged, they shifted the transmission to move the car, checked their surroundings for any dangers, then eased out of the driveway and onto the road.  Once on the road, they followed the fastest route and (presumably) obeyed all the laws and street signs, even going so far as to check for dangers along the way and switch lanes when the car in front of them was going too slow.

In essence, they have a 'Going To the Store' model that has been tried and tested over and over until the best, most efficient path to the store was discovered.  

Trickling Science back into the mix, every trip they took to the store was noted by their hippocampus, the part of the brain that automates tasks to make life easier.  It processes data as it comes in and tests it against prior data and outcomes.  Eventually, it comes up with the best way to the store for you so that you no longer have to think about all the steps in getting there.

That's what our model is doing: taking in data, testing outcomes, weighing effects, and then repeating the process until we tell it to stop.  Once training stops, it has figured out / predicted our best way to get to the store.

**___________________________________________________________________________________________________________________________________**

## Acquiring the Data

Thanks to our aforementioned functions, we were able to filter all the original information from the US Census Bureau into over 8,382 points of data across twenty-nine (29) features describing the building permit activities in 390 domestic markets.  To see the original data from which we culled everything down, please feel free to visit the website [here](https://www.census.gov/construction/bps/).    

**___________________________________________________________________________________________________________________________________**

## Preparing the Data 

"Give me six hours to chop down a tree, and I'll spend the first hour sharpening the axe." - *Abe Lincoln*

Now that we had the data, we had to clean it up a bit.  Whenever you obtain data - even data that is well maintained by someone like the Census Bureau - its structure gets skewed when you move it over to a software library like Pandas.  Not only that, but a lot of times, the types of data that migrate over don't make sense to Pandas.  For instance, if you have an XL spreadsheet full of numbers and you do what's necessary to import it into a Pandas DataFrame, Pandas will see some of those numbers a strings ('words') and not integers ('numbers').  

Another important thing to look at when preparing data for analysis in Pandas is missing values in the original dataset.  Missing values can be a nightmare if you don't handle them at the beginning stages of the Data Science Pipeline - in this Data Prep stage, you have to decide what to do with them.  Do you drop the features containing them, or do you insert a value and just move on with the project?

If the percentage is low enough so that removing a certain number of values won't affect the integrity of the data (if, for instance, you have 100 missing values but the other 900,000 datapoints are all there), you can drop them.  But what if the percentage is large or you really think those values would help you find the truth in your investigation?  In that case, you would have to find a way to impute (or substitute) values where they're missing.

Since the issue of how or what to impute could be a doctoral thesis (oh, look: [here's one!](https://repository.upenn.edu/dissertations/AAI3158644/), diving into the details of imputation is far beyond the scope of this project README.  However, what's neat about this dataset from the Census Bureau is that they actually have their own method of imputation, so however they dealt with the missing values was a non-issue for us - they had it handled before they published the dataset.

**___________________________________________________________________________________________________________________________________**

## Splitting and Scaling the Data

To 'split' data simply means to divide the dataset up into two main segments: 'train, and 'test.'  The purpose of this is to train the model against a certain portion of the dataset, and then put that model up against the 'testing' set.  Just like boxing, the training portion is where the data learns to hit a heavy bag, slip under a punch, work its feet around the ring, etc.  The testing portion is when the data actually steps into the ring to spar.  How well it does sparring is a direct reflection of how well it will perfrom in the world of real-time data. 

But within the 'training' dataset, you have a sub-dataset known as 'validation.'  Validation data is like the boxing coach: "You say you can hit a heavy bag, huh?  Alright, show me how well you hit a heavy bag."  The 'validation' portion takes a look at the model and evaluates its training each step of the way so that it is at its peak performance by the time it goes up against the testing data.

There are several suggestions on how much of the data should be reserved for training and how much for testing, but a good rule of thumb is that 75% of the dataset go to training, while 25% of it be set aside for testing.  And we say 'set aside' because you don't want to touch that test data if you don't have to.  It's raw and unrefined, which is good, because that's the best way to replicate the data our model will see when it is deployed into the real world.

Another reason we validate as we train is to prevent something called 'overfitting.'  Even though you want to run your training data through the grinder, you don't want to do it to the extent that all the model can do is perform well against that same training data.  Otherwise, it will do poorly against that raw, untouched and real-world-like testing data.  Back to the boxing reference: if all a fighter knows how to do is hit a heavy bag, they will get destroyed when they finally step into the ring. 

'Scaling' data removes redundancies in a dataset while simultaneously applying a logic that groups our data together.  It also normalizes all the data so that its easier to read and interpret.  If you have a dataset that includes prices and quantities, '6' can represent either the price of the unit or the number of units you're looking at.  Because all that would do is lead to confusion, the data has to be scaled prior to label creation.  

In our case, the MinMaxScaler tool from the Python library 'sklearn' was selected because we wanted to normalize all our data into values between 0 and 1. 

### Creating Labels for the Model

A model is all well and good, but unless you have labels for the data, you won't understand any of the model's outputs.  

There are two types of machine learning: supervised and unsupervised.  Supervised learning is where all the data is labeled - you know which feature is which, you build the model, and the output is easy to read because the dataset from which you are working has all the data packaged in nice, clean labels.

This project's model is formed from UNSUPERVISED learning - our model will work just fine, but without any inherent labels, we don't know what it's telling us.  Thus, we had to label the data ourselves.  How?  Well that segues nicely into our next topic: clustering.

### Clustering and the K-Nearest Neighbors Algorithm

'Clustering' is grouping 'scaled' data according to certain traits shared between the datapoints.  Even though there may be millions of points of data, we have several tools at our disposal that can look at all that data and find out where those commonalities lie.

One such tool: the K-Nearest Neighbors Algorithm.  

All an algorithm really is is a set of steps to get a desired outcome.  You see them all the time: recipe cards, assembly instructions, 'How To Lose 5 Pounds In 10 Days' websites... all of those are just processes on how to get something you want done.  While vastly more complicated with regards to computers and technology (because each step must be explicit; computers lack the human ability of inference), that's all they really are: "To get from Point A to Point B, do steps 1, 2, and 3."

When training your model, the algorithm you select directs the computer on what to look for during the training.  Using the K-Nearest Neighbors algorithm, we told our model to look at all the datapoints in the training set and find all their shared features.  The features with the strongest similarities - the ones nearest each other - get clustered together in a group that tells us a lot about those points of data.

As a result of our clustering, the following labels were developed:

- cluster0 - underperforming markets building an average number of units per building
- cluster1 - markets outpacing the population building an average number of units per building
- cluster2 - mixed growth markets building an average number of units per building
- cluster3 - mixed growth markets building a high number of units per building
- cluster4 - underperforming markets building a low number of units per building
- cluster5 - mixed growth markets building a low number of units per building

It was from this K-Nearest Neighbors clustering algorithm that we were able to determine the labels for our data: cities with 23-years of consecutive permit documentation regarding five-or-more units per building.  This filtering effort reduced our DataFrame into a more managable size of 2,860 observations across 17 features.  While the features are the columns themselves, the term 'observation' here means 'city, state, and year.'  Details as to why several features were lost after clustering are in the [Final Notebook](https://github.com/hud-capstone/capstone/blob/master/final_notebook.ipynb).

**___________________________________________________________________________________________________________________________________**

## Exploring the Data

Now that the data is cleaned and split, we can start playing with the training data: finding correlations, types of distributions, and various relationships between the different features.  For this dataset, an example exploration question is "How many 'total_high_density_units' were in the 'city' of Abilene, TX in the 'year' 2001?"  the values 'total_high_density_units,' 'city,' and 'year' are all features (columns) in the cleaned DataFrame.

Just like any scientific exploration, our exploration begins here with a couple of questions:

1.) Do the total number of high-density units (recall: units of five or more) vary based on cluster; and 

2.) What does the evolution index look like for each cluster?

From these questions we develop 'yes' or 'no' hypothetical predictions, referred to as 'null' and 'alternate' hypotheses, respectively, in Data Science.  A 'null hypothesis' is a guess that the status quo will be maintained.  Or, to put it plainly, it's like assuming everything will stay the same.  An 'alternate hypothesis' is the exact opposite; it assumes that things change.

So to phrase the aforemention questions more completely:

1.) Do the total number of high-density units (recall: units of five or more) vary based on cluster?

<span style="color:blue">Null Hypothesis: The mean units for all clusters is the same.</span><br/>
<span style="color:red">Alternate Hypothesis: The mean units for all clusters is different.</span><br/>

2.) What does the evolution index look like for each cluster?

<span style="color:blue">Null Hypothesis: The mean evolution index is the same for all clusters.</span><br/>
<span style="color:red">Alternate Hypothesis: The mean evolution index is different for all clusters.</span><br/>

You may wonder how these simple hypotheses help guide data exploration.  The first explanation is clarity's sake.  Data exploration can be rather tedious, and keeping simple questions and predictions like these in mind is a good way to interpret the results as they come about.  

The other is testing: the null hypothesis is the thing we're testing against - the thing we're trying to disprove.  We either 'accept' it (prove it to be true) or 'reject' it (proving it NOT to be true).  A more familiar example of this is the legal phrase 'innocent until proven guilty.'  In a courtroom, the jury is to assume innocence of the person on trial until the prosecution can prove they are NOT innocent (ie: guilty).

Statistical tests (Chi-Square, Binomial, etc.) are run by researchers to see if this null hypothesis is true or not.  In keeping with the legal analogy: Researchers are the prosecution, and we run statistical tests in an effort to build our case against the person (hypothesis) on trial.  The two most impactful statistical tests run in this project were the 'One Sample T-Test' and the 'ANOVA' or 'Analysis of Variance' Test.  

Because of the nature of the Evolution Index, the T-Test was selected to help us gain insight on how well the average growth of our sample market (say, Atlanta) compared to the average growth of all the other major markets in the US.  But because aspects of demographic and economic distribution are unique to each market, the markets themselves are independent of each other - what happens in Atlanta does not determine what happens in Boise, Idaho.  Therefore, an ANOVA test was run to see if there was any statisical significance in market growth amongst all our sample markets.

Once the tests had been run, we had to determine whether to accept or reject each null hypothesis.  Deciding what to do revolves around what's known as a probability value, or **p-value** for short.  P-values tell us if the results of our statistical tests were significant or just happened by chance: were we actually on to something, or did we just luck out on a wild guess?  In sports, it would be like putting your money on the underdog to win, and they actually win.  Did all your pre-game analysis work, or did you just get lucky?  A low p-value means all your pre-work was accurate and effective; a high p-value means you just got lucky. 

Now the question is, "How do we know our p-value is low enough to reject our null hypothesis?"  There is a value known as 'alpha' (ususally either .05 and .01) we set at the beginning of testing.  Due to the limitless alternatives to every outcome, there is no 'truth' in Science, and the best researchers can do is make an assumption (the null hypothesis) and try to disprove it.  The 'alpha' is the 'significance level' we want to be correct in our assumption of the null hypothesis - we know there's no absolute truth, so we set a tolerance level at the beginning of testing to account for this.  Our 'alpha' value for this project was .01, which means we will reject our null hypothesis even if the probability of chance (the p-value) is less than 1%.   

**___________________________________________________________________________________________________________________________________**

### Modeling Findings After Exploration

**___________________________________________________________________________________________________________________________________**

## Results

After implementing, reviewing, and discussing the results produced by our model, we were able to divide cities into three markets: those with high growth, those that are high-density and have growth potential, and markets that are already hot.  Below is a snippet of our findings (details are in both the presentation and this repository):

| Markets with Greatest ROI Potential  | High Density with Potential  | Markets Already Hot |
|:------------------------------------:|:----------------------------:|:-------------------:|
| Appleton, WI                         | Albany, NY                   | Albuquerque, NM     |
| Bloomington, IL                      | Allentown, PA                | Atlanta, GA         |
| Columbia, SC                         | Anchorage, AL                | Baltimore, MD       |
