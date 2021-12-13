# ZIllow-regression-project


## Project Planning

## Goals 
My goal in this project is to:

- Identify the key Drivers of home prices and improve the baseline RSME.

### Questions 
####  1. What is driving house prices?
- Is there a relationship between number of bedrooms per home and county?
- Is there a relationship between square footage(area) and tax value ?
#### 2. Which County has the highest numbers of transactions for houses?
#### 3. Which months has the highest number of transactions?
#### 4. Is the tax rate the same in each county?
#### 5. What is the best model for prediction?


## Acquiring and Preparing
Understanding home market can be chanlenging, especially when there many variable that can influence home prices such as location, average income, assessed taxes and many more. In the case of Zillow, in order to have a clear picture Housing market and the key drivers of prices, I will perform the following:

- Acquire the Zillow data from using acquire.py.
- Prepare the data using prepare.py
- Clearly define two hypotheses, set an alpha, run the statistical tests needed, reject or fail to reject the Null Hypothesis, and document findings and takeaways
- Explore Data
- Establish a baseline accuracy
- Model building and Evaluation

- Deliver pressentation, recommendations and next steps


## Executive Summary
### Findings Include:
- Original data size was 52,442 
- After cleaning , dropping duplicates and outliers, I ended using 47892, about 91.3% of the data 
- drivers of tax value are area(sqr ft), numbers of bedrooms, bathrooms and year built
- 2nd degree Polynomial Regression beat the baseline RSME by 54,361
- 2nd degree Polynomial RMSE is 264,859 with r2 of .3 ,and the mean baseline is 319220
- The log error has some outliers


## Data Dictionary

 
 
| Features            |    Description                                                 |Data Type|
|---------------------|----------------------------------------------------------------|---------|
|area                 |Calculated Square footage of residence                          |float64  |
|bathrooms            |indicates the Number of bathrooms in residence                  |float64  |
|bedrooms             |indicates  the Number of Bedrooms in residence                  |float64  |
|parcelid             |indicates  the Lot id number.                                   |float64  |
|tax_value            |indicates  the The estimated value of the home                  |float64  |
|year_built           |indicates the The year the home was built                       |float64  |
|taxamount            |indicates the The amount of taxes payed the previous year (2016)|float64  |
|fips                 |indicates the County code resident resides within               |float64  |
|logerror             |indicates  the log error pulled from the 2017 predictions       |float64  |
|transactiondate      |indicates  the date the transaction was cloase on the home      |object   |
|tax_rate             |indicates the tax rate as a percentage of tax value             |float64  |
|county               |indicates the county is which the home is located               |object   |
| state               |indicates the state in which the property is located            |object   |


## Data Exploration
Goal: Explore the data, come to a greater understanding for feature selection.

Think about the following in this stage:

I've found that there was a lot of missing values for features that could have been useful in modeling

I like SQFT, Bathrooms, Bedrooms, Pool and year built as features.

## Modeling
- OLS
- LassoLars setting Alpha to 1
- 2 degree Polynomial regression

## Conclusion
- 
- Best performing model was 2nd degree polynomial
- Test Finished with r2 score of .3 and RMSE2 of 264,859 which was 54,361 more acurate than basline
- I believe there are a lot to unpack from this this dataset, and this model Definitely a steeping stone to handle this this dataset, an can be finetune with time. 
- given more time i would work on engineering better features for better perfopmance




