# Credit Card Risk Analysis using Machine Learning
## SUMMARY
Using a credit card application credit risk dataset provided by the Lending Club, a peer-to-peer lending company, I have used various unbalanced Machine Learning techniques to determine
a model for predicting whether or not an applicant will be a good or bad credit risk.

### Dataset and Data Preparation
The data set includes 68,817 rows (Observations) and 86 columns (Variables).  There is no data dictionary to describe what is in the variables so I am keeping them all and letting the machine learning algorithm try and find the relevant variables.  

The target variable is "Loan_Status" which describes whether the borrower is "low risk" or "high risk".  I set the y target variable to be equal to just this column.  
To prepare the dataset for Machine learning, there are several feature variables that need to be converted to numeric. These are the ones that are highlighted as "objects" in the variable list.  I have identified the objects using Dtypes and used the get.dummies command to convert the specific columns.  I have left out loan_status because I don't want that changed over.  It would also be possible to define the X features by dropping the "loan status" then running get_dummies with enumerating the columns and it will convert all yes/no columns to numeric.  But I took the long route so I could see what was happening in each step.  The feature variables do not need to be scaled since I am using a classifier model.  

Once the y variable and X features dataset are defined, I can look and see how the target variable splits out to see if we have balanced data or not.  The results are as follows:

|Target Class|Number of Records|
|____________|_________________|
| low_Risk|68,470|
_________________
|High_Risk|347|






