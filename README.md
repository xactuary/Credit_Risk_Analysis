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
| :---   |----:|
|low_risk | 68,470|
|high_risk| 347|

This result shows that this is a very unbalanced dataset.  So the following is an analysis using imbalanced maching learning techniques.

1. ### Oversampling Analysis

Oversampling chooses more instances from the "high_risk" category to increase the size of this class.  
After running the oversampling model, the counts in each class are now:


|Target Class|Number of Records|
| :---   |----:|
|low_risk | 51,366|
|high_risk| 51,366|

Now the machine learning model is run on the newly sampled testing dataset which results in the following Statistics:

|Metric|Result|Result|
| :---   |----:|----:|
|Accuracy | .65|--|
|Confusion Matrix| 71|30|
|--|6,883|10,221|


|Class|Precision|Recall|F1|
| :---   |----:|----:|----:|
|high_risk|.01|.70|.02|
|low_risk|1.0|.60|.75|

The precision is really low for the high_risk and perfect for the low_risk.  This is a problem with such an imbalanced dataset and the over-sampling did not help this.  Although the F1 is reasonable for the low_risk, it basically says the model can only predict 2% of the high_risk which makes the model fairly useless.  These results suggest that this model is not good for predicting which applicant might be high_risk which is really the goal of the study.

2. ### SMOTE Oversampling

SMOTE - synthetic minority oversampling technique - increases the size of the high_risk data by interpolating between the high_risk data points and adding these interpolated values in.  So they are close to existing high_risk data points but not equal to them.  So SMOTE creates synthetic sample values. 

After running the SMOTE model, the counts in each class are now:


|Target Class|Number of Records|
| :---   |----:|
|low_risk | 51,366|
|high_risk| 51,366|

Now the machine learning model is run on the newly sampled testing dataset which results in the following Statistics:

|Metric|Result|Result|
| :---   |----:|----:|
|Accuracy | .66|--|
|Confusion Matrix| 64|37|
|--|5,287|11,817|


|Class|Precision|Recall|F1|
| :---   |----:|----:|----:|
|high_risk|.01|.63|.02|
|low_risk|1.0|.69|.63|

So the accuracy went up just a little bit but now the F1 for low_risk went down.  So this model is even worse than the oversampling one.  It still does a poor job at predicting the high risk values.

3.  ### Undersampling

Undersampling techniques reduce the majority class rather than increasing the minority class.  Our random undersampling model reduces the majority class by randomly selecting instances from the majority class to remove from the dataset.  

After running the Cluster Centroids undersampling model, the counts in each class are now:


|Target Class|Number of Records|
| :---   |----:|
|low_risk | 246|
|high_risk| 246|

Now the machine learning model is run on the newly sampled testing dataset which results in the following Statistics:

|Metric|Result|Result|
| :---   |----:|----:|
|Accuracy | .547|--|
|Confusion Matrix| 69|32|
|--|10,073|7,031|


|Class|Precision|Recall|F1|
| :---   |----:|----:|----:|
|high_risk|.01|.68|.01|
|low_risk|1.0|.41|.58|



