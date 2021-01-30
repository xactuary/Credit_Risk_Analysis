# Credit Card Risk Analysis using Machine Learning
## SUMMARY
Using a credit card application credit risk dataset provided by the Lending Club, a peer-to-peer lending company, I have used various unbalanced Machine Learning techniques to determine
a model for predicting whether or not an applicant will be a good or bad credit risk.

### Dataset and Data Preparation
The data set includes 68,817 rows (Observations) and 86 columns (Variables).  There is no data dictionary to describe what is in the variables so I am keeping them all and letting the machine learning algorithm try and find the relevant variables.  
  
The target variable is "loan_status" which describes whether the borrower is "low risk" or "high risk".  I set the y target variable to be equal to just this column. 

      # Create our target
      y = dummies.loan_status
  
To prepare the dataset for Machine learning, there are several feature variables that need to be converted to numeric. These are the ones that are highlighted as "objects" in the variable list.  I have identified the objects using Dtypes and used the get.dummies command to convert the specific columns.  I have left out loan_status because I don't want that changed over.  It would also be possible to define the X features by dropping the "loan status" then running get_dummies with enumerating the columns and it will convert all yes/no columns to numeric.  But I took the long route so I could see what was happening in each step.  The feature variables do not need to be scaled since I am using a classifier model.  

      dummies = pd.get_dummies(df,columns=["home_ownership","verification_status","issue_d","pymnt_plan","initial_list_status","next_pymnt_d","application_type","hardship_flag","debt_settlement_flag"])
      X = dummies.drop(columns="loan_status",axis=1)
Once the y variable and X features dataset are defined, I can look and see how the target variable splits out to see if we have balanced data or not.  The results are as follows:

![](https://github.com/xactuary/Credit_Risk_Analysis/blob/master/Resources/ycounter1.PNG)

This result shows that this is a very unbalanced dataset.  So the following is an analysis using imbalanced machine learning techniques.

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

3.  ### Undersampling using Cluster Centroids

Undersampling techniques reduce the majority class rather than increasing the minority class.  Our random undersampling model reduces the majority class by randomly selecting instances from the majority class to remove from the dataset. Cluster Centroids identifies clusters in the majority class then creates new data points by picking data that would be representative of the clusters.  Then the majority class is undersampled to reduce the dataset.   

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

The accuracy on this model went way down.  This is also reflected in the F1 score decreasing for the low_risk.  This model is also a poor performer for predicting the high_class applications.

4.  ### Combination (Over and Under) Sampling using SMOTEEN

SMOTEEN is a technique that combines over and under sampling.  It combines the SMOTE method with the Edited Nearest Neighbors to both increase the high_risk sample and decrease the low_risk sample.  

After running the SMOTEEN model, the counts in each class are now:

|Target Class|Number of Records|
| :---   |----:|
|low_risk | 46,653|
|high_risk| 51,361|

Now the machine learning model is run on the newly sampled testing dataset which results in the following Statistics:

|Metric|Result|Result|
| :---   |----:|----:|
|Accuracy | .651|--|
|Confusion Matrix| 73|28|
|--|7,184|9,920|


|Class|Precision|Recall|F1|
| :---   |----:|----:|----:|
|high_risk|.01|.72|.02|
|low_risk|1.0|.58|.73|

This model still fails to show that it can properly predict the high_risk applications.  The accuracy improved over the Clustered Centroid method but the other statistics still show a very low F1 for high-risk so there is virtually no ability for this model to predict the high_risk.  Based on these 4 models, we would reject all of them as having any reliability for projecting the high_risk clients.  

5. ### Balanced Random Forest Classifier - BRFC

This model uses an extended decision tree to try and categorize the data into high_risk versus low_risk.  Note that we are using the original data for this rather than the over and under sampled data.  
The results of the BRFC model are as follows:

|Metric|Result|Result|
| :---   |----:|----:|
|Accuracy | .788|--|
|Confusion Matrix| 71|30|
|--|2,153|14,951|


|Class|Precision|Recall|F1|
| :---   |----:|----:|----:|
|high_risk|.03|.70|.06|
|low_risk|1.0|.87|.93|

This model improves the accuracy and the F1 score for the low_risk class.  In fact, it is a very good performing for predicting the low_risk.  However, it still has very poor statistics for predicting the high_risk class.  The F1 improved but is still incredibly low so again this model cannot be used to predict which applications will be high_risk.

6.  ### Easy Ensemble AdaBoost Classifier

AdaBoost is adaptive boosting which runs the model multiple times and each time improves on the prior by giving extra weight to the errors of the previous model.

The results of the BRFC model are as follows:

|Metric|Result|Result|
| :---   |----:|----:|
|Accuracy | .93|--|
|Confusion Matrix| 93|8|
|--|983|116,121|


|Class|Precision|Recall|F1|
| :---   |----:|----:|----:|
|high_risk|.09|.92|.16|
|low_risk|1.0|.94|.97|

So this model has very high accuracy and precision. However, the recall is very low for high_risk so the F1 is still too low to be a reliable model.  The F1 has increased to double digits for the high_risk but is still very low.  So again, this model is not going to be useful for predicting the high_risk applications.  

## Summary

The following table shows how the different methods perform comparatively via selected performance metrics.

| Method                 | Class     | Accuracy | Precision | Recall | F1  |
|------------------------|-----------|----------|-----------|--------|-----|
| Oversampling           | High_risk | .65      | .01       | .70    | .02 |
|                        | Low_risk  |          | 1.0       | .60    | .75 |
| SMOTE                  | High_risk | .66      | .01       | .63    | .02 |
|                        | Low_risk  |          | 1.0       | .69    | .63 |
| Cluster Centroids      | High_risk | .547     | .01       | .68    | .01 |
|                        | Low_risk  |          | 1.0       | .41    | .58 |
| SMOTEEN                | High_risk | .651     | .01       | .72    | .02 |
|                        | Low_risk  |          | 1.0       | .58    | .73 |
| Balanced Random Forest | High_risk | .788     | .03       | .70    | .06 |
|                        | Low_risk  |          | 1.0       | .87    | .93 |
| Easy Ensemble AdaBoost | High_risk | .93      | .09       | .92    | .16 |
|                        | Low_risk  |          | 1.0       | .94    | .97 |

None of the models performs well to predict the high_risk applicatons.  The Easy Ensemble ADABoost gives the best results metrics for getting the low_risk correct and closest on the high_risk.  Given the low F1 scores for high_risk, I would not use any of these models for making reliable predictions.  I would consider developing some more potential predictive features plus gathering more data that includes more high_risk real data points.  
