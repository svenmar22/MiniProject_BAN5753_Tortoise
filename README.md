# MiniProject_BAN5753_Tortoise
PySpark Mini project- Banking Dataset
Deposit opening classification problem

The deposit opening classification problem deals with dataset from marketing campaign by a bank. The objective of the study is to predict customers who will subscribe to a term deposit based on the previous campaign dataset. This will help in planning future campaigns to be more successful by targeting the customers who are most likely to subscribe.

Dataset information:

•	Data is about an XYZ bank’s direct marketing campaign. Marketing campaigns were driven by telephone calls.
•	Data Set In many cases, more than one contact for the same client was required., in order to access if the product (deposit) would be ('yes') or not ('no') subscribed
•	The purpose of the classification is to forecast whether the customer will signup (yes/no) a term deposit (variable y).
•	The dataset: XYZ_Bank_Deposit_Data_Classification.csv, 20 entries/columns, sorted by date between May 2008 and November 2010. 
Attributes information:
1 - Age (Numeric)
2 - Job: type of job (categorical)
3 - Marital: marital status (categorical)
4 - Education (categorical)
5 - Default: has credit in default? (categorical)
6 - Housing: has housing loan? (categorical)
7 - Loan: has personal loan? (categorical)

regarding the latest contact in the ongoing campaign:

8 - Contact: contact communication type (categorical)
9 - Month: last contact month of year (categorical)
10 - Day_of_week: last contact day of the week (categorical)
11 - Duration: last contact duration, in seconds (numeric)
other attributes:
12 - Campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)
13 - Pdays: number of days that passed by after the client was last contacted from a previous campaign (numeric; 999 means client was not previously contacted)
14 - Previous: number of contacts performed before this campaign and for this client (numeric)
15 - Poutcome: outcome of the previous marketing campaign (categorical)
social and economic context attributes
16 - Emp.var.rate: employment variation rate - quarterly indicator (numeric)
17 - Cons.price.idx: consumer price index - monthly indicator (numeric) 
18 - Cons.conf.idx: consumer confidence index - monthly indicator (numeric) 
19 - Euribor3m: euribor 3 month rate - daily indicator (numeric)
20 - Nr.employed: number of employees - quarterly indicator (numeric)


Exploratory Data Analysis:

The exploratory data analysis was carried out to understand the variables and trends in the data.
Numerical and categorical type of variables are present: 
Numerical variables:   Age, duration, campaign, pdays, previous, emp.var.rate,cons.price.idx, cons.conf.idx, euribor3m,mr.employed
        
Categorical Variables:  job, marital, education, default', housing, loan, contact, month, day_of_week, poutcome,y

Summary statistics and plots for each of the numerical and categorical variables were generated.
Checked for null values and cardinality

Pearson Correlation: To identify the relation between the variables, Pearson correlation is used. This measures the degree of linear relationship between the variables and shows whether it’s correlated positively or negatively.

Prepare Data for Machine Learning:

Created new columns: Age group and pdays_contact. The Age column was grouped in to different categories. Also, pdays variable was replaced with pdays_contact with values o and 1 to denote previously contacted or not.
String Indexer and Onehot Encoder is used for categorical data to convert it into 0’s and 1’s.
Normalized the data using standard scaler. The transformers/estimators are applied in a pipeline. 

Train/test data Split: The data set is then divided into training/ Test data of the ration 80:20.

Four predictive modeling algorithms were then generated.


Classification predictive Modeling:

Following predictive modeling algorithms were compared:
1)	Logistic Regression- a supervised classification algorithm generally used for binary classification which forecasts the possibility of a target variable. Here the target variable is whether the customer will subscribe   for the term deposit as “yes” or “no”.
2)	Decision Tree model: The second supervised classification model built is Decision tree model
3)	Random Forest Model: Another model created was Random Forest Model which creates several decision trees and fit the data to one of them.
4)	K-Means Model: 

The accuracies of all the above models were compared. 
     
Summary

Of all the classification predictive models created, the logistic regression is selected as the champion model based on accuracy.


