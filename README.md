# Prosper Loan Data
Online peer-to-peer (P2P) lending markets enable individual consumers to borrow from, and lend money to, one another directly. We study the borrower-, loan- and group- related determinants of performance predictability in an online P2P lending market by conceptualizing financial and social strength to predict borrower rate and whether the loan would be timely paid.

Our goal is to explore a sample of Prosper loan data to uncover borrower motivations when applying for loans, and identify several factors that may influence loan favorability.


## Understanding the Dataset
**Initial Notes on Dataset Structure:**
The dataframe comprises **113,937 rows** and **81 columns (features)**. The most of these 81 columns contain numeric data.

**Features of Interest:**
The dataset currently contains loads of information. However, the goal of this exploration is to understand the different borrower motivations when applying for loans, including the different factors that may influence loan favorability. As a result, we will direct our exploratory efforts towards the following features:

- **ListingCreationDate:** The date the listing was created.

- **ListingCategory (numeric):** The category of the listing that the borrower selected when posting their listing: 0 - Not Available, 1 - Debt Consolidation, 2 - Home Improvement, 3 - Business, 4 - Personal Loan, 5 - Student Use, 6 - Auto, 7- Other, 8 - Baby&Adoption, 9 - Boat, 10 - Cosmetic Procedure, 11 - Engagement Ring, 12 - Green Loans, 13 - Household Expenses, 14 - Large Purchases, 15 - Medical/Dental, 16 - Motorcycle, 17 - RV, 18 - Taxes, 19 - Vacation, 20 - Wedding Loans.

- **BorrowerState:** The two letter abbreviation of the state of the address of the borrower at the time the Listing was created.

- **isBorrowerHomeowner:** A Borrower will be classified as a homowner if they have a mortgage on their credit profile or provide documentation confirming they are a homeowner.

- **IncomeRange:** The income range of the borrower at the time the listing was created.

- **IncomeVerifiable:** The borrower indicated they have the required documentation to support their income.

- **DebtToIncomeRatio:** The debt to income ratio of the borrower at the time the credit profile was pulled. This value is Null if the debt to income ratio is not available. This value is capped at 10.01 (any debt to income ratio larger than 1000% will be returned as 1001%).

- **StatedMonthlyIncome:** The monthly income the borrower stated at the time the listing was created.

- **ProsperRating (Alpha):** The Prosper Rating assigned at the time the listing was created between AA - HR. Applicable for loans originated after July 2009.

- **Term:** The length of the loan expressed in months.

- **EmploymentStatus:** The employment status of the borrower at the time they posted the listing.

- **LoanStatus:** The current status of the loan: Cancelled, Chargedoff, Completed, Current, Defaulted, FinalPaymentInProgress, PastDue. The PastDue status will be accompanied by a delinquency bucket.

- **LoanOriginalAmount:** The origination amount of the loan.

- **BorrowerAPR:** The Borrower's Annual Percentage Rate (APR) for the loan.

Generally, we aim to measure loan favorability in terms of Prosper rating and Annual percentage rate (borrower APR). They will be our dependent variables.

## Preprocessing and Sentiment Analysis

1. Key features need to be isolated from the dataset.
2. There are 871 duplicate records in the dataset.
3. ListingCategory (numeric) and ProsperRating (Alpha) can be reassigned with column names that are easier to work with.
4. ListingCreationDate is stored with the wrong datatype. It should be a pandas datetime object.
5. The numeric information in ListingCategory (numeric) could be better expanded to reflect the actual reasons for the loan. The data dictionary contains helpful information for this.
6. 'Not employed' entries in IncomeRange could be safely replaced with 0.
7. The dependent variables (BorrowerAPR and ProsperRating (Alpha)) contain null values. The DebtToIncomeRatio column also contains null values.
8. ProsperRating and IncomeRange are ordinal categorical variables and should be stored in ordered form.

Before performing the EDA, we did data cleaning and addressed the null values in our dataframe.
```
import pandas as pd
import numpy as np

# Filter out null values from the dataframe
for col in null_columns:
    clean_df = clean_df[clean_df[col].notnull()]
```


## EDA
**Introduction:**

- **clear_data** dataset comprises of 113936 rows and 14 columns.
- Dataset comprises of bool(2), float64(3), int64(3), object(6) data types. 

**Information of Dataset:**

We asked many questions for each variabele during studying it in those phases Univariate, Bivariate, Multivariate Analysis and had some observations for each question.

**Univariate Analysis:**

We Plottd histogams and countplots to see the distributions of the features in our dataset.
<br/>
Ex: 
- How are Borrower APR values distributed in the dataset, do the majority of loans have high or low APR values?
- How are the values for Debt-to-income ratio distributed, are borrowers taking more debt than their income could possibly handle?
- What is the distribution of loan amounts requested by borrowers?

**Descriptive Statistics:**

Using **describe()** we could get the following result for the numerical features

||DebtToIncomeRatio|StatedMonthlyIncome|Term|LoanOriginalAmount|BorrowerAPR|
|:----|:----|:----|:----|:----|:----|
|Count|76768|83982|83982|83982|83982|
|Mean|0.258692|5930.614|42.462813|9061.224381|0.226945|
|Standard Deviation|0.319727|8268.432|11.639032|6279.649648|0.080047|
|Minimum|0.000000|0.000000|12.000000|1000.000000|0.045830|
|25%|0.150000|3426.938|36.000000|4000.000000|0.163610|
|50%|0.220000|5000.000|36.000000|7500.000000|0.219450|
|75%|0.320000|7083.333|60.000000|13500.000000|0.292540|
|Maximum|10.010000|1.750003e+06|60.000000|35000.000000|0.423950|


**Bivariate Analysis:**

We Plottd histogams and countplots to see the distributions of the features in our dataset.
<br/>
Ex: 
- Is there any relationship between numerical features such as BorrowerAPR, StatedMonthlyIncome, LoanOriginalAmount, DebtToIncomeRatio and categorical features like Term, ProsperRating, and IncomeRange?
- What is the relationship between ListingCategory and LoanOriginalAmount?
- What is the relationship between Term and BorrowerAPR?
    
    
**Multivariate Analysis:**

We Plottd histogams and countplots to see the distributions of the features in our dataset.
<br/>
Ex:
- What is the relationship between LoanOriginalAmount, ProsperRating and IsBorrowerHomeowner?
- What is the relation between BorrowerAPR, LoanOriginalAmount, and IncomeRange?
- What is the relationship between ProsperRating, LoanOriginalAmount, loan Term?

**Visualisation of Variables:**

- For a particular day, the opening and closing cost does not have much difference.
- Upon plotting box plot between **Volume** and **Label** we could see that there are outliers. Other numnerical features doesnot have any outliers in them.
- Observed outliers in few categorical columns as well.
 
 
## Preprocessing Again

we do feature engineering before the modelling part:

- Create new column for **LoanStatus** with only two values: Accepted, High Risk
- Convert **LoanStatus** from categorical to numerical

And we appied Mutual Information Scores (MI) and created principal components transformation (PCA) for our features.


## Model Building

#### Metrics considered for Model Evaluation
**Accuracy , Precision , Recall and F1 Score**
- Accuracy: What proportion of actual positives and negatives is correctly classified?
- Precision: What proportion of predicted positives are truly positive ?
- Recall: What proportion of actual positives is correctly classified ?
- F1 Score : Harmonic mean of Precision and Recall

#### Logistic Regression
- Logistic Regression helps find how probabilities are changed with actions.
- The function is defined as P(y) = 1 / 1+e^-(A+Bx) 
- Logistic regression involves finding the **best fit S-curve** where A is the intercept and B is the regression coefficient. The output of logistic regression is a probability score.

#### Random Forest Classifier
- The random forest is a classification algorithm consisting of **many decision trees.** It uses bagging and features randomness when building each individual tree to try to create an uncorrelated forest of trees whose prediction by committee is more accurate than that of any individual tree.
- **Bagging and Boosting**: In this method of merging the same type of predictions. Boosting is a method of merging different types of predictions. Bagging decreases variance, not bias, and solves over-fitting issues in a model. Boosting decreases bias, not variance.
- **Feature Randomness**:  In a normal decision tree, when it is time to split a node, we consider every possible feature and pick the one that produces the most separation between the observations in the left node vs. those in the right node. In contrast, each tree in a random forest can pick only from a random subset of features. This forces even more variation amongst the trees in the model and ultimately results in lower correlation across trees and more diversification.


### Choosing the features
After choosing LDA model based on confusion matrix here where **choose the features** taking in consideration the deployment phase.

We know from the EDA that all the features are highly correlated and almost follows the same trend among the time.

When we apply the **Logistic Regression** model the accuracy was 79%.
```
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error,mean_absolute_error,accuracy_score,classification_report,confusion_matrix

# Split the data into features (X) and target (y)
X = df.drop('ProsperRating',axis = 1)
y = df['ProsperRating']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fitting the data
lr = LogisticRegression(penalty='l2', C=1.0, solver='saga')
lr.fit(X_train, y_train)
```

When we apply **Random Forest** model for classification the accuracy increased to 88%.
```
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint

# Tree Visualisation
from sklearn.tree import export_graphviz
from IPython.display import Image
import graphviz
     
# Split the data into features (X) and target (y)
X = df.drop('ProsperRating',axis = 1)
y = df['ProsperRating']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
     
# Fitting the data
rf = RandomForestClassifier()
rf.fit(X_train, y_train)

```

## Deployment
you can access our app by following this link [P2P-Data-Application-Streamlit](https://jainhimanshu908-p2p-data-app-st-za88jb.streamlit.app)
### Streamlit
- It is a tool that lets you creating applications for your machine learning model by using simple python code.
- We write a python code for our app using Streamlit; the app asks the user to enter the following data (**isBorrowerHomeowner**, **LoanStatus**, **EmploymentStatus**,**BorrowerAPR**,**Term**,**StatedMonthlyIncome**,**DebtToIncomeRatio**,**IncomeVerifiable**,**IncomeRange**,**BorrowerState**).
- The output of our app will indicates (**Borrower Rating out of 7**, **Borrower Category**, **Getting the Loan Chance in %**)
- To deploy it on the internt we have to deploy it to Streamlit.
