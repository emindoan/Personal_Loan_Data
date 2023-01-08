#!/usr/bin/env python
# coding: utf-8

# # Import Libraries

# In[212]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import roc_curve, auc
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns',None)
pd.set_option('display.max_rows',None)


# In[160]:


get_ipython().system('pip install imbalanced-learn==0.6.0')


# #  data description
# 
# About this file
# Attribute Information:
# 
# ID : Customer ID
# 
# Age : Customer's age in completed years
# 
# Experience : #years of professional experience
# 
# Income : Annual income of the customer ($000)
# 
# ZIP Code : Home Address ZIP code.
# 
# Family : Family size of the customer
# 
# CCAvg : Avg. spending on credit cards per month ($000)
# 
# Education : Education Level.
# 1: Undergrad;
# 2: Graduate;
# 3: Advanced/Professional
# 
# Mortgage : Value of house mortgage if any. ($000)
# 
# 10.Personal Loan : Did this customer accept the personal loan offered in the last campaign?
# 
# 11.Securities Account : Does the customer have a securities account with the bank?
# 
# 12.CD Account : Does the customer have a certificate of deposit (CD) account with the bank?
# 
# 13.Online : Does the customer use internet banking facilities?
# 
# 14.Credit card : Does the customer use a credit card

# In[102]:


df = pd.read_csv('Bank_Personal_Loan_Modelling.csv')


# In[3]:


df.head(10)


# In[103]:


df.rename(columns ={"CCAvg": "Monthly_Avg_CC_Spending", 
                    "ZIP Code": "ZIP_Code",
                    "Personal Loan": "Personal_Loan_Offer",
                    "Securities Account": "Securities_Account",
                    "CD Account": "CD_Account"}, inplace = True)
df.head()


# # EDA Analysis

# #### There are no missing and duplicate values in the dataset.
# 

# In[5]:


df.info()


# In[6]:


df.isnull().sum()


# In[7]:


df.duplicated().sum()


# In[8]:


df.head()


# #### One-hot Encoding
# 

# In[104]:


df["Undergrad_Edu_Level"]    = df["Education"].apply(lambda x:1 if x==1 else 0)
df["Graduate_Edu_Level"]     = df["Education"].apply(lambda x:1 if x==2 else 0)
df["Professional_Edu_Level"] = df["Education"].apply(lambda x:1 if x==3 else 0)
df.head()


# ### Descriptive Statistics
# #####  Income has a high std #####
# ##### Monthly_Avg_CC_Spending as almost already normal distribution shape #####
# ##### There are negative values in experience #####

# In[105]:


df.columns
df_list = df[["Age","Experience","Income","Family","Monthly_Avg_CC_Spending","Mortgage"]]
df_list.head(10)
df_list.describe().T


# In[106]:


df.loc[df["Experience"]<0]
df.loc[df["Experience"]<0].count() ## 52 observations


# In[107]:


# mean and median are almost already the same
df_pst_exp = df.loc[df['Experience'] > 0]
df_pst_exp_mean = df_pst_exp["Experience"].mean() # 20.31
df_pst_exp_median = df_pst_exp["Experience"].median() # 20
print(df_pst_exp_mean)
print(df_pst_exp_median)


# In[108]:



# Step 1: Group data by column 'Age'
grouped = df.groupby(by='Age')

# Step 2: Define a custom function to replace negative values with the mean of positive values for each group
def replace_with_mean(group):
    # Select positive values in the group
    mask = group > 0
    # Calculate the mean of positive values
    mean = group[mask].mean()
    # Replace negative values with the mean
    group[~mask] = mean
    return group

# Step 3: Apply the custom function to each group
df['Experience'] = grouped['Experience'].apply(replace_with_mean)

# Step 4: Replace NA values with 0
df = df.fillna(0)

display(df)


# #### Converting the average credit card spengind from monthly average to annual average like income column.
# #### There is no relation between ID columns or ZIP Code coulumn and any ather variable.

# In[109]:


df['Ann_Avg_CC_Spending'] = df['Monthly_Avg_CC_Spending'] * 12
df.drop(['ID', 'ZIP_Code','Monthly_Avg_CC_Spending'], axis = 1, inplace=True)
df.head()


# In[110]:


df.corr()


# ### Correlation Analysis
# #### There is a almost perfect correlation between Experience and Age.
# #### The correlation between income, monthly average credit card spending, and deposit accounts might seem meaningful. 

# In[111]:


plt.figure(figsize=(16,8))
sns.heatmap(df.corr(),annot=True,cmap='Blues')
plt.show()


# # Comments
# 'Age' and 'Experience' are correlated with each other.
# 
# 'Income' and ‘Ann_Avg_CC_Spending' correlated with each other.
# 
# 'CD_Account' has a correlation with 'Credit_Card', 'Securities Account', 'Online', ‘ann_CCAvg' and 'Income'.
# 
# 'Personal_Loan_Offer' has correlation with 'Income’, 'Ann_Avg_CC_Spending', 'CD_Account', 'Mortgage', and 'Education'.
# 
# 'Mortgage' has moderate correlation with 'Income'
# 
# 'Income' influences ‘Ann_Avg_CC_Spending', 'Personal_Loan_Offer', 'CD_Account' and 'Mortgage'.
# 

# In[112]:


#only 9% of customers accepted the loan offer.
df.Personal_Loan_Offer.value_counts()
df.Personal_Loan_Offer.value_counts(normalize=True)


# In[113]:


df.head()


# In[120]:


df.Personal_Loan_Offer.value_counts().plot(kind="pie")
plt.title("Value counts of Personal_Loan_Offer")
plt.xlabel("Income")
plt.xticks(rotation=1)
plt.ylabel("Count")
plt.show()


# In[118]:


c = ["red","blue"]
sns.scatterplot(x = 'Age', y = 'Income', data = df, c=c, hue = 'Personal_Loan_Offer')


# Clients with income more than 100k are more likely to get loan

# In[122]:


sns.scatterplot(x = 'Age', y = 'Ann_Avg_CC_Spending', data = df, hue = 'Personal_Loan_Offer')


# In[134]:


sns.lmplot(x = "Income", y="Ann_Avg_CC_Spending", data=df, fit_reg = False, hue="Personal_Loan_Offer",
        legend = False, markers =["o","x"] )
plt.legend(loc="lower right")
plt.show()


# Clients with annual CC spending average more than 30 are more likely to get loan

# In[123]:


sns.countplot(x='Experience', hue = 'Personal_Loan_Offer', data = df)


# In[124]:


sns.countplot(x='Family', hue = 'Personal_Loan_Offer', data = df)


# As it seems in previous two graph the Family and Experience has a low effect in the personal loan attribute

# In[126]:


sns.countplot(x='CreditCard', hue = 'Personal_Loan_Offer', data = df)


# In[127]:


sns.countplot(x='Securities_Account', hue = 'Personal_Loan_Offer', data = df)


# In[128]:


sns.countplot(x='CD_Account', hue = 'Personal_Loan_Offer', data = df)


# In[129]:


sns.catplot(x='Securities_Account', y = 'CD_Account', data = df, kind = 'bar', hue = 'Personal_Loan_Offer' )


# In[130]:


sns.catplot(x='CreditCard', y = 'CD_Account', data = df, kind = 'bar', hue = 'Personal_Loan_Offer' )


# After investigating previous plots, after drop the 'ID' and 'ZIP Code' coulmns have influence on each other

# In[131]:


# with regression
sns.pairplot(df, kind="reg")
plt.show()

 
# without regression
sns.pairplot(df, kind="scatter")
plt.show()


# In[135]:


df_desc = df[["Age","Experience","Income","Family","Ann_Avg_CC_Spending","Mortgage",
             "Securities_Account","CD_Account","Online","CreditCard","Personal_Loan_Offer"]]


# In[136]:


df_desc.groupby("Personal_Loan_Offer").describe()


# In[137]:


df.groupby("Personal_Loan_Offer").mean()


# In[237]:


fig, axes = plt.subplots(2,2, figsize=(14,7))
sns.distplot(df['Age'], ax=axes[0,0])
sns.distplot(df['Ann_Avg_CC_Spending'], ax=axes[0,1])
sns.distplot(df['Income'], ax=axes[1,0])
sns.distplot(df['Experience'], ax=axes[1,1])
fig.tight_layout()


# In[138]:


print(df['Age'].describe())
plt.figure(figsize=(9, 8))
sns.distplot(df['Age'], color='b', bins=50, hist_kws={'alpha': 0.5});


# In[141]:


print(df['Ann_Avg_CC_Spending'].describe())
plt.figure(figsize=(9, 8))
sns.distplot(df['Ann_Avg_CC_Spending'], color='y', bins=50, hist_kws={'alpha': 0.5});


# In[142]:


print(df['Experience'].describe())
plt.figure(figsize=(9, 8))
sns.distplot(df['Experience'], color='b', bins=50, hist_kws={'alpha': 0.5});


# In[143]:


print(df['Income'].describe())
plt.figure(figsize=(9, 8))
sns.distplot(df['Income'], color='g', bins=100, hist_kws={'alpha': 0.4});


# In[144]:


df_corr = df.corr()['Personal_Loan_Offer'][:-1] 
golden_features_list = df_corr[abs(df_corr) >= 0.2].sort_values(ascending=False)
print("There is {} strongly correlated values with Personal_Loan_offer:\n{}".format(len(golden_features_list),
                                                                                    golden_features_list))


# In[145]:


corr = df.drop('Personal_Loan_Offer', axis=1).corr() 
plt.figure(figsize=(12, 10))

sns.heatmap(corr[(corr >= 0.3) | (corr <= -0.3)], 
            cmap='crest', vmax=2.0, vmin=-1.0, linewidths=0.3,
            annot=True, annot_kws={"size": 10}, square=False);


# ### Splitting the Data
# ##### Training and Testing Set in the ratio of 70:30

# In[240]:



X = df.drop('Personal_Loan_Offer', axis = 1)
y = df['Personal_Loan_Offer']
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=34)

# X_train.shape, X_test.shape
print("Training Dataset Shape:",X_train.shape)
r, c = X_train.shape
print("Rows= ",r )
print("Columns= ",c)
print("Testing Dataset Shape:",X_test.shape)
r, c = X_test.shape
print("Rows= ",r )
print("Columns= ",c)


# #### Determine Mutual Information
# ##### Calculate the mutual information between the variables and the target. The smaller the value of MI, the less information we can infer from the feature about the target.

# In[148]:


mi = mutual_info_classif(X_train, y_train)
mi


# Let's capture the above array in a pandas series add the variable names in the index sort the features based on their mutual information value and make a plot

# In[241]:


mi = pd.Series(mi)
mi.index = X_train.columns
mi.sort_values(ascending=False).plot.bar(figsize=(18, 4))
plt.ylabel('Mutual Information')


# The  features of left of the plot have higher mutual information values whereas features of the right of the plot have almost zero mutual information (mi) values.
# 
# To determine a threshold or cut-off value for the mutual information values in order to select features
# 
# To determine a treshold, select top k features, where k is an arbitrary number of features

# ### Select top k features based on Mutual Information¶
# #### Here we will select the top 8 features based on their mutual information value

# In[243]:


# select features
sel_ = SelectKBest(mutual_info_classif, k=8).fit(X_train, y_train)

# display features
X_train.columns[sel_.get_support()]


# In[244]:


# X_train.shape,X_test.shape
r, c = X_train.shape
print("Train Dataset:")
print("Rows=",r)
print("Column=",c)
r, c = X_test.shape
print("Test Dataset:")
print("Rows=",r)
print("Column=",c)


# ### SMOTE
# Here we can see the data is Unbalance. The lable which contain '0' is greater than the label containg '1'.
# 
# So here we need to Balance the dataset in this way, the model gets train in similar ways on both the labels.

# In[245]:


print("Before OverSampling, counts of label '1': {}".format(sum(y_train == 1)))
print("Before OverSampling, counts of label '0': {} \n".format(sum(y_train == 0)))


# In[246]:



sm = SMOTE(random_state = 2)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)


print('After OverSampling, the shape of X_train: {}'.format(X_train_res.shape))
print('After OverSampling, the shape of y_train: {} \n'.format(y_train_res.shape))

print("After OverSampling, counts of label '1': {}".format(sum(y_train_res == 1)))
print("After OverSampling, counts of label '0': {}".format(sum(y_train_res == 0)))


# In[247]:


#checking the length of the Training data before balancing
print(len(X_train))


# In[248]:


#checking the length of the Test data after balancing
print(len(X_train_res))


# # Algorithm
# On our work we will use five kind of algorithms to the find algorithm with highest f1_score
# 
# #### LogisticRegression
# #### SVM
# #### K-NN
# #### DecisionTreeClassifier
# #### RandomForestClassifier

# ###  Logestic Regression

# In[249]:


classifier = LogisticRegression(random_state = 34)
classifier.fit(X_train, y_train)


# In[250]:


# logistic regression object
lr = LogisticRegression()

# train the model on train set
lr.fit(X_train_res, y_train_res)

predictions = lr.predict(X_test)

#import classification report
from sklearn.metrics import confusion_matrix, classification_report

# print classification report
print(classification_report(y_test, predictions))


# In[251]:


y_pred = classifier.predict(X_test)
print(y_pred)


# In[252]:


y_pred = classifier.predict(X_test)
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)


# In[253]:


sns.heatmap(cm,annot=True)
plt.show()


# In[254]:


LR_acc1 = accuracy_score(y_test, y_pred)
print("Accuracy score for Logistic Regression Model: {:.2f} %".format(LR_acc1*100))


# ###  ROC Curve

# In[255]:


#---find the predicted probabilities using the test set
probs = classifier.predict_proba(X_test)
preds = probs[:,1]

#---find the FPR, TPR, and threshold---
fpr, tpr, threshold = roc_curve(y_test, preds)

roc_auc = auc(fpr, tpr)

import matplotlib.pyplot as plt
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate (TPR)')
plt.xlabel('False Positive Rate (FPR)')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc = 'lower right')
plt.show()


# In[256]:


# By using SMOTE

parameters = [{'penalty': ['l1','l2'], 'C': np.arange(1,10) }]

grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)

grid_search.fit(X_train_res, y_train_res)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_

print("Best Accuracy: {:.2f} %".format(best_accuracy*100))
print("Best Parameters:", best_parameters)


# # SVM

# In[257]:


SVMclassifier = SVC(kernel = 'linear', random_state = 0)
SVMclassifier.fit(X_train, y_train)

y_pred = SVMclassifier.predict(X_test)
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)


# In[258]:


sns.heatmap(cm,annot=True)
plt.show()


# In[259]:


SVM_acc1 = accuracy_score(y_test, y_pred)
print("Accuracy score for SVM Model: {:.2f} %".format(SVM_acc1*100))


# In[260]:


# By using SMOTE

classifier2 = SVC(kernel = 'linear', random_state = 0)
SVM_classifier = classifier2.fit(X_train_res, y_train_res)

y_pred = SVM_classifier.predict(X_test)
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)


# In[273]:


sns.heatmap(cm,cmap="YlGn",annot=True)
plt.show()


# In[274]:


SVM_acc2 = accuracy_score(y_test, y_pred)
print("Accuracy score for SVM Model: {:.2f} %".format(SVM_acc2*100))


# # Hyperparameter Tuning

# In[275]:


parameters = [{'C': np.arange(1,10) }]

grid_search = RandomizedSearchCV(estimator = SVMclassifier,
                            param_distributions = parameters,
                            scoring = 'accuracy',
                            cv = 10,
                            n_jobs = -1)


# In[276]:


grid_search.fit(X_train_res, y_train_res)
SVM_acc_sorte = grid_search.best_score_
best_parameters = grid_search.best_params_

print("Best Accuracy of SVM: {:.2f} %".format(SVM_acc_sorte*100))
print("Best Parameters of SVM:", best_parameters)


# #  K-NN

# In[277]:


KNNclassifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
KNNclassifier.fit(X_train, y_train)

y_pred = KNNclassifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)


# In[278]:


sns.heatmap(cm,cmap = "RdBu",annot=True)
plt.show()


# In[279]:


KNN_acc1 = accuracy_score(y_test, y_pred)
print("Best Accuracy of K-NN: {:.2f} %".format(KNN_acc1*100))


# In[280]:


KNN_classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
KNN_classifier.fit(X_train_res, y_train_res)


# In[281]:


y_pred = KNN_classifier.predict(X_test)
y_pred


# In[282]:


cm = confusion_matrix(y_test, y_pred)
print(cm)


# In[285]:


sns.heatmap(cm,cmap = "OrRd",annot=True)
plt.show()


# In[286]:


KNN_acc2 = accuracy_score(y_test, y_pred)
print("Best Accuracy of K-NN: {:.2f} %".format(KNN_acc2*100))


# Hyperparameter Tuning

# In[287]:


parameters = [{ 'n_neighbors' :  np.arange(1,10)  }]

grid_search = GridSearchCV(estimator = KNN_classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)


# In[288]:


grid_search.fit(X_train_res, y_train_res)
KNN_acc3 = grid_search.best_score_
best_parameters3 = grid_search.best_params_


# In[289]:


print("Best Accuracy of KNN after Hyperparameter tuning: {:.2f} %".format(KNN_acc3*100))
print("Best Parameters of KNN:", best_parameters3)


# # Decision Tree Classifier

# In[290]:


DTclassifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
DTclassifier.fit(X_train, y_train)


# In[291]:


y_pred = DTclassifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)


# In[292]:


y_pred = classifier.predict(X_test)
DT_acc1 = accuracy_score(y_test, y_pred)
print(f"Accuracy score for Decision Tree: {DT_acc1*100}")


# By using SMOTE

# In[293]:


DT_classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
DT_classifier.fit(X_train_res, y_train_res)


# In[294]:


y_pred = DT_classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)


# In[297]:


sns.heatmap(cm,cmap= "PuBu",annot=True)
plt.show()


# In[298]:


DT_acc2 = accuracy_score(y_test, y_pred)
print("Accuracy score for Decision Tree: {:.2f} %".format(DT_acc2*100))


# Hyperparameter Tuning

# In[300]:


parameters = [{  }]

grid_search = GridSearchCV(estimator = DT_classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)


# In[301]:


grid_search.fit(X_train_res, y_train_res)
DT_acc3 = grid_search.best_score_
best_parameters5 = grid_search.best_params_


# In[302]:


print("Best Accuracy of Decision Tree Classifier: {:.2f} %".format(DT_acc3*100))
print("Best Parameters of Decision Tree Classifier:", best_parameters5)


# # Random Forest Classifier

# In[303]:


RFclassifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
RFclassifier.fit(X_train, y_train)


# In[304]:


y_pred = RFclassifier.predict(X_test)
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)


# In[305]:


RF_acc1 = accuracy_score(y_test, y_pred)
print(f"Random Forest Classification accuracy: {RF_acc1*100}")


# In[306]:


RF_classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
RF_classifier.fit(X_train_res, y_train_res)


# In[307]:


y_pred = RF_classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)


# In[309]:


sns.heatmap(cm,cmap="YlOrBr",annot=True)
plt.show()


# In[310]:


RF_acc2 = accuracy_score(y_test, y_pred)
print("Accuracy score for Random Forest: {:.2f} %".format(RF_acc2*100))


# Hyperparameter Tuning

# In[311]:


parameters = [{'n_estimators' : [10, 50, 100, 200], 'max_depth' : [3, 10, 20, 40]}]

grid_search = GridSearchCV(estimator = RF_classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)


# In[312]:


grid_search.fit(X_train_res, y_train_res)
RF_acc3 = grid_search.best_score_
best_parameters6 = grid_search.best_params_


# In[313]:


print("Best Accuracy of Random Forest with hyperparameter tuning: {:.2f} %".format(RF_acc3*100))
print("Best Parameters of Random Forest:", best_parameters6)


# In[314]:


model_list=[]
model_list2=[]
model_list.append(LR_acc1)
model_list2.append("Logistic Regression")
model_list.append(SVM_acc1)
model_list2.append("SVM")
model_list.append(KNN_acc1)
model_list2.append("K-NN")
model_list.append(DT_acc1)
model_list2.append("DTC")
model_list.append(RF_acc1)
model_list2.append("RFC")

plt.rcParams['figure.figsize']=22,10
sns.set_style("darkgrid")
ax = sns.barplot(x=model_list2, y=model_list, palette = "coolwarm", saturation =1.5)
plt.xlabel("Classification Models", fontsize = 20 )
plt.ylabel("Accuracy", fontsize = 20)
plt.title("Accuracy of different Classification Models with Unbalance Data", fontsize = 20)
plt.xticks(fontsize = 11, horizontalalignment = 'center', rotation = 8)
plt.yticks(fontsize = 13)
for p in ax.patches:
    width, height = p.get_width(), p.get_height()
    x, y = p.get_xy() 
    ax.annotate(f'{height:.2%}', (x + width/2, y + height*1.02), ha='center', fontsize = 'x-large')
plt.show()


# To reduce the type-2 error SMOTE and Hyperparameter Tuning will be used for more Accuracy

# In[315]:


model_list=[]
model_list2=[]
model_list.append(LR_acc1)
model_list2.append("Logistic Regression")
model_list.append(SVM_acc2)
model_list2.append("SVM")
model_list.append(KNN_acc2)
model_list2.append("K-NN")
model_list.append(DT_acc2)
model_list2.append("DTC")
model_list.append(RF_acc2)
model_list2.append("RFC")

plt.rcParams['figure.figsize']=22,10
sns.set_style("darkgrid")
ax = sns.barplot(x=model_list2, y=model_list, palette = "coolwarm", saturation =1.5)
plt.xlabel("Classification Models", fontsize = 20 )
plt.ylabel("Accuracy", fontsize = 20)
plt.title("Accuracy of different Classification Models with SMOTE ", fontsize = 20)
plt.xticks(fontsize = 11, horizontalalignment = 'center', rotation = 8)
plt.yticks(fontsize = 13)
for p in ax.patches:
    width, height = p.get_width(), p.get_height()
    x, y = p.get_xy() 
    ax.annotate(f'{height:.2%}', (x + width/2, y + height*1.02), ha='center', fontsize = 'x-large')
plt.show()


# In[316]:


model_list=[]
model_list2=[]
model_list.append(LR_acc1)
model_list2.append("Logistic Regression")
model_list.append(SVM_acc2)
model_list2.append("SVM")
model_list.append(KNN_acc3)
model_list2.append("K-NN")
model_list.append(DT_acc3)
model_list2.append("DTC")
model_list.append(RF_acc3)
model_list2.append("RFC")

plt.rcParams['figure.figsize']=22,10
sns.set_style("darkgrid")
ax = sns.barplot(x=model_list2, y=model_list, palette = "coolwarm", saturation =1.5)
plt.xlabel("Classification Models", fontsize = 20 )
plt.ylabel("Accuracy", fontsize = 20)
plt.title("Accuracy of different Classification Models with SMOTE and Hyperparameter Tuning", fontsize = 20)
plt.xticks(fontsize = 11, horizontalalignment = 'center', rotation = 8)
plt.yticks(fontsize = 13)
for p in ax.patches:
    width, height = p.get_width(), p.get_height()
    x, y = p.get_xy() 
    ax.annotate(f'{height:.2%}', (x + width/2, y + height*1.02), ha='center', fontsize = 'x-large')
plt.show()


# # Conclusion
# #####  Random Forest Classifier s the best algoritham to analyse with Unblance Data, SMOTE and SMOTE with Hyperparameter Tuning .
# 

# In[ ]:




