import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
%config InlineBackend.figure_format = "retina" #enable 2x images

dataset = pd.read_csv("train.csv")
dataset.head()

dataset.shape

dataset.describe()
#notice that for features whose min=0, there are null values

#data visulization
dataset.hist(figsize=(16, 14))

dataset.plot(kind="box", subplots=True, layout=(3,3), sharex=False, sharey=False, figsize=(16,14))
#subplots=True：plot every feature as subplots

#check if features are correlated 
correlation = dataset[dataset.columns].corr(method='pearson') #dataframe.corr: attribute of pandas

plt.subplots(figsize=(16,14))
sns.heatmap(correlation, annot = True) #annot=true writes the data in each cell

#Feature Extraction
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
#chi2 measures dependence between stochastic variables, so using this function “weeds out” the features that are the most likely to be independent of class and therefore irrelevant for classification.

X = dataset.iloc[:, 1:8]
y = dataset.iloc[:, 9]

Select_top_4 = SelectKBest(score_func=chi2, k=4) #form the selector 
X_new = Select_top_4.fit_transform(X, y) #fit the selector to X and transform X to X_new
X_new[0:5]

dataset.head()
#we can see that no.2, no.5, no.6 and no.8 are the best features

X_features = pd.DataFrame(data=X_new, columns=["no_times_pregnant", "glucose_concentration", "serum_insulin", "bmi"])

X_features.head() #now we got the new dataframe with the most important features

#Standardizing features
from sklearn.preprocessing import StandardScaler
rescaled_X = StandardScaler().fit_transform(X_features )

X = pd.DataFrame(data=rescaled_X, columns=X_features.columns)
X.head()

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score #calculate corss-validation scores 
#trying multiple models here 
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

models = [] #making model dictionary 
models.append(("LR", LogisticRegression()))
models.append(("KN", KNeighborsClassifier()))
models.append(("DT", DecisionTreeClassifier()))
models.append(("SVC", SVC()))

results = []
names = []
for name, model in models:
    kfold = KFold(n_splits=10, random_state=22, shuffle=True) #splitting trainning data into 10 folds
    cv_result = cross_val_score(model, X, y, cv= kfold, scoring="accuracy") #calculate accuracy score of every cross validation using 4 models respectively
    names.append(name)
    results.append(cv_result)
print(len(names))

print(results)  #axis= accuracy score of every cross validation using the models 

results=np.mean(results, axis=0) #calculate the mean score of each model

for i in range(len(names)):
    print(names[i], results[i]) 

#optimizing parameters of decision tree
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score 

tree_para = {'criterion':['gini','entropy'],'max_depth':[4,5,6,7,8,9,10,11,12,15,20,30,40,50,70,90,120,150]}
grid= GridSearchCV(DecisionTreeClassifier(), tree_para, cv=5)
#GridSearchCV: Exhaustive search over specified parameter values for an estimator. 

grid.fit(X, y)
print("Best: %f using %s" % (grid.best_score_, grid.best_params_))

dataset_test = pd.read_csv("test.csv")
dataset_test.head()

X_test=pd.DataFrame(data=dataset_test, columns=["no_times_pregnant", "glucose_concentration", "serum_insulin", "bmi"])
rescaled_X_test = StandardScaler().fit_transform(X_test)
X_test = pd.DataFrame(data=rescaled_X_test, columns=X_test.columns)
X_test.head()

predictions = grid.predict(X_test)
print(predictions)

result = pd.DataFrame({'p_id':dataset_test['p_id'].as_matrix(), 'diabetes':predictions.astype(np.int32)})
result.to_csv("pima_diabetes_predictions.csv", index=False)
print(pd.read_csv("pima_diabetes_predictions.csv"))
