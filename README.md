# pima_diabetes
Codes for Kaggle contest of Pima diabetes predictions: https://www.kaggle.com/uciml/pima-indians-diabetes-database

This is also a classification problem.
But compared to Titanic, I tried a couple of new things: data visualization techniques; decision tree model; techniques of predictor selection and model assembling.

# Steps
1. Loading train data

2. Explore & visualize dataset
* Check descriptive parameters
* Plot histograms
* Histogrms did not fit well for some feaures, therefore I also did box plot

3. Selecting and pre-processing predictor
* Check features' correlation by Pearson's r
* Check dependence between stochastic variables and y by chi square. And select the top 4 features as the predcitor.
* Standardizing features

4. Assembling multiple models.
* Here I chose LogisticRegression, KNeighborsClassifier, DecisionTreeClassifier and SVC. 

5. Cross validation test the models and calculate accuracy score of each model.Then calculating the means of each model.
* I tried different numbers of folds here. Turned out SVC had the best score, and other models' score did not vary too much. Because at that time I was focusing on decision tree, so I chose to continue with decision tree model.

6. Optimizing parameters of decision tree using GridSearchCV. 
* Results returned that best score of 0.739331 using {'criterion': 'gini', 'max_depth': 5}

7. Fit the optimized model to train data.

6. Loading test data

7. Preprocessing test data 
* same as part of step 3

8. Implementing model on test data

9. Loading prediction results to a csv file

  
