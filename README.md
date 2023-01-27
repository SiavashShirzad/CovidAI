# Artificial intelligence approach to evaluate the risk of mortality for Covid-19 patients.
In this study we used patients' medical history and laboratory data to make accurate AI models for prediction of Covid-19 patients' mortality risk.

In order to remove collinearity from our dataset we generated two datasets with two feature selection methods (LASSO regression and Boruta).

Then, we trained 6 AI models including Neural Network, Support vector machine, Logistic Regression, Gradient Boosted Decision Tree,and Random Forest. with features selected using each feature selection methods.

We selected the best model based on metrics such as AUC, Precision, Recall, Sensitivity,and Specificity on both internal validation and external validation datasets.

The main and external validation dataset are available upon reasonable request.

## The article related to this repository has been accepted for publication in Nature Scientific Reports.

## You can find the preprint version of the article related to this repository in [researchsquare](https://assets.researchsquare.com/files/rs-2152771/v1/5c5584d7-a784-4fe8-81b9-60b77d29e726.pdf?c=1666708980)