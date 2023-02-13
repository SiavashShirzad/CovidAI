This page contain code and info about article entitled "Generalizable machine learning approach for COVID-19 mortality risk prediction using on-admission clinical and laboratory features" (Published by Nature Scientific Report) by Siavash Shirzadeh Barough, Seyed Amir Ahmad Safavi-Naini, Fatemeh Siavoshi, Atena Tamimi, Saba Ilkhani, Setareh Akbari, Sadaf Ezzati, Hamidreza Hatamabadi & Mohamad Amin Pourhoseingholi. This study is part of ["Tehran COVID-19 Cohort"](https://github.com/Sdamirsa/Tehran_COVID_Cohort) Project and the data will be accebile in the near future. Any inquiry can be made to corresponding author, Dr. Mohamad Amin Pourhoseingholi, PhD (aminphg@gmail.com), or to Seyed Amir Ahmad Safavi-Naini (sdamirsa@gmail.com) with details of usage, required data, and ethical committee approval.

In this study we used patients' medical history and laboratory data to make accurate AI models for prediction of Covid-19 patients' mortality risk.

In order to remove collinearity from our dataset we generated two datasets with two feature selection methods (LASSO regression and Boruta).

Then, we trained 6 AI models including Neural Network, Support vector machine, Logistic Regression, Gradient Boosted Decision Tree,and Random Forest. with features selected using each feature selection methods.

We selected the best model based on metrics such as AUC, Precision, Recall, Sensitivity,and Specificity on both internal validation and external validation datasets.

The main and external validation dataset are available upon reasonable request.

## The article related to this repository has been accepted for publication in Nature Scientific Reports.

## You can find the published version of the article related to this repository in [Nature, Scientific Reports](https://www.nature.com/articles/s41598-023-28943-z)
