import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import pickle
from sklearn import metrics
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
def results(CM):
    TN = CM[0][0]
    FN = CM[1][0]
    TP = CM[1][1]
    FP = CM[0][1]
    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP / (TP + FN)
    # Specificity or true negative rate
    TNR = TN / (TN + FP)
    # Precision or positive predictive value
    PPV = TP / (TP + FP)
    # Negative predictive value
    NPV = TN / (TN + FN)
    return ('sensetivity: ', TPR, ' specifity: ', TNR, ' PPV: ', PPV, ' NPV: ', NPV)

standard = StandardScaler()

lasso_features = ['age' , 'cough','muscle pain' , 'LOC' , 'rhinorrhea' , 'anosmia' , 'vomitting' , 'arthralgia' , 'headache', 'sore throat'  , 'alcohol use' , 'opium' , 'IHD','CHF' , 'COPD' , 'DM' , 'CVA' , 'GI problems','RA','cancer','HLP','hepatitis c','neurological problems','parkinson','alzheimer','WBC','NEUT','HB','CR','ph','hco3','inr','k','ca','mg','o2sat','rr']
boruta_features = ['o2sat','age','NEUT' , 'CR','troponin','LOC','LYMPHH','ph','hco3','WBC','na' , 'alzheimer','PLT','AST','HB','pco2','inr','ca','rr','pt','k','sbp','dbp','cpk','pr','ALT']

final_validation_data = pd.read_excel("Downloads/final_valid_eval.xlsx")
valid_y = final_validation_data['Mortality']

valid_boruta = final_validation_data[boruta_features]
valid_lasso = final_validation_data[lasso_features]

imp = IterativeImputer()
impute_boruta = imp.fit_transform(valid_boruta)
impute_lasso =  imp.fit_transform(valid_lasso)
impute_boruta = standard.fit_transform(impute_boruta)
impute_lasso = standard.fit_transform(impute_lasso)
lassso_test_y = valid_y
boruta_test_y = valid_y

#loading models

model = tf.keras.Sequential([
    tf.keras.layers.Dense(37, activation='relu', input_dim=37),
    tf.keras.layers.Dropout(0.6),
    tf.keras.layers.Dense(128, activation='relu', kernel_regularizer='l2'),
    tf.keras.layers.Dropout(0.6),
    tf.keras.layers.Dense(64, activation='relu', kernel_regularizer='l2'),
    tf.keras.layers.Dropout(0.6),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.load_weights('Downloads/CheckPoint_LASSO')
lasso_svm = pickle.load(open('Downloads/lasso_svm.sav', 'rb'))
lasso_rf = pickle.load(open('Downloads/lasso_rf.sav', 'rb'))
lasso_gbc = pickle.load(open('Downloads/lasso_gbc.sav', 'rb'))
lasso_knn = pickle.load(open('Downloads/lasso_knn.sav', 'rb'))
lasso_lr = pickle.load(open('Downloads/lasso_lr.sav', 'rb'))

boruta_model = tf.keras.Sequential([
    tf.keras.layers.Dense(26, activation='relu', input_dim=26),
    tf.keras.layers.Dropout(0.6),
    tf.keras.layers.Dense(128, activation='relu', kernel_regularizer='l2'),
    tf.keras.layers.Dropout(0.6),
    tf.keras.layers.Dense(64, activation='relu', kernel_regularizer='l2'),
    tf.keras.layers.Dropout(0.6),
    tf.keras.layers.Dense(1, activation='sigmoid', kernel_regularizer='l2')
])
boruta_model.load_weights('Downloads/CheckPoint_Boruta')
boruta_svm = pickle.load(open('Downloads/boruta_svm.sav', 'rb'))
boruta_rf = pickle.load(open('Downloads/boruta_rf.sav', 'rb'))
boruta_gbc = pickle.load(open('Downloads/boruta_gbc.sav', 'rb'))
boruta_knn = pickle.load(open('Downloads/boruta_knn.sav', 'rb'))
boruta_lr = pickle.load(open('Downloads/boruta_lr.sav', 'rb'))

#LASSO feature selection features external validation dataset
#DNN model for lasso
impuet_pred = ((1- model.predict(impute_lasso))>=0.5)
impute_pred_prob = 1 - model.predict(impute_lasso)
print("DNN evaluation results for lasso: ",results(metrics.confusion_matrix(valid_y,impuet_pred)))

#SVM model
impute_pred_prob_lasso_svm  = lasso_svm.predict_proba(impute_lasso)
impute_pred_lasso_svm = lasso_svm.predict(impute_lasso)
np.set_printoptions()
impute_pred_prob_lasso_svm =  impute_pred_prob_lasso_svm.reshape(2370  ,1)[::2]
print("SVM evaluation results for lasoo: ",results(metrics.confusion_matrix(lassso_test_y,impute_pred_lasso_svm)))

#RF model
impute_lasso_rf_pred_prob = lasso_rf.predict_proba(impute_lasso)
impute_lasso_rf_pred = lasso_rf.predict(impute_lasso)
impute_lasso_rf_pred_prob =  impute_lasso_rf_pred_prob.reshape(2370 ,1)[::2]
print("RF model evaluation results for lasso: ",results(metrics.confusion_matrix(lassso_test_y,impute_lasso_rf_pred)))

#GBC model
impute_lasso_gbc_pred_prob = lasso_gbc.predict_proba(impute_lasso)
impute_lasso_gbc_pred = lasso_gbc.predict(impute_lasso)
impute_lasso_gbc_pred_prob =  impute_lasso_gbc_pred_prob.reshape(2370 ,1)[::2]
print("GBC model evaluation results for lasso: ", results(metrics.confusion_matrix(lassso_test_y,impute_lasso_gbc_pred)))

#KNN model
impute_lasso_knn_pred_prob = lasso_knn.predict_proba(impute_lasso)
impute_lasso_knn_pred = lasso_knn.predict(impute_lasso)
impute_lasso_knn_pred_prob =  impute_lasso_knn_pred_prob.reshape(2370 ,1)[::2]
print("KNN evaluation results for lasso: ",results(metrics.confusion_matrix(lassso_test_y,impute_lasso_knn_pred)))

#LR model
impute_lasso_lr_pred_prob = lasso_lr.predict_proba(impute_lasso)
impute_lasso_lr_pred = lasso_lr.predict(impute_lasso)
print("LR evaluation results for lasso: ",results(metrics.confusion_matrix(lassso_test_y,impute_lasso_lr_pred)))
impute_lasso_lr_pred_prob =  impute_lasso_lr_pred_prob.reshape(2370 ,1)[::2]

fpr1, tpr1, thresh1 = roc_curve(lassso_test_y, impute_pred_prob, pos_label=1)
fpr2, tpr2, thresh2 = roc_curve(lassso_test_y, impute_pred_prob_lasso_svm, pos_label=1)
fpr3, tpr3, thresh3 = roc_curve(lassso_test_y, impute_lasso_rf_pred_prob, pos_label=1)
fpr4, tpr4, thresh4 = roc_curve(lassso_test_y, impute_lasso_gbc_pred_prob, pos_label=1)
fpr5, tpr5, thresh5 = roc_curve(lassso_test_y, impute_lasso_knn_pred_prob, pos_label=1)
fpr6, tpr6, thresh6 = roc_curve(lassso_test_y, impute_lasso_lr_pred_prob, pos_label=1)

auc_score1 = roc_auc_score(lassso_test_y, impute_pred_prob)
auc_score2 = roc_auc_score(lassso_test_y, impute_pred_prob_lasso_svm)
auc_score3 = roc_auc_score(lassso_test_y, impute_lasso_rf_pred_prob)
auc_score4 = roc_auc_score(lassso_test_y, impute_lasso_gbc_pred_prob)
auc_score5 = roc_auc_score(lassso_test_y, impute_lasso_knn_pred_prob)
auc_score6 = roc_auc_score(lassso_test_y, impute_lasso_lr_pred_prob)

print("LASSO DNN AUC: ",auc_score1)
print("LASSO SVM AUC: ",auc_score2)
print("LASSO RF AUC: ",auc_score3)
print("LASSO GBC AUC: ",auc_score4)
print("LASSO KNN AUC: ",auc_score5)
print("LASSO LR AUC: ",auc_score6)
#print(auc_score7)
random_probs = [0 for i in range(len(lassso_test_y))]
p_fpr, p_tpr, _ = roc_curve(lassso_test_y, random_probs, pos_label=1)
plt.plot(fpr1, tpr1, linestyle='--',color='orange', label='lasso_DNN')
plt.plot(fpr2, tpr2, linestyle='--',color='red', label='lasso_svm')
plt.plot(fpr3, tpr3, linestyle='--',color='blue', label='lasso_rf')
plt.plot(fpr4, tpr4, linestyle='--',color='green', label='lasso_gbc')
plt.plot(fpr5, tpr5, linestyle='--',color='yellow', label='lasso_knn')
plt.plot(fpr6, tpr6, linestyle='--',color='purple', label='lasso_lr')
#plt.plot(fpr7, tpr7, linestyle='--',color='pink', label='lasso_nb')
plt.plot(p_fpr, p_tpr, linestyle='--', color='black')
plt.legend(loc="lower right")
plt.savefig('Downloads/valid_lasso.pdf', dpi=None,
                          facecolor='w',
                          edgecolor='w',
                          orientation='portrait',
                          papertype=None,
                          format=None,
                          transparent=False,
                          bbox_inches=None,
                          pad_inches=0.1,
                          frameon=None,
                          metadata=None)


#Boruta feaature selection features external validation
#DNN model
impute_boruta_pred_prob = 1- boruta_model.predict(impute_boruta)
impute_boruta_pred =  1- boruta_model.predict(impute_boruta)
impute_boruta_pred = (impute_boruta_pred>=0.5)
print("DNN evaluation for boruta: ",results(metrics.confusion_matrix(valid_y,impute_boruta_pred)))

#svm model
impute_pred_prob_boruta_svm  = boruta_svm.predict_proba(impute_boruta)
impute_pred_boruta_svm = boruta_svm.predict(impute_boruta)
np.set_printoptions()
print("SVM evaluation for boruta: ",results(metrics.confusion_matrix(boruta_test_y,impute_pred_boruta_svm)))
impute_pred_prob_boruta_svm =  impute_pred_prob_boruta_svm.reshape(2370  ,1)[::2]

#RF model
impute_boruta_rf_pred_prob = boruta_rf.predict_proba(impute_boruta)
impute_boruta_rf_pred = boruta_rf.predict(impute_boruta)
impute_boruta_rf_pred_prob =  impute_boruta_rf_pred_prob.reshape(2370  ,1)[::2]
print("RF evaluation results for boruta",results(metrics.confusion_matrix(boruta_test_y,impute_boruta_rf_pred)))

#gbc model
impute_boruta_gbc_pred_prob = boruta_gbc.predict_proba(impute_boruta)
impute_boruta_gbc_pred = boruta_gbc.predict(impute_boruta)
impute_boruta_gbc_pred_prob =  impute_boruta_gbc_pred_prob.reshape(2370  ,1)[::2]
print("GBC evaluation results for boruta: ",results(metrics.confusion_matrix(boruta_test_y,impute_boruta_gbc_pred)))

#KNN model
impute_boruta_knn_pred_prob = boruta_knn.predict_proba(impute_boruta)
impute_boruta_knn_pred = boruta_knn.predict(impute_boruta)
impute_boruta_knn_pred_prob =  impute_boruta_knn_pred_prob.reshape(2370  ,1)[::2]
print("KNN evaluation results for boruta: ",results(metrics.confusion_matrix(boruta_test_y,impute_boruta_knn_pred)))

#LR model
impute_boruta_lr_pred_prob = boruta_lr.predict_proba(impute_boruta)
impute_boruta_lr_pred = boruta_lr.predict(impute_boruta)
print("LR evaluation reults for boruta: ",results(metrics.confusion_matrix(boruta_test_y,impute_boruta_lr_pred)))
impute_boruta_lr_pred_prob =  impute_boruta_lr_pred_prob.reshape(2370  ,1)[::2]

fpr8, tpr8, thresh8 = roc_curve(boruta_test_y, impute_boruta_pred_prob, pos_label=1)
fpr9, tpr9, thresh9 = roc_curve(boruta_test_y, impute_pred_prob_boruta_svm, pos_label=1)
fpr10, tpr10, thresh10 = roc_curve(boruta_test_y, impute_boruta_rf_pred_prob, pos_label=1)
fpr11, tpr11, thresh11 = roc_curve(boruta_test_y, impute_boruta_gbc_pred_prob, pos_label=1)
fpr12, tpr12, thresh12 = roc_curve(boruta_test_y, impute_boruta_knn_pred_prob, pos_label=1)
fpr13, tpr13, thresh13 = roc_curve(boruta_test_y, impute_boruta_lr_pred_prob, pos_label=1)

auc_score8 = roc_auc_score(boruta_test_y, impute_boruta_pred_prob)
auc_score9 = roc_auc_score(boruta_test_y, impute_pred_prob_boruta_svm)
auc_score10 = roc_auc_score(boruta_test_y, impute_boruta_rf_pred_prob)
auc_score11 = roc_auc_score(boruta_test_y, impute_boruta_gbc_pred_prob)
auc_score12 = roc_auc_score(boruta_test_y, impute_boruta_knn_pred_prob)
auc_score13 = roc_auc_score(boruta_test_y, impute_boruta_lr_pred_prob)

print("Boruta DNN AUC: ",auc_score8)
print("Boruta SVM AUC: ",auc_score9)
print("Boruta RF AUC: ",auc_score10)
print("Boruta GBC AUC: ",auc_score11)
print("Boruta KNN AUC: ",auc_score12)
print("Boruta LR AUC: ",auc_score13)

random_probs = [0 for i in range(len(lassso_test_y))]
p_fpr, p_tpr, _ = roc_curve(lassso_test_y, random_probs, pos_label=1)
plt.plot(fpr8, tpr8, linestyle='--',color='orange', label='boruta_DNN')
plt.plot(fpr9, tpr9, linestyle='--',color='red', label='boruta_svm')
plt.plot(fpr10, tpr10, linestyle='--',color='blue', label='boruta_rf')
plt.plot(fpr11, tpr11, linestyle='--',color='green', label='boruta_gbc')
plt.plot(fpr12, tpr12, linestyle='--',color='yellow', label='boruta_knn')
plt.plot(fpr13, tpr13, linestyle='--',color='purple', label='boruta_lr')
#plt.plot(fpr14, tpr14, linestyle='--',color='pink', label='lasso_nb')
plt.plot(p_fpr, p_tpr, linestyle='--', color='black')
plt.legend(loc="lower right")
plt.savefig('Downloads/valid_bruta.pdf', dpi=None,
                          facecolor='w',
                          edgecolor='w',
                          orientation='portrait',
                          papertype=None,
                          format=None,
                          transparent=False,
                          bbox_inches=None,
                          pad_inches=0.1,
                          frameon=None,
                          metadata=None)
