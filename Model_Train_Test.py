
import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import pickle
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


data = pd.read_excel('Downloads/Final_covid_ML_Data.xlsx')

lasso_features = ['age', 'cough', 'muscle pain', 'LOC', 'rhinorrhea', 'anosmia', 'vomitting', 'arthralgia', 'headache',
                  'sore throat', 'alcohol use', 'opium', 'IHD', 'CHF', 'COPD', 'DM', 'CVA', 'GI problems', 'RA',
                  'cancer', 'HLP', 'hepatitis c', 'neurological problems', 'parkinson', 'alzheimer', 'WBC', 'NEUT',
                  'HB', 'CR', 'ph', 'hco3', 'inr', 'k', 'ca', 'mg', 'o2sat', 'rr']

lasso_x = data[lasso_features]
lasso_y = data['deaths']

lasso_x_train, lasso_x_test, lasso_y_train, lasso_y_test = train_test_split(lasso_x, lasso_y, test_size=0.2,
                                                                            random_state=123)
test_lasso = lasso_x_test.copy()
test_lasso['death'] = lasso_y_test

test_lasso.to_excel("Downloads/lasso_test_models_predictions.xlsx")

standard = StandardScaler()
lasso_x_train = standard.fit_transform(lasso_x_train)
lasso_x_test = standard.fit_transform(lasso_x_test)

model_checkpoint_callback_LASSO = tf.keras.callbacks.ModelCheckpoint(
    filepath='Downloads/CheckPoint_LASSO',
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)
#neural network model for lasso
model = tf.keras.Sequential([
    tf.keras.layers.Dense(37, activation='relu', input_dim=37),
    tf.keras.layers.Dropout(0.6),
    tf.keras.layers.Dense(128, activation='relu', kernel_regularizer='l2'),
    tf.keras.layers.Dropout(0.6),
    tf.keras.layers.Dense(64, activation='relu', kernel_regularizer='l2'),
    tf.keras.layers.Dropout(0.6),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
sgd = tf.keras.optimizers.SGD(learning_rate=0.01, decay=1e-7, momentum=0.9, nesterov=True)
model.compile(loss='binary_crossentropy', optimizer="adam", metrics=['accuracy'])

history = model.fit(lasso_x_train, lasso_y_train, validation_data=(lasso_x_test, lasso_y_test), epochs=100,callbacks=[model_checkpoint_callback_LASSO])

model.evaluate(lasso_x_test, lasso_y_test)

pred = (model.predict(lasso_x_test) > 0.5)

pred_prob = model.predict(lasso_x_test)
print("DNN evaluation results for lassso: ", results(metrics.confusion_matrix(lasso_y_test, pred)))

#Support vector machine model for lasso

lasso_svm = SVC(decision_function_shape='ovo')
lasso_svm.probability = True
lasso_svm.fit(lasso_x_train, lasso_y_train)
pred_prob_lasso_svm = lasso_svm.predict_proba(lasso_x_test)
pred_lasso_svm = lasso_svm.predict(lasso_x_test)
np.set_printoptions()
print("SVM evaluation results for lasso: ",results(metrics.confusion_matrix(lasso_y_test, pred_lasso_svm)))
pred_prob_lasso_svm = 1 - pred_prob_lasso_svm.reshape(1446, 1)[::2]
pickle.dump(lasso_svm, open('Downloads/lasso_svm.sav', 'wb'))

#random forest model for lasso
lasso_rf = RandomForestClassifier(oob_score=True, criterion='gini', max_depth=13)
lasso_rf.fit(lasso_x_train, lasso_y_train)
lasso_rf_pred_prob = lasso_rf.predict_proba(lasso_x_test)
lasso_rf_pred = lasso_rf.predict(lasso_x_test)
lasso_rf_pred_prob = 1 - lasso_rf_pred_prob.reshape(1446, 1)[::2]
print("RF evaluation results for lasso:",results(metrics.confusion_matrix(lasso_y_test, lasso_rf_pred)))
lasso_rf_pred = lasso_rf_pred_prob > 0.5
pickle.dump(lasso_rf, open('Downloads/lasso_rf.sav', 'wb'))

#GBC model for lasso
lasso_gbc = GradientBoostingClassifier(loss='deviance', n_estimators=400, learning_rate=0.01, criterion='mse',
                                       max_depth=6)
lasso_gbc.fit(lasso_x_train, lasso_y_train)
lasso_gbc_pred_prob = lasso_gbc.predict_proba(lasso_x_test)
lasso_gbc_pred = lasso_gbc.predict(lasso_x_test)
lasso_gbc_pred_prob = 1 - lasso_gbc_pred_prob.reshape(1446, 1)[::2]
print("GBC evaluation results for lasso: ",results(metrics.confusion_matrix(lasso_y_test, lasso_gbc_pred)))
pickle.dump(lasso_gbc, open('Downloads/lasso_gbc.sav', 'wb'))

#Nearest neighbor for lasso
lasso_knn = KNeighborsClassifier(n_neighbors=3, leaf_size=30, weights='distance')
lasso_knn.fit(lasso_x_train, lasso_y_train)
lasso_knn_pred_prob = lasso_knn.predict_proba(lasso_x_test)
lasso_knn_pred = lasso_knn.predict(lasso_x_test)
lasso_knn_pred_prob = 1 - lasso_knn_pred_prob.reshape(1446, 1)[::2]
print("KNN evaluation results for lasso: ",results(metrics.confusion_matrix(lasso_y_test, lasso_knn_pred)))
pickle.dump(lasso_knn, open('Downloads/lasso_knn.sav', 'wb'))

#Logistic regression model for lasso
lasso_lr = LogisticRegression(penalty='l2', tol=1e-5)
lasso_lr.fit(lasso_x_train, lasso_y_train)
lasso_lr_pred_prob = lasso_lr.predict_proba(lasso_x_test)
lasso_lr_pred = lasso_lr.predict(lasso_x_test)
print("LR evaluation results for lasso: ",results(metrics.confusion_matrix(lasso_y_test, lasso_lr_pred)))
lasso_lr_pred_prob = 1 - lasso_lr_pred_prob.reshape(1446, 1)[::2]
pickle.dump(lasso_lr, open('Downloads/lasso_lr.sav', 'wb'))


lasso_nb = GaussianNB()
lasso_nb.fit(lasso_x_train, lasso_y_train)
lasso_nb_pred_prob = lasso_nb.predict_proba(lasso_x_test)
lasso_nb_pred = lasso_nb.predict(lasso_x_test)
print(results(metrics.confusion_matrix(lasso_y_test, lasso_nb_pred)))
lasso_nb_pred_prob = 1 - lasso_nb_pred_prob.reshape(1446, 1)[::2]


fpr1, tpr1, thresh1 = roc_curve(lasso_y_test, pred_prob, pos_label=1)
fpr2, tpr2, thresh2 = roc_curve(lasso_y_test, pred_prob_lasso_svm, pos_label=1)
fpr3, tpr3, thresh3 = roc_curve(lasso_y_test, lasso_rf_pred_prob, pos_label=1)
fpr4, tpr4, thresh4 = roc_curve(lasso_y_test, lasso_gbc_pred_prob, pos_label=1)
fpr5, tpr5, thresh5 = roc_curve(lasso_y_test, lasso_knn_pred_prob, pos_label=1)
fpr6, tpr6, thresh6 = roc_curve(lasso_y_test, lasso_lr_pred_prob, pos_label=1)
fpr7, tpr7, thresh7 = roc_curve(lasso_y_test, lasso_nb_pred_prob, pos_label=1)
auc_score1 = roc_auc_score(lasso_y_test, pred_prob)
auc_score2 = roc_auc_score(lasso_y_test, pred_prob_lasso_svm)
auc_score3 = roc_auc_score(lasso_y_test, lasso_rf_pred_prob)
auc_score4 = roc_auc_score(lasso_y_test, lasso_gbc_pred_prob)
auc_score5 = roc_auc_score(lasso_y_test, lasso_knn_pred_prob)
auc_score6 = roc_auc_score(lasso_y_test, lasso_lr_pred_prob)
auc_score7 = roc_auc_score(lasso_y_test, lasso_nb_pred_prob)
print("LASSO DNN AUC: ",auc_score1)
print("LASSO SVM AUC: ",auc_score2)
print("LASSO RF AUC: ",auc_score3)
print("LASSO GBC AUC: ",auc_score4)
print("LASSO KNN AUC: ",auc_score5)
print("LASSO LR AUC: ",auc_score6)
#print("",auc_score7)
random_probs = [0 for i in range(len(lasso_y_test))]
p_fpr, p_tpr, _ = roc_curve(lasso_y_test, random_probs, pos_label=1)
plt.plot(fpr1, tpr1, linestyle='--', color='orange', label='lasso_DNN')
plt.plot(fpr2, tpr2, linestyle='--', color='red', label='lasso_svm')
plt.plot(fpr3, tpr3, linestyle='--', color='blue', label='lasso_rf')
plt.plot(fpr4, tpr4, linestyle='--', color='green', label='lasso_gbc')
plt.plot(fpr5, tpr5, linestyle='--', color='yellow', label='lasso_knn')
plt.plot(fpr6, tpr6, linestyle='--', color='purple', label='lasso_lr')
# plt.plot(fpr7, tpr7, linestyle='--',color='pink', label='lasso_nb')
plt.plot(p_fpr, p_tpr, linestyle='--', color='black')
plt.legend(loc="lower right")
plt.savefig('Downloads/lasso.pdf', dpi=None,
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

test_lasso['nn'] = model.predict(lasso_x_test)
test_lasso['SVM'] = lasso_svm.predict(lasso_x_test)
test_lasso['rf'] = lasso_rf_pred_prob
test_lasso['knn'] = lasso_knn_pred_prob
test_lasso['lr'] = lasso_lr_pred_prob
test_lasso['gbc'] = lasso_gbc_pred_prob


# boruta
boruta_features = ['o2sat', 'age', 'NEUT', 'CR', 'troponin', 'LOC', 'LYMPHH', 'ph', 'hco3', 'WBC', 'na', 'alzheimer',
                   'PLT', 'AST', 'HB', 'pco2', 'inr', 'ca', 'rr', 'pt', 'k', 'sbp', 'dbp', 'cpk', 'pr', 'ALT']


boruta_x = data[boruta_features]
boruta_y = data['deaths']
boruta_x_train, boruta_x_test, boruta_y_train, boruta_y_test = train_test_split(boruta_x, boruta_y, test_size=0.2)


boruta_test = boruta_x_test.copy()
boruta_test['death'] = boruta_y_test

standard = StandardScaler()
boruta_x_train = standard.fit_transform(boruta_x_train)
boruta_x_test = standard.fit_transform(boruta_x_test)

model_checkpoint_callback_Boruta = tf.keras.callbacks.ModelCheckpoint(
    filepath='Downloads/CheckPoint_Boruta',
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)
#DNN model for boruta
boruta_model = tf.keras.Sequential([
    tf.keras.layers.Dense(26, activation='relu', input_dim=26),
    tf.keras.layers.Dropout(0.6),
    tf.keras.layers.Dense(128, activation='relu', kernel_regularizer='l2'),
    tf.keras.layers.Dropout(0.6),
    tf.keras.layers.Dense(64, activation='relu', kernel_regularizer='l2'),
    tf.keras.layers.Dropout(0.6),
    tf.keras.layers.Dense(1, activation='sigmoid', kernel_regularizer='l2')
])
sgd = tf.keras.optimizers.SGD(lr=0.02, decay=1e-6, momentum=0.8, nesterov=True)
boruta_model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
boruta_history = boruta_model.fit(boruta_x_train, boruta_y_train, validation_data=(boruta_x_test, boruta_y_test),
                                  epochs=100,callbacks=[model_checkpoint_callback_Boruta])

boruta_pred_prob = boruta_model.predict(boruta_x_test)
boruta_pred = boruta_model.predict(boruta_x_test)
boruta_pred = (boruta_pred >= 0.45)
print("DNN evaluation results for boruta: ",results(metrics.confusion_matrix(boruta_y_test, boruta_pred)))

#Support vector machine model for boruta
boruta_svm = SVC(decision_function_shape='ovo')
boruta_svm.probability = True
boruta_svm.fit(boruta_x_train, boruta_y_train)
pred_prob_boruta_svm = boruta_svm.predict_proba(boruta_x_test)
pred_boruta_svm = boruta_svm.predict(boruta_x_test)
np.set_printoptions()
print("SVM evaluation results for boruta: ",results(metrics.confusion_matrix(boruta_y_test, pred_boruta_svm)))
pred_prob_boruta_svm = 1 - pred_prob_boruta_svm.reshape(1446, 1)[::2]
pickle.dump(boruta_svm, open('Downloads/boruta_svm.sav', 'wb'))

#rf model for boruta
boruta_rf = RandomForestClassifier(oob_score=True, criterion='gini', max_depth=13)
boruta_rf.fit(boruta_x_train, boruta_y_train)
boruta_rf_pred_prob = boruta_rf.predict_proba(boruta_x_test)
boruta_rf_pred = boruta_rf.predict(boruta_x_test)
boruta_rf_pred_prob = 1 - boruta_rf_pred_prob.reshape(1446, 1)[::2]
print("RF evaluation results for boruta: ",results(metrics.confusion_matrix(boruta_y_test, boruta_rf_pred)))
pickle.dump(boruta_svm, open('Downloads/boruta_rf.sav', 'wb'))

#GBC model for boruta
boruta_gbc = GradientBoostingClassifier(loss='deviance', n_estimators=400, learning_rate=0.01, criterion='mse',
                                        max_depth=6)
boruta_gbc.fit(boruta_x_train, boruta_y_train)
boruta_gbc_pred_prob = boruta_gbc.predict_proba(boruta_x_test)
boruta_gbc_pred = boruta_gbc.predict(boruta_x_test)
boruta_gbc_pred_prob = 1 - boruta_gbc_pred_prob.reshape(1446, 1)[::2]
print("GBC evaluation results for boruta: ", results(metrics.confusion_matrix(boruta_y_test, boruta_gbc_pred)))
pickle.dump(boruta_gbc, open('Downloads/boruta_gbc.sav', 'wb'))

#KNN model for boruta
boruta_knn = KNeighborsClassifier(n_neighbors=3)
boruta_knn.fit(boruta_x_train, boruta_y_train)
boruta_knn_pred_prob = boruta_knn.predict_proba(boruta_x_test)
boruta_knn_pred = boruta_knn.predict(boruta_x_test)
boruta_knn_pred_prob = 1 - boruta_knn_pred_prob.reshape(1446, 1)[::2]
print("KNN evaluation results for boruta",results(metrics.confusion_matrix(boruta_y_test, boruta_knn_pred)))
pickle.dump(boruta_knn, open('Downloads/boruta_knn.sav', 'wb'))

#LR model for boruta
boruta_lr = LogisticRegression(penalty='l2', tol=1e-6)
boruta_lr.fit(boruta_x_train, boruta_y_train)
boruta_lr_pred_prob = boruta_lr.predict_proba(boruta_x_test)
boruta_lr_pred = boruta_lr.predict(boruta_x_test)
print("LR evaluation results for boruta: ",results(metrics.confusion_matrix(boruta_y_test, boruta_lr_pred)))
boruta_lr_pred_prob = 1 - boruta_lr_pred_prob.reshape(1446, 1)[::2]
pickle.dump(boruta_lr, open('Downloads/boruta_lr.sav', 'wb'))

boruta_nb = GaussianNB()
boruta_nb.fit(boruta_x_train, boruta_y_train)
boruta_nb_pred_prob = boruta_nb.predict_proba(boruta_x_test)
boruta_nb_pred = boruta_nb.predict(boruta_x_test)
print(results(metrics.confusion_matrix(boruta_y_test, boruta_nb_pred)))
boruta_nb_pred_prob = 1 - boruta_nb_pred_prob.reshape(1446, 1)[::2]


fpr8, tpr8, thresh8 = roc_curve(boruta_y_test, boruta_pred_prob, pos_label=1)
fpr9, tpr9, thresh9 = roc_curve(boruta_y_test, pred_prob_boruta_svm, pos_label=1)
fpr10, tpr10, thresh10 = roc_curve(boruta_y_test, boruta_rf_pred_prob, pos_label=1)
fpr11, tpr11, thresh11 = roc_curve(boruta_y_test, boruta_gbc_pred_prob, pos_label=1)
fpr12, tpr12, thresh12 = roc_curve(boruta_y_test, boruta_knn_pred_prob, pos_label=1)
fpr13, tpr13, thresh13 = roc_curve(boruta_y_test, boruta_lr_pred_prob, pos_label=1)
fpr14, tpr14, thresh14 = roc_curve(boruta_y_test, boruta_nb_pred_prob, pos_label=1)
auc_score8 = roc_auc_score(boruta_y_test, boruta_pred_prob)
auc_score9 = roc_auc_score(boruta_y_test, pred_prob_boruta_svm)
auc_score10 = roc_auc_score(boruta_y_test, boruta_rf_pred_prob)
auc_score11 = roc_auc_score(boruta_y_test, boruta_gbc_pred_prob)
auc_score12 = roc_auc_score(boruta_y_test, boruta_knn_pred_prob)
auc_score13 = roc_auc_score(boruta_y_test, boruta_lr_pred_prob)
auc_score14 = roc_auc_score(boruta_y_test, boruta_nb_pred_prob)
print("Boruta DNN AUC: ",auc_score8)
print("Boruta SVM AUC: ",auc_score9)
print("Boruta RF AUC: ",auc_score10)
print("Boruta GBC AUC: ",auc_score11)
print("Boruta KNN AUC: ",auc_score12)
print("Boruta LR AUC: ",auc_score13)
random_probs = [0 for i in range(len(lasso_y_test))]
p_fpr, p_tpr, _ = roc_curve(lasso_y_test, random_probs, pos_label=1)
plt.plot(fpr8, tpr8, linestyle='--', color='orange', label='boruta_DNN')
plt.plot(fpr9, tpr9, linestyle='--', color='red', label='boruta_svm')
plt.plot(fpr10, tpr10, linestyle='--', color='blue', label='boruta_rf')
plt.plot(fpr11, tpr11, linestyle='--', color='green', label='boruta_gbc')
plt.plot(fpr12, tpr12, linestyle='--', color='yellow', label='boruta_knn')
plt.plot(fpr13, tpr13, linestyle='--', color='purple', label='boruta_lr')

plt.plot(p_fpr, p_tpr, linestyle='--', color='black')
plt.legend(loc="lower right")
plt.savefig('Downloads/bruta.pdf', dpi=None,
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
boruta_test['nn'] = boruta_pred_prob
boruta_test['svm'] = pred_prob_boruta_svm
boruta_test['rf'] = boruta_rf_pred_prob
boruta_test['gbc'] = boruta_gbc_pred_prob
boruta_test['knn'] = boruta_knn_pred_prob
boruta_test['lr'] = boruta_lr_pred_prob

