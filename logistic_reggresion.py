from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_curve, auc
import numpy as np
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from plots import plot_roc_curve, plot_confusion_matrix


mushrooms = pd.read_csv("./mushrooms.csv")
mushrooms = mushrooms.drop("veil-type", axis=1)
label_encoder = LabelEncoder()
for col in mushrooms.columns:
    mushrooms[col] = label_encoder.fit_transform(mushrooms[col])
X = mushrooms.iloc[:, 1:22]
y = mushrooms.iloc[:, 0]
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=4)
# Default Logistic Regression

model_LR = LogisticRegression(solver='lbfgs')
model_LR.fit(X_train, y_train)
print("Logistic Regression model: ", model_LR)
# this will give positive class prediction probabilities
y_prob = model_LR.predict_proba(X_test)[:, 1]
# this will threshold the probabilities to give class predictions
y_pred = np.where(y_prob > 0.5, 1, 0)
print("Logistic Regression score: ", model_LR.score(X_test, y_pred))

confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
print("Confusion Matrix")
print(confusion_matrix)

metric = metrics.roc_auc_score(y_test, y_pred)
print("AUC ROC score: ", metric)

scores = cross_val_score(model_LR, X, y, cv=10, scoring='accuracy')
print("Cross validation scores: ", scores)
print("Mean cross validation score: ", scores.mean())

false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(false_positive_rate, true_positive_rate)
print("AUC: ", roc_auc)

plot_roc_curve(false_positive_rate, true_positive_rate, roc_auc)
plot_confusion_matrix(confusion_matrix, ['edible', 'poisonous'])

# Logistic Regression (Tuned Model)
LR_model = LogisticRegression(solver='saga')
tuned_parameters = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
                    'penalty': ['l1', 'l2']}

# L1 and L2 are regularization parameters. They're used to avoid overfiting.
# Both L1 and L2 regularization prevents overfitting by shrinking
# (imposing a penalty) on the coefficients.
# L1 is the first moment norm |x1-x2| (|w| for regularization case)
# that is simply the absolute dıstance between two points
# where L2 is second moment norm corresponding to Eucledian Distance
# that is |x1-x2|^2 (|w|^2 for regularization case).
# In simple words, L2 (Ridge) shrinks all the coefficient by the same
# proportions but eliminates none, while L1 (Lasso) can shrink some
# coefficients to zero, performing variable selection.
# If all the features are correlated with the label, ridge outperforms lasso,
# as the coefficients are never zero in ridge. If only a subset of features
# are correlated with the label, lasso outperforms ridge as in lasso model
# some coefficient can be shrunken to zero.

LR = GridSearchCV(LR_model, tuned_parameters, cv=10)
LR.fit(X_train, y_train)
print("Tuned Logistic Regression")
print(LR)
print("Best paremeters: ", LR.best_params_)
y_prob = LR.predict_proba(X_test)[:, 1]
y_pred = np.where(y_prob > 0.5, 1, 0)
print("Tuned Logistic Regression score: ", LR.score(X_test, y_pred))
confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
print("Confusion Matrix")
print(confusion_matrix)
metric = metrics.classification_report(y_test, y_pred)
print("Classification Report")
print(metric)

metric = metrics.roc_auc_score(y_test, y_pred)
print("AUC ROC score: ", metric)

# scores = cross_val_score(LR, X, y, cv=10, scoring='accuracy')
# print("Cross validation scores: ", scores)
# print("Mean cross validation score: ", scores.mean())

false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(false_positive_rate, true_positive_rate)
print("AUC: ", roc_auc)

plot_roc_curve(false_positive_rate, true_positive_rate, roc_auc)
plot_confusion_matrix(confusion_matrix, ['edible', 'poisonous'])

# Logistic Regression with Rigde penalty
LR_ridge = LogisticRegression(penalty='l2', solver='saga')
LR_ridge.fit(X_train, y_train)
print("Logistic Regression with Ridge penalty: ", LR_ridge)
y_prob = LR_ridge.predict_proba(X_test)[:, 1]
y_pred = np.where(y_prob > 0.5, 1, 0)
print("Logistic Regression with Ridge penalty score: ", LR_ridge.score(X_test,
                                                                       y_pred))
confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
print("Confusion Matrix")
print(confusion_matrix)

metric = metrics.classification_report(y_test, y_pred)
print("Classification Report")
print(metric)

metric = metrics.roc_auc_score(y_test, y_pred)
print("AUC ROC score: ", metric)

scores = cross_val_score(LR_ridge, X, y, cv=10, scoring='accuracy')
print("Cross validation scores: ", scores)
print("Mean cross validation score: ", scores.mean())

false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(false_positive_rate, true_positive_rate)
print("AUC: ", roc_auc)

plot_roc_curve(false_positive_rate, true_positive_rate, roc_auc)
plot_confusion_matrix(confusion_matrix, ['edible', 'poisonous'])
