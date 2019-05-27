from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve, auc
from sklearn import metrics
import matplotlib.pyplot as plt


def plot_roc_curve(false_positive_rate, true_positive_rate, auc):
    plt.figure(figsize=(10, 10))
    plt.title("ROC - Receiver Operating Characteristic")
    plt.plot(false_positive_rate, true_positive_rate, color='red',
             label='AUC = %0.2f' % auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.axis('tight')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()


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

svm_model = SVC()
tuned_parameters = {
    'C': [1, 10, 100, 500, 1000], 'kernel': ['linear', 'rbf'],
    'C': [1, 10, 100, 500, 1000], 'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
    'kernel': ['rbf']
    # 'degree': [2,3,4,5,6] , 'C':[1,10,100,500,1000] , 'kernel':['poly']
}
model_svm = RandomizedSearchCV(svm_model, tuned_parameters, cv=10,
                               scoring='accuracy', n_iter=20)
model_svm.fit(X_train, y_train)
print("SVM model: ", model_svm)
print("Best score: ", model_svm.best_score_)
print("Grid scores: ", model_svm.grid_scores_)
print("Best parameters: ", model_svm.best_params_)
y_pred = model_svm.predict(X_test)
print("Accuracy score: ", metrics.accuracy_score(y_pred, y_test))

# Confusion Matrix
confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
print("Confusion Matrix")
print(confusion_matrix)

metric = metrics.classification_report(y_test, y_pred)
print("Classification Report")
print(metric)

metric = metrics.roc_auc_score(y_test, y_pred)
print("ROC AUC score: ", metric)

false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
roc_auc = auc(false_positive_rate, true_positive_rate)
print("AUC: ", roc_auc)

plot_roc_curve(false_positive_rate, true_positive_rate, roc_auc)

# Support Vector Machine with polynomial Kernel
