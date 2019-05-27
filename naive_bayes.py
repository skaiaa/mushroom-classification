from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve, auc
from sklearn import metrics
import matplotlib.pyplot as plt
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

model_naive = GaussianNB()
model_naive.fit(X_train, y_train)
print("Naive Bayes model: ", model_naive)
y_prob = model_naive.predict_proba(X_test)[:, 1]
y_pred = np.where(y_prob > 0.5, 1, 0)
print("Naive Bayes score: ", model_naive.score(X_test, y_pred))
print("Number of mislabeled points from %d points: %d"
      % (X_test.shape[0], (y_test != y_pred).sum()))

# Cross validation
scores = cross_val_score(model_naive, X, y, cv=10, scoring='accuracy')
print("Cross validation scores: ", scores)
print("Mean cross validation score: ", scores.mean())

# Confusion Matrix & Classification Report
confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
print("Confusion Matrix")
print(confusion_matrix)

metric = metrics.classification_report(y_test, y_pred)
print("Classification Report")
print(metric)

metric = metrics.roc_auc_score(y_test, y_pred)
print("ROC AUC score: ", metric)
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(false_positive_rate, true_positive_rate)
print("AUC: ", roc_auc)

plot_roc_curve(false_positive_rate, true_positive_rate, roc_auc)
plot_confusion_matrix(confusion_matrix, ['edible', 'poisonous'])
# print("Real number of edible: ", (np.where(y_test == 0, 1, 0)).sum())
# print("Real number of poisonous: ", (np.where(y_test == 1, 1, 0)).sum())
