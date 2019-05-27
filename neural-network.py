from sklearn.metrics import roc_curve, auc
import numpy as np
from sklearn import metrics
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from plots import plot_roc_curve, plot_confusion_matrix
from sklearn.neural_network import MLPClassifier


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

model_MLP = MLPClassifier()
model_MLP.fit(X_train, y_train)
print("Random Forest model: ", model_MLP)

y_prob = model_MLP.predict_proba(X_test)[:, 1]
y_pred = np.where(y_prob > 0.5, 1, 0)
print("Random Forest score: ", model_MLP.score(X_test, y_pred))

confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
print("Confusion Matrix")
print(confusion_matrix)

metric = metrics.roc_auc_score(y_test, y_pred)
print("ROC AUC score: ", metric)

scores = cross_val_score(model_MLP, X, y, cv=10, scoring='accuracy')
print("Cross validation scores: ", scores)
print("Mean cross validation score: ", scores.mean())

false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(false_positive_rate, true_positive_rate)
print("AUC: ", roc_auc)

plot_roc_curve(false_positive_rate, true_positive_rate, roc_auc)
plot_confusion_matrix(confusion_matrix, ['edible', 'poisonous'])
