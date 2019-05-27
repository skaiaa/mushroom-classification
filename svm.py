from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn import metrics
from plots import plot_roc_curve, plot_confusion_matrix, plot_feature_importance


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
# print("Grid scores: ", model_svm.grid_scores_)
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

# plot_roc_curve(false_positive_rate, true_positive_rate, roc_auc)
# plot_confusion_matrix(confusion_matrix, ['edible', 'poisonous'])
col_names = [x for x in mushrooms.columns]
col_names = col_names[1:]
feature_imp = pd.Series(model_svm.best_estimator_.coef_,
                        index=col_names).sort_values(
                        ascending=False)
# print(feature_imp)
# plot_feature_importance(feature_imp, feature_imp.index)
# Support Vector Machine with polynomial Kernel
