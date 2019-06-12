from sklearn.model_selection import RandomizedSearchCV
from pprint import pprint
from sklearn.ensemble import RandomForestRegressor
import pydot
import matplotlib.pyplot as plt
from sklearn.tree import export_graphviz
from sklearn.metrics import roc_curve, auc
import numpy as np
from sklearn import metrics
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from plots import plot_roc_curve, plot_confusion_matrix, plot_feature_importance
from sklearn.ensemble import RandomForestClassifier


mushrooms = pd.read_csv("./mushrooms.csv")
mushrooms = mushrooms.drop("veil-type", axis=1)
mushrooms = mushrooms.drop("gill-attachment", axis=1)
mushrooms = mushrooms.drop("veil-color", axis=1)
mushrooms = mushrooms.drop("cap-shape", axis=1)
label_encoder = LabelEncoder()
for col in mushrooms.columns:
    mushrooms[col] = label_encoder.fit_transform(mushrooms[col])
X = mushrooms.iloc[:, 1:22]
y = mushrooms.iloc[:, 0]
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=4)
# Random Forest


rf = RandomForestClassifier(random_state=42)


# Look at parameters used by our current forest
print('Parameters currently in use:\n')
pprint(rf.get_params())


# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]

# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

pprint(random_grid)

# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestClassifier()
# Random search of parameters, using 3 fold cross validation,
# search across 100 different combinations, and use all available cores
model_RF = RandomizedSearchCV(estimator=rf, param_distributions=random_grid,
                              n_iter=100, cv=3, verbose=2, random_state=42, n_jobs=-1)


# Fit the random search model
model_RF.fit(X_train, y_train)
pprint(model_RF.best_params_)
print("Random Forest model: ", model_RF)

y_prob = model_RF.predict_proba(X_test)[:, 1]
y_pred = np.where(y_prob > 0.5, 1, 0)
print("Random Forest score: ", model_RF.score(X_test, y_pred))

confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
print("Confusion Matrix")
print(confusion_matrix)

metric = metrics.roc_auc_score(y_test, y_pred)
print("ROC AUC score: ", metric)

scores = cross_val_score(model_RF, X, y, cv=10, scoring='accuracy')
print("Cross validation scores: ", scores)
print("Mean cross validation score: ", scores.mean())

false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(false_positive_rate, true_positive_rate)
print("AUC: ", roc_auc)

plot_roc_curve(false_positive_rate, true_positive_rate, roc_auc)
plot_confusion_matrix(confusion_matrix, ['edible', 'poisonous'])
col_names = [x for x in mushrooms.columns]
col_names = col_names[1:]
feature_imp = pd.Series(model_RF.feature_importances_,
                        index=col_names).sort_values(
    ascending=False)
# print(feature_imp)
plot_feature_importance(feature_imp, feature_imp.index)
# Export as dot file
estimator = model_RF.estimators_[5]
export_graphviz(estimator, out_file='tree.dot',
                feature_names=col_names,
                rounded=True, proportion=False,
                precision=2, filled=True)

# Convert to png

(graph,) = pydot.graph_from_dot_file('./report/random-forest/tree.dot')
graph.write_png('./report/random-forest/tree.png')

# Display in python
plt.figure(figsize=(14, 18))
plt.imshow(plt.imread('./report/random-forest/tree.png'))
plt.axis('off')
plt.show()
