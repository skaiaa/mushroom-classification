from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

mushrooms = pd.read_csv("./mushrooms.csv")
# display first 6 rows
print(mushrooms.head(6))

# checking for null values (missing data)
print(mushrooms.isnull().sum())

# checking for values of target(class) attribute
print(mushrooms['class'].unique())

# print data shape (# of samples, # of attributes (columns))
print(mushrooms.shape)

# we have 22 features (1 is a label) and 8124 samples
# we need to change data from chars (strings) into integers
label_encoder = LabelEncoder()
for col in mushrooms.columns:
    mushrooms[col] = label_encoder.fit_transform(mushrooms[col])

print(mushrooms.head(6))

# checking encoded values
print(mushrooms['stalk-color-above-ring'].unique())
print(mushrooms.groupby('class').size())

# plotting boxplot to see distribution of data
ax = sns.boxplot(x='class', y='stalk-color-above-ring', data=mushrooms)
ax = sns.stripplot(x='class', y='stalk-color-above-ring',
                   data=mushrooms, jitter=True, edgecolor="gray")
ax.set_title("Class w.r.t stalkcolor above ring", fontsize=12)
plt.show()

# separating features and label (class)

# all rows, all features, no labels
X = mushrooms.iloc[:, 1:23]
y = mushrooms.iloc[:, 0]
print(X.head())
print(y.head())

print(X.describe())
print(mushrooms.corr())

# standardising the data
scaler = StandardScaler()
X = scaler.fit_transform(X)
print(X)

# Principal Component Analysis (PCA)
pca = PCA()
pca.fit_transform(X)
covariance = pca.get_covariance()
explained_variance = pca.explained_variance_
print(explained_variance)

with plt.style.context('dark_background'):
    plt.figure(figsize=(6, 4))
    plt.bar(range(22), explained_variance, alpha=0.5, align='center',
            label='idividual explained variance')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal components')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()

# take first two principal components and visualize it using
# K-means clustering
N = mushrooms.values
pca = PCA(n_components=2)
x = pca.fit_transform(N)
plt.figure(figsize=(5, 5))
plt.scatter(x[:, 0], x[:, 1])
plt.show()

kmeans = KMeans(n_clusters=2, random_state=5)
X_clustered = kmeans.fit_predict(N)
COLOR_MAP = {0: 'g',
             1: 'y'}
label_color = [COLOR_MAP[l] for l in X_clustered]
plt.figure(figsize=(5, 5))
plt.scatter(x[:, 0], x[:, 1], c=label_color)
plt.show()

# performing PCA with 17 components with maximum variance
pca_modified = PCA(n_components=17)
pca_modified.fit_transform(X)
print("PCA with 17 components: ", pca_modified)
# p - 1
# e - 0
