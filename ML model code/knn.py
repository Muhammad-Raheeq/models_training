# K-Nearest Neighbors example
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

data = load_iris()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=5)
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)
print('KNN accuracy:', accuracy_score(y_test, model.predict(X_test)))
