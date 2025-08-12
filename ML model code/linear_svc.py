# LinearSVC example
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC

X, y = make_classification(n_samples=250, n_features=10, n_informative=6, random_state=12)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=12)
model = LinearSVC(max_iter=10000)
model.fit(X_train, y_train)
print('Linear svc accuracy:', accuracy_score(y_test, model.predict(X_test)))
