from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

X, y = make_classification(n_samples=300, n_features=5, n_informative=3, random_state=11)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=11)
model = SVC(kernel='rbf', gamma='scale')
model.fit(X_train, y_train)
print('svc accuracy:', accuracy_score(y_test, model.predict(X_test)))
