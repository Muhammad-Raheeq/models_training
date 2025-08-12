# Gradient Boosting Classifier example
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import GradientBoostingClassifier

X, y = make_classification(n_samples=400, n_features=10, n_informative=6, random_state=8)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=8)
model = GradientBoostingClassifier(n_estimators=100, random_state=8)
model.fit(X_train, y_train)
print('Gradient boosting accuracy:', accuracy_score(y_test, model.predict(X_test)))
