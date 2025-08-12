# Random Forest example
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, random_state=7)
model = RandomForestClassifier(n_estimators=100, random_state=7)
model.fit(X_train, y_train)
print('Random forest accuracy:', accuracy_score(y_test, model.predict(X_test)))
 