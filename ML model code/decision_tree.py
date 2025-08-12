# Decision Tree Classifier example
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

data = load_wine()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, random_state=6)
model = DecisionTreeClassifier(max_depth=5, random_state=6)
model.fit(X_train, y_train)
print('Decision tree accuracy:', accuracy_score(y_test, model.predict(X_test)))
