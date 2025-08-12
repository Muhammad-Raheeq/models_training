# Logistic Regression example (classification)
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

X, y = make_classification(n_samples=300, n_features=5, n_informative=3, n_redundant=0, random_state=4)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=4)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
print('Logistic regression accuracy:', accuracy_score(y_test, model.predict(X_test)))
