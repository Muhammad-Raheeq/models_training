# Lasso Regression example
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso

X, y = make_regression(n_samples=150, n_features=10, noise=30, random_state=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
model = Lasso(alpha=0.5, max_iter=10000)
model.fit(X_train, y_train)
print('Lasso mse:', mean_squared_error(y_test, model.predict(X_test)))
