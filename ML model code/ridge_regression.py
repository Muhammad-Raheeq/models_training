# Ridge Regression example
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge

X, y = make_regression(n_samples=200, n_features=5, noise=20, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
model = Ridge(alpha=1.0)
model.fit(X_train, y_train)
print('Ridge mse:', mean_squared_error(y_test, model.predict(X_test)))
