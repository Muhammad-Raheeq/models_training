# ElasticNet example
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import ElasticNet

X, y = make_regression(n_samples=180, n_features=8, noise=25, random_state=2)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=2)
model = ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=10000)
model.fit(X_train, y_train)
print('Elastic net mse:', mean_squared_error(y_test, model.predict(X_test)))
