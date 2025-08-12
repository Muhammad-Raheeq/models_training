from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

X, y = make_regression(n_samples=200, n_features=1, noise=15, random_state=3)
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, random_state=3)
model = LinearRegression()
model.fit(X_train, y_train)
print('Polymomial mse:', mean_squared_error(y_test, model.predict(X_test)))
