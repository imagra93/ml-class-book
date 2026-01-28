import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Example data
X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]])
y = np.array([2.0, 3.0, 3.5, 5.0, 5.5, 6.2])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("coef_ =", model.coef_)
print("intercept_ =", model.intercept_)
print("RMSE =", mean_squared_error(y_test, y_pred, squared=False))
print("R2 =", r2_score(y_test, y_pred))
