
# Linear Regression

Linear Regression is one of the simplest and most useful supervised learning models. It learns a relationship between input features (X) and a continuous target (y) by fitting a line (or hyperplane) that minimizes prediction error.

---

## 1) Problem setup and notation

We have:

* Dataset with (n) examples
* (d) features per example
* Feature matrix (X \in \mathbb{R}^{n \times d})
* Target vector (y \in \mathbb{R}^{n})

### Model

**Without intercept (bias):**
[
\hat{y} = Xw
]

**With intercept:**
[
\hat{y} = Xw + b
]

Where:

* (w \in \mathbb{R}^{d}) are the weights
* (b \in \mathbb{R}) is the bias
* (\hat{y}) are the predictions

---

## 2) Loss function: Mean Squared Error (MSE)

A standard choice is **MSE**:

[
\text{MSE}(w,b)=\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y}_i)^2
]

Sometimes you will see a factor (\frac{1}{2n}) (it makes derivatives cleaner):

[
J(w,b)=\frac{1}{2n}\sum_{i=1}^{n}(y_i-(x_i^\top w+b))^2
]

Goal:
[
(w^*,b^*)=\arg\min_{w,b} J(w,b)
]

---

## 3) Closed-form solution (Normal Equation)

If we include the intercept into the matrix by adding a column of ones:

[
\tilde{X} = [\mathbf{1}\ \ X] \in \mathbb{R}^{n \times (d+1)}, \quad \tilde{w} = \begin{bmatrix} b \ w \end{bmatrix}
]
[
\hat{y}=\tilde{X}\tilde{w}
]

The least-squares optimal parameters are:

[
\tilde{w}^* = (\tilde{X}^\top \tilde{X})^{-1}\tilde{X}^\top y
]

### Notes

* This requires (\tilde{X}^\top \tilde{X}) to be invertible.
* In practice, numerical methods (like QR, SVD) are preferred over directly computing the inverse.
* For many features (large (d)) or massive (n), gradient methods are often used.

---

## 4) Gradient Descent solution

Using:
[
J(w,b)=\frac{1}{2n}\sum_{i=1}^{n}(y_i-(x_i^\top w+b))^2
]

Gradients:

[
\frac{\partial J}{\partial w} = -\frac{1}{n}X^\top (y-\hat{y})
]
[
\frac{\partial J}{\partial b} = -\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y}_i)
]

Update rules (learning rate (\alpha)):

[
w \leftarrow w - \alpha \frac{\partial J}{\partial w},
\quad
b \leftarrow b - \alpha \frac{\partial J}{\partial b}
]

---

## 5) Interpreting the coefficients

For a single feature (x):

[
\hat{y} = wx + b
]

* (w): expected change in (\hat{y}) for a +1 change in (x)
* (b): predicted value when (x=0) (sometimes not meaningful depending on feature scale)

For multiple features:

[
\hat{y} = w_1 x_1 + \cdots + w_d x_d + b
]

Each (w_j) explains the effect of feature (x_j) **assuming other features are held constant**.

---

## 6) Assumptions (classical linear regression)

In many ML contexts we care more about prediction than strict statistical assumptions, but it’s good to know the classical ones:

1. **Linearity**: relationship is linear in parameters
2. **Independent errors**
3. **Constant variance (homoscedasticity)**
4. **Normally distributed errors** (mainly for confidence intervals)
5. **No perfect multicollinearity** (features not perfectly correlated)

Violations often hurt interpretability and sometimes prediction quality.

---

## 7) Evaluation metrics

### Mean Squared Error (MSE)

[
\text{MSE}=\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y}_i)^2
]

### Root Mean Squared Error (RMSE)

[
\text{RMSE}=\sqrt{\text{MSE}}
]

### Mean Absolute Error (MAE)

[
\text{MAE}=\frac{1}{n}\sum_{i=1}^{n}|y_i-\hat{y}_i|
]

### (R^2) (Coefficient of determination)

[
R^2 = 1 - \frac{\sum_{i=1}^n (y_i-\hat{y}*i)^2}{\sum*{i=1}^n (y_i-\bar{y})^2}
]

* (R^2 = 1) is perfect fit
* (R^2 = 0) means “no better than predicting the mean”
* Can be negative if the model is worse than predicting (\bar{y})

---

## 8) Common pitfalls

* **Feature scaling:** gradient descent can be slow if features are on very different scales.
* **Outliers:** MSE strongly penalizes large errors → outliers can dominate the fit.
* **Multicollinearity:** correlated features can make coefficients unstable.
* **Data leakage:** do not fit preprocessing (scalers) on test data.
* **Non-linearity:** linear regression won’t capture curved relationships unless you add features (e.g., polynomial terms).

---

# Code examples

## A) Pure NumPy: closed-form solution

```python
import numpy as np

# Example data
X = np.array([[1.0], [2.0], [3.0], [4.0]])     # shape (n, d)
y = np.array([2.0, 3.0, 3.5, 5.0])             # shape (n,)

# Add bias column of ones
X_tilde = np.hstack([np.ones((X.shape[0], 1)), X])  # shape (n, d+1)

# Normal equation (using pseudo-inverse for stability)
w_tilde = np.linalg.pinv(X_tilde) @ y

b = w_tilde[0]
w = w_tilde[1:]

print("b =", b)
print("w =", w)

# Predictions
y_hat = X_tilde @ w_tilde
print("preds:", y_hat)
```

Why `pinv`? It uses a stable SVD-based pseudo-inverse, and works even when ((X^TX)) is not invertible.

---

## B) Pure NumPy: gradient descent from scratch

```python
import numpy as np

def train_linear_regression_gd(X, y, lr=0.1, epochs=1000):
    n, d = X.shape
    w = np.zeros(d)
    b = 0.0

    for _ in range(epochs):
        y_hat = X @ w + b
        error = y - y_hat

        # gradients for J(w,b) = (1/(2n)) * sum(error^2)
        grad_w = -(1/n) * (X.T @ error)
        grad_b = -(1/n) * np.sum(error)

        w -= lr * grad_w
        b -= lr * grad_b

    return w, b

# Example data
X = np.array([[1.0], [2.0], [3.0], [4.0]])
y = np.array([2.0, 3.0, 3.5, 5.0])

w, b = train_linear_regression_gd(X, y, lr=0.05, epochs=5000)
print("w =", w, "b =", b)

y_hat = X @ w + b
mse = np.mean((y - y_hat) ** 2)
print("MSE =", mse)
```

Tip: if training is unstable, reduce `lr` or normalize features.

---

## C) scikit-learn: the practical baseline

```python
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
```

---

# (Optional) Extensions

## Ridge Regression (L2 regularization)

Adds a penalty on large weights:

[
J_{\text{ridge}}(w,b)=\frac{1}{2n}\sum_{i=1}^{n}(y_i-(x_i^\top w+b))^2 + \lambda |w|_2^2
]

* Helps with multicollinearity
* Reduces overfitting

In scikit-learn:

```python
from sklearn.linear_model import Ridge

model = Ridge(alpha=1.0)  # alpha corresponds to lambda (scaled)
model.fit(X_train, y_train)
```

## Lasso Regression (L1 regularization)

[
J_{\text{lasso}}(w,b)=\frac{1}{2n}\sum_{i=1}^{n}(y_i-(x_i^\top w+b))^2 + \lambda |w|_1
]

* Encourages sparsity (some weights become exactly 0)
* Useful for feature selection

```python
from sklearn.linear_model import Lasso

model = Lasso(alpha=0.01)
model.fit(X_train, y_train)
```

---

## Quick checklist (what to remember)

* Linear regression predicts: (\hat{y}=Xw+b)
* It usually minimizes MSE
* You can solve it with:

  * **Normal equation / pseudo-inverse** (small to medium problems)
  * **Gradient descent** (large-scale)
* Evaluate with RMSE/MAE and (R^2)
* Consider Ridge/Lasso when overfitting or multicollinearity appear

---

If you want, I can also write a second page for your book: **“Linear Regression in practice: feature scaling, train/test splits, plots, residual analysis”** (with a small synthetic dataset + visualizations).
