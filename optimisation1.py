import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linprog, minimize

# Сітка значень
x1 = np.linspace(-2, 6, 100)
x2 = np.linspace(-4, 6, 100)
X1, X2 = np.meshgrid(x1, x2)

# Функції
functions = {
    "f1": {
        "function": lambda x: -2 * x[0] + 2 * x[1],
        "linear_coeffs": [-2, 2]
    },
    "f2": {
        "function": lambda x: (x[0] - 5)**2 + (x[1] + 1)**2,
        "linear_coeffs": None
    }
}

# Обмеження
constraints = [
    {"type": "ineq", "fun": lambda x: -x[0] + x[1] - 3},  # -x1 + x2 <= 3
    {"type": "ineq", "fun": lambda x: 3*x[0] + x[1] - 11},  # 3x1 + x2 <= 11
    {"type": "ineq", "fun": lambda x: -(3*x[0] + 5*x[1]) + 7}  # 3x1 + 5x2 >= 7 -> -3x1 - 5x2 <= -7
]

# Результати оптимізації
results = {}
for key, details in functions.items():
    function = details["function"]
    if details["linear_coeffs"] is not None:  # Лінійна оптимізація
        res_min = linprog(details["linear_coeffs"], A_ub=[[-1, 1], [3, 1], [-3, -5]], b_ub=[3, 11, -7])
        res_max = linprog([-c for c in details["linear_coeffs"]], A_ub=[[-1, 1], [3, 1], [-3, -5]], b_ub=[3, 11, -7])
    else:  # Нелінійна оптимізація
        res_min = minimize(function, x0=[0, 0], constraints=constraints)
        res_max = minimize(lambda x: -function(x), x0=[0, 0], constraints=constraints)
    results[key] = {"min": res_min, "max": res_max}

# Графічне представлення
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
x_vals = np.linspace(-2, 6, 200)

for ax, (key, details) in zip(axes, functions.items()):
    F = np.vectorize(lambda x1, x2: details["function"]([x1, x2]))(X1, X2)
    contour = ax.contour(X1, X2, F, levels=20, cmap="coolwarm")
    ax.set_title(f'Contours of ${key}(x_1, x_2)$ with Solutions')
    ax.set_xlabel(r'$x_1$')
    ax.set_ylabel(r'$x_2$')
    fig.colorbar(contour, ax=ax, label=f'${key}(x_1, x_2)$')
    
    # Додаємо обмеження як лінії
    ax.plot(x_vals, x_vals + 3, 'r--', label=r'$-x_1 + x_2 = 3$')
    ax.plot(x_vals, 11 - 3*x_vals, 'g--', label=r'$3x_1 + x_2 = 11$')
    ax.plot(x_vals, (7 - 3*x_vals) / 5, 'b--', label=r'$3x_1 + 5x_2 = 7$')
    
    ax.scatter(*results[key]["min"].x, color='blue', marker='o', label=f'Min {key}')
    ax.scatter(*results[key]["max"].x, color='red', marker='o', label=f'Max {key}')
    ax.legend()

plt.show()

# Вивід значень
for key, res in results.items():
    print(f"Analytical Min {key}: {res['min'].fun} at {res['min'].x}")
    print(f"Analytical Max {key}: {-res['max'].fun} at {res['max'].x}")
