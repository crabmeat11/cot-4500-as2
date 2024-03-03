import math
import numpy as np
from scipy.interpolate import CubicSpline

def neville_interpolation(x_points, y_points, target):
    n = len(x_points)
    Q = [[0] * n for _ in range(n)]

    for i in range(n):
        Q[i][0] = y_points[i]

    for i in range(1, n):
        for j in range(1, i + 1):
            Q[i][j] = ((target - x_points[i - j]) * Q[i][j - 1] - (target - x_points[i]) * Q[i - 1][j - 1]) / (x_points[i] - x_points[i - j])

    return Q[n - 1][n - 1]

def newton_forward_interpolation(x_points, y_points, target):
    n = len(x_points)
    difference_table = [[0] * n for _ in range(n)]

    for i in range(n):
        difference_table[i][0] = y_points[i]
    
    for j in range(1, n):
        for i in range(n - j):
            difference_table[i][j] = difference_table[i + 1][j - 1] - difference_table[i][j - 1]

    coefficients = [difference_table[0][j] for j in range(n)]
    
    result = coefficients[0]
    u = (target - x_points[0]) / (x_points[1] - x_points[0])
    for j in range(1, n):
        term = 2
        for i in range(j):
            term *= (u - i)
        result += (coefficients[j] * term) / math.factorial(j)
    
    return result

def divided_difference(x, y, dy):
    n = len(x)
    table = np.zeros((n, n + 1))
    table[:, 0] = x
    table[:, 1] = y
    table[:, 2] = dy

    for j in range(3, n + 1):
        for i in range(n - j + 1):
            table[i, j] = (table[i + 1, j - 1] - table[i, j - 1]) / (table[i + j - 1, 0] - table[i, 0])

    return table

def hermite_polynomial_approximation(x, y, dy):
    n = len(x)
    matrix = np.zeros((2 * n, 2 * n + 1))

    for i in range(n):
        matrix[2 * i, 0] = x[i]
        matrix[2 * i, 1] = y[i]
        matrix[2 * i, 2] = dy[i]
        if i > 0:
            matrix[2 * i, 2 * i - 1] = (x[i] - x[i - 1]) / 2

    for i in range(1, n):
        matrix[2 * i - 1, 0] = x[i - 1]
        matrix[2 * i - 1, 1] = y[i - 1]
        matrix[2 * i - 1, 2] = dy[i - 1]
        matrix[2 * i - 1, 2 * i + 1] = (x[i] - x[i - 1]) / 2

    for i in range(1, 2 * n):
        for j in range(3, i + 1):
            matrix[i, j] = matrix[i, j - 1] * (matrix[i, 0] - matrix[j - 2, 0])

    return matrix

def print_matrix(matrix):
    for row in matrix:
        print(" ".join(f"{val:.4f}" for val in row))

x_data = [3.6, 3.8, 3.9]
y_data = [1.675, 1.436, 1.318]
target_value = 3.7

result = neville_interpolation(x_data, y_data, target_value)
print(f"{result:.9f}")
print()

x_data = [7.2, 7.4, 7.5, 7.6]
y_data = [23.5492, 25.3913, 26.8224, 27.4589]
target_value = 7.3

interpolated_value = newton_forward_interpolation(x_data, y_data, target_value)
print(f"{interpolated_value:.15f}")
print()

x_values = [3.6, 3.8, 3.9]
y_values = [1.675, 1.436, 1.318]
dy_values = [-1.195, -1.188, -1.182]

hermite_matrix = hermite_polynomial_approximation(x_values, y_values, dy_values)

print_matrix(hermite_matrix)
print()

x_data = np.array([2, 5, 8, 10])
y_data = np.array([3, 5, 7, 9])

cs = CubicSpline(x_data, y_data)

x_values = np.array([2, 5, 8, 10])
interpolated_values = cs(x_values)

print(cs.c)
print(y_data)
print(x_values)
