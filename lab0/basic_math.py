import numpy as np


def matrix_multiplication(matrix_a, matrix_b):
    if len(matrix_a[0]) != len(matrix_b):
        raise ValueError("Матрицы нельзя умножить из-за неправильных размеров")
    else:
        result = []

        for _ in range(len(matrix_a)):
            result += [[0] * len(matrix_b[0])]

        for i in range(len(matrix_a)):
            for j in range(len(matrix_b[0])):
                for k in range(len(matrix_b)):
                    result[i][j] += matrix_a[i][k] * matrix_b[k][j]

        return result


def functions(a_1, a_2):
    if a_1 == a_2:
        return None

    a11, a12, a13 = map(float, a_1.split())
    a21, a22, a23 = map(float, a_2.split())

    def F(x):
        return a11 * x ** 2 + a12 * x + a13

    def P(x):
        return a21 * x ** 2 + a22 * x + a23

    x_extr_f = (-a12) / (2 * a11) if a11 != 0 else None
    x_extr_p = (-a22) / (2 * a21) if a21 != 0 else None

    solutions = []

    a = a11 - a21
    b = a12 - a22
    c = a13 - a23
    discriminant = b ** 2 - 4 * a * c

    if a == 0:
        if b != 0:
            x = -c / b
            solutions.append((x, F(x)))
    elif discriminant > 0:
        x1 = (-b + (discriminant) ** 0.5) / (2 * a)
        x2 = (-b - (discriminant) ** 0.5) / (2 * a)
        solutions.append((x1, F(x1)))
        solutions.append((x2, F(x2)))

    return solutions


def skew(x):
    x = np.array(x)
    x_e = np.mean(x)
    m3 = np.sum((x - x_e) ** 3) / len(x)
    d2 = np.sum((x - x_e) ** 2) / len(x)
    A3 = m3 / (d2 ** 1.5)
    return round(A3, 2)


def kurtosis(x):
    x = np.array(x)
    x_e = np.mean(x)
    m4 = np.sum((x - x_e) ** 4) / len(x)
    d2 = np.sum((x - x_e) ** 2) / len(x)
    E4 = m4 / (d2 ** 2) - 3
    return round(E4, 2)
