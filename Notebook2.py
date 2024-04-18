from collections import deque
from typing import Any

import numpy as np
from numpy import ndarray, dtype
import matplotlib.pyplot as plt


def task1():
	print("""
	Определитель исходной матрицы и определитель обратной матрицы связаны следующим образом: 
	если матрица A обратима (то есть имеет обратную матрицу), 
	то определитель обратной матрицы равен обратному определителю исходной матрицы, 
	то есть det(A^{-1}) = frac{1}{det(A)}. 
	
	Это следует из свойства определителя, 
	который для обратимых матриц можно выразить через определитель исходной матрицы.
	""")

	print("""
	Обратная матрица A^{-1} существует, если и только если определитель матрицы A отличен от нуля, то есть det(A != 0. 
	
	Если определитель матрицы A равен нулю det(A) == 0, 
	то матрица A считается вырожденной, и у нее нет обратной матрицы.
	""")


def task2():
	# Задаем координаты трех точек
	P1 = (1, 2)  # (x1, y1)
	P2 = (3, 5)  # (x2, y2)
	P3 = (4, 7)  # (x3, y3)

	# Строим и рисуем прямую, проходящую через P1 и P2
	# Уравнение прямой y = kx + b, где k - наклон прямой, b - свободный член
	# Находим k и b из уравнения системы уравнений с двумя точками
	A = np.array([[P1[0], 1], [P2[0], 1]])
	b = np.array([P1[1], P2[1]])
	k, b = np.linalg.solve(A, b)

	# Создаем точки для построения прямой
	x = np.linspace(0, 5, 100)
	y = k * x + b

	# Рисуем прямую, проходящую через P1 и P2
	plt.plot(x, y, label='Line through P1 and P2')

	# Строим и рисуем прямую, проходящую через P1 и P3
	A = np.array([[P1[0], 1], [P3[0], 1]])
	b = np.array([P1[1], P3[1]])
	k, b = np.linalg.solve(A, b)

	# Создаем точки для построения прямой
	x = np.linspace(0, 5, 100)
	y = k * x + b

	# Рисуем прямую, проходящую через P1 и P3
	plt.plot(x, y, label='Line through P1 and P3')

	# Строим и рисуем прямую, проходящую через P2 и P3
	A = np.array([[P2[0], 1], [P3[0], 1]])
	b = np.array([P2[1], P3[1]])
	k, b = np.linalg.solve(A, b)

	# Создаем точки для построения прямой
	x = np.linspace(0, 5, 100)
	y = k * x + b

	# Рисуем прямую, проходящую через P2 и P3
	plt.plot(x, y, label='Line through P2 and P3')

	# Рисуем точки P1, P2, P3
	plt.scatter([P1[0], P2[0], P3[0]], [P1[1], P2[1], P3[1]], color='red', label='Points')
	plt.legend()
	plt.show()


def task3():
	# Задаем коэффициенты для трех прямых
	k1, l1 = 2, 1
	k2, l2 = -1, 4
	k3, l3 = 0.5, 0

	# Находим точки пересечения прямых (решаем системы уравнений)
	A = np.array([[k1, -1], [k2, -1]])
	b = np.array([-l1, -l2])
	x, y = np.linalg.solve(A, b)

	# Находим точку пересечения прямой 1 и 3
	x_13 = (l3 - l1) / (k1 - k3)
	y_13 = k1 * x_13 + l1

	# Находим точку пересечения прямой 2 и 3
	x_23 = (l3 - l2) / (k2 - k3)
	y_23 = k2 * x_23 + l2

	# Создаем массивы для построения прямых
	x_values = np.linspace(-10, 10, 100)
	y_values1 = k1 * x_values + l1
	y_values2 = k2 * x_values + l2
	y_values3 = k3 * x_values + l3

	# Рисуем прямые
	plt.plot(x_values, y_values1, label='y = {}x + {}'.format(k1, l1))
	plt.plot(x_values, y_values2, label='y = {}x + {}'.format(k2, l2))
	plt.plot(x_values, y_values3, label='y = {}x + {}'.format(k3, l3))

	# Рисуем точки пересечения
	plt.scatter(x, y, color='red', label='Intersection point 1&2')
	plt.scatter(x_13, y_13, color='green', label='Intersection point 1&3')
	plt.scatter(x_23, y_23, color='blue', label='Intersection point 2&3')

	plt.legend()
	plt.grid()
	plt.show()


def task4():
	# Функция для определения определителя матрицы
	def check_determinant(n):
		A = np.zeros((n, n))
		b = np.arange(1, n + 1)
		for i in range(n):
			A[i, i] = 1
			if i == n - 1:
				A[i, 0] = 1
			else:
				A[i, i + 1] = 1

		determinant = np.linalg.det(A)
		return determinant

	# Найдем максимальное n, для которого система имеет решение
	n = 3
	max_n = 3
	while n <= 30:
		determinant = check_determinant(n)
		if determinant != 0:
			max_n = n
		n += 1

	# Находим решение для максимального n
	A_max = np.zeros((max_n, max_n))
	b_max = np.arange(1, max_n + 1)
	for i in range(max_n):
		A_max[i, i] = 1
		if i == max_n - 1:
			A_max[i, 0] = 1
		else:
			A_max[i, i + 1] = 1
	x_max = np.linalg.solve(A_max, b_max)

	# Отображаем точки (i, x_i) на графике
	plt.scatter(np.arange(1, max_n + 1), x_max, color='b')
	plt.xlabel('i')
	plt.ylabel('x_i')
	plt.title('Solution for n = {}'.format(max_n))
	plt.grid()
	plt.show()

	# Выводим множество решений для системы
	solution = [f'x_{i} = {x_max[i]}' for i in range(len(x_max))]
	print('Множество решений для системы с n = {}:'.format(max_n))
	for s in solution:
		print(s)


def task5():
	def check_determinant(n, b):
		A = np.zeros((n, n))

		# Заполнение матрицы A в зависимости от количества уравнений
		for i in range(n):
			A[i, (i + 1) % n] = 1
			A[i, (i + 2) % n] = 1
			A[i, i] = 1

		# Проверка определителя
		determinant = np.linalg.det(A)

		if determinant != 0:  # Если определитель не равен нулю, есть решение
			x = np.linalg.solve(A, b)
			return x
		else:
			return None

	# Исследование первой системы уравнений
	print("Исследование первой системы уравнений:")
	n = 3
	max_n = 3
	x_max = None

	while n <= 30:
		b = np.arange(1, n)
		x = check_determinant(n, b)
		if x is not None:
			max_n = n
			x_max = x
		n += 1

	if x_max is not None:
		plt.scatter(np.arange(1, max_n + 1), x_max, color='b')
		plt.xlabel('i')
		plt.ylabel('x_i')
		plt.title('Solution for n = {}'.format(max_n))
		plt.grid()
		plt.show()

		print('Множество решений для первой системы с n = {}:'.format(max_n))
		for i, val in enumerate(x_max):
			print(f'x_{i + 1} = {val}')
	else:
		print('Для всех n от 3 до 30 первая система не имеет решения.')

	# Исследование второй системы уравнений
	print("\nИсследование второй системы уравнений:")
	n = 3
	max_n = 3
	x_max = None

	while n <= 30:
		b = np.arange(1, n - 1)
		b = np.append(b, n)
		x = check_determinant(n, b)
		if x is not None:
			max_n = n
			x_max = x
		n += 1

	if x_max is not None:
		plt.scatter(np.arange(1, max_n + 1), x_max, color='b')
		plt.xlabel('i')
		plt.ylabel('x_i')
		plt.title('Solution for n = {}'.format(max_n))
		plt.grid()
		plt.show()

		print('Множество решений для второй системы с n = {}:'.format(max_n))
		for i, val in enumerate(x_max):
			print(f'x_{i + 1} = {val}')
	else:
		print('Для всех n от 3 до 30 вторая система не имеет решения.')


def main():
	task1()
	task2()
	task3()
	task4()
	task5()


task2()
