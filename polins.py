import numpy as np
import matplotlib.pyplot as plt


def polin_for_n(n):
	X = np.arange(n)
	Y = np.sin(2 * X)

	R = np.empty([n, n])

	for i in range(n):
		for j in range(n):
			R[i][j] = X[i] ** j

	A = np.linalg.solve(R, Y)
	linsp = np.linspace(X.min(), X.max(), num=100)
	f = np.poly1d(np.flip(A))

	fun = [f(x) for x in linsp]

	plt.plot(linsp, fun)
	plt.plot(X, Y, 'ro')
	plt.plot(linsp, np.sin(2 * linsp), 'g--')

	plt.grid(True)

	for i in range(len(linsp)):
		if abs(fun[i]) > 1:
			print(i, end='')


def handle_line_by_2_points():
	def line_by_points(P1, P2):
		X = np.array([P1[0], P2[0]])
		Y = np.array([P1[1], P2[1]])
		A = np.vstack([X, np.ones(2)]).T  # Склейка матриц
		print(A, Y, sep='\n')

		K = np.linalg.solve(A, Y)
		# K = np.linalg.inv(A).dot(Y) # В явном виде через обрантую матрицу
		# Замечание: Формально Y в обоих вариантах должен быть столбцом, а не строкой, но функции и так работают правильно.

		return K

	P1 = [2, 0]
	P2 = [-2, 4]
	K = line_by_points(P1, P2)
	print(K)
	x = np.linspace(-5, 5)  # Множество значений от -5 до +5 (с шагом 50 по умолчанию)
	y = K[0] * x + K[1]
	plt.plot(x, y)  # Строим график прямой
	plt.plot(*P1, 'ro')  # Рисуем красную (r) точку (o)
	plt.plot(*P2, 'go')  # Рисуем зеленую (g) точку (o)
	plt.axis([-5, 5, -5, 5])  # Устанавливаем масштабы осей
	plt.grid(True)  # Отображаем сетку


def handle_point_by_lines():
	def point_by_lines(R1, R2):
		K = np.array([R1[0], R2[0]])
		L = np.array([R1[1], R2[1]])
		A = np.vstack([K, -np.ones(2)]).T
		D = np.linalg.solve(A, -L)
		return D

	R1 = [2, -1]  # y = 2x - 1
	R2 = [-1, 1]  # y = -x + 1

	x = np.linspace(-5, 5)
	plt.plot(x, R1[0] * x + R1[1])
	plt.plot(x, R2[0] * x + R2[1])

	D = point_by_lines(R1, R2)
	print(D)
	plt.plot(*D, 'ro')

	plt.axis([-5, 5, -5, 5])
	plt.grid(True)



