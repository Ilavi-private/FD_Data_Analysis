from collections import deque
from typing import Any

import numpy as np
from numpy import ndarray, dtype


def task1():
	def snake1(n, m):
		n = np.arange(1, n * m + 1).reshape(n, m)  # Создаём из массива матрицу нужного размера

		for i in range(1, len(n)):
			n[i] = n[i][::(1 if i % 2 == 0 else -1)]

		return n

	a, b = map(int, input('snake 1: ').split())
	print(snake1(a, b))


def task2():
	def snake2(n):
		# Создаем пустую матрицу n x n, заполненную нулями
		mat = np.zeros((n, n), dtype=int)

		num = 1
		# Задаем начальные значения для строк и столбцов
		top, bottom, left, right = 0, n - 1, 0, n - 1

		while top <= bottom and left <= right:
			# Заполняем верхнюю строку слева направо
			for i in range(left, right + 1):
				mat[top][i] = num
				num += 1
			top += 1

			# Заполняем правый столбец сверху вниз, но меняем направление каждый второй раз
			direction = 1 if (right - left) % 2 == 0 else -1
			for i in range(top, bottom + 1):
				mat[right][i] = num
				num += 1
			right += direction

			# Заполняем нижнюю строку справа налево
			for i in range(right, left - 1, -1):
				mat[i][bottom] = num
				num += 1
			bottom -= 1

			# Заполняем левый столбец снизу вверх, но меняем направление каждый второй раз
			direction = 1 if (right - left) % 2 == 0 else -1
			for i in range(bottom, top - 1, -1):
				mat[i][left] = num
				num += 1
			left += direction

		return mat

	a = int(input('snake 2: '))
	print(snake2(a))


def task3():
	def snake3(n, m):
		# Создаем пустую матрицу размером n x m
		matrix = np.zeros((n, m), dtype=int)

		x, y = 0, 0  # Начальные координаты
		dx, dy = 0, 1  # Начальное направление движения (вправо)

		for num in range(1, n * m + 1):
			matrix[x, y] = num  # Записываем число в текущую ячейку

			# Проверяем следующую ячейку на выход за границы матрицы или уже заполненную
			if 0 <= x + dx < n and 0 <= y + dy < m and matrix[x + dx, y + dy] == 0:
				x, y = x + dx, y + dy
			else:
				# Поворачиваем направление движения на 90 градусов по часовой стрелке
				dx, dy = dy, -dx
				x, y = x + dx, y + dy

		return matrix

	a, b = map(int, input('snake 3: ').split())
	print(snake3(a, b))


def task4():
	def paraboloid(n):
		# Отображаем результат с округлением до двух знаков после запятой
		np.set_printoptions(precision=2)
		# Создаем массив индексов для каждой оси
		x = np.linspace(-n, n, 2 * n + 1)
		y = np.linspace(-n, n, 2 * n + 1)

		# Создаем двумерный массив сеток, содержащий координаты точек
		X, Y = np.meshgrid(x, y)

		# Вычисляем расстояние от точки (0, 0) до каждой точки на сетке
		distances = np.sqrt((X / n) ** 2 + (Y / n) ** 2)

		return distances

	a = int(input('paraboloid: '))
	print(paraboloid(a))


def task5():
	def chess_queen(n, m, i, j):
		# Создаем матрицу размера n x m, заполненную нулями
		board = np.zeros((n, m), dtype=np.int64)

		# Поставляем значение 2 в ячейку (i, j)
		board[i, j] = 2

		# Определяем направления, в которых может ходить ферзь: горизонталь, вертикаль, диагонали
		directions = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (-1, -1), (1, -1), (-1, 1)]

		# Помечаем клетки, куда может ударить ферзь, значением 1
		for dx, dy in directions:
			x, y = i, j

			while 0 <= x + dx < n and 0 <= y + dy < m:
				x, y = x + dx, y + dy
				board[x, y] = 1

		return board

	a, b, c, d = map(int, input('chess_queen: ').split())
	print(chess_queen(a, b, c, d))


def task6():
	def chess_knight(n, m, i, j):
		# Создаем матрицу заполненную -1
		visited = np.full((n, m), -1)
		# Устанавливаем значение в стартовую позицию коня
		visited[i][j] = 0

		# Создаем массив с возможными ходами коня
		moves = [(2, 1), (2, -1), (-2, 1), (-2, -1), (1, 2), (1, -2), (-1, 2), (-1, -2)]

		# Создаем список координат, начиная с исходной позиции коня
		queue = [(i, j)]

		while queue:
			# Берем первую координату из очереди
			x, y = queue.pop(0)

			# Перебираем все возможные ходы коня
			for dx, dy in moves:
				new_x, new_y = x + dx, y + dy
				# Проверяем, что новые координаты находятся на поле и еще не посещены
				if 0 <= new_x < n and 0 <= new_y < m and visited[new_x][new_y] == -1:
					visited[new_x][new_y] = visited[x][y] + 1
					queue.append((new_x, new_y))

		return visited

	a, b, c, d = map(int, input('chess_knight: ').split())
	print(chess_knight(a, b, c, d))


def task7():
	def generate(a, b, c):
		n, m = a.shape
		p, q = b.shape
		result = np.zeros((n * p, m * q), dtype=np.int64)

		for i in range(n):
			for j in range(m):
				# Проверяем значение элемента матрицы a
				if a[i, j] == 0:
					# Если значение 0, то на соответствующем месте в результате ставим матрицу b
					result[i * p:(i + 1) * p, j * q:(j + 1) * q] = b
				else:
					# Если значение 1, то на соответствующем месте в результате ставим матрицу c
					result[i * p:(i + 1) * p, j * q:(j + 1) * q] = c

		return result

	a = np.array([[1, 1, 0], [0, 1, 1]])
	b = np.array([[1, 2], [3, 4]])
	c = np.array([[3, 3], [3, 3]])

	print(generate(a, b, c))


def task8():
	def is_symmetric(A):
		# Получаем размерность матрицы
		n = len(A)

		# Проверяем условие симметричности матрицы
		for i in range(n):
			for j in range(i + 1, n):
				if A[i][j] != A[j][i]:
					return False

		return True

	A = [[1,2,3], [2,5,4]]
	print(is_symmetric(A))  # Ожидаемый результат: False


def task9():
	def transpose(A):
		# Получаем размеры исходной матрицы A
		n = len(A)
		m = len(A[0])

		# Создаем пустую матрицу размера m x n для транспонированной матрицы
		result = [[0 for _ in range(n)] for _ in range(m)]

		# Заполняем транспонированную матрицу значениями из исходной
		for i in range(n):
			for j in range(m):
				result[j][i] = A[i][j]

		return result

	A = [[1, 2, 3],
	     [4, 5, 6],
	     [7, 8, 9]]
	A = transpose(A)
	for i in A:
		print(i)


def task10():
	def mult(A, B):
		# Проверяем возможность умножения матриц - соответствие высоты и ширины.
		if len(A[0]) != len(B):
			return "Ошибка: Умножение невозможно"

		# Определяем размеры результатирующей матрицы
		n = len(A)
		m = len(B[0])
		p = len(A[0])

		# Создаем пустую матрицу размера n x m для результата умножения
		result = [[0 for _ in range(m)] for _ in range(n)]

		# Вычисляем произведение матриц
		for i in range(n):
			for j in range(m):
				for k in range(p):
					result[i][j] += A[i][k] * B[k][j]

		return result

	A = [[1, 2], [3, 4]]
	B = [[2, 0], [1, 2]]
	print(mult(A, B))


def task11():
	def max_in_rows_and_min_in_columns(A):
		# Находим максимальные значения в каждой строке и сохраняем их в матрицу B размера n x 1
		B = np.max(A, axis=1, keepdims=True)

		# Находим минимальные значения в каждом столбце и сохраняем их в матрицу C размера 1 x m
		C = np.min(A, axis=0, keepdims=True)

		return B, C

	A = [[1, 3], [4, 1], [5, 6]]
	B, C = max_in_rows_and_min_in_columns(A)
	# B = [[3],
	#      [4],
	#      [6]]
	#
	# C = [[1, 1]]
	print("B:\n", B)
	print("C:\n", C)


def task12():
	def sum_in_rows_and_product_in_columns(A):
		# Находим суммы элементов в каждой строке и сохраняем их в матрицу B размера n x 1
		B = np.sum(A, axis=1, keepdims=True)

		# Находим произведения элементов в каждом столбце и сохраняем их в матрицу C размера 1 x m
		C = np.prod(A, axis=0, keepdims=True)

		return B, C

	A = [[1, 3], [4, 1], [5, 6]]

	B, C = sum_in_rows_and_product_in_columns(A)
#  	B = [[4],
# 	     [5],
# 	     [11]]
#
# 	C = [[20, 18]]
	print("B:\n", B)
	print("C:\n", C)


def task13():
	def normalize_matrix(A):
		# Находим сумму всех элементов матрицы A
		sum_A = np.sum(A)

		# Нормируем матрицу A, разделяя каждый элемент на сумму всех элементов
		normalized_A = A / sum_A

		return normalized_A

	A = np.array([[1, 2], [3, 4]])
	print(normalize_matrix(A))


def task14():
	def normalize_columns(A):
		# Считаем сумму элементов в каждом столбце
		col_sums = np.sum(A, axis=0)

		# Нормируем каждый столбец матрицы A, деля каждый элемент на сумму элементов столбца
		normalized_A = A / col_sums

		return normalized_A

	A = np.array([[1, 2], [3, 4]])
	print(normalize_columns(A))


def task15():
	def process_columns(A):
		# Создаем массив из индексов столбцов, начиная с 0
		column_indices = np.arange(A.shape[1])

		# Используем операции по модулю для определения четных и нечетных индексов
		even_columns = A[:, column_indices % 2 == 0] * 5
		odd_columns = A[:, column_indices % 2 == 1] / -5

		# Объединяем результаты обработки четных и нечетных столбцов
		processed_A = np.hstack((even_columns, odd_columns))

		return processed_A

	A = np.array([[1, 2, 3], [4, 5, 6]])
	print(process_columns(A))


def main():
	task1()
	task2()
	task3()
	task4()
	task5()
	task6()
	task7()
	task8()
	task9()
	task10()
	task11()
	task12()
	task13()
	task14()
	task15()


task8()