import numpy as np

m = 2
p = 3
n = 4

A = np.array([[1, 2, 3], [4, 5, 6]])
B = np.array([[1, 2, 3, 4], [3, 4, 7, 8], [9, 0, 1, 2]])
C = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])

print(f'A:\n {A}')
print(f'B:\n {B}')
print(f'C:\n {C}')
print(f'Shape A: {A.shape}')
print(f'Shape B: {B.shape}')
print(f'Shape C: {C.shape}')

print(f'AB:\n {A @ B}')

D = A @ B + C
print(f'D:\n {D}')
print(f'Shape D: {D.shape}')

a = 2
b = 3

D = a * (A @ B) + b * C
print(f'D:\n {D}')
print(f'Shape D: {D.shape}')