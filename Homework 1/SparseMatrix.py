from typing import List
import numpy as np
from itertools import chain

MatrixList = List[List[float]]


class SparseMatrix:
    def __init__(self, data: MatrixList, indices: MatrixList) -> None:
        self._data = data
        self._indices = indices

    def convertToFull(self):
        """Converts matrix to full representation in numpy"""
        adjusted_indices = [
            np.array(row) + i * len(self._indices)
            for i, row in enumerate(self._indices)
        ]
        adjusted_indices = np.array(list(chain(*adjusted_indices)))

        adjusted_data = np.array(list(chain(*self._data)))

        full_matrix = np.zeros(len(self._data) ** 2)
        full_matrix[adjusted_indices] = adjusted_data
        full_matrix = full_matrix.reshape(len(self._data), len(self._data))
        return full_matrix

    def __matmul__(self, x: List[float] | np.ndarray):
        res = np.zeros_like(x).astype(float)
        for i in range(len(self._indices)):
            for j in range(len(self._indices[i])):
                res[i] += self._data[i][j] * x[self._indices[i][j]]
        return res.tolist()

    def __repr__(self) -> str:
        return str(self.convertToFull())

    def __getitem__(self, key):
        if type(key) != tuple or len(key) != 2:
            raise ValueError("Element access must be in the from [row, col]")

        for i in range(len(self._indices[key[0]])):
            if self._indices[key[0]][i] == key[1]:
                return self._data[key[0]][i]
            elif self._indices[key[0]][i] > key[1]:
                break
        return 0.0

    def __len__(self):
        return len(self._data)

    @staticmethod
    def fromFull(matrix: MatrixList | np.ndarray):
        """Takes full matrix in either list or numpy formats and constructs SparseMatrix"""
        matrix = np.array(matrix).astype(float)
        data = [row[row != 0].tolist() for row in matrix]
        indices = [np.nonzero(row)[0].tolist() for row in matrix]
        return SparseMatrix(data, indices)


def generateSparseSDDMatrix(size: int, min_val=-100, max_val=100, sparsity=0.5):
    """
    Generates strictly diagonal dominant matrix of size*size. i.e. absolute values of diagonal
    elements are strictly larger then the row-wise sum of absolute values of other
    elements.
    
    Arguments:
    - size: size of the matrix
    - min_val and max_val: min and max value in matrix
    - sparsity: probability of generating zero element 
    Outputs:
    - sparse SDD matrix
    """
    A = np.random.rand(size, size) * (max_val - min_val) + min_val
    A *= np.random.choice([0, 1], size, p=[sparsity, 1 - sparsity])
    diagonal = (
        abs(A).sum(1)
        - abs(A).diagonal()
        + np.random.random(A.diagonal().shape[0])
        + 0.1
    ) #To assure that the matrix is SDD
    np.fill_diagonal(A, diagonal)
    return A

def sor(A, b, x0, omega, tol=1e-10):
    """
    Solves linear system with SOR iteration.
    
    Arguments:
    - A: input matrix
    - b: right side of equation
    - x0: initial approximation
    - omega: relaxation factor
    - tol: error tolerance
    Outputs:
    - x: solution of a linear system
    - it: number of iterations until tolerance is met
    
    To assure convergence the input must be strictly diagonal dominant matrix.
    """
    x = x0

    it = 0
    while np.max(np.abs(A @ x - b)) > tol:
        for i in range(len(A)):
            x_new = b[i]
            for j in range(len(A)):
                if i != j:
                    x_new -= (A[i,j] * x[j])
            x[i] = (1-omega) * x[i] + omega/A[i,i] * x_new
        it += 1
    return x, it