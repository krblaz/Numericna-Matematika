from typing import List, Dict
import numpy as np
import networkx as nx
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

def sor(A: SparseMatrix, b: np.ndarray | List[float], x0: np.ndarray | List[float] = None, omega: float = 0.5, tol:float = 1e-10, max_iter: int = 1000):
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
    - max_iter: max number of iterations before quitting
    
    To assure convergence the input must be strictly diagonal dominant matrix.
    """
    
    if x0 is None:
        x = np.random.rand(len(A))
    else:
        x = x0.copy()

    it = 0
    while np.max(np.abs(A @ x - b)) > tol and it < max_iter:
        for i in range(len(A)):
            x_new = b[i]
            for j in range(len(A)):
                if i != j:
                    x_new -= (A[i,j] * x[j])
            x[i] = (1-omega) * x[i] + omega/A[i,i] * x_new
        it += 1
    return x, it

def getGraphPosSOR(G: nx.Graph, initial_pos: Dict[int, List[float]], omega: float = 0.5, tol: float = 1e-10):
  """
  Sets position of vertices in a graphs according to Force-Directed Graph Drawing method in euclidean space using SOR method
  
  Arguments:
  - G: graph in a NetworkX format
  - initial_pos: optional dict where keys are vertex labels and values are initial 2D coordinates of vertices
  - omega: relaxation factor for SOR method
  - tol: error tolerance for SOR method
  Outputs:
  - pos: same as initial_pos only that all vertices all present
  - it: number of iterations until tolerance is met
  """
  A = SparseMatrix.fromFull(nx.adjacency_matrix(G).todense())
  st = np.array(nx.degree(G))[:,1]
  
  pos_filter = np.zeros_like(st, dtype=bool)
  pos_filter[list(initial_pos.keys())] = 1

  pos = np.random.rand(pos_filter.shape[0],2)
  initial_pos_array = np.array(list(initial_pos.values()))
  pos[pos_filter] = initial_pos_array
  
  pos_prev = np.ones_like(pos) * np.Inf
  it = 0
  while np.max(np.abs(pos_prev - pos)) > tol:
    pos_prev = pos.copy()
    for i in range(len(A)):
      pos_new = 0
      for j in range(len(A)):
        if i != j:
          pos_new += (A[i,j] * pos[j])
      pos[i] = (1-omega) * pos[i] + omega/st[i] * pos_new
    pos[pos_filter] = initial_pos_array
    it+=1
  return {k:v.tolist() for k,v in enumerate(pos)}, it