"""
Module: matrix_library
Purpose: Object-oriented custom matrix operations without external libraries.
"""

class Matrix:
    """Base class for all matrices."""
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
        self.data = [[0.0 for _ in range(cols)] for _ in range(rows)]

    def get_val(self, i, j):
        return self.data[i][j]

    def set_val(self, i, j, val):
        self.data[i][j] = val

    def transpose(self):
        """Returns the transpose of the matrix."""
        transposed = Matrix(self.cols, self.rows)
        for i in range(self.rows):
            for j in range(self.cols):
                transposed.set_val(j, i, self.get_val(i, j))
        return transposed

    def multiply(self, other):
        """Multiplies this matrix with another matrix or vector."""
        if isinstance(other, list):  # Matrix-Vector multiplication
            if self.cols != len(other):
                raise ValueError("Dimension mismatch.")
            result = [0.0] * self.rows
            for i in range(self.rows):
                for j in range(self.cols):
                    result[i] += self.get_val(i, j) * other[j]
            return result
            
        # Matrix-Matrix multiplication
        if self.cols != other.rows:
            raise ValueError("Dimension mismatch.")
        result = Matrix(self.rows, other.cols)
        for i in range(self.rows):
            for j in range(other.cols):
                val = sum(self.get_val(i, k) * other.get_val(k, j) for k in range(self.cols))
                result.set_val(i, j, val)
        return result

class RectangularMatrix(Matrix):
    """Class for MxN matrices (M != N)."""
    def __init__(self, rows, cols):
        if rows == cols:
            raise ValueError("Rows and columns must be different for a RectangularMatrix.")
        super().__init__(rows, cols)

class SquareMatrix(Matrix):
    """Class for NxN matrices."""
    def __init__(self, size):
        super().__init__(size, size)

class SymmetricMatrix(SquareMatrix):
    """Class for symmetric matrices, storing only the upper triangle to save space."""
    def __init__(self, size):
        self.rows = size
        self.cols = size
        self.data = {} # Only store upper triangle

    def set_val(self, i, j, val):
        if i <= j:
            self.data[(i, j)] = val
        else:
            self.data[(j, i)] = val

    def get_val(self, i, j):
        if i <= j:
            return self.data.get((i, j), 0.0)
        else:
            return self.data.get((j, i), 0.0)

class SparseMatrix(SquareMatrix):
    """
    Class for Sparse matrices utilizing Dictionary of Keys (DOK) storage scheme.
    Highly efficient for Global Stiffness Matrix assembly.
    """
    def __init__(self, size):
        self.rows = size
        self.cols = size
        self.data = {}  # Format: {(row, col): value}

    def set_val(self, i, j, val):
        if val != 0.0:
            self.data[(i, j)] = val
        elif (i, j) in self.data:
            del self.data[(i, j)]

    def get_val(self, i, j):
        return self.data.get((i, j), 0.0)

    def add_val(self, i, j, val):
        """Adds to an existing value, crucial for matrix assembly."""
        if val != 0.0:
            current = self.get_val(i, j)
            self.set_val(i, j, current + val)

    def solve(self, b_vector):
        """
        Solves [A]{x} = {b} using Gaussian Elimination adapted for Sparse Data.
        Inputs: b_vector (list)
        Outputs: x_vector (list)
        """
        n = self.rows
        # Convert to dense locally just for the exact solve step to ensure stability 
        # (A fully sparse direct solver is highly complex; this uses sparse storage but solves directly)
        A = [[self.get_val(i, j) for j in range(n)] for i in range(n)]
        b = list(b_vector)

        # Forward elimination
        for i in range(n):
            pivot = A[i][i]
            if pivot == 0:
                raise ValueError("Zero pivot encountered.")
            for j in range(i + 1, n):
                factor = A[j][i] / pivot
                for k in range(i, n):
                    A[j][k] -= factor * A[i][k]
                b[j] -= factor * b[i]

        # Back substitution
        x = [0.0] * n
        for i in range(n - 1, -1, -1):
            sum_ax = sum(A[i][j] * x[j] for j in range(i + 1, n))
            x[i] = (b[i] - sum_ax) / A[i][i]
            
        return x