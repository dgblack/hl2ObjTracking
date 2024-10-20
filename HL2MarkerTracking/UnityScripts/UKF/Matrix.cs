using System;
using System.Security.Claims;
using UnityEngine;

public class Matrix
{
    public double[,] mat { get; }
    public int M { get; }
    public int N { get; }

    public Matrix(int m, int n)
    {
        M = m;
        N = n;
        mat = new double[m, n];
    }

    public Matrix(double[,] m)
    {
        M = m.GetLength(0);
        N = m.GetLength(1);
        mat = m;
    }

    public static Matrix Identity(int m)
    {
        Matrix M = new Matrix(m, m);

        for (int i = 0; i < m; i++)
            M[i, i] = 1;

        return M;
    }

    public double[,] Mat()
    {
        double[,] outd = new double[M, N];
        for (int i = 0; i < M; i++)
        {
            for (int j = 0; j < N; j++)
            {
                outd[i, j] = mat[i, j];
            }
        }
        return outd;
    }

    public double this[int row, int col]
    {
        get => mat[row, col];
        set => mat[row, col] = value;
    }

    public Vector this[int col]
    {
        get => GetCol(col);
        set => SetCol(col, value);
    }
    private Vector GetCol(int col)
    {
        Vector v = new Vector(M);
        for (int i = 0; i < M; i++)
        {
            v[i] = mat[i, col];
        }
        return v;
    }
    private void SetCol(int col, Vector v)
    {
        if (v.Length != M)
            throw new System.Exception("Column is the wrong size");

        for (int i = 0; i < M; i++)
        {
            mat[i, col] = v[i];
        }
    }
    public void SetColumn(double[] col, int i)
    {
        if (col.Length != M)
            throw new System.Exception("Column does not match matrix dimension");
        for (int j = 0; j < M; j++)
        {
            mat[j, i] = col[j];
        }
    }
    public static Matrix Eye(int n)
    {
        Matrix m = new Matrix(n, n);
        for (int i = 0; i < n; i++)
            m[i, i] = 1;
        return m;
    }

    public bool IsDiag(double tol)
    {
        for (int i = 0; i < M; i++)
        {
            for (int j = 0; j < N; j++)
            {
                if (i != j && Math.Abs(mat[i, j]) > tol)
                {
                    return false;
                }
            }
        }
        return true;
    }
    public void UpperTriangle(out double[,] outM)
    {
        int dim = (N > M) ? M : N;
        outM = new double[dim, dim];

        for (int row = 0; row < dim; row++)
        {
            int count = 0;
            for (int col = N - dim; col < N; col++)
            {
                outM[row, count] = mat[row, col];
            }
        }
    }

    public static Matrix Skew(Vector v)
    {
        Matrix M = new Matrix(v.Length, v.Length);
        M[0, 1] = -v[2];
        M[1, 0] = v[2];
        M[0, 2] = v[1];
        M[2, 0] = -v[1];
        M[1, 2] = -v[0];
        M[2, 1] = v[0];
        return M;
    }

    public Matrix Transpose()
    {
        Matrix m = new Matrix(N, M);
        for (int i = 0; i < M; i++)
        {
            for (int j = 0; j < N; j++)
            {
                m[j, i] = mat[i, j];
            }
        }
        return m;
    }

    public void RemoveNumericalErr(double threshold)
    {
        for (int i = 0; i < M; i++)
        {
            for (int j = 0; j < N; j++)
            {
                if (System.Math.Abs(mat[i, j]) < threshold)
                    mat[i, j] = 0;
            }
        }
    }

    public bool AnyIsNan()
    {
        for (int i = 0; i < M; i++)
            for (int j = 0; j < N; j++)
                if (double.IsNaN(mat[i, j]))
                    return true;
        return false;
    }

    public static Matrix Concat(Matrix A, Matrix B, Matrix C, int dim)
    {
        if (dim == 0)
        {
            // Concat along row
            if (A.M != B.M || A.M != C.M)
                throw new System.Exception("Matrix dimension mismatch");

            int nCols = A.N + B.N + C.N;
            Matrix M = new Matrix(A.M, nCols);

            for (int row = 0; row < A.M; row++)
            {
                for (int col = 0; col < nCols; col++)
                {
                    if (col < A.N)
                    {
                        M[row, col] = A[row, col];
                    }
                    else if (col < A.N + B.N)
                    {
                        M[row, col] = B[row, col - A.N];
                    }
                    else
                    {
                        M[row, col] = C[row, col - A.N - B.N];
                    }
                }
            }

            return M;
        }
        else
        {
            // Concat along column
            if (A.N != B.N || A.N != C.N)
                throw new System.Exception("Matrix dimension mismatch");

            int nRows = A.M + B.M + C.M;
            Matrix M = new Matrix(nRows, A.N);

            for (int row = 0; row < nRows; row++)
            {
                for (int col = 0; col < A.N; col++)
                {
                    if (row < A.M)
                    {
                        M[row, col] = A[row, col];
                    }
                    else if (row < A.M + B.M)
                    {
                        M[row, col] = B[row - A.M, col];
                    }
                    else
                    {
                        M[row, col] = C[row - A.M - B.M, col];
                    }
                }
            }

            return M;
        }
    }

    public static Matrix Concat(Matrix A, Matrix B, int dim)
    {
        if (dim == 0)
        {
            // Concat along row
            if (A.M != B.M)
                throw new System.Exception("Matrix dimension mismatch");

            int nCols = A.N + B.N;
            Matrix M = new Matrix(A.M, nCols);

            for (int row = 0; row < A.M; row++)
            {
                for (int col = 0; col < nCols; col++)
                {
                    if (col < A.N)
                    {
                        M[row, col] = A[row, col];
                    }
                    else
                    {
                        M[row, col] = B[row, col - A.N];
                    }
                }
            }

            return M;
        }
        else
        {
            // Concat along column
            if (A.N != B.N)
                throw new System.Exception("Matrix dimension mismatch");

            int nRows = A.M + B.M;
            Matrix M = new Matrix(nRows, A.N);

            for (int row = 0; row < nRows; row++)
            {
                for (int col = 0; col < A.N; col++)
                {
                    if (row < A.M)
                    {
                        M[row, col] = A[row, col];
                    }
                    else
                    {
                        M[row, col] = B[row - A.M, col];
                    }
                }
            }

            return M;
        }
    }

    public static Matrix VecTimesTranspose(Vector3 v, Vector3 w)
    {
        Matrix m = new Matrix(3, 3);
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                m[i, j] = v[i] * w[j];
            }
        }
        return m;
    }

    public static Matrix VecTimesTranspose(Vector v, Vector w)
    {
        // v*w'
        Matrix m = new Matrix(v.Length, w.Length);
        for (int i = 0; i < v.Length; i++)
        {
            for (int j = 0; j < w.Length; j++)
            {
                m[i, j] = v[i] * w[j];
            }
        }
        return m;
    }

    public static Matrix operator *(Matrix A, Matrix B)
    {
        if (A.N != B.M)
            throw new System.Exception("Invalid dimensions for matrix multiplication: [" + A.M.ToString() + "x" + A.N.ToString() + "] * [" + B.M.ToString() + "x" + B.N.ToString() + "]");


        Matrix C = new Matrix(A.M, B.N);
        for (int i = 0; i < B.N; i++) // cols of B
        {
            for (int j = 0; j < A.M; j++) // rows of A
            {
                for (int k = 0; k < A.N; k++)
                {
                    C[j, i] += A[j, k] * B[k, i];
                }
            }
        }
        return C;
    }

    public static Matrix BackSub(Matrix A, Matrix B, bool AUpper)
    {
        // Solve AX=B -> X = B/A = inv(A)*B   [Ma x Na] * [Na x Nb] = [Mb x Nb] -> requires Ma = Mb and A square
        // Uses backsubstitution -> assumes A is already upper or lower triangular

        if (A.M != B.M || A.M != A.N)
            throw new System.Exception("Invalid dimensions for matrix division: [" + B.M.ToString() + "x" + B.N.ToString() + "] / [" + A.M.ToString() + "x" + A.N.ToString() + "]");

        // Solve inverse for each column
        Matrix X = new Matrix(A.N, B.N);
        bool isSingular = false;
        for (int i = 0; i < B.N; i++)
        {
            Vector x = new Vector(A.M);
            for (int j = 0; j < A.M; j++)
            {
                if (AUpper)
                {
                    double sum = 0;
                    for (int k = 0; k < j; k++)
                    {
                        sum += A[A.M - j - 1, A.M - k - 1] * x[A.M - k - 1];
                    }
                    if (A[A.M - j - 1, A.M - j - 1] == 0)
                    {
                        Console.WriteLine("Caution, matrix being inverted is singular. Switching to LSQ");
                        isSingular = true;
                        break;
                    }
                    x[A.M - j - 1] = (B[A.M - j - 1, i] - sum) / A[A.M - j - 1, A.M - j - 1];
                }
                else
                {
                    double sum = 0;
                    for (int k = 0; k < j; k++)
                    {
                        sum += A[j, k] * x[k];
                    }
                    if (A[j, j] == 0)
                    {
                        Console.WriteLine("Caution, matrix being inverted is singular. Switching to LSQ");
                        isSingular = true;
                        break;
                    }
                    x[j] = (B[j, i] - sum) / A[j, j];
                }
            }
            if (isSingular) break;
            // Assign column of X
            X[i] = x;
        }

        if (isSingular) return SolveLSQ(A, B);

        return X;
    }

    public static Matrix SolveLSQ(Matrix A, Matrix B)
    {
        Matrix At = A.Transpose();
        Matrix ATA = At * A;
        double[,] ata = ATA.mat;
        alglib.rmatrixinverse(ref ata, out int info1, out _);
        Matrix ATAinv = new Matrix(ata);

        return ATAinv * At * B;
    }

    public static Vector operator *(Matrix A, Vector v)
    {
        if (A.N != v.Length)
            throw new System.Exception("Invalid dimensions for matrix/vector multiplication");

        Vector c = new Vector(A.M);
        for (int i = 0; i < A.M; i++) // rows of A
        {
            for (int j = 0; j < v.Length; j++) // length of v
            {
                c[i] += A[i, j] * v[j];
            }
        }
        return c;
    }

    public static Vector operator *(Matrix A, Vector3 v)
    {
        if (A.N != 3)
            throw new System.Exception("Invalid dimensions for matrix/vector multiplication");

        Vector c = new Vector(A.M);
        for (int i = 0; i < A.M; i++) // rows of A
        {
            for (int j = 0; j < 3; j++) // length of v
            {
                c[i] += A[i, j] * v[j];
            }
        }
        return c;
    }

    public static Matrix operator *(double k, Matrix A)
    {
        Matrix C = new Matrix(A.M, A.N);
        for (int i = 0; i < A.M; i++) // rows of A
        {
            for (int j = 0; j < A.N; j++) // cols of A
            {
                C[i, j] = A[i, j] * k;
            }
        }
        return C;
    }

    public static Matrix operator +(Matrix A, Matrix B)
    {
        if (A.N != B.N || A.M != B.M)
            throw new System.Exception("Invalid dimensions for matrix addition");

        Matrix C = new Matrix(A.M, A.N);
        for (int i = 0; i < A.M; i++) // rows of A
        {
            for (int j = 0; j < A.N; j++) // cols of A
            {
                C[i, j] = A[i, j] + B[i, j];
            }
        }
        return C;
    }

    public override string ToString()
    {
        string s = "[";
        for (int i = 0; i < M; i++)
        {
            for (int j = 0; j < N; j++)
            {
                s += mat[i, j].ToString("e");
                if (j != N - 1)
                    s += ", ";
            }
            if (i != M - 1)
                s += ";\n";
        }
        s += "]";
        return s;
    }

    public string ToString(int dec)
    {
        string s = "[";
        for (int i = 0; i < M; i++)
        {
            for (int j = 0; j < N; j++)
            {
                s += mat[i, j].ToString("n" + dec.ToString());
                if (j != N - 1)
                    s += ", ";
            }
            if (i != M - 1)
                s += ";\n";
        }
        s += "]";
        return s;
    }
}
