using System;
using UnityEngine;

using Complex = alglib.complex;

public class CMatrix
{
    public Complex[,] mat { get; }
    public int M { get; }
    public int N { get; }

    public CMatrix(int m, int n)
    {
        M = m;
        N = n;
        mat = new Complex[m, n];
    }

    public CMatrix(Complex[,] m)
    {
        M = m.GetLength(0);
        N = m.GetLength(1);
        mat = m;
    }

    public CMatrix(double[,] m)
    {
        M = m.GetLength(0);
        N = m.GetLength(1);
        mat = new Complex[M, N];
        for (int i = 0; i < M; i++)
        {
            for (int j = 0; j < N; j++)
            {
                mat[i, j] = m[i, j];
            }
        }
    }

    public Complex[,] Mat()
    {
        Complex[,] outd = new Complex[M, N];
        for (int i = 0; i < M; i++)
        {
            for (int j = 0; j < N; j++)
            {
                outd[i, j] = mat[i, j];
            }
        }
        return outd;
    }

    public Complex this[int row, int col]
    {
        get => mat[row, col];
        set => mat[row, col] = value;
    }

    public CVector this[int col]
    {
        get => GetCol(col);
        set => SetCol(col, value);
    }
    private CVector GetCol(int col)
    {
        CVector v = new CVector(M);
        for (int i = 0; i < M; i++)
        {
            v[i] = mat[i, col];
        }
        return v;
    }
    private void SetCol(int col, CVector v)
    {
        if (v.Length != M)
            throw new System.Exception("Column is the wrong size");

        for (int i = 0; i < M; i++)
        {
            mat[i, col] = v[i];
        }
    }
    public void SetColumn(Complex[] col, int i)
    {
        if (col.Length != M)
            throw new System.Exception("Column does not match CMatrix dimension");
        for (int j = 0; j < M; j++)
        {
            mat[j, i] = col[j];
        }
    }
    public static CMatrix Eye(int n)
    {
        CMatrix m = new CMatrix(n, n);
        for (int i = 0; i < n; i++)
            m[i, i] = 1;
        return m;
    }
    public Matrix Real()
    {
        Matrix m = new Matrix(M, N);
        for (int i = 0; i < M; i++)
        {
            for (int j = 0; j < N; j++)
            {
                m[i, j] = mat[i, j].x;
            }
        }
        return m;
    }
    public Matrix Magnitude()
    {
        Matrix m = new Matrix(M, N);
        for (int i = 0; i < M; i++)
        {
            for (int j = 0; j < N; j++)
            {
                m[i, j] = mat[i, j].Magnitude();
            }
        }
        return m;
    }
    public bool AnyIsNan()
    {
        for (int i = 0; i < M; i++)
            for (int j = 0; j < N; j++)
                if (mat[i, j].IsNan())
                    return true;
        return false;
    }
    public bool IsDiag(double tol)
    {
        for (int i = 0; i < M; i++)
        {
            for (int j = 0; j < N; j++)
            {
                if (i != j && mat[i, j].Magnitude() > tol)
                {
                    return false;
                }
            }
        }
        return true;
    }
    public void UpperTriangle(out Complex[,] outM)
    {
        int dim = (N > M) ? M : N;
        outM = new Complex[dim, dim];

        for (int row = 0; row < dim; row++)
        {
            int count = 0;
            for (int col = N - dim; col < N; col++)
            {
                outM[row, count] = mat[row, col];
            }
        }
    }

    public static CMatrix Skew(CVector v)
    {
        CMatrix M = new CMatrix(v.Length, v.Length);
        M[0, 1] = -v[2];
        M[1, 0] = v[2];
        M[0, 2] = v[1];
        M[2, 0] = -v[1];
        M[1, 2] = -v[0];
        M[2, 1] = v[0];
        return M;
    }

    public CMatrix Transpose()
    {
        CMatrix m = new CMatrix(N, M);
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
                if (mat[i, j].Magnitude() < threshold)
                    mat[i, j] = 0;
            }
        }
    }

    public static CMatrix Concat(CMatrix A, CMatrix B, CMatrix C, int dim)
    {
        if (dim == 0)
        {
            // Concat along row
            if (A.M != B.M || A.M != C.M)
                throw new System.Exception("CMatrix dimension mismatch");

            int nCols = A.N + B.N + C.N;
            CMatrix M = new CMatrix(A.M, nCols);

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
                throw new System.Exception("CMatrix dimension mismatch");

            int nRows = A.M + B.M + C.M;
            CMatrix M = new CMatrix(nRows, A.N);

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

    public static CMatrix Concat(CMatrix A, CMatrix B, int dim)
    {
        if (dim == 0)
        {
            // Concat along row
            if (A.M != B.M)
                throw new System.Exception("CMatrix dimension mismatch");

            int nCols = A.N + B.N;
            CMatrix M = new CMatrix(A.M, nCols);

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
                throw new System.Exception("CMatrix dimension mismatch");

            int nRows = A.M + B.M;
            CMatrix M = new CMatrix(nRows, A.N);

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
    public static CMatrix Concat(Matrix A, CMatrix B, int dim)
    {
        if (dim == 0)
        {
            // Concat along row
            if (A.M != B.M)
                throw new System.Exception("CMatrix dimension mismatch");

            int nCols = A.N + B.N;
            CMatrix M = new CMatrix(A.M, nCols);

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
                throw new System.Exception("CMatrix dimension mismatch");

            int nRows = A.M + B.M;
            CMatrix M = new CMatrix(nRows, A.N);

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

    public static CMatrix Concat(CMatrix A, Matrix B, int dim)
    {
        if (dim == 0)
        {
            // Concat along row
            if (A.M != B.M)
                throw new System.Exception("CMatrix dimension mismatch");

            int nCols = A.N + B.N;
            CMatrix M = new CMatrix(A.M, nCols);

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
                throw new System.Exception("CMatrix dimension mismatch");

            int nRows = A.M + B.M;
            CMatrix M = new CMatrix(nRows, A.N);

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

    public static CMatrix VecTimesTranspose(Vector3 v, Vector3 w)
    {
        CMatrix m = new CMatrix(3, 3);
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                m[i, j] = v[i] * w[j];
            }
        }
        return m;
    }

    public static CMatrix VecTimesTranspose(CVector v, CVector w)
    {
        // v*w'
        CMatrix m = new CMatrix(v.Length, w.Length);
        for (int i = 0; i < v.Length; i++)
        {
            for (int j = 0; j < w.Length; j++)
            {
                m[i, j] = v[i] * w[j];
            }
        }
        return m;
    }

    public static CMatrix operator *(CMatrix A, CMatrix B)
    {
        if (A.N != B.M)
            throw new System.Exception("Invalid dimensions for CMatrix multiplication: [" + A.M.ToString() + "x" + A.N.ToString() + "] * [" + B.M.ToString() + "x" + B.N.ToString() + "]");


        CMatrix C = new CMatrix(A.M, B.N);
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
    public static CMatrix operator *(Matrix A, CMatrix B)
    {
        if (A.N != B.M)
            throw new System.Exception("Invalid dimensions for CMatrix multiplication: [" + A.M.ToString() + "x" + A.N.ToString() + "] * [" + B.M.ToString() + "x" + B.N.ToString() + "]");


        CMatrix C = new CMatrix(A.M, B.N);
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
    public static CMatrix operator *(CMatrix A, Matrix B)
    {
        if (A.N != B.M)
            throw new System.Exception("Invalid dimensions for CMatrix multiplication: [" + A.M.ToString() + "x" + A.N.ToString() + "] * [" + B.M.ToString() + "x" + B.N.ToString() + "]");


        CMatrix C = new CMatrix(A.M, B.N);
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

    public static CMatrix BackSub(CMatrix A, CMatrix B, bool AUpper)
    {
        // Solve AX=B -> X = B/A = inv(A)*B   [Ma x Na] * [Na x Nb] = [Mb x Nb] -> requires Ma = Mb and A square
        // Uses backsubstitution -> assumes A is already upper or lower triangular

        if (A.M != B.M || A.M != A.N)
            throw new System.Exception("Invalid dimensions for CMatrix division: [" + B.M.ToString() + "x" + B.N.ToString() + "] / [" + A.M.ToString() + "x" + A.N.ToString() + "]");

        // Solve inverse for each column
        CMatrix X = new CMatrix(A.N, B.N);
        bool isSingular = false;
        for (int i = 0; i < B.N; i++)
        {
            CVector x = new CVector(A.M);
            for (int j = 0; j < A.M; j++)
            {
                if (AUpper)
                {
                    Complex sum = 0;
                    for (int k = 0; k < j; k++)
                    {
                        sum += A[A.M - j - 1, A.M - k - 1] * x[A.M - k - 1];
                    }
                    if (A[A.M - j - 1, A.M - j - 1] == 0)
                    {
                        Console.WriteLine("Caution, CMatrix being inverted is singular. Switching to LSQ");
                        isSingular = true;
                        break;
                    }
                    x[A.M - j - 1] = (B[A.M - j - 1, i] - sum) / A[A.M - j - 1, A.M - j - 1];
                }
                else
                {
                    Complex sum = 0;
                    for (int k = 0; k < j; k++)
                    {
                        sum += A[j, k] * x[k];
                    }
                    if (A[j, j] == 0)
                    {
                        Console.WriteLine("Caution, CMatrix being inverted is singular. Switching to LSQ");
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

    public static CMatrix BackSub(CMatrix A, Matrix B, bool AUpper)
    {
        // Solve AX=B -> X = B/A = inv(A)*B   [Ma x Na] * [Na x Nb] = [Mb x Nb] -> requires Ma = Mb and A square
        // Uses backsubstitution -> assumes A is already upper or lower triangular

        if (A.M != B.M || A.M != A.N)
            throw new System.Exception("Invalid dimensions for CMatrix division: [" + B.M.ToString() + "x" + B.N.ToString() + "] / [" + A.M.ToString() + "x" + A.N.ToString() + "]");

        // Solve inverse for each column
        CMatrix X = new CMatrix(A.N, B.N);
        bool isSingular = false;
        for (int i = 0; i < B.N; i++)
        {
            CVector x = new CVector(A.M);
            for (int j = 0; j < A.M; j++)
            {
                if (AUpper)
                {
                    Complex sum = 0;
                    for (int k = 0; k < j; k++)
                    {
                        sum += A[A.M - j - 1, A.M - k - 1] * x[A.M - k - 1];
                    }
                    if (A[A.M - j - 1, A.M - j - 1] == 0)
                    {
                        Console.WriteLine("Caution, CMatrix being inverted is singular. Switching to LSQ");
                        isSingular = true;
                        break;
                    }
                    x[A.M - j - 1] = (B[A.M - j - 1, i] - sum) / A[A.M - j - 1, A.M - j - 1];
                }
                else
                {
                    Complex sum = 0;
                    for (int k = 0; k < j; k++)
                    {
                        sum += A[j, k] * x[k];
                    }
                    if (A[j, j] == 0)
                    {
                        Console.WriteLine("Caution, CMatrix being inverted is singular. Switching to LSQ");
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

    public static CMatrix SolveLSQ(CMatrix A, CMatrix B)
    {
        CMatrix At = A.Transpose();
        CMatrix ATA = At * A;
        Complex[,] ata = ATA.mat;
        alglib.cmatrixinverse(ref ata, out int info1, out _);
        CMatrix ATAinv = new CMatrix(ata);

        return ATAinv * At * B;
    }

    public static CMatrix SolveLSQ(CMatrix A, Matrix B)
    {
        CMatrix At = A.Transpose();
        CMatrix ATA = At * A;
        Complex[,] ata = ATA.mat;
        alglib.cmatrixinverse(ref ata, out int info1, out _);
        CMatrix ATAinv = new CMatrix(ata);

        return ATAinv * At * B;
    }

    public static CVector operator *(CMatrix A, CVector v)
    {
        if (A.N != v.Length)
            throw new System.Exception("Invalid dimensions for CMatrix/CVector multiplication");

        CVector c = new CVector(A.M);
        for (int i = 0; i < A.M; i++) // rows of A
        {
            for (int j = 0; j < v.Length; j++) // length of v
            {
                c[i] += A[i, j] * v[j];
            }
        }
        return c;
    }

    public static CVector operator *(CMatrix A, Vector3 v)
    {
        if (A.N != 3)
            throw new System.Exception("Invalid dimensions for CMatrix/CVector multiplication");

        CVector c = new CVector(A.M);
        for (int i = 0; i < A.M; i++) // rows of A
        {
            for (int j = 0; j < 3; j++) // length of v
            {
                c[i] += A[i, j] * v[j];
            }
        }
        return c;
    }

    public static CMatrix operator *(Complex k, CMatrix A)
    {
        CMatrix C = new CMatrix(A.M, A.N);
        for (int i = 0; i < A.M; i++) // rows of A
        {
            for (int j = 0; j < A.N; j++) // cols of A
            {
                C[i, j] = A[i, j] * k;
            }
        }
        return C;
    }

    public static CMatrix operator +(CMatrix A, CMatrix B)
    {
        if (A.N != B.N || A.M != B.M)
            throw new System.Exception("Invalid dimensions for CMatrix addition");

        CMatrix C = new CMatrix(A.M, A.N);
        for (int i = 0; i < A.M; i++) // rows of A
        {
            for (int j = 0; j < A.N; j++) // cols of A
            {
                C[i, j] = A[i, j] + B[i, j];
            }
        }
        return C;
    }

    public static CMatrix operator *(double k, CMatrix A)
    {
        CMatrix C = new CMatrix(A.M, A.N);
        for (int i = 0; i < A.M; i++) // rows of A
        {
            for (int j = 0; j < A.N; j++) // cols of A
            {
                C[i, j] = A[i, j] * k;
            }
        }
        return C;
    }

    public static CMatrix operator +(CMatrix A, Matrix B)
    {
        if (A.N != B.N || A.M != B.M)
            throw new System.Exception("Invalid dimensions for CMatrix addition");

        CMatrix C = new CMatrix(A.M, A.N);
        for (int i = 0; i < A.M; i++) // rows of A
        {
            for (int j = 0; j < A.N; j++) // cols of A
            {
                C[i, j] = A[i, j] + B[i, j];
            }
        }
        return C;
    }

    public static CMatrix operator +(Matrix A, CMatrix B)
    {
        if (A.N != B.N || A.M != B.M)
            throw new System.Exception("Invalid dimensions for CMatrix addition");

        CMatrix C = new CMatrix(A.M, A.N);
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
                s += mat[i, j].x.ToString("e") + "+ " + mat[i, j].y.ToString("e") + "i";
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
                s += mat[i, j].x.ToString("n" + dec.ToString()) + "+ " + mat[i, j].y.ToString("n" + dec.ToString()) + "i";
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
