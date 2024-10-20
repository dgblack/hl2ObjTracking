using System.Collections;
using System.Collections.Generic;
using System;
using UnityEngine;
using Complex = alglib.complex;

public class UKF : MonoBehaviour
{
    [NonSerialized] public double timeSinceLastMes = 0;
    public List<Transform> expected;
    public List<Transform> alternative;
    public RigidGeometry geom;
    public RenderManager renderManager;

    // Measurements
    protected bool hasIMU = false;
    protected bool hasPoints = false;
    protected Vector imuSample;
    protected Vector pointSample;

    // Kalman sigma point parameters
    protected int nPoints = 4; // Number of tracking points
    protected int n; // Size of state vector x
    protected int nMes; // Size of output vector y
    protected readonly double beta = 2;
    protected readonly double alpha = 1e-3;
    protected double kappa;
    protected double lambda;
    protected double dt;
    [NonSerialized] public double sigmaR = Math.Sqrt(0.00025); // CV image points & extrinsics transform OR CV position
    [NonSerialized] public double sigmaW = Math.Sqrt(0.1); // IMU gyroscope
    [NonSerialized] public double sigmaQ = Math.Sqrt(0.1); // CV orientation

    // State vector and sqrt covariances
    protected Vector Xk; // State vector
    protected Vector xmk; // State vector
    protected Vector yk; // Output vector

    protected Vector Wc; // Covariance unscented weights
    protected Vector Wm; // Mean unscented weights
    protected Matrix Xmk; // Model-propagated sigma points
    protected Matrix Ymk; // Output-propagated sigma points

    protected Matrix Sk; // Sqrt of process covariance
    protected Matrix Sy; // Sqrt of measurement covariance
    protected Matrix Qy; // Measurement noise covariance sqrt

    protected Matrix Rk; // Rotation matrix
    protected Vector3 ck; // Centroid of points

    protected bool runFilter = false;

    protected void SetUpGeometry()
    {
        // Create the expected geometry
        Vector3[] points = new Vector3[expected.Count];
        for (int i = 0; i < points.Length; i++)
        {
            points[i] = expected[i].localPosition;
        }
        geom = new RigidGeometry(points, renderManager.trackerTips[renderManager.toolIdx].transform.localPosition);
    }

    // Start is called before the first frame update
    protected void SetUpUKF()
    {
        SetUpGeometry();

        // Set unscented transform weights
        Wc = new Vector(2 * n + 1, 1 / (2 * (n + lambda)));
        Wm = new Vector(2 * n + 1, 1 / (2 * (n + lambda)));
        Wc[0] = lambda / (n + lambda) + 1 - alpha * alpha + beta;
        Wm[0] = lambda / (n + lambda);

        // Set up measurement arrays
        pointSample = new Vector(nPoints);
        imuSample = new Vector(3);

        Rk = Matrix.Identity(3);
    }

    public void UpdateSigma(bool RnotW)
    {
        if (RnotW)
        {
            for (int i = 0; i < nPoints * 3; i++)
                Qy[i, i] = sigmaR;
        }
        else
        {
            for (int i = nPoints * 3; i < nPoints * 3 + 3; i++)
                Qy[i, i] = sigmaW;
        }
    }


    protected virtual void StateUpdate()
    {
        // Determine Sigma points
        FindSigmaPoints(Xk, out Vector[] Chi); // Sk gets messed up which messes this up

        // Predicted state x and covariance P
        F(Chi, out Xmk, dt); // Propagate sigma points through model
        xmk = Xmk * Wm; // Calculate mean (Xk prior estimate)

        Matrix xmx = new Matrix(Xmk.M, 2 * n);
        double sw2 = Math.Sqrt(Wc[1]);
        for (int i = 0; i < 2 * n; i++)
        {
            xmx[i] = sw2 * (Xmk[i + 1] - xmk);
        }

        ProcessCovariance(out Matrix P);
        Matrix Qx = Sqrtm(P);

        Sk = QR(Matrix.Concat(xmx, Qx, 0));
        CholeskyUpdate(ref Sk, Xmk[0] - xmk, Wc[0]);

        // Predicted output
        H(Xmk, out Ymk);
        var ymk = Ymk * Wm;
        // Renormalize the quaternion
        double mag = ymk.Slice(3, 7).Norm();
        for (int i = 3; i < 7; i++)
            ymk[i] /= mag;
        yk = ymk;
        Matrix ymy = new Matrix(Ymk.M, 2 * n);
        for (int i = 0; i < 2 * n; i++)
        {
            ymy[i] = sw2 * (Ymk[i + 1] - ymk);
        }
        Sy = QR(Matrix.Concat(ymy, Qy, 0));
        CholeskyUpdate(ref Sy, Ymk[0] - yk, Wc[0]);
    }

    protected Matrix CalcGain(Matrix Sy, Matrix Pxy)
    {
        // Kk*Sy*Sy' = Pxy  -> Kk = Pxy*inv(Sy')*inv(Sy)
        // Sy*Sy'*Kk' = Pxy'
        // Sy*(Sy'*Kk') = Pxy'
        // Sy'Kk' = Pxy'/Sy
        // Kk' = Pxy'/Sy/Sy'
        var SyKk = Matrix.BackSub(Sy, Pxy.Transpose(), true);
        return Matrix.BackSub(Sy.Transpose(), SyKk, false).Transpose();
    }

    #region Virtual Methods
    protected virtual void MeasurementUpdate(Vector yMes)
    {
        Debug.Log("Not implemented");
    }
    public virtual void StartFilter(Pose init)
    {
        Debug.Log("Not implemented");
    }
    protected virtual void H(Matrix X, out Matrix Y)
    {
        Debug.Log("Not implemented");
        Y = new Matrix(nPoints * 3 + 3, X.N);
    }
    protected virtual void F(Vector[] X, out Matrix Xp1, double dt)
    {
        Debug.Log("Not implemented");
        Xp1 = new Matrix(n, X.Length);
    }
    public virtual void OnNewIMUSample(Vector3 newImuSample)
    {
        Debug.Log("Not implemented");
        imuSample = new Vector(newImuSample);
        hasIMU = true;
    }
    public virtual void OnNewPointSample(List<Vector3> points, int[] idxs)
    {
        Debug.Log("Not implemented");
    }
    public virtual Vector3[] CurrentPoints()
    {
        Debug.Log("Not implemented");
        return new Vector3[nPoints];
    }
    public virtual Pose CurrentPose()
    {
        Debug.Log("Not implemented");
        return new Pose();
    }
    protected virtual void ProcessCovariance(out Matrix P)
    {
        Debug.Log("Not implemented");
        P = new Matrix(n, n);
    }
    #endregion

    #region Helper Methods
    public bool Running()
    {
        return runFilter;
    }

    public void Stop()
    {
        runFilter = false;
    }

    protected void FindSigmaPoints(Vector X, out Vector[] Chi)
    {
        Chi = new Vector[2 * n + 1];
        Chi[0] = X;
        double fac = Math.Sqrt(n + lambda);

        for (int i = 0; i < n; i++)
        {
            Chi[i + 1] = X + fac * Sk[i];
            Chi[i + 1 + n] = X - fac * Sk[i];
        }
    }

    protected Matrix QR(Matrix A)
    {
        double[,] mat = A.Transpose().Mat();
        alglib.rmatrixqr(ref mat, A.N, A.M, out _);
        alglib.rmatrixqrunpackr(mat, A.N, A.M, out double[,] r); // swap n and m since transposed

        // Upper triangle is in the upper mxm submatrix of r
        Matrix Rtri = new Matrix(A.M, A.M);
        for (int i = 0; i < A.M; i++)
        {
            for (int j = 0; j < A.M; j++)
            {
                Rtri[i, j] = r[i, j];
            }
        }

        return Rtri;
    }

    protected Matrix Sqrtm(Matrix M)
    {
        if (M.M != M.N)
            throw new System.Exception("Matrix must be square");

        double[,] T = M.Mat(); // copy the data so M is unaffected by the Schur decomp 
        alglib.rmatrixschur(ref T, M.N, out double[,] S);

        CMatrix Q = new CMatrix(S);
        CMatrix U = new CMatrix(M.M, M.M);
        if (M.IsDiag(1e-18))
        {
            for (int i = 0; i < M.M; i++)
            {
                U[i, i] = SqrtC(M[i, i]);
            }
        }
        else
        {
            for (int os = 0; os < M.N; os++)
            {
                for (int i = 0; i < M.N - os; i++)
                {
                    int j = i + os;

                    if (os == 0)
                    {
                        // Uii = +/- sqrt(Tii)
                        U[i, j] = SqrtC(T[i, j]);
                    }
                    else
                    {
                        Complex sum = 0;
                        for (int k = i + 1; k < j; k++)
                        {
                            sum += U[i, k] * U[k, j];
                        }

                        // Avoid div by 0
                        if ((U[i, i] + U[j, j]).Magnitude() < 1e-12)
                        {
                            U[i, j] = 0; //Debug.Log("Set to zero");
                        }
                        else
                            U[i, j] = (T[i, j] - sum) / (U[i, i] + U[j, j]);
                    }
                }
            }
        }
        U = (Q * U) * Q.Transpose();
        return U.Real();
    }
    protected double Sqrt(double x)
    {
        return Math.Sqrt(Math.Abs(x));
    }

    protected Complex SqrtC(double x)
    {
        if (x < 0)
            return new Complex(0, Math.Sqrt(-x));
        else return new Complex(Math.Sqrt(x), 0);
    }

    protected void Cholesky(ref Matrix M)
    {
        M.UpperTriangle(out double[,] mat);
        if (!alglib.spdmatrixcholesky(ref mat, M.N, true))
            Console.WriteLine("Cholesky factorization failed");
        M = new Matrix(mat);
    }
    protected void CholeskyUpdate(ref Matrix M, Matrix X, double factor)
    {
        for (int i = 0; i < X.N; i++)
        {
            CholeskyUpdate(ref M, X[i], factor);
        }
    }

    protected void CholeskyUpdate(ref Matrix M, Vector x, double factor) // rank one cholesky update subroutine is NOT working. Had to replace with just doing the full decomp
    {
        // Algorithm from Wikipedia: https://en.wikipedia.org/wiki/Cholesky_decomposition#Rank-one_update
        // https://doc.lagout.org/science/0_Computer%20Science/2_Algorithms/Matrix%20Algorithms%20(Vol.%201_%20Basic%20Decompositions)%20%5BStewart%201998-12%5D.pdf page 347

        Matrix backup = new Matrix(M.Mat());
        bool fail = false;

        int sgn = (factor > 0) ? 1 : -1;
        x = Sqrt(factor) * x;

        int nVec = x.Length;
        for (int k = 0; k < nVec; k++)
        {
            if (M[k, k] == 0) // We were getting divide by 0 errors
            {
                continue;
            }

            double r = Sqrt(M[k, k] * M[k, k] + sgn * x[k] * x[k]);
            if (r == 0)
            {
                fail = true;
                break;
            }

            double c = r / M[k, k];
            double s = x[k] / M[k, k];
            M[k, k] = r;
            for (int j = k + 1; j < nVec; j++)
            {
                M[k, j] = (M[k, j] + sgn * s * x[j]) / c;
                x[j] = c * x[j] - s * M[k, j];
            }
        }

        // If something went wrong, don't use this one
        if (fail || M.AnyIsNan() || AnyLarge(M))
        {
            //Debug.Log("Cholesky update failed");
            M = backup;
        }
        //else Debug.Log("Cholesky update succeeded");
    }

    protected bool AnyLarge(Matrix M)
    {
        return AnyLarge(M, 10000);
    }
    protected bool AnyLarge(Matrix M, double threshold)
    {
        for (int i = 0; i < M.M; i++)
        {
            for (int j = 0; j < M.N; j++)
            {
                if (Math.Abs(M[i, j]) > threshold)
                    return true;
            }
        }
        return false;
    }
    protected Matrix Kabsch(Vector r)
    {
        // Convert list of points to matrix
        Matrix rMat = new Matrix(3, nPoints);

        // Calculate centroid
        Vector mean = new Vector(3);
        for (int i = 0; i < nPoints; i++)
        {
            mean += r.Slice(i * 3, i * 3 + 3);
        }
        mean /= nPoints;

        // Place vectors in a matrix
        for (int i = 0; i < nPoints; i++)
        {
            rMat[i] = r.Slice(i * 3, i * 3 + 3) - mean;
        }

        // Calculate rotation
        Matrix R = Kabsch(rMat);

        // Save centroid
        ck = mean.ToV3() - (R * geom.centroid).ToV3();

        return R;
    }

    protected Matrix Kabsch(Matrix r)
    {
        // Find rotation matrix (orthogonal, determinant 1) to map pointsFrom to pointsTo with minimum RMSE
        // Assumes the centroid of r is at [0,0,0]
        // Covariance H = A'*B
        // Where B = r' and A = geomMatrix
        // So H' = B'*A = r * geomMatirx
        Matrix H = (r * geom.mat).Transpose();

        // SVD of covariance matrix
        SVD(H, out Matrix U, out Matrix Vt);

        // Find determinant of VU' = (UV')'
        Matrix Ut = U.Transpose();
        Matrix V = Vt.Transpose();
        Matrix VUt = V * Ut;

        double d = alglib.rmatrixdet(VUt.mat);

        // Check determinant (should be 1, but it's numerical, so won't be precisely 1)
        if (Mathf.Abs((float)d - 1) < 0.01f)
            return VUt;
        else
        {
            Matrix sign = Matrix.Eye(3);
            sign[2, 2] = -1; // d

            return V * sign * Ut;
        }
    }
    protected bool SVD(Matrix mat, out Matrix U, out Matrix Vt)
    {
        // Calculate SVD
        double[,] Ud, Vtd;
        bool ret = alglib.rmatrixsvd(mat.mat, mat.M, mat.N, 1, 1, 2, out _, out Ud, out Vtd);
        U = new Matrix(Ud);
        Vt = new Matrix(Vtd);

        return ret;
    }

    protected Pose FindTransformFromPoints(Vector mesPts)
    {
        Matrix R = Kabsch(mesPts);
        Vector3 t = ck;

        // Create homogeneous transformation matrix
        Matrix4x4 pose = Matrix4x4.identity;
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                pose[i, j] = (float)R[i, j];
            }
        }
        for (int j = 0; j < 3; j++)
        {
            pose[j, 3] = t[j];
        }

        return new Pose(t,pose.rotation);
    }

    #endregion
}
