using System;
using System.Collections.Generic;
using UnityEngine;
using Complex = alglib.complex;

public class UKFPoints : UKF
{
    // More Kalman filter parameters
    readonly double sigmaA = Math.Sqrt(0.5); // Acceleration
    readonly double sigmaAl = Math.Sqrt(0.5); // Angular acceleration

    void Start()
    {
        // Initialize UKF parameters
        n = 3 * nPoints + 6;
        nMes = nPoints * 3 + 3;
        kappa = 3 - n;
        lambda = alpha * alpha * (n + kappa) - n;

        // Set measurement noise
        Qy = new Matrix(nPoints * 3 + 3, nPoints * 3 + 3);
        for (int i = 0; i < nPoints * 3; i++)
            Qy[i, i] = sigmaR;
        for (int i = nPoints * 3; i < nPoints * 3 + 3; i++)
            Qy[i, i] = sigmaW;

        SetUpUKF();
    }
    public override void StartFilter(Pose init)
    {
        timeSinceLastMes = 0;
        dt = Time.deltaTime;
        runFilter = true;
        hasIMU = false;
        hasPoints = false;

        Xk = new Vector(n);
        for (int i = 0; i < nPoints; i++)
        {
            Vector3 ri = init.position + init.rotation * geom.relativePositions[i];
            for (int j = 0; j < 3; j++)
                Xk[i * 3 + j] = ri[j];
        }

        Rk = Kabsch(Xk.Slice(0, 3 * nPoints));

        ProcessCovariance(out Matrix P);
        Sk = Sqrtm(P);
    }

    // Update is called once per frame
    void FixedUpdate()
    {
        if (!runFilter) return;
        
        // Check for numerical issues 
        bool isnan = false;
        for (int i = 0; i < Xk.Length; i++)
        {
            if (double.IsNaN(Xk[i]) || (i < nPoints * 3 && Xk[i] > 10))
            { isnan = true; break; }
        }
        if (isnan)
        {
            Stop();
            return;
        }

        dt = Time.fixedDeltaTime; // 0.0025;
        timeSinceLastMes += dt;

        if (hasIMU || hasPoints)
        {
            // Create measurement vector, initialize with prior values
            Vector yMes = new Vector(nMes, double.NaN);

            // Update with measured values, if available
            Qy = new Matrix(nPoints * 3 + 3, nPoints * 3 + 3);
            
            if (hasIMU)
            {
                for (int i = nPoints * 3; i < nPoints * 3 + 3; i++)
                    Qy[i, i] = sigmaW;
                yMes.Assign(imuSample, nPoints * 3);
                hasIMU = false;
            } else
            {
                for (int i = nPoints * 3; i < nPoints * 3 + 3; i++)
                    Qy[i, i] = 99999;
            }
            if (hasPoints)
            {
                timeSinceLastMes = 0;
                for (int i = 0; i < nPoints * 3; i++)
                {
                    if (!double.IsNaN(pointSample[i]))
                    {
                        Qy[i, i] = sigmaR;
                        yMes[i] = pointSample[i];
                    }
                    else Qy[i, i] = 99999;
                }
                    
                hasPoints = false;
            } else
            {
                for (int i = 0; i < nPoints * 3; i++)
                    Qy[i, i] = 99999;
            }


            // Prior estimate state update
            StateUpdate();

            // Posterior state estimate from measurement
            MeasurementUpdate(yMes);
        }
    }

    public void SwitchGeometry()
    {
        for (int i = 0; i < alternative.Count; i++)
        {
            (alternative[i], expected[i]) = (expected[i], alternative[i]);
        }

        SetUpGeometry();
    }

    protected override void MeasurementUpdate(Vector yMes)
    {
        // For measurements that were missed, set those elements to equal yk
        // This way they have no effect on the update

        //Debug.Log("PERFORMING MEASUREMENT UPDATE: " + yMes.ToString());

        // Calculate covariance and cross - covariance
        Matrix Pxy = new Matrix(n, yk.Length);
        for (int j = 0; j < 2 * n; j++)
        {
            Pxy += Wc[j] * Matrix.VecTimesTranspose(Xmk[j] - xmk, Ymk[j] - yk);
        }

        // Calculate near - optimal Kalman gain
        Matrix Kk = CalcGain(Sy, Pxy);

        // Update the state.
        for (int i = 0; i < yk.Length; i++)
            if (double.IsNaN(yMes[i])) yMes[i] = yk[i];
        Xk = xmk + Kk * (yMes - yk);

        CholeskyUpdate(ref Sk, Kk * Sy, -1);
        CholeskyUpdate(ref Sk, Kk * Sy, -1);
        CholeskyUpdate(ref Sk, Kk * Sy, -1);
    }


    protected override void StateUpdate()
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
        Rk = Kabsch(xmk.Slice(0, nPoints * 3));

        // Predicted output
        H(Xmk, out Ymk);
        var ymk = Ymk * Wm;
        yk = ymk;
        Matrix ymy = new Matrix(Ymk.M, 2 * n);
        for (int i = 0; i < 2 * n; i++)
        {
            ymy[i] = sw2 * (Ymk[i + 1] - ymk);
        }
        Sy = QR(Matrix.Concat(ymy, Qy, 0));
        CholeskyUpdate(ref Sy, Ymk[0] - yk, Wc[0]);
    }
    protected override void H(Matrix X, out Matrix Y)
    {
        Y = new Matrix(nPoints * 3 + 3, X.N);
        for (int i = 0; i < X.N; i++)
        {
            Vector v = X[i];
            Vector y = Vector.Concat(v.Slice(0, nPoints * 3), v.Slice(nPoints * 3 + 3, v.Length));
            Y[i] = y;
        }
    }

    protected override void F(Vector[] X, out Matrix Xp1, double dt)
    {
        Xp1 = new Matrix(n, X.Length);
        int count = 0;
        foreach (Vector v in X)
        {
            Vector r = v.Slice(0, nPoints * 3);
            Vector cdot = v.Slice(v.Length - 6, v.Length - 3);
            Matrix wx = Matrix.Skew(v.Slice(v.Length - 3, v.Length));

            Matrix R = Kabsch(r);

            Vector add = new Vector(v.Length);
            for (int i = 0; i < nPoints; i++)
            {
                Vector3 pos = geom.relativePositions[i] - geom.tipOffset;
                add.Assign(dt * (cdot + wx * R * pos), 3 * i);
            }

            Vector y = v + add;
            Xp1[count] = y;
            count++;
        }
    }

    public override void OnNewPointSample(List<Vector3> points, int[] idxs)
    {
        pointSample = new Vector(nPoints * 3, double.NaN);
        for (int i = 0; i < idxs.Length; i++)
        {
            if (idxs[i] == -2)
            {
                Debug.Log("A -2 snuck thru to OnNewPointSample");
                continue;
            }
            if (idxs[i] != -1)
                for (int j = 0; j < 3; j++)
                    pointSample[idxs[i] * 3 + j] = points[i][j];
        }
        //Debug.Log(pointSample.ToString());
        hasPoints = true;
    }

    public override void OnNewIMUSample(Vector3 newImuSample)
    {
        Pose p = CurrentPose();
        Matrix4x4 M = Matrix4x4.TRS(p.position, p.rotation, Vector3.one);
        imuSample = new Vector(M.MultiplyVector(newImuSample));
        hasIMU = true;
    }

    public override Vector3[] CurrentPoints()
    {
        Vector3[] points = new Vector3[nPoints];
        for (int i = 0; i < nPoints; i++)
        {
            points[i] = new Vector3((float)Xk[i * 3], (float)Xk[i * 3 + 1], (float)Xk[i * 3 + 2]);
        }
        return points;
    }

    public override Pose CurrentPose()
    {
        Quaternion q = Quaternion.LookRotation(Rk[2].ToV3(), Rk[1].ToV3());
        return new Pose(ck, q);
    }
    protected override void ProcessCovariance(out Matrix P)
    {
        Matrix I = Matrix.Eye(3);

        Matrix Eaa = dt * dt * sigmaAl * sigmaAl * I;
        Matrix Ecc = dt * dt * sigmaA * sigmaA * I;
        Matrix Eca = new Matrix(3, 3);

        Vector w = Xk.Slice(n - 3, n);
        Matrix wx = Matrix.Skew(w);
        Matrix wx2 = wx * wx;

        Matrix Err = new Matrix(n - 6, n - 6);
        for (int i = 0; i < nPoints; i++)
        {
            for (int j = 0; j < nPoints; j++)
            {
                // Create block matrix
                Matrix lMat = Matrix.VecTimesTranspose(geom.relativePositions[i], geom.relativePositions[j]);
                Matrix M = sigmaA * sigmaA * I + sigmaAl * sigmaAl * G(i, j) + wx2 * Rk * lMat * Rk.Transpose() * wx2;

                // Assign it to the block
                for (int row = 0; row < 3; row++)
                    for (int col = 0; col < 3; col++)
                        Err[3 * i + row, 3 * j + col] = M[row, col] * 0.25 * System.Math.Pow(dt, 4);
            }
        }

        Matrix Erc = new Matrix(3, n - 6);
        Matrix ErcBlock = 0.5 * System.Math.Pow(dt, 3) * sigmaA * sigmaA * I;
        for (int j = 0; j < n - 6; j++)
        {
            for (int row = 0; row < 3; row++)
                Erc[row, j] = ErcBlock[row, j % 3];
        }

        Matrix Era = new Matrix(3, n - 6);
        Vector R1 = Rk[0]; Vector R2 = Rk[1]; Vector R3 = Rk[2];
        for (int i = 0; i < nPoints; i++)
        {
            Matrix M = new Matrix(3, 3);
            Vector3 l = geom.relativePositions[i];
            M[2, 1] = R1.Dot(l);
            M[0, 2] = R2.Dot(l);
            M[1, 0] = R3.Dot(l);
            M[1, 2] = -M[2, 1];
            M[2, 0] = -M[0, 2];
            M[0, 1] = -M[1, 0];

            for (int row = 0; row < 3; row++)
                for (int col = 0; col < 3; col++)
                    Era[row, 3 * i + col] = M[row, col] * sigmaAl * sigmaAl * 0.5 * System.Math.Pow(dt, 3);
        }

        Matrix m1 = Matrix.Concat(Err, Erc.Transpose(), Era.Transpose(), 0);
        Matrix m2 = Matrix.Concat(Erc, Ecc, Eca, 0);
        Matrix m3 = Matrix.Concat(Era, Eca, Eaa, 0);

        P = Matrix.Concat(m1, m2, m3, 1);
    }

    private Matrix G(int i, int j)
    {
        Matrix M = new Matrix(3, 3);
        M[0, 0] = g(2, i, j) - g(1, i, j);
        M[1, 1] = g(2, i, j) - g(0, i, j);
        M[2, 2] = g(1, i, j) + g(0, i, j);
        return M;
    }

    private double g(int a, int i, int j)
    {
        Vector col = Rk[a];
        return col.Dot(geom.relativePositions[i]) * col.Dot(geom.relativePositions[j]);
    }
}