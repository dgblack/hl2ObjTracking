using System;
using System.Collections.Generic;
using UnityEngine;


public class UKFPose : UKF
{
    // More Kalman filter parameters
    readonly double sigmaV = Math.Sqrt(0.5); // Acceleration
    private Pose mesPose;

    void Start()
    {
        // Initialize UKF parameters
        n = 13;
        nMes = 10;
        kappa = 3 - n;
        lambda = alpha * alpha * (n + kappa) - n;

        // Set measurement noise
        Qy = new Matrix(nMes, nMes);
        for (int i = 0; i < 3; i++)
            Qy[i, i] = sigmaR;
        for (int i = 3; i < 7; i++)
            Qy[i, i] = sigmaQ;
        for (int i = 7; i < 10; i++)
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
        // Fill the state vector with initial values
        for (int i = 0; i < 3; i++)
        {
            Xk[i] = init.position[i];
            Xk[i + 3] = init.rotation[i];
        }
        Xk[6] = init.rotation[3];
        for (int i = 7; i < n; i++) Xk[i] = 0;

        ProcessCovariance(out Matrix P);
        Sk = Sqrtm(P);
    }

    // Update is called once per frame
    void FixedUpdate()
    {
        if (!runFilter) return;

        dt = Time.fixedDeltaTime; // 0.0025;
        timeSinceLastMes += dt;

        // Prior estimate state update
        StateUpdate();

        if (hasIMU || hasPoints)
        {
            // Create measurement vector, initialize with prior values
            Vector yMes = new Vector(nMes, double.NaN);

            // Update with measured values, if available
            if (hasIMU)
            {
                yMes.Assign(imuSample, nMes - 3);
                hasIMU = false;
            }
            if (hasPoints)
            {
                timeSinceLastMes = 0;
                hasPoints = false;
                // Put measured pose into measurement vector
                for (int i = 0; i < 3; i++)
                {
                    yMes[i] = mesPose.position[i];
                    yMes[i + 3] = mesPose.rotation[i];
                }
                yMes[6] = mesPose.rotation[3];
            }

            // Posterior state estimate from measurement
            MeasurementUpdate(yMes);
        }
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

        // Renormalize quaternion
        double mag = Xk.Slice(3, 7).Norm();
        for (int i = 3; i < 7; i++)
        {
            Xk[i] /= mag;
        }

        CholeskyUpdate(ref Sk, Kk * Sy, -1);
    }

    protected override void H(Matrix X, out Matrix Y)
    {
        Y = new Matrix(nMes, X.N);
        for (int i = 0; i < X.N; i++)
        {
            Vector v = X[i];
            // Combine position, orientation, angular velocity
            Vector y = Vector.Concat(v.Slice(0, 7), v.Slice(v.Length-3, v.Length));
            Y[i] = y;
        }
    }

    protected override void F(Vector[] X, out Matrix Xp1, double dt)
    {
        Xp1 = new Matrix(n, X.Length);
        int count = 0;
        foreach (Vector v in X)
        {
            Vector p = v.Slice(0, 3);
            Vector q = v.Slice(3, 7);
            Vector vel = v.Slice(7, 10);
            Vector w = v.Slice(10, 13);

            // Compute small rotation quaternion
            double wnorm = w.Norm();
            double realPart = Math.Cos(wnorm * dt / 2);
            double imPart = Math.Sin(wnorm * dt / 2);
            Vector dq = new Vector(new double[] { imPart * w[0] / wnorm, imPart * w[1] / wnorm, imPart * w[2] / wnorm, realPart });

            // Updated values (vel and w stay the same)
            Vector newP = p + dt * vel;
            Vector newQ = QuaternionMultiply(dq,q);
            newQ /= newQ.Norm(); // Normalize

            // Stick into the return matrix
            Xp1[count] = Vector.Concat(Vector.Concat(Vector.Concat(newP, newQ), vel), w);
            count++;
        }
    }

    public override void OnNewPointSample(List<Vector3> points, int[] idxs)
    {
        
        pointSample = new Vector(nPoints * 3, double.NaN);
        for (int i = 0; i < idxs.Length; i++)
        {
            if (idxs[i] == -1)
            {
                //Debug.Log("Point wasn't matched");
                continue;
            }
            for (int j = 0; j < 3; j++)
                pointSample[idxs[i] * 3 + j] = points[i][j];
        }

        mesPose = FindTransformFromPoints(pointSample);
        hasPoints = true;
    }

    public override Vector3[] CurrentPoints()
    {
        Quaternion q = new Quaternion();
        for (int i = 0; i < 4; i++) q[i] = (float)Xk[i + 3];
        Vector3 p = new Vector3((float)Xk[0], (float)Xk[1], (float)Xk[2]);

        Matrix4x4 M = Matrix4x4.TRS(p, q, Vector3.one);

        Vector3[] points = new Vector3[nPoints];
        for (int i = 0; i < nPoints; i++)
        {
            points[i] = M.MultiplyPoint3x4(geom.relativePositions[i]);
        }
        return points;
    }

    public override Pose CurrentPose()
    {
        Quaternion q = new Quaternion();
        for (int i = 0; i < 4; i++) q[i] = (float)Xk[i + 3];
        Vector3 p = new Vector3((float)Xk[0], (float)Xk[1], (float)Xk[2]);
        return new Pose(p, q);
    }
    protected override void ProcessCovariance(out Matrix P)
    {
        P = new Matrix(n, n);
        for (int i = 0; i < 3; i++) P[i, i] = sigmaR * sigmaR;
        for (int i = 3; i < 7; i++) P[i, i] = sigmaQ * sigmaQ;
        for (int i = 7; i < 10; i++) P[i, i] = dt * dt * sigmaV * sigmaV;
        for (int i = 10; i < 13; i++) P[i, i] = dt * dt * sigmaW * sigmaW;
    }

    private Vector QuaternionMultiply(Vector q1, Vector q2)
    {
        if (q1.Length != 4 || q2.Length != 4) throw new Exception("Quaternions have incorrect dimension");

        Vector q = new Vector(4);
        // w is stored last in Unity
        q[0] = q1[3] * q2[0] + q1[0] * q2[3] + q1[1] * q2[2] - q1[2] * q2[1];
        q[1] = q1[3] * q2[1] - q1[0] * q2[2] + q1[1] * q2[3] + q1[2] * q2[0];
        q[2] = q1[3] * q2[2] + q1[0] * q2[1] - q1[1] * q2[0] + q1[2] * q2[3];
        q[3] = q1[3] * q2[3] - q1[0] * q2[0] - q1[1] * q2[1] - q1[2] * q2[2];

        return q;
    }
}