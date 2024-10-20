using Microsoft.MixedReality.Toolkit.UI;
using System;
using System.Collections;
using System.Collections.Generic;
using TMPro;
using UnityEngine;

public class PoseTracker : MonoBehaviour
{
    public UKFPoints ukf;
    public IRTracker ir;
    public Transform cam;
    public Signaler signaler;
    public bool extrapolate = true;
    private float distThresh = 0.8f; // 8 cm
    private float ukfHorizon = 1.0f; // 1 second 
    private float matchThresh = 0.008f; // was 0.005
    private float voteThresh = 1;
    private float deltaTime = 0.1f;
    public GameObject warningText;

    public ButtonConfigHelper smoothButton;
    public int NumRunning { get; private set; }
    
    [System.NonSerialized] public float smoothing = 0.9f;
    [System.NonSerialized] public Pose pose = new Pose();
    [System.NonSerialized] public Vector3 vel = Vector3.zero;
    [System.NonSerialized] public Quaternion angularVel = Quaternion.identity;
    private bool updated = false;
    [System.NonSerialized] public bool hasPose = false;
    [System.NonSerialized] public bool hadPose = false;
    [System.NonSerialized] public long framesSinceLastPose = 0;
    [System.NonSerialized] public int maxFramesSincePose = 100;

    private List<List<int>> permutations5;
    private List<List<int>> permutations4;
    private List<List<int>> permutations3;
    private List<List<int>> permutations2;
    int nBadJumps = 0;
    List<Vector3> lastPoints = new List<Vector3>();
    private Pose lastMeas = new Pose();
    private long lastMeasTime = 0;
    int nPoints;

    private Camera mainCamera;

    // 0 = no UKF, 1 = Points
    [System.NonSerialized] public int USE_UKF = 0;

    private void Start()
    {
        mainCamera = Camera.main;
        NumRunning = 0;
        permutations5 = RecursiveCreateCombo(new int[] { 0, 1, 2, 3, 4 });
        permutations4 = RecursiveCreateCombo(new int[] { 0, 1, 2, 3 });
        permutations3 = RecursiveCreateCombo(new int[] { 0, 1, 2 });
        permutations2 = new List<List<int>>(2);
        permutations2.Add(new List<int> { 0, 1 });
        permutations2.Add(new List<int> { 1, 0 });

        // Using ukf.geom.relativePoints[i] causes a null pointer exception, so just initialize to zero
        for (int i = 0; i < 4; i++)
            lastPoints.Add(Vector3.zero);

        nPoints = 4;//ukf.geom.nPoints;
        smoothButton.MainLabelText = "Smooth " + smoothing.ToString("N2");
    }

    private void Update()
    {
        try
        {
            if (deltaTime == 0.1)
                deltaTime = Time.deltaTime;
            if (updated)
            {
                updated = false;
            }
            else if (extrapolate)
            {
                if (framesSinceLastPose < 50)
                {
                    pose.position += vel;
                    pose.rotation *= angularVel;

                    // slightly decrease the velocity and angular velocity so it doesn't run away
                    vel *= 0.9f;
                    angularVel = Quaternion.Slerp(Quaternion.identity, angularVel, 0.9f);
                }
            }

            if (warningText.activeInHierarchy)
            {
                Vector3 directionToCamera = warningText.transform.position - mainCamera.transform.position;
                if (directionToCamera != Vector3.zero)
                {
                    Quaternion buttonRot = Quaternion.LookRotation(directionToCamera);
                    warningText.transform.rotation = buttonRot;
                }
            }
        }
        catch(Exception e)
        {
            Debug.LogError("Error in Update: " + e.ToString());
        }

    }

    public void IMUMeasurementUpdate(float temp, Vector3 gyro, Vector3 accel)
    {
        if (USE_UKF == 1)
            ukf.OnNewIMUSample(gyro);
    }

    // Run this function asynchronously
    public int[] UpdatePose(List<Vector3> mes)
    {
        NumRunning++;

        // ----------------------------------- Try to find point correspondences ---------------------------------------------
        // If the ith element of idxs equals j, then the ith measured point corresponds to the jth known point

        // Use Euclidean matching first
        int[] idxs = new int[mes.Count];
        bool success = TryEuclideanMatching(mes, ref idxs);

        // Check if we got enough inliers - if we have mostly outliers, we can't do anything else so just return
        // I've never seen this condition triggered
        if (success && BadMatch(idxs)) {
            //Debug.Log($"{Timer.GetTime()}: Failed with bad match");
            PoseCalcFailed();
            return new int[0];
        }

        // Check that the fit is reasonable - not fitting to some outliers
        if (success) {
            float err = Mathf.Abs(MeanMatchError(mes, idxs));
            success = (err < matchThresh);
            if (!success) {
                // The match was possibly wrong so make everything unknown
                for (int i = 0; i < idxs.Length; i++)
                    if (idxs[i] > 0)
                        idxs[i] = -2;

                //Debug.Log($"{Timer.GetTime()}: Failed with match error {err}");
            }
            //else Debug.Log($"{Timer.GetTime()}: SUCCESS: Euclidean Matching with error {err}");
        }

        //PrintList(idxs, Timer.GetTime().ToString());

        // Everything failed so far - try brute force
        if (!success) {
            //Debug.Log($"{Timer.GetTime()}: Failed Euclidean matching");

            // Remove outliers
            List<Vector3> mps = new List<Vector3>();
            List<int> idxsMinusOutliers = new List<int>();
            for (int i = 0; i < mes.Count; i++)
            {
                if (idxs[i] != -1)
                {
                    mps.Add(mes[i]);
                    idxsMinusOutliers.Add(idxs[i]);
                }
            }

            // If more than 5, this becomes too slow. If <= 2, this is garbage anyway
            if (mps.Count <= 5 && mps.Count > 2) {
                double err = BruteForceMatch(mps, ref idxs, idxsMinusOutliers, false);
                success = (err < matchThresh);
                //if (!success) Debug.Log($"{Timer.GetTime()}: Failed brute force with error {err}"); // Happens frequently with 0.005 thresh
                //else Debug.Log($"{Timer.GetTime()}: SUCCESS: brute force with error {err}");
            }
        }

        // Try filling in any missing markers based on the ones we know
        if (success) {
            double err = FillInMissing(ref mes, ref idxs);
            success = (err < matchThresh && err > 0);
            //if (!success) Debug.Log($"{Timer.GetTime()}: Failed with error {err} after filling in missing"); // Never happens with 0.005 thresh
            //else Debug.Log($"{Timer.GetTime()}: SUCCESS: Filled in missing with error {err}");
            //PrintList(idxs, "");
        }

        // Look for markers bouncing around badly
        if (success) {
            success = NoBadJumps(mes, idxs);
            //if (!success) Debug.Log($"{Timer.GetTime()}: Detected illegal jump");
        }

        // Calculate and return the resulting pose from the match
        if (success)
        {
            // We found a good match
            //Debug.Log($"{Timer.GetTime()}: Succeeded!!");
            hasPose = true;
            hadPose = true;
            framesSinceLastPose = 0;
            SetPoseFromResult(mes, idxs);
            warningText.SetActive(false);
            return idxs;
        }
        //Debug.Log($"{Timer.GetTime()}: Failed");
        PoseCalcFailed();
        return new int[0];
    }

    double FillInMissing(ref List<Vector3> mes, ref int[] idxs)
    {
        // Find how many indices were matched
        int nPos = 0;
        int nUA = 0;
        for (int i = 0; i < idxs.Length; i++)
        {
            if (idxs[i] >= 0) nPos++;
            else if (idxs[i] == -2) nUA++;
        }

        if (nPos == 4) // All points are accounted for. Just calculate the fit error
        {
            // Find the transform
            List<Vector3> mesPts = new List<Vector3>();
            List<Vector3> expPts = new List<Vector3>();
            for (int i = 0; i < idxs.Length; i++)
            {
                if (idxs[i] != -1)
                {
                    mesPts.Add(mes[i]);
                    expPts.Add(ukf.geom.relativePositions[idxs[i]]);
                }
            }
            Matrix4x4 M = FindTransformFromPoints(mesPts, expPts);
            
            // Compute the error
            double err = 0;
            for (int i = 0; i < mesPts.Count; i++)
            {
                err += (mesPts[i] - M.MultiplyPoint3x4(expPts[i])).magnitude;
            }
            return err/mesPts.Count;
        }
        else if (nPos == 3) // All but 1 accounted for. Should be easy to find the fourth
        {
            // Find the transform
            List<Vector3> mesPts = new List<Vector3>();
            List<Vector3> expPts = new List<Vector3>();
            for (int i = 0; i < idxs.Length; i++)
            {
                if (idxs[i] >= 0)
                {
                    mesPts.Add(mes[i]);
                    expPts.Add(ukf.geom.relativePositions[idxs[i]]);
                }
            }
            Matrix4x4 M = FindTransformFromPoints(mesPts, expPts);

            // Determine which point is missing
            int missing = -1;
            for (int i = 0; i < 4; i++)
            {
                if (!Contains(idxs, i)) { missing = i; break; }
            }

            if (missing != -1)
            {
                // If this point was actually missing, add it
                if (nUA == 0)
                {
                    // Add the missing point to the lists
                    Vector3 pt = M.MultiplyPoint3x4(ukf.geom.relativePositions[missing]);
                    mesPts.Add(pt);
                    mes.Add(pt); // Add to the referenced list, which affects the call to ukf.OnNewPointSample
                    expPts.Add(ukf.geom.relativePositions[missing]);

                    // Add the new index to the end of the idxs list as well to match the new point
                    int[] idxs2 = new int[idxs.Length + 1];
                    idxs.CopyTo(idxs2, 0);
                    idxs2[idxs.Length] = missing;
                    idxs = idxs2;
                }
                // Otherwise, if we had this point and it was just unmatched, change it
                else if (nUA == 1)
                {
                    // Find the unmatched point
                    int idx = -1;
                    for (int i = 0; i < idxs.Length; i++)
                        if (idxs[i] == -2) idx = i;

                    // Add it / change it in the various lists
                    if (idx != -1)
                    {
                        Vector3 pt = M.MultiplyPoint3x4(ukf.geom.relativePositions[missing]);
                        mesPts.Add(pt);
                        expPts.Add(ukf.geom.relativePositions[missing]);
                        mes[idx] = pt;
                        idxs[idx] = missing;
                    }
                }
            }

            // Compute the error
            double err = 0;
            for (int i = 0; i < mesPts.Count; i++)
            {
                err += (mesPts[i] - M.MultiplyPoint3x4(expPts[i])).magnitude;
            }
            return err / mesPts.Count * Mathf.Sign(missing);
        } 
        else if (nPos == 2 && nUA == 1)
        {
            // We know which marker is ok, we just don't know what geometry point that fits to
            // There are only two options, so try both and compare the error
            // Determine which geometry points are unassigned
            int[] missing = { -1, -1 };
            int count = 0;
            for (int i = 0; i < 4; i++)
            {
                if (!Contains(idxs, i)) { missing[count] = i; count++; }
            }

            // For each unassigned geometry point, assign it and check the fit error
            double minErr = double.MaxValue;
            int minIdx = -1;
            for (int ua = 0; ua < 2; ua++)
            {
                // Find the transform
                List<Vector3> mesPts = new List<Vector3>();
                List<Vector3> expPts = new List<Vector3>();
                for (int i = 0; i < idxs.Length; i++)
                {
                    if (idxs[i] >= 0)
                    {
                        mesPts.Add(mes[i]);
                        expPts.Add(ukf.geom.relativePositions[idxs[i]]);
                    } else if (idxs[i] == -2)
                    {
                        mesPts.Add(mes[i]);
                        expPts.Add(ukf.geom.relativePositions[missing[ua]]);
                    }
                }
                Matrix4x4 M = FindTransformFromPoints(mesPts, expPts);

                // Compute the error
                double err = 0;
                for (int i = 0; i < mesPts.Count; i++)
                {
                    err += (mesPts[i] - M.MultiplyPoint3x4(expPts[i])).magnitude;
                }
                err /= mesPts.Count;

                // Compare the error
                if (err < minErr)
                {
                    minErr = err;
                    minIdx = missing[ua];
                }
            }

            if (minErr < matchThresh)
            {
                // If the new error is acceptable, assign the new index 
                for (int i = 0; i < idxs.Length; i++)
                    if (idxs[i] == -2) idxs[i] = minIdx;

                // Try to fill in the final point
                return FillInMissing(ref mes, ref idxs);
            }
            else return -1;
        }
        else return -1;
    }

    #region Non-Brute Force Matching
    bool TryMatchLastPoints(List<Vector3> mesPts, Vector3[] lastPts, out int[] idxs)
    {
        idxs = new int[mesPts.Count];
        double[] dists = new double[mesPts.Count];
        int numInliers = 0;

        // ---------------------- Find closest previous points to each point ------------------------
        if (lastPts.Length > 0) // if previous points are available
        {
            for (int i = 0; i < mesPts.Count; i++)
            {
                float minDist = float.MaxValue;
                int minIdx = -1; // Outliers get a -1 value
                for (int j = 0; j < lastPts.Length; j++)
                {
                    float dist = (mesPts[i] - lastPts[j]).magnitude;
                    if (dist > distThresh) // If a single prev point is > distThresh, call it an outlier
                    {
                        minIdx = -1;
                        break;
                    }
                    if (dist < minDist)
                    {
                        minDist = dist;
                        minIdx = j;
                        numInliers++;
                    }
                }
                idxs[i] = minIdx;
                dists[i] = minDist;
            }

            // Check for redundant indices if we've matched enough of the points
            bool badMatch = true;
            if (numInliers < 3)
            {
                badMatch = false;
                for (int i = 0; i < idxs.Length; i++)
                {
                    for (int j = i + 1; j < idxs.Length; j++)
                    {
                        if (idxs[i] == idxs[j] && idxs[i] != -1)
                        {
                            badMatch = true;
                            break;
                        }
                    }
                    if (badMatch) break;
                }
            }

            if (!badMatch) return true;
        }

        return false;
    }

    bool TryEuclideanMatching(List<Vector3> mesPts, ref int[] idxs)
    {
        // ----------------------- Euclidean Outlier Removal -----------------------------------
        // Remove outliers from mesPts and look for possible matches
        try
        {
            idxs = EuclideanVotingOutliers(mesPts);
        } catch (IndexOutOfRangeException e)
        {
            Debug.Log("Caught index out of bounds exception in EuclideanVotingOutliers");
            return false;
        }

        bool hasRedundant = false;
        int numUnassigned = 0;
        int numAssigned = 0;
        for (int i = 0; i < idxs.Length; i++)
        {
            if (idxs[i] == -2) numUnassigned++;
            else if (idxs[i] != -1) numAssigned++;
            for (int j = i + 1; j < idxs.Length; j++)
            {
                if (idxs[i] != -1 && idxs[i] == idxs[j])
                {
                    hasRedundant = true;
                    idxs[i] = -2;
                    idxs[j] = -2;
                    break;
                }
            }
        }

        // Assign the single unknown point to the single remaining geometry point (if possible)
        if (!hasRedundant && numAssigned == nPoints - 1 && numUnassigned == 1)
        {
            //Debug.Log("Filling in single missing marker");
            for (int i = 0; i < nPoints; i++)
            {
                if (!Contains(idxs, i))
                {
                    // This geometry point is not included yet
                    // Find the unassigned measured point
                    for (int j = 0; j < idxs.Length; j++)
                    {
                        // And set it to this geometry point
                        if (idxs[j] == -2)
                        {
                            idxs[j] = i;
                            numAssigned++;
                            break;
                        }
                    }

                    break;
                }
            }
        }

        // If the point correspondence is already all good, return it
        if (!hasRedundant && numAssigned == nPoints) return true;
        return false;
    }

    int[] EuclideanVotingOutliers(List<Vector3> mesPts)
    {
        if (mesPts.Count == 0) return new int[0];

        int np = mesPts.Count;
        int mp = ukf.geom.nPoints;
        int[] idxs = new int[np];

        //Debug.Log("Creating " + np.ToString() + "x" + np.ToString() + " diffs matrix");
        // Create inter-point distance matrix
        // The same matrix for the known geometry is in ukf.geom.diffs
        float[,] D = new float[np, np];
        for (int i = 0; i < np; i++)
        {
            for (int j = i + 1; j < np; j++)
            {
                D[i, j] = (mesPts[i] - mesPts[j]).magnitude;
                D[j, i] = D[i, j];
            }
        }
        //PrintMat(D, "");

        // Voting matrix
        int[,] V = new int[mp, np];
        for (int i = 0; i < mp; i++)
        {
            for (int j = i + 1; j < mp; j++)
            {
                // Find closest inter-point distance
                int bestK = -1;
                int bestL = -1;
                float bestDist = float.MaxValue;
                for (int k = 0; k < np; k++)
                {
                    for (int l = k + 1; l < np; l++)
                    {
                        float dist = Mathf.Abs(D[k, l] - ukf.geom.diffs[i, j]);
                        if (dist < bestDist)
                        {
                            bestDist = dist;
                            bestK = k; bestL = l;
                        }
                    }
                }

                // Vote on the corresponding indices
                if (bestK != -1 && bestL != -1)
                    V[i, bestK]++; V[i, bestL]++; V[j, bestK]++; V[j, bestL]++;
            }
        }
        //PrintMat(V,Timer.GetTime().ToString());

        //Debug.Log("Counting votes");
        // Find outliers with few votes and likely point correspondences
        bool[] leq1 = new bool[np];
        for (int i = 0; i < np; i++)
        {
            // Find sum of votes for this measured point
            // As well as the most-matched known point, if there is an obvious max
            leq1[i] = true;
            int sm = 0;
            int mostMatches = 0;
            int mostMatched = -1;
            bool isRedundant = false;
            for (int j = 0; j < mp; j++)
            {
                sm += V[j, i];
                if (V[j, i] > mostMatches)
                {
                    mostMatches = V[j, i];
                    mostMatched = j;
                    isRedundant = false;
                }
                // Check if two points have the same number of matches
                else if (V[j, i] == mostMatches)
                    isRedundant = true;

                if (V[j, i] > 1) leq1[i] = false;
            }

            if ((float)sm / mp < voteThresh)
                idxs[i] = -1;
            else if (!isRedundant)
                idxs[i] = mostMatched;
            else idxs[i] = -2;
        }

        // Check if we're still including an outlier and an included column of votes has only 1s
        // and no other included column had only 1s. In this case we can flag that column as an outlier
        int nInliers = 0; int nLeq1 = 0; int leq1Idx = -1;
        for (int i = 0; i < np; i++)
        {
            if (idxs[i] != -1) nInliers++;
            if (leq1[i] && idxs[i] != -1) { nLeq1++; leq1Idx = i; }
        }
        if (leq1Idx != -1 && nInliers > mp && nLeq1 == 1)
            idxs[leq1Idx] = -1;

        //PrintList(idxs,"");
        return idxs;
    }

    bool TryFrobeniusMatching(List<Vector3> mesPts, Vector3[] lastPts, ref int[] idxs)
    {
        // -------------------------------- Frobenius Matching -----------------------------------
        // Remove outliers and move to Frobenius matching
        List<Vector3> mes = new List<Vector3>();
        for (int i = 0; i < mesPts.Count; i++)
        {
            if (idxs[i] != -1)
                mes.Add(mesPts[i]);
        }

        // Try Frobenius matching, and otherwise fall back on brute force
        if (mes.Count <= nPoints) // If there are still outliers, Frobenius will fail
        {
            int[] frobIdxs = FrobeniusMatch(mes, lastPts);

            // frobIdxs doesn't include outliers, so fill it back into the original idxs array
            int j = 0;
            for (int i = 0; i < idxs.Length; i++)
            {
                if (idxs[i] != -1)
                {
                    idxs[i] = frobIdxs[j];
                    j++;
                }
                else idxs[i] = -1;
            }

            return true;
        }

        return false;
    }

    int[] FrobeniusMatch(List<Vector3> mesPts, Vector3[] lastPts)
    {
        if (mesPts.Count > nPoints)
            throw new System.Exception("Measured points still include outliers");

        int numMissing = nPoints - mesPts.Count;
        double bestSum = 0;
        int[] indices = new int[mesPts.Count];
        if (numMissing == 1)
        {
            // Measured points is missing one point
            int[] skipIdxs = new int[1];
            for (int i = 0; i < nPoints; i++)
            {
                skipIdxs[0] = i;
                int[] idxs;
                double sm = FrobeniusOpt(mesPts, lastPts, skipIdxs, out idxs);

                if (sm > bestSum)
                {
                    idxs.CopyTo(indices, 0);
                    bestSum = sm;
                }
            }
        } else if (numMissing == 2)
        {
            // Measured points is missing two points
            int[] skipIdxs = new int[2];
            for (int i = 0; i < nPoints; i++)
            {
                for (int j = i+1; j < nPoints; j++)
                {
                    skipIdxs[0] = i; skipIdxs[1] = j;
                    int[] idxs;
                    double sm = FrobeniusOpt(mesPts, lastPts, skipIdxs, out idxs);

                    if (sm > bestSum)
                    {
                        idxs.CopyTo(indices, 0);
                        bestSum = sm;
                    }
                }
            }
        } else if (numMissing == 0)
        {
            int[] skipIdxs = new int[0];
            FrobeniusOpt(mesPts, lastPts, skipIdxs, out indices);
        } else
        {
            throw new System.Exception("Too many points missing to match");
        }

        return indices;
    }

    double FrobeniusOpt(List<Vector3> mesPts, Vector3[] lastPts, int[] skipIdxs, out int[] idxs)
    {
        // Create matrices
        Matrix X = new Matrix(3, mesPts.Count);
        Matrix Y = new Matrix(3, mesPts.Count);
        int count = 0;
        for (int i = 0; i < nPoints;  i++)
        {
            if (!Contains(skipIdxs,i))
            {
                X[count] = new Vector(mesPts[count]);
                Y[count] = new Vector(lastPts[i]);
                count++;
            }
        }
        Matrix Z = X.Transpose() * Y;
        
        // Get list of permutations
        List<List<int>> perms;
        if (mesPts.Count == 4)
            perms = permutations4;
        else if (mesPts.Count == 3)
            perms = permutations3;
        else
        {
            perms = new List<List<int>>();
            perms.Add(new List<int> { 0, 1 });
            perms.Add(new List<int> { 1, 0 });
        }

        // Find best permutation
        double bestSum = 0;
        int[] bestPerm = new int[mesPts.Count];
        foreach (List<int> perm in perms)
        {
            double e = 0;
            for (int i = 0; i < mesPts.Count; i++)
            {
                e += Z[i, perm[i]];
            }

            if (e > bestSum)
            {
                bestSum = e;
                bestPerm = perm.ToArray();
            }
        }

        idxs = bestPerm;
        return bestSum;
    }
    #endregion

    #region Brute Force Matching
    double BruteForceMatch(List<Vector3> mesPts, ref int[] idxsWithOutliers, List<int> idxs, bool useFixedIdxs)
    {
        int np = mesPts.Count;
        float bestErr = float.MaxValue;
        int bestPermIdx = -1;

        // Generate list of permutations
        List<List<int>> permutations;
        if (np <= 4) permutations = permutations4;
        // Outliers should be removed already. If not, this becomes much slower. > 6 becomes infeasibly slow
        else if (np == 5) permutations = permutations5;
        else
        {
            Debug.Log("Too many outliers - exiting brute force match");
            return 100;
            //int[] ints = new int[np];
            //for (int i = 0; i < np; i++)
            //    ints[i] = i;
            //permutations = RecursiveCreateCombo(ints);
        }
        
        // Try all the permutations
        int count = 0;
        foreach (var perm in permutations)
        {
            if (useFixedIdxs)
            {
                // Only consider the perm if it matches the known fixed indices 
                bool valid = true;
                for (int i = 0; i < idxs.Count; i++)
                {
                    if (idxs[i] >= 0 && idxs[i] != perm[i])
                    {
                        valid = false;
                        break;
                    }
                }
                if (!valid)
                {
                    count++;
                    continue;
                }
            }

            // Make the lists of points in the new order
            List<Vector3> exp = new List<Vector3>();
            List<Vector3> mes = new List<Vector3>();
            int nInclude = 0;
            for (int i = 0; i < np; i++)
            {
                // if perm[i] >= 4, mes[i] is an outlier so we ignore it
                if (perm[i] < nPoints)
                {
                    nInclude++;
                    mes.Add(mesPts[i]);
                    exp.Add(ukf.geom.relativePositions[perm[i]]);
                }
                //exp.Add(ukf.geom.relativePositions[perm[i]]);
            }

            // Calculate transform
            Matrix4x4 M = FindTransformFromPoints(mes, exp);

            // Calculate the error
            float err = 0;
            for (int i = 0; i < mes.Count; i++)
            {
                Vector3 fp = M.MultiplyPoint3x4(exp[i]);
                err += (fp - mes[i]).magnitude;
            }
            err /= nInclude;

            // Save if minimal
            if (err < bestErr)
            {
                bestErr = err;
                bestPermIdx = count;
            }

            count++;
        }

        if (bestPermIdx == -1) return float.MaxValue;

        // Mark the outliers with -1
        List<int> bestPerm = permutations[bestPermIdx];
        int j = 0;
        for (int i = 0; i < idxsWithOutliers.Length; i++)
        {
            if (idxsWithOutliers[i] == -1) continue;
            if (bestPerm[j] < 4)
                idxsWithOutliers[i] = bestPerm[j];
            else idxsWithOutliers[i] = -1;
            j++;
        }
        return bestErr;
    }

    Pose TransformFromIdxs(List<Vector3> mes, int[] idxs)
    {
        List<Vector3> mesPts = new List<Vector3>();
        List<Vector3> expPts = new List<Vector3>();
        for (int i = 0; i < idxs.Length; i++)
        {
            if (idxs[i] != -1)
            {
                mesPts.Add(mes[i]);
                expPts.Add(ukf.geom.relativePositions[idxs[i]]);
            }
        }

        Matrix4x4 M = FindTransformFromPoints(mesPts, expPts);
        return new Pose(M.MultiplyPoint3x4(Vector3.zero), M.rotation);
    }

    Matrix4x4 FindTransformFromPoints(List<Vector3> mes, List<Vector3> exp)
    {
        Vector3[] _mes = new Vector3[mes.Count];
        Vector3[] _exp = new Vector3[exp.Count];

        // Find average of each cluster of points
        Vector3 mesAvg = Vector3.zero;
        Vector3 expAvg = Vector3.zero;
        for (int i = 0; i < mes.Count; i++)
        {
            mesAvg += mes[i];
            expAvg += exp[i];
        }
        mesAvg /= mes.Count;
        expAvg /= exp.Count;

        // Move the expected points so the centroids are at the origin
        for (int i = 0; i < mes.Count; i++)
        {
            _exp[i] = exp[i] - expAvg;
            _mes[i] = mes[i] - mesAvg;
        }

        // Now the difference should be a pure rotation
        Matrix R = Kabsch(_mes, _exp);
        // R*(exp-expAvg) = (mes-mesAvg)
        // mes = R*exp + (mesAvg - R*expAvg) = R*exp + t
        Vector3 t = mesAvg - (R*expAvg).ToV3();

        // Create homogeneous transformation matrix
        Matrix4x4 pse = Matrix4x4.identity;
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                pse[i, j] = (float)R[i, j];
            }
        }
        for (int j = 0; j < 3; j++)
        {
            pse[j, 3] = t[j];
        }

        return pse;
    }

    private Matrix Kabsch(Vector3[] mes, Vector3[] exp)
    {
        // Create matrices
        Matrix At = new Matrix(3, exp.Length);
        Matrix B = new Matrix(mes.Length, 3);
        for (int i = 0; i < exp.Length; i++)
        {
            for (int j = 0; j < 3; j++)
                At[j, i] = exp[i][j]; 
        }
        for (int i = 0; i < mes.Length; i++)
        {
            for (int j = 0; j < 3; j++)
                B[i, j] = mes[i][j];
        }

        // Find rotation matrix (orthogonal, determinant 1) to map pointsFrom to pointsTo with minimum RMSE
        // Assumes the centroid of r is at [0,0,0]
        // Covariance H = A'*B
        Matrix H = At * B;

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
    bool SVD(Matrix mat, out Matrix U, out Matrix Vt)
    {
        // Calculate SVD
        double[,] Ud, Vtd;
        bool ret = alglib.rmatrixsvd(mat.mat, mat.M, mat.N, 1, 1, 2, out _, out Ud, out Vtd);
        U = new Matrix(Ud);
        Vt = new Matrix(Vtd);

        return ret;
    }

    #endregion

    #region Helper Methods
    private bool NoBadJumps(List<Vector3> mes, int[] idxs)
    {
        if (nBadJumps > 5)
        {
            // Been "bad" for a while so it probably actually should be this way
            for (int i = 0; i < mes.Count; i++)
            {
                if (idxs[i] >= 0 && idxs[i] < nPoints)
                {
                    lastPoints[idxs[i]] = new Vector3(mes[i].x, mes[i].y, mes[i].z);
                }
            }
            nBadJumps = 0;
            return true;
        }

        // Check if points are reasonably close to their last locations
        bool hasBad = false;
        for (int i = 0; i < idxs.Length; i++)
        {
            if (idxs[i] >= 0 && idxs[i] < nPoints)
            {
                float dist = (mes[i] - lastPoints[idxs[i]]).magnitude;
                if (dist > 0.04f) // -------------------------------------------------- 40 mm - good threshold?
                {
                    hasBad = true;
                } else
                {
                    // Update the good point positions
                    lastPoints[idxs[i]] = new Vector3(mes[i].x, mes[i].y, mes[i].z);
                }
            }
        }

        if (hasBad) nBadJumps++;
        else nBadJumps = 0;

        return !hasBad;
    }

    private void PoseCalcFailed()
    {
        // Nothing worked. Don't use this sample
        framesSinceLastPose++;

        // Restart the UKF?
        if (USE_UKF == 1 && ukf.timeSinceLastMes > ukfHorizon)
        {
            hasPose = false;
        }

        if (framesSinceLastPose > 4 && !warningText.activeInHierarchy)
        {
            warningText.SetActive(true);
        }

        NumRunning--;
    }

    private void SetPoseFromResult(List<Vector3> mes, int[] idxs)
    {
        Pose p = TransformFromIdxs(mes, idxs);

        // Compare to current pose and ignore if it's a big jump
        //Quaternion rotFromLast = p.rotation * Quaternion.Inverse(pose.rotation);
        //rotFromLast.ToAngleAxis(out float angleFromLast, out _);
        //if (Mathf.Abs(angleFromLast) > 30) {
        //    if (nBadPoses < 5)
        //    {
        //        nBadPoses++;
        //        lastBadPose = p;
        //        return;
        //    }
        //    else nBadPoses = 0; // We've been here for a bit, so accept this pose and move on
        //}

        //Debug.Log("__NKF:" + Timer.GetTime().ToString("N") + ":" + p.position.x.ToString("N4") + "," + p.position.y.ToString("N4") + "," + p.position.z.ToString("N4") + "," + p.rotation.x.ToString("N4") + "," + p.rotation.y.ToString("N4") + "," + p.rotation.z.ToString("N4") + "," + p.rotation.w.ToString("N4"));

        if (USE_UKF == 1)
        {
            // Make sure the UKF is running properly
            if (!ukf.Running())
                ukf.StartFilter(TransformFromIdxs(mes, idxs));
            else if (ukf.timeSinceLastMes > ukfHorizon)
            {
                ukf.Stop();
                ukf.StartFilter(TransformFromIdxs(mes, idxs));
            }

            // Perform measurement update
            ukf.OnNewPointSample(mes, idxs);

            // Set the pose from the UKF
            pose = ukf.CurrentPose();
        }
        else
        {
            // Remove jitter in marker positions
            //for (int i = 0; i < idxs.Length; i++)
            //{
            //    if (idxs[i] != -1)
            //    {
            //        // If it has barely moved, remove noise. Otherwise, update the last points
            //        if ((mes[i] - lastPoints[idxs[i]]).magnitude < 0.0025)
            //        {
            //            mes[i] = lastPoints[idxs[i]];
            //        } else 
            //            lastPoints[idxs[i]] = mes[i];
            //    }
            //}
            //p = TransformFromIdxs(mes, idxs);

            // Remove jitter in overall position
            //(p.rotation * Quaternion.Inverse(pose.rotation)).ToAngleAxis(out float angle, out _);
            //if ((p.position - pose.position).magnitude > 0.01f || Mathf.Abs(angle) > 15)
            //{
            //    // This was a sharp motion, so don't try to smooth it
            //}
            //else
            //{
            //    // Not moving much, so eliminate noise
            //    var q = Quaternion.Slerp(pose.rotation, p.rotation, smoothing);
            //    p = new Pose(p.position * (1 - smoothing) + pose.position * smoothing, q);
            //}

            // First order low-pass filter
            var q = Quaternion.Slerp(pose.rotation, p.rotation, 1-smoothing);
            p = new Pose(p.position * (1 - smoothing) + pose.position * smoothing, q);

            pose = p;

            // Approximate the velocity to extrapolate between readings
            if (extrapolate)
            {
                float dt = Timer.GetTime() - lastMeasTime;
                vel = (pose.position - lastMeas.position) / dt * deltaTime;
                angularVel = Quaternion.Slerp(Quaternion.identity, (q * Quaternion.Inverse(lastMeas.rotation)).normalized, deltaTime / dt);
                
                lastMeas = new Pose(pose.position, pose.rotation);
                lastMeasTime = Timer.GetTime();
            }

            updated = true;
        }

        NumRunning--;
    }
    float MeanMatchError(List<Vector3> mesPts, int[] idxs)
    {
        // Make the lists of points in the new order
        List<Vector3> exp = new List<Vector3>();
        List<Vector3> mes = new List<Vector3>();
        for (int i = 0; i < mesPts.Count; i++)
        {
            // Remove outliers
            if (idxs[i] >= 0)
            {
                mes.Add(mesPts[i]);
                exp.Add(ukf.geom.relativePositions[idxs[i]]);
            }
        }

        // Calculate transform
        Matrix4x4 M = FindTransformFromPoints(mes, exp);

        // Calculate the error
        float err = 0;
        for (int i = 0; i < mes.Count; i++)
        {
            Vector3 fp = M.MultiplyPoint3x4(exp[i]);
            err += (fp - mes[i]).magnitude;
        }

        return err / mesPts.Count;
    }

    public void ChangeSmoothing()
    {
        smoothing = (smoothing + 0.05f) % 1.0f;
        smoothButton.MainLabelText = "Smooth " + smoothing.ToString("N2");
    }

    public void ToggleExtrapolation()
    {
        if (signaler.IsConnected())
            extrapolate = !extrapolate;
    }

    bool Contains(int[] vec, int val)
    {
        foreach (int v in vec)
            if (v == val) return true;
        return false;
    }

    bool BadMatch(int[] lst)
    {
        int nGood = 0;
        foreach (int i in lst)
            if (i != -1) nGood++;
        return nGood <= 2;
    }

    public static List<List<int>> RecursiveCreateCombo(int[] ns)
    {
        if (ns.Length == 2)
        {
            List<List<int>> l = new List<List<int>>(2);
            l.Add(new List<int> { ns[0], ns[1] }); 
            l.Add(new List<int> { ns[1], ns[0] });
            return l;
        }
        else
        {
            List<List<int>> arrs = new List<List<int>>();

            // Take each element to be the first element, arrange the remaining elements
            for (int i = 0; i < ns.Length; i++)
            {
                // Create list not including chosen first element
                int[] newList = new int[ns.Length - 1];
                int count = 0;
                for (int j = 0; j < ns.Length; j++)
                {
                    if (j != i)
                    {
                        newList[count] = ns[j];
                        count++;
                    }
                }

                // Recursively find arrangements of the sublist
                List<List<int>> subarrs = RecursiveCreateCombo(newList);

                // Add the chosen first element to all the subarrs and add to master list
                foreach (List<int> lst in subarrs)
                {
                    lst.Insert(0, ns[i]);
                    arrs.Add(lst);
                }
            }

            return arrs;
        }
    }

    void PrintPose(Pose p)
    {
        Vector3 v = p.position;
        Quaternion q = p.rotation;
        string s = v.x.ToString("n4") + ", " + v.y.ToString("n4") + ", " + v.z.ToString("n4") + "\n";
        s += q.x.ToString("n4") + ", " + q.y.ToString("n4") + ", " + q.z.ToString("n4") + ", " + q.w.ToString("n4");
        Debug.Log(s);
    }

    void PrintList(int[] lst, string msg)
    {
        string s = msg + ": ";
        for (int i = 0; i < lst.Length; i++)
            s += lst[i].ToString() + ", ";
        Debug.Log(s);
    }

    void PrintList(List<int> lst, string msg)
    {
        string s = msg + ": ";
        for (int i = 0; i < lst.Count; i++)
            s += lst[i].ToString() + ", ";
        Debug.Log(s);
    }

    void PrintMat(int[,] lst, string msg)
    {
        string s = msg + ":\n";
        for (int i = 0; i < lst.GetLength(0); i++)
        {
            for (int j = 0; j < lst.GetLength(1); j++)
                s += lst[i, j].ToString() + ", ";
            s += "\n";
        }
        Debug.Log(s);
    }

    void PrintMat(float[,] lst, string msg)
    {
        string s = msg + ":\n";
        for (int i = 0; i < lst.GetLength(0); i++)
        {
            for (int j = 0; j < lst.GetLength(1); j++)
                s += lst[i, j].ToString("N3") + ", ";
            s += "\n";
        }
        Debug.Log(s);
    }
    #endregion
}
