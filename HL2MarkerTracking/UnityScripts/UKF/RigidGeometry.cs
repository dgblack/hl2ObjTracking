using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class RigidGeometry
{
    public Vector3[] relativePositions;
    public float[,] diffs;
    public int nPoints;
    public Vector3 centroid;
    public Matrix mat;
    public Vector3 tipOffset;
    public RigidGeometry(int nPoints)
    {
        relativePositions = new Vector3[nPoints];
        diffs = new float[nPoints, nPoints];
        this.nPoints = nPoints;
        this.centroid = Vector3.zero;
    }
    public RigidGeometry(Vector3[] points, Vector3 tipPos)
    {
        tipOffset = tipPos;
        relativePositions = points;
        nPoints = points.Length;
        diffs = new float[nPoints, nPoints];
        centroid = Vector3.zero;

        for (int i = 0; i < points.Length; i++)
        {
            diffs[i, i] = 0;

            for (int j = i+1; j < points.Length; j++)
            {
                diffs[i, j] = (points[i] - points[j]).magnitude;
                diffs[j, i] = diffs[i, j];
            }

            centroid += points[i];
        }

        centroid /= nPoints;

        mat = new Matrix(points.Length, 3);
        for (int i = 0; i < points.Length; i++)
        {
            for (int j = 0; j < 3; j++)
                mat[i,j] = points[i][j]-centroid[j];
        }
    }
}
