using UnityEngine;

public class Vector
{
    public double[] vec { get; }
    public int Length { get; }

    public Vector(int n)
    {
        Length = n;
        vec = new double[n];
    }

    public Vector(int n, double val)
    {
        Length = n;
        vec = new double[n];
        for (int i = 0; i < n; i++)
            vec[i] = val;
    }

    public Vector(Vector3 v)
    {
        Length = 3;
        vec = new double[3];
        for (int i = 0; i < 3; i++)
            vec[i] = v[i];
    }

    public Vector(double[] v)
    {
        Length = v.Length;
        vec = new double[Length];
        for (int i = 0; i < Length; i++)
            vec[i] = v[i];
    }

    public double this[int key]
    {
        get => vec[key];
        set => vec[key] = value;
    }

    public Vector Slice(int startIdx, int endIdx)
    {
        Vector v = new Vector(endIdx - startIdx);
        int count = 0;
        for (int i = startIdx; i < endIdx; i++)
        {
            v[count] = vec[i];
            count++;
        }
        return v;
    }

    public void Assign(Vector v, int startIdx)
    {
        int count = 0;
        for (int i = startIdx; i < startIdx + v.Length; i++)
        {
            vec[i] = v[count];
            count++;
        }
    }

    public static Vector Concat(Vector v, Vector w)
    {
        Vector vout = new Vector(v.Length + w.Length);
        for (int i = 0; i < v.Length; i++)
            vout[i] = v[i];
        for (int i = 0; i < w.Length; i++)
            vout[i + v.Length] = w[i];
        return vout;
    }

    public double Dot(Vector other)
    {
        if (other.Length != Length)
            throw new System.Exception("Vector dimension mismatch");

        double sum = 0;
        for (int i = 0; i < Length; i++)
        {
            sum += vec[i] * other[i];
        }

        return sum;
    }

    public double Dot(Vector3 other)
    {
        if (3 != Length)
            throw new System.Exception("Vector dimension mismatch");

        double sum = 0;
        for (int i = 0; i < 3; i++)
        {
            sum += vec[i] * other[i];
        }

        return sum;
    }
    public void SetVal(double val, int startIdx, int endIdx)
    {
        for (int i = startIdx; i < endIdx; i++)
        {
            vec[i] = val;
        }
    }
    public static Vector operator +(Vector v, Vector w)
    {
        if (v.Length != w.Length)
            throw new System.Exception("Vector dimension mismatch");

        Vector vout = new Vector(v.Length);

        for (int i = 0; i < v.Length; i++)
            vout[i] = v[i] + w[i];

        return vout;
    }
    public static Vector operator -(Vector v, Vector w)
    {
        if (v.Length != w.Length)
            throw new System.Exception("Vector dimension mismatch");

        Vector vout = new Vector(v.Length);

        for (int i = 0; i < v.Length; i++)
            vout[i] = v[i] - w[i];

        return vout;
    }

    public static Vector operator *(double k, Vector v)
    {
        Vector vout = new Vector(v.Length);

        for (int i = 0; i < v.Length; i++)
            vout[i] = v[i] * k;

        return vout;
    }

    public static Vector operator /(Vector v, double k)
    {
        Vector vout = new Vector(v.Length);

        for (int i = 0; i < v.Length; i++)
            vout[i] = v[i] / k;

        return vout;
    }

    public double Norm()
    {
        double ret = 0;
        foreach (double d in vec)
            ret += d * d;
        return System.Math.Sqrt(ret);
    }
    public Vector3 ToV3()
    {
        if (Length != 3)
            throw new System.Exception("Cannot convert non 3-vector to Vector3");

        return new Vector3((float)vec[0], (float)vec[1], (float)vec[2]);
    }

    public override string ToString()
    {
        string s = "[";
        for (int i = 0; i < Length; i++)
        {
            s += vec[i].ToString("e");
            if (i != Length - 1)
                s += ",";
        }
        s += "]'";
        return s;
    }
}
