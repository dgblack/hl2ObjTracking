using UnityEngine;

using Complex = alglib.complex;

public class CVector
{
    public Complex[] vec { get; }
    public int Length { get; }

    public CVector(int n)
    {
        Length = n;
        vec = new Complex[n];
    }

    public CVector(int n, Complex val)
    {
        Length = n;
        vec = new Complex[n];
        for (int i = 0; i < n; i++)
            vec[i] = val;
    }

    public CVector(Vector3 v)
    {
        Length = 3;
        vec = new Complex[3];
        for (int i = 0; i < 3; i++)
            vec[i] = v[i];
    }

    public static implicit operator CVector(Vector _x)
    {
        CVector v = new CVector(_x.Length);
        for (int i = 0; i < _x.Length; i++)
            v[i] = _x[i];
        return v;
    }

    public Complex this[int key]
    {
        get => vec[key];
        set => vec[key] = value;
    }

    public CVector Slice(int startIdx, int endIdx)
    {
        CVector v = new CVector(endIdx - startIdx);
        int count = 0;
        for (int i = startIdx; i < endIdx; i++)
        {
            v[count] = vec[i];
            count++;
        }
        return v;
    }

    public Vector Real()
    {
        Vector v = new Vector(Length);
        for (int i = 0; i < Length; i++)
            v[i] = vec[i].x;
        return v;
    }

    public Vector Magnitude()
    {
        Vector v = new Vector(Length);
        for (int i = 0; i < Length; i++)
            v[i] = vec[i].Magnitude();
        return v;
    }

    public void Assign(CVector v, int startIdx)
    {
        int count = 0;
        for (int i = startIdx; i < startIdx + v.Length; i++)
        {
            vec[i] = v[count];
            count++;
        }
    }

    public static CVector Concat(CVector v, CVector w)
    {
        CVector vout = new CVector(v.Length + w.Length);
        for (int i = 0; i < v.Length; i++)
            vout[i] = v[i];
        for (int i = 0; i < w.Length; i++)
            vout[i + v.Length] = w[i];
        return vout;
    }

    public Complex Dot(CVector other)
    {
        if (other.Length != Length)
            throw new System.Exception("Vector dimension mismatch");

        Complex sum = 0;
        for (int i = 0; i < Length; i++)
        {
            sum += vec[i] * other[i];
        }

        return sum;
    }

    public Complex Dot(Vector3 other)
    {
        if (3 != Length)
            throw new System.Exception("Vector dimension mismatch");

        Complex sum = 0;
        for (int i = 0; i < 3; i++)
        {
            sum += vec[i] * other[i];
        }

        return sum;
    }
    public void SetVal(Complex val, int startIdx, int endIdx)
    {
        for (int i = startIdx; i < endIdx; i++)
        {
            vec[i] = val;
        }
    }
    public static CVector operator +(CVector v, CVector w)
    {
        if (v.Length != w.Length)
            throw new System.Exception("Vector dimension mismatch");

        CVector vout = new CVector(v.Length);

        for (int i = 0; i < v.Length; i++)
            vout[i] = v[i] + w[i];

        return vout;
    }
    public static CVector operator -(CVector v, CVector w)
    {
        if (v.Length != w.Length)
            throw new System.Exception("Vector dimension mismatch");

        CVector vout = new CVector(v.Length);

        for (int i = 0; i < v.Length; i++)
            vout[i] = v[i] - w[i];

        return vout;
    }

    public static CVector operator *(Complex k, CVector v)
    {
        CVector vout = new CVector(v.Length);

        for (int i = 0; i < v.Length; i++)
            vout[i] = v[i] * k;

        return vout;
    }

    public static CVector operator /(CVector v, Complex k)
    {
        CVector vout = new CVector(v.Length);

        for (int i = 0; i < v.Length; i++)
            vout[i] = v[i] / k;

        return vout;
    }

    public Vector3 ToV3()
    {
        if (Length != 3)
            throw new System.Exception("Cannot convert non 3-vector to Vector3");

        return new Vector3((float)vec[0].Magnitude(), (float)vec[1].Magnitude(), (float)vec[2].Magnitude());
    }

    public override string ToString()
    {
        string s = "[";
        for (int i = 0; i < Length; i++)
        {
            s += vec[i].x.ToString("e") + "+ " + vec[i].y.ToString("e") + "i";
            if (i != Length - 1)
                s += ",";
        }
        s += "]'";
        return s;
    }
}
