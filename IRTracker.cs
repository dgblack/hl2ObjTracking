using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Events;
using System;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
using AOT;
using System.Diagnostics;

#if ENABLE_WINMD_SUPPORT
using HL2MarkerTracking;
#else
using System.IO;
#endif

public class IRTracker : MonoBehaviour
{
#if ENABLE_WINMD_SUPPORT
    MarkerTracker researchMode;
#endif

    enum DepthSensorMode
    {
        ShortThrow,
        LongThrow,
        None
    };

    public enum StereoSaveMode
    {
        Both,
        Left,
        Right,
        None
    }

    // Callbacks for images
    public UnityEvent<byte[]> OnIrImage;
    public UnityEvent<UInt16[]> OnRawIrImage;
    public UnityEvent<byte[]> OnDepthMap;
    public UnityEvent<UInt16[]> OnRawDepthMap;
    public UnityEvent<byte[], byte[]> OnStereoPair;

    // Pose tracking preview
    public GameObject sphereMarker; // Used to preview the raw marker positions. Set it to a sphere the size of a IR reflective sphere
    private List<GameObject> markers; // Used to preview the raw marker positions
    public Material matchedMat;
    public Material outlierMat;
    public int np = 8;

    // CV params
    public float smoothing = 0.9f;
    public int maxArea = 1000;
    public int minArea = 2;
    public int binThreshold = 222;
    public float convexity = 0.75f;
    public float circularity = 0.6f;
    public bool contours = true;
    public bool saveIrImages = false;
    public bool saveDepthMaps = false;
    public StereoSaveMode saveStereoImages = StereoSaveMode.None;
    public bool saveRaw = false;
    public bool showRawMarkers = false;
    public bool showPreview = false;
    public bool verbose = true;
    [NonSerialized] public Vector3 trackOffset = new Vector3(-0.0375f, 0.0494f, 0.0522f); // From experimentation
    [NonSerialized] public Vector3 eulerOffset = new Vector3(-1, 5.5f, -1.1f); // From experimentation
    [NonSerialized] public bool hasPose = false;    

    public List<Transform> geometry; // Used to define the marker geometry
    public Transform cameraPose; // Set to the MainCamera so it gives the HoloLens's pose

    // Pose memory for correlation with captured sensor frames
    Vector3 lastPosition = Vector3.zero;
    Quaternion lastRotation = Quaternion.identity;
    long lastUpdateTime = 0;

    // ROI for searching in
    [NonSerialized] public float roiBuffer = 1.25f;

#if ENABLE_WINMD_SUPPORT
    Windows.Perception.Spatial.SpatialCoordinateSystem unityWorldOrigin;
#endif

    private void Awake()
    {
        // This doesn't really seem to work so it's not used
#if ENABLE_WINMD_SUPPORT
    unityWorldOrigin = Microsoft.MixedReality.OpenXR.PerceptionInterop.GetSceneCoordinateSystem(UnityEngine.Pose.identity) as Windows.Perception.Spatial.SpatialCoordinateSystem;
#endif
    }

    void Start()
    {
        float[] geomVec = new float[geometry.Count * 3];
        for (int i = 0; i < geometry.Count; i++)
        {
            geomVec[i * 3] = geometry[i].localPosition[0];
            geomVec[i * 3 + 1] = geometry[i].localPosition[1];
            geomVec[i * 3 + 2] = geometry[i].localPosition[2];
        }

        markers = new List<GameObject>(np);
        markers.Add(sphereMarker);
        for (int i = 1; i < np; i++)
        {
            markers.Add(GameObject.Instantiate(sphereMarker));
        }
        SetMarkers();

        Matrix4x4 extrinsicsOffset = Matrix4x4.TRS(trackOffset, Quaternion.Euler(eulerOffset.x, eulerOffset.y, eulerOffset.z), Vector3.one);
        float[] extOffset = new float[16];
        for (int i = 0; i < 16; i++)
            extOffset[i] = extrinsicsOffset[i];

#if ENABLE_WINMD_SUPPORT
        researchMode = new MarkerTracker(geomVec, extOffset, verbose);
        SetParams();

        researchMode.InitializeDepthSensor();
        if (saveStereoImages != StereoSaveMode.None) {
            researchMode.InitializeStereoCamerasFront();
        }

        researchMode.SetReferenceCoordinateSystem(unityWorldOrigin);

        researchMode.StartDepthSensorLoop();
        if (saveStereoImages != StereoSaveMode.None) {
            researchMode.StartStereoCamerasFrontLoop();
        }

        Debug.Log("Initialized ResearchMode APIs");
#endif
    }

    private void Update()
    {
        // Get current HoloLens pose as vector
        Matrix4x4 T = Matrix4x4.TRS(cameraPose.position, cameraPose.rotation, Vector3.one);
        Matrix4x4 objPose = Matrix4x4.identity;
        float[] Tvec = new float[16];
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 4; j++)
                Tvec[i * 4 + j] = T[i, j];

#if ENABLE_WINMD_SUPPORT
        researchMode.SetDevicePose(Tvec);

        double[] poseVec = researchMode.GetObjectPose();
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 4; j++)
                objPose[i,j] = (float)poseVec[i * 4 + j];
#endif

        if (objPose != Matrix4x4.identity)
        {
            hasPose = true;
            transform.SetPositionAndRotation(objPose.GetPosition(), objPose.rotation);
            if (transform.position != lastPosition) 
            {
                lastPosition = transform.position;
                lastRotation = transform.rotation;
                lastUpdateTime = Timer.GetTime();
                
            }
        }
    }

    async void LateUpdate()
    {
#if ENABLE_WINMD_SUPPORT // Is there a synchronization problem if we use depth and reflectivity images from different times? (i.e. a new depth image arrives while we're processing the refl image)
        // update short-throw reflectivity texture - actually we don't have any use for the images
        if (saveIrImages) {
            if (researchMode.IrImageUpdated())
            {
                if (saveRaw && OnRawIrImage != null) {
                    UInt16[] frameTexture = researchMode.GetRawIrImage();
                    if (frameTexture.Length > 0)
                            OnRawIrImage.Invoke(frameTexture);
                } else if (OnIrImage != null) {
                    byte[] frameTexture = researchMode.GetProcessedIrImage();
                    if (frameTexture.Length > 0 && OnIrImage != null)
                            OnIrImage.Invoke(frameTexture);
                }
            }
        }

        if (saveDepthMaps) {
            if (saveRaw && OnRawDepthMap != null) {
                UInt16[] frameTexture = researchMode.GetRawDepthMap();
                if (frameTexture.Length > 0)
                    OnRawDepthMap.Invoke(frameTexture);
            } else if (OnDepthMap != null) {
                byte[] frameTexture = researchMode.GetProcessedDepthMap();
                if (frameTexture.Length > 0)
                    OnDepthMap.Invoke(frameTexture);
            }
        }
        
        // update LF camera texture
        if (saveStereoImages == StereoSaveMode.Left) {
            if (researchMode.LfImageUpdated() && OnStereoPair != null)
            {
                long ts;
                byte[] frameTexture = researchMode.GetLfImage(out ts);
                if (frameTexture.Length > 0)
                    OnStereoPair.Invoke(frameTexture, null);
            }
        }
        // update RF camera texture
        else if (saveStereoImages == StereoSaveMode.Right) {
            if (researchMode.RfImageUpdated() && OnStereoPair != null)
            {
                long ts;
                byte[] frameTexture = researchMode.GetRfImage(out ts);
                if (frameTexture.Length > 0)
                    OnStereoPair.Invoke(null, frameTexture);
            }
        }
        // Update both stereo camera images
        else if (saveStereoImages == StereoSaveMode.Both) {
            if (researchMode.LfImageUpdated() && researchMode.RfImageUpdated() && OnStereoPair != null)
            {
                long tsl, tsr;
                byte[] frameTexture = researchMode.GetLrfImages(out tsl, out tsr);
                int imLength = frameTexture.Length / 2;
                byte[] leftImage = new byte[imLength];
                byte[] rightImage = new byte[imLength];
                Buffer.BlockCopy(frameTexture, 0, leftImage, 0, imLength);
                Buffer.BlockCopy(frameTexture, imLength+1, rightImage, 0, imLength);

                if (frameTexture.Length > 0)
                    OnStereoPair.Invoke(leftImage, rightImage);
            }
        }
#endif
    }

    public void SetParams()
    {
#if ENABLE_WINMD_SUPPORT
        if (saveStereoImages == StereoSaveMode.Both)
            researchMode.SetParams(minArea, maxArea, binThreshold, convexity, circularity, smoothing, contours, saveIrImages, saveDepthMaps, true, true, saveRaw);
        else if (saveStereoImages == StereoSaveMode.Left)
            researchMode.SetParams(minArea, maxArea, binThreshold, convexity, circularity, smoothing, contours, saveIrImages, saveDepthMaps, true, false, saveRaw);
        else if (saveStereoImages == StereoSaveMode.Right)
            researchMode.SetParams(minArea, maxArea, binThreshold, convexity, circularity, smoothing, contours, saveIrImages, saveDepthMaps, false, true, saveRaw);
        else
            researchMode.SetParams(minArea, maxArea, binThreshold, convexity, circularity, smoothing, contours, saveIrImages, saveDepthMaps, false, false, saveRaw);
#endif
    }

    public void ChangeOffset(int idx, float delta, bool isRotation = false)
    {
        if (!isRotation)
        {
            trackOffset[idx] += delta;
            Debug.Log("Offset: " + trackOffset.ToString("n4"));
        }
        else
        {
            eulerOffset[idx] += delta;
            Debug.Log("Euler Offset: " + eulerOffset.ToString("n4"));
        }

        Matrix4x4 extrinsicsOffset = Matrix4x4.TRS(trackOffset, Quaternion.Euler(eulerOffset.x, eulerOffset.y, eulerOffset.z), Vector3.one);
        float[] extOffset = new float[16];
        for (int i = 0; i < 16; i++)
            extOffset[i] = extrinsicsOffset[i];

#if ENABLE_WINMD_SUPPORT
        researchMode.SetExtrinsicsOffset(extOffset);
#endif
    }

    public void ToggleMarkers()
    {
        showRawMarkers = !showRawMarkers;
        SetMarkers();
    }

    private void SetMarkers()
    {
        foreach (var m in markers)
        {
            m.GetComponent<MeshRenderer>().enabled = showRawMarkers;
        }
    }

    public void ToggleContours()
    {
        contours = !contours;
        SetParams();
    }

    public void ToggleIRIms()
    {
        saveIrImages = !saveIrImages;
        SetParams();
    }

    public void ToggleLFCam()
    {
        if (!stereoCamsInited)
        {
#if ENABLE_WINMD_SUPPORT
        researchMode.InitializeStereoCamerasFront();
        researchMode.StartStereoCamerasFrontLoop();
#endif
            stereoCamsInited = true;
        }

        saveStereoImages = (saveStereoImages == StereoSaveMode.None) ? StereoSaveMode.Both : StereoSaveMode.None;
    }

    private void ShowPoints(List<Vector3> pos, int[] idxs)
    {
        // Render the points
        for (int j = 0; j < pos.Count; j++)
        {
            //if (idxs.Length > j && idxs[j] != -1) {
            if (j < np)
            {
                var mr = markers[j].GetComponent<MeshRenderer>();
                markers[j].transform.position = pos[j];
                mr.enabled = true;
                if (idxs.Length > j && idxs[j] != -1)
                    mr.material = matchedMat;
                else mr.material = outlierMat;
            }
            else
            {
                markers.Add(GameObject.Instantiate(sphereMarker));
                var mr = markers[j].GetComponent<MeshRenderer>();
                markers[j].transform.position = pos[j];
                mr.enabled = true;
                if (idxs.Length > j && idxs[j] != -1)
                    mr.material = matchedMat;
                else mr.material = outlierMat;
                np++;
            }
        }
        // Hide any extra markers
        for (int j = pos.Count; j < np; j++)
        {
            markers[j].GetComponent<MeshRenderer>().enabled = false;
        }
    }

    public void StopSensorsEvent()
    {
#if ENABLE_WINMD_SUPPORT
        researchMode.StopAllSensorDevice();
#endif
    }

#if UNITY_EDITOR
    public void OnApplicationQuit() { StopSensorsEvent(); }
#else
        public void OnApplicationFocus(bool hasFocus) {if (!hasFocus) StopSensorsEvent(); }
#endif
}