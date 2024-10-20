using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
using AOT;

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

    private class KeyPoints
    {
        public int Count { get => points3d.Count; }
        public List<Vector3> points3d;
        public List<Vector2> imCoords;

        public KeyPoints()
        {
            points3d = new List<Vector3>();
            imCoords = new List<Vector2>();
        }

        public KeyPoints(List<Vector3> points3d, List<Vector2> imCoords)
        {
            this.points3d = points3d;
            this.imCoords = imCoords;
        }
    }

    // Images
    private byte[] irImage = null;
    private byte[] lfImage = null;

    // Pose tracking
    public GameObject sphereMarker;
    private List<GameObject> markers;
    private float[] kps;
    public Transform hlPose;
    public Transform cameraPose;
    public PoseTracker ptrack;
    public int np = 8;

    // Video preview
    public Renderer preview;
    private Texture2D previewTex;
    private bool newReflImage = false;
    private int newLfImage = 0;
    private int newIrImage = 0;
    [NonSerialized] public int vidSendInterval = 2;
    [NonSerialized] public bool streaming = false;
    public bool getLFCam = false;
    private bool spatialCamsInited = false;

    // CV params
    public int maxArea = 1000;
    public int minArea = 2;
    public int binThreshold = 222;
    public float convexity = 0.75f;
    public float circularity = 0.6f;
    public bool contours = true;
    public bool saveIrImages = false;
    public bool useTimedPose = true;
    public bool showRawMarkers = false;
    public bool showPreview = false;
    [NonSerialized] public bool editMode = false;
    public Microsoft.MixedReality.Toolkit.UI.PinchSlider slider;
    [NonSerialized] public Vector3 trackOffset = new Vector3(-0.0477f, 0.0433f, 0.0492f); //-0.0105f
    [NonSerialized] public Vector3 eulerOffset = new Vector3(5.5f, -1, 1);

    private Matrix4x4 pos2unity;
    private Matrix4x4 rot2unity;

    public Material matchedMat;
    public Material outlierMat;

    // Pose memory for correlation with captured sensor frames
    DictQueue timedPoses;
    Pose nowPose;
    Pose lastPose;
    Pose lastLastPose;

    // ROI for searching in
    [NonSerialized] public float roiBuffer = 1.25f;

    public void OnChangeOffset(int idx)
    {
        if (editMode)
        {
            trackOffset[Math.Abs(idx)-1] += Math.Sign(idx)* 0.001f;
            Debug.Log("Offset: " + trackOffset.x.ToString("n4") + ", " + trackOffset.y.ToString("n4") + ", " + trackOffset.z.ToString("n4"));
        }
        else {
            eulerOffset[Math.Abs(idx)-1] += Math.Sign(idx) * 0.5f;
            Debug.Log("Euler Offset: " + eulerOffset.x.ToString("n4") + ", " + eulerOffset.y.ToString("n4") + ", " + eulerOffset.z.ToString("n4"));
        }
    }

#if ENABLE_WINMD_SUPPORT
    Windows.Perception.Spatial.SpatialCoordinateSystem unityWorldOrigin;
#endif

    private void Awake()
    {
#if ENABLE_WINMD_SUPPORT
    unityWorldOrigin = Microsoft.MixedReality.OpenXR.PerceptionInterop.GetSceneCoordinateSystem(UnityEngine.Pose.identity) as Windows.Perception.Spatial.SpatialCoordinateSystem;
#endif
    }

    void Start()
    {
        timedPoses = new DictQueue();
        markers = new List<GameObject>(np);
        markers.Add(sphereMarker);
        for (int i = 1; i < np; i++)
        {
            markers.Add(GameObject.Instantiate(sphereMarker));
        }
        SetMarkers();

        // From some recorded data and a least squares calculation in MATLAB
        // To transform positions from cpp to unity. Is this even constant?
        // Dont fkn ask why pos2unity and rot2unity arent the same
        pos2unity = new Matrix4x4();
        pos2unity.m00 = 0.7936f;
        pos2unity.m01 = 0.0465f;
        pos2unity.m02 = 0.0514f;
        pos2unity.m03 = 0.0399f;
        pos2unity.m10 = -0.1700f;
        pos2unity.m11 = 0.5803f;
        pos2unity.m12 = -0.1647f;
        pos2unity.m13 = -0.0370f;
        pos2unity.m20 = 0.0446f;
        pos2unity.m21 = -0.0907f;
        pos2unity.m22 = -0.4225f;
        pos2unity.m23 = -0.0177f;
        pos2unity.m30 = 0;
        pos2unity.m31 = 0;
        pos2unity.m32 = 0;
        pos2unity.m33 = 1;

        rot2unity = new Matrix4x4();
        rot2unity.m00 = 0.126f;
        rot2unity.m01 = -0.9494f;
        rot2unity.m02 = 0.2878f;
        rot2unity.m03 = 0;
        rot2unity.m10 = -0.9711f;
        rot2unity.m11 = -0.0588f;
        rot2unity.m12 = 0.2311f;
        rot2unity.m13 = 0;
        rot2unity.m20 = -0.2025f;
        rot2unity.m21 = -0.3087f;
        rot2unity.m22 = -0.9294f;
        rot2unity.m23 = 0;
        rot2unity.m30 = 0;
        rot2unity.m31 = 0;
        rot2unity.m32 = 0;
        rot2unity.m33 = 1;

#if ENABLE_WINMD_SUPPORT
        // To preview the LF image        
        //previewTex = new Texture2D(640, 480, TextureFormat.Alpha8, false);
        // To preview the IR image
        previewTex = new Texture2D(512, 512, TextureFormat.Alpha8, false);
        preview.material.mainTexture = previewTex;

        researchMode = new MarkerTracker();
        researchMode.SetParams(minArea, maxArea, binThreshold, convexity, circularity, contours, saveIrImages);

        researchMode.InitializeDepthSensor();
        if (getLFCam) {
            researchMode.InitializeSpatialCamerasFront();
        }

        researchMode.SetReferenceCoordinateSystem(unityWorldOrigin);

        researchMode.StartDepthSensorLoop();
        if (getLFCam) {
            researchMode.StartSpatialCamerasFrontLoop();
        }

        Debug.Log("Initialized ResearchMode APIs");
#endif
    }

    Vector3 lastPosition = Vector3.zero;
    Quaternion lastRotation = Quaternion.identity;
    long lastUpdateTime = 0;
    private void Update()
    {
        if (lastPosition != ptrack.pose.position || lastRotation != ptrack.pose.rotation)
        {
            long t = Timer.GetTimeUs();
            //Debug.Log("Update:" + (t-lastUpdateTime).ToString("N"));
            lastUpdateTime = t;
            lastPosition = ptrack.pose.position;
            lastRotation = ptrack.pose.rotation;
        }

#if ENABLE_WINMD_SUPPORT
        transform.SetPositionAndRotation(ptrack.pose.position, ptrack.pose.rotation);
#endif
        if (!useTimedPose) return;

#if ENABLE_WINMD_SUPPORT
        // read in all timestamps 
        float ts = 1;
        bool gotNewTs = false;
        Pose newPose = new Pose(cameraPose.position, cameraPose.rotation);
        while (ts > 0) {
            ts = Mathf.Round(researchMode.GetLastTimestamp()*100)/100;

            if (ts > 0) {
                gotNewTs = true;
                timedPoses.Add(ts,newPose);
            }
        }

        if (gotNewTs) {
            if (nowPose != null)
                lastPose = nowPose;
            nowPose = newPose;
        }
#endif
    }

    async void LateUpdate()
    {

#if ENABLE_WINMD_SUPPORT // Is there a synchronization problem if we use depth and reflectivity images from different times? (i.e. a new depth image arrives while we're processing the refl image)
        string s = researchMode.GetLogs();
        if (s != null && s != "") Debug.Log(s);

        // update short-throw reflectivity texture - actually we don't have any use for the images
        if (saveIrImages) {
            if (researchMode.ShortAbImageTextureUpdated())
            {
                byte[] frameTexture = researchMode.GetShortAbImageTextureBuffer();
                if (frameTexture.Length > 0)
                {
                    newIrImage++;
                    if (irImage == null)
                    {
                        irImage = frameTexture;
                    }
                    else
                    {
                        System.Buffer.BlockCopy(frameTexture, 0, irImage, 0, irImage.Length);
                    }
                }
            }
        }
        
        // update LF camera texture
        if (getLFCam) {
            if (researchMode.LFImageUpdated())
            {
                long ts;
                byte[] frameTexture = researchMode.GetLFCameraBuffer(out ts);
                if (frameTexture.Length > 0)
                {
                    newLfImage++;
                    if (lfImage == null)
                    {
                        lfImage = frameTexture;
                    }
                    else
                    {
                        System.Buffer.BlockCopy(frameTexture, 0, lfImage, 0, lfImage.Length);
                    }
                }
            }
        }
#endif

        await FindPose();

        if (getLFCam && lfImage != null && newLfImage != 0 && newLfImage % vidSendInterval == 0)
        {
            newLfImage = 0;

            // Preview locally
            //previewTex.LoadRawTextureData(lfImage);
            //previewTex.Apply();
        }

        if (saveIrImages && irImage != null && newIrImage != 0 && newIrImage % vidSendInterval == 0)
        {
            newIrImage = 0;

            // Preview locally
            previewTex.LoadRawTextureData(irImage);
            previewTex.Apply();
        }
    }

    public void SetParams()
    {
#if ENABLE_WINMD_SUPPORT
        researchMode.SetParams(minArea, maxArea, binThreshold, convexity, circularity, contours, saveIrImages);
#endif
    }

    public void SwitchGeometry()
    {
        if (ptrack.USE_UKF == 1)
        {
            ptrack.ukf.Stop();
        }

        ptrack.ukf.SwitchGeometry();
        
        if (ptrack.USE_UKF == 1)
        {
            ptrack.ukf.StartFilter(ptrack.ukf.CurrentPose());
        }
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
        preview.gameObject.SetActive(saveIrImages);
        SetParams();
    }

    public void ToggleLFCam()
    {
        if (!spatialCamsInited)
        {
#if ENABLE_WINMD_SUPPORT
        researchMode.InitializeSpatialCamerasFront();
        researchMode.StartSpatialCamerasFrontLoop();
#endif
            spatialCamsInited = true;
        }

        getLFCam = !getLFCam;
        //preview.gameObject.SetActive(getLFCam);
    }

    private async Task<bool> FindPose()
    {
        // Find the marker points
        KeyPoints kps = FindMarkers();

        if (kps.Count > 0)
        {
            var pos = kps.points3d;
            var imcoords = kps.imCoords;

            // Compute the pose
            int[] idxs = await Task.Run(() => ptrack.UpdatePose(pos));
            //ptrack.UpdatePose(pos);

            // Generate ROI for searching in the next frame
            int minX = 512; int minY = 512; int maxX = 0; int maxY = 0;
            List<Vector2> markerImPoints = new List<Vector2>();
            for (int j = 0; j < imcoords.Count; j++)
            {
                if (idxs.Length > j && idxs[j] != -1)
                {
                    if (imcoords[j].x < minX) minX = (int)imcoords[j].x;
                    if (imcoords[j].y < minY) minY = (int)imcoords[j].y;
                    if (imcoords[j].x > maxX) maxX = (int)imcoords[j].x;
                    if (imcoords[j].y > maxY) maxY = (int)imcoords[j].y;
                }
            }

            if (minX == 0 && minY == 0 && maxX == 512 && maxY == 512)
            {
                //Debug.Log("No inlying points found");
                // Don't reset ROI here. It should only happen in one place, otherwise it gets out of hand. See the else for kps.Count < 0
            } else
            {
                // Add a buffer around the markers to allow for some motion. Also make sure there's no overflow
                int width = (int)(Mathf.Max(maxX - minX, maxY - minY) * roiBuffer);
                int xCrop = (maxX + minX) / 2;
                int yCrop = (maxY + minY) / 2;

                if (xCrop < 0 || yCrop < 0 || width < 0)
                {
                    // something went wrong so set to defaults
                    xCrop = 256; yCrop = 256; width = 512;
                } 

#if ENABLE_WINMD_SUPPORT
                researchMode.SetROI(xCrop, yCrop, width);
#endif
            }

            if (showRawMarkers)
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
                    //} else if (j < np) // hide outliers
                    //{
                    //    markers[j].GetComponent<MeshRenderer>().enabled = false;
                    //    markers.Remove(markers[j]);
                    //}
                }
                // Hide any extra markers
                for (int j = pos.Count; j < np; j++)
                {
                    markers[j].GetComponent<MeshRenderer>().enabled = false;
                }
            }

            return true;
        }
        else
        {
            // Open up the roi again in case this was the issue
            ptrack.hasPose = false;
            ptrack.framesSinceLastPose++;
            if (ptrack.framesSinceLastPose > 25)
            {
                ptrack.hadPose = false;
#if ENABLE_WINMD_SUPPORT
                //Debug.Log("Didn't find anything for a while. Resetting ROI");
                researchMode.SetROI(256, 256, 512);
#endif
            }
            return false;
        }
    }

    private KeyPoints FindMarkers()
    {
        kps = GetKeypoints();

        if (kps == null)
        {
            //Debug.Log("Returned keypoints are null");
            return new KeyPoints();
        }

        int n;
        if ((kps.Length - 1) % 5 != 0)
        {
            //Debug.Log($"{Timer.GetTime()}: Incomplete list of keypoints received. Ignoring...");
            return new KeyPoints();
        }
        else n = (int)((kps.Length - 1) / 5);

        // Process marker positions
        return GetMarkersFromKeypoints(kps, n);
    }

    private float[] GetKeypoints()
    {
#if ENABLE_WINMD_SUPPORT
        int flag = 2;
        int numWaiting = 0;

        kps = researchMode.GetLastKeypoints(out flag, out numWaiting);

        if (flag == 1) 
            return kps;
        else {
            if (flag == 2)
                Debug.Log("Flag not returned");
            else if (flag == 0) {
                //Debug.Log("No points found");
            } else if (flag == -1)
                Debug.Log("No clusters found");
            else if (flag == -2)
                Debug.Log("Error reading in image");
        }
#endif
        return null;
    }
    private KeyPoints GetMarkersFromKeypoints(float[] kps, int n)
    {
        //Debug.Log($"{Timer.GetTime()}: {n} keypoints found");

        // Extract HL2 pose at time of measurement
        float tstamp = Mathf.Round(kps[0] * 100) / 100;
        if (timedPoses.TryGet(tstamp, out Pose p))
            hlPose.SetPositionAndRotation(p.position, p.rotation);
        else
            hlPose.SetPositionAndRotation(cameraPose.position, cameraPose.rotation);

        // Process marker positions
        if (n >= 2)
        {
            // For each keypoint (potential marker)
            List<Vector3> pos = new List<Vector3>(n);
            List<Vector2> imCoords = new List<Vector2>(n);
            int i = 0;
            for (int j = 1; j < n*3 + 1; j += 3)
            {
                // Get rid of really distant ones and really close ones
                if (kps[j + 2] > 2 || kps[j + 2] < -2 || Mathf.Abs(kps[j+2]) < 0.1f )
                    continue;

                // Some transformation to make the axes align
                pos.Add(new Vector3(-kps[j + 1], -kps[j], kps[j + 2]) + trackOffset); // Offset for inaccurate extrinsics parameters
                pos[i] = Quaternion.Euler(eulerOffset) * pos[i];
                pos[i] = hlPose.TransformPoint(pos[i]);

                // Also keep the image coordinates
                imCoords.Add(new Vector2(kps[n*3 + 1 + 2*i], kps[n*3 + 2 + 2*i]));

                i++;
            }

            //Debug.Log($"{Timer.GetTime()}: Keeping {pos.Count} keypoints for processing");

            return new KeyPoints(pos, imCoords);
        }
        else
        {
            return new KeyPoints();
        }
    }

#region Button Event Functions
    public void StopSensorsEvent()
    {
#if ENABLE_WINMD_SUPPORT
        researchMode.StopAllSensorDevice();
#endif
    }

    #endregion
#if UNITY_EDITOR
    public void OnApplicationQuit() { StopSensorsEvent(); }
#else
        public void OnApplicationFocus(bool hasFocus) {if (!hasFocus) StopSensorsEvent(); }
#endif
}