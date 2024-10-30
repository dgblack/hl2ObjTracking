#include <unknwn.h>
#include <winrt/Windows.Foundation.h>
#include <winrt/Windows.Foundation.Collections.h>

#include <Eigen/Dense>

#include<winrt/Windows.Perception.Spatial.h>
#include<winrt/Windows.Perception.Spatial.Preview.h>

#include "MarkerTracker.h"
#include "MarkerTracker.g.cpp"

static ResearchModeSensorConsent camAccessCheck;
static HANDLE camConsentGiven;
static ResearchModeSensorConsent imuAccessCheck;
static HANDLE imuConsentGiven;

using namespace DirectX;
using namespace winrt::Windows::Perception;
using namespace winrt::Windows::Perception::Spatial;
using namespace winrt::Windows::Perception::Spatial::Preview;

namespace winrt::HL2MarkerTracking::implementation
{
    void debugLog(std::string st) {
        std::cout << "[HL2MarkerTracking] " << st << std::endl;
    }

    MarkerTracker::MarkerTracker(array_view<float const> geometry, array_view<float const> extrinsicsCorrection, float markerDiameter, bool verbose)
    {
        // Load Research Mode library
        camConsentGiven = CreateEvent(nullptr, true, false, nullptr);
        imuConsentGiven = CreateEvent(nullptr, true, false, nullptr);
        HMODULE hrResearchMode = LoadLibraryA("ResearchModeAPI");
        HRESULT hr = S_OK;

        if (hrResearchMode)
        {
            typedef HRESULT(__cdecl* PFN_CREATEPROVIDER) (IResearchModeSensorDevice** ppSensorDevice);
            PFN_CREATEPROVIDER pfnCreate = reinterpret_cast<PFN_CREATEPROVIDER>(GetProcAddress(hrResearchMode, "CreateResearchModeSensorDevice"));
            if (pfnCreate)
            {
                winrt::check_hresult(pfnCreate(&m_pSensorDevice));
            }
            else
            {
                winrt::check_hresult(E_INVALIDARG);
            }
        }

        // get spatial locator of rigNode
        GUID guid;
        IResearchModeSensorDevicePerception* pSensorDevicePerception;
        winrt::check_hresult(m_pSensorDevice->QueryInterface(IID_PPV_ARGS(&pSensorDevicePerception)));
        winrt::check_hresult(pSensorDevicePerception->GetRigNodeId(&guid));
        pSensorDevicePerception->Release();
        m_locator = SpatialGraphInteropPreview::CreateLocatorForNode(guid);

        size_t sensorCount = 0;

        winrt::check_hresult(m_pSensorDevice->QueryInterface(IID_PPV_ARGS(&m_pSensorDeviceConsent)));
        winrt::check_hresult(m_pSensorDeviceConsent->RequestCamAccessAsync(MarkerTracker::CamAccessOnComplete));

        m_pSensorDevice->DisableEyeSelection();

        winrt::check_hresult(m_pSensorDevice->GetSensorCount(&sensorCount));
        m_sensorDescriptors.resize(sensorCount);
        winrt::check_hresult(m_pSensorDevice->GetSensorDescriptors(m_sensorDescriptors.data(), m_sensorDescriptors.size(), &sensorCount));

        // Extract the marker geometry
        std::vector<Eigen::Vector3d> points;
        for (size_t i = 0; i <= geometry.size()-3; i += 3) {
            points.emplace_back(geometry[i], geometry[i + 1], geometry[i + 2]);
        }

        // Extract extrinsics correction
        if (extrinsicsCorrection.size() == 16) {
            Matrix4d ec;
            for (int i = 0; i < 16; i++) {
                ec(i % 4,(int)(i / 4.0)) = extrinsicsCorrection[i];
            }
            m_extrinsicsCorrection = Isometry3d(ec);
        }
        else LOG << "Extrinsics correction matrix had incorrect size of " << extrinsicsCorrection.size();

        // Create the IR image processor and pose tracker
        m_irTracker = std::make_shared<IrTracker>(512, 512, (verbose) ? IrTracker::LogLevel::VeryVerbose : IrTracker::LogLevel::Silent);
        m_poseTracker = std::make_shared<PoseTracker>(m_irTracker, points, markerDiameter, true);
    }

    void MarkerTracker::SetROI(int x, int y, int w) {
        m_irTracker->setROI(x, y, w, w);
    }

    void MarkerTracker::SetJumpSettings(bool doFilter, float threshold, int nFrames) {
        m_poseTracker->setJumpSettings(doFilter, threshold, nFrames);
    }

    void MarkerTracker::SetParams(int minArea, int maxArea, int binThreshold, float convexity, float circularity, float smoothing, bool contours, bool saveIrImages, bool saveDepthImages, bool saveLeftImages, bool saveRightImages, bool saveRaw) {
        if (m_irTracker)
            m_irTracker->setSearchParams(minArea, maxArea, binThreshold, convexity, circularity, (contours) ? IrTracker::DetectionMode::Contour : IrTracker::DetectionMode::Blob);
        if (m_poseTracker)
            m_poseTracker->setSmoothing(smoothing);
        
        m_saveIrImages = saveIrImages;
        m_saveDepthImages = saveDepthImages;
        m_saveLeftImages = saveLeftImages;
        m_saveRightImages = saveRightImages;
        m_saveRaw = saveRaw;
    }

    void MarkerTracker::SetExtrinsicsOffset(array_view<float const> ext) {
        if (ext.size() == 16) {
            // Convert the correction          
            Matrix4d ec;
            for (int i = 0; i < 16; i++) {
                ec(i % 4,(int)(i / 4.0)) = ext[i];
            }
            m_extrinsicsCorrection = Isometry3d(ec);

            // Also get the actual extrinsics
            Matrix4d T;
            XMFLOAT4X4 tmp;
            XMStoreFloat4x4(&tmp, m_depthCameraPoseInvMatrix);
            for (int i = 0; i < 4; i++)
                for (int j = 0; j < 4; j++)
                    T(j, i) = tmp.m[i][j];

            // Set it to the product
            auto xt = Isometry3d(ec * T);
            m_irTracker->setCameraExtrinsics(xt);
        }
        else LOG << "Extrinsics correction matrix had incorrect size of " << ext.size();
    }

    bool MarkerTracker::HasNewPose() {
        return m_poseTracker->hasNewPose();
    }

    com_array<double> MarkerTracker::GetObjectPose() {
        auto pose = m_poseTracker->getPose();

        auto arr = com_array<double>(16);
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 4; j++)
                arr[i * 4 + j] = pose(i, j);

        return arr;
    }

    com_array<double> MarkerTracker::GetObjectPoseAndMarkers() {
        auto pose = m_poseTracker->getLastMeasurement();

        auto arr = com_array<double>(16 + pose.markerPositions.size()*3);
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 4; j++)
                arr[i * 4 + j] = pose.pose.matrix()(i, j);

        for (int i = 0; i < pose.markerPositions.size(); i++) {
            arr[16 + i * 3] = pose.markerPositions[0][0];
            arr[16 + i * 3 + 1] = pose.markerPositions[0][1];
            arr[16 + i * 3 + 2] = pose.markerPositions[0][2];
        }

        return arr;
    }

    void MarkerTracker::SetDevicePose(array_view<float const> pose) {
        std::scoped_lock<std::mutex> lck(m_poseMutex);

        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 4; j++)
                m_devicePose(i,j) = pose[i * 4 + j];
    }

    void MarkerTracker::InitializeDepthSensor()
    {
        for (auto sensorDescriptor : m_sensorDescriptors)
        {
            if (sensorDescriptor.sensorType == DEPTH_AHAT)
            {
                winrt::check_hresult(m_pSensorDevice->GetSensor(sensorDescriptor.sensorType, &m_depthSensor));
                winrt::check_hresult(m_depthSensor->QueryInterface(IID_PPV_ARGS(&m_pDepthCameraSensor)));
                winrt::check_hresult(m_pDepthCameraSensor->GetCameraExtrinsicsMatrix(&m_depthCameraPose));
                m_depthCameraPoseInvMatrix = XMMatrixInverse(nullptr, XMLoadFloat4x4(&m_depthCameraPose));
                m_depthCameraPoseMatrix = XMLoadFloat4x4(&m_depthCameraPose);

                // Set the world_T_camera matrix
                Matrix4d T;
                XMFLOAT4X4 tmp; // Will be row-major
                XMStoreFloat4x4(&tmp, m_depthCameraPoseInvMatrix);
                for (int i = 0; i < 4; i++)
                    for (int j = 0; j < 4; j++)
                        T(j,i) = tmp.m[i][j]; // Transpose it as well
                auto ext = Isometry3d(m_extrinsicsCorrection * T);
                m_irTracker->setCameraExtrinsics(ext);

                // Set the camera intrinsics
                m_irTracker->setCameraIntrinsics([this](const std::array<double, 2>& UV, std::array<double, 2>& XY) {
                    float uv[2] = { UV[0], UV[1] };
                    float xy[2] = { 0,0 };
                    m_pDepthCameraSensor->MapImagePointToCameraUnitPlane(uv, xy);
                    XY[0] = xy[0]; XY[1] = xy[1];
                });

                m_irTracker->setCameraBoundaries(depthCamRoi.kRowLower * 512, depthCamRoi.kRowUpper * 512, depthCamRoi.kColLower * 512, depthCamRoi.kColUpper * 512, depthCamRoi.depthNearClip, depthCamRoi.depthFarClip);

                break;
            }
        }
    }

    void MarkerTracker::InitializeStereoCamerasFront()
    {
        for (auto sensorDescriptor : m_sensorDescriptors)
        {
            if (sensorDescriptor.sensorType == LEFT_FRONT)
            {
                winrt::check_hresult(m_pSensorDevice->GetSensor(sensorDescriptor.sensorType, &m_LFSensor));
                winrt::check_hresult(m_LFSensor->QueryInterface(IID_PPV_ARGS(&m_LFCameraSensor)));
                winrt::check_hresult(m_LFCameraSensor->GetCameraExtrinsicsMatrix(&m_LFCameraPose));
                m_LFCameraPoseInvMatrix = XMMatrixInverse(nullptr, XMLoadFloat4x4(&m_LFCameraPose));
            }
            if (sensorDescriptor.sensorType == RIGHT_FRONT)
            {
                winrt::check_hresult(m_pSensorDevice->GetSensor(sensorDescriptor.sensorType, &m_RFSensor));
                winrt::check_hresult(m_RFSensor->QueryInterface(IID_PPV_ARGS(&m_RFCameraSensor)));
                winrt::check_hresult(m_RFCameraSensor->GetCameraExtrinsicsMatrix(&m_RFCameraPose));
                m_RFCameraPoseInvMatrix = XMMatrixInverse(nullptr, XMLoadFloat4x4(&m_RFCameraPose));
            }
        }
    }


    void MarkerTracker::StartDepthSensorLoop()
    {
        m_pDepthUpdateThread = new std::thread(MarkerTracker::DepthSensorLoop, this);
    }

    long MarkerTracker::GetTimeUs() {
        auto now_us = std::chrono::time_point_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now());
        auto value = now_us.time_since_epoch();
        return value.count();
    }

    void MarkerTracker::DepthSensorLoop(MarkerTracker* pMarkerTracker)
    {
        // prevent starting loop for multiple times
        if (!pMarkerTracker->m_depthSensorLoopStarted)
        {
            pMarkerTracker->m_depthSensorLoopStarted = true;
        }
        else return;

        pMarkerTracker->m_depthSensor->OpenStream();

        try
        {
            UINT64 lastTs = 0;
            while (pMarkerTracker->m_depthSensorLoopStarted)
            {
                // Request a sensor frame
                IResearchModeSensorFrame* pDepthSensorFrame = nullptr;
                ResearchModeSensorResolution resolution;
                pMarkerTracker->m_depthSensor->GetNextBuffer(&pDepthSensorFrame);
                pDepthSensorFrame->GetResolution(&resolution);
                pMarkerTracker->m_depthResolution = resolution;
                IResearchModeSensorDepthFrame* pDepthFrame = nullptr;
                winrt::check_hresult(pDepthSensorFrame->QueryInterface(IID_PPV_ARGS(&pDepthFrame)));

                // Get the raw depth and IR buffers
                size_t outBufferCount = 0;
                const UINT16* pDepth = nullptr;
                pDepthFrame->GetBuffer(&pDepth, &outBufferCount);
                pMarkerTracker->m_depthBufferSize = outBufferCount;
                size_t outAbBufferCount = 0;
                const UINT16* pAbImage = nullptr;
                pDepthFrame->GetAbDepthBuffer(&pAbImage, &outAbBufferCount);

                // Get the timestamp and ignore the frame if we already used it
                ResearchModeSensorTimestamp timestamp;
                pDepthSensorFrame->GetTimeStamp(&timestamp);
                if (timestamp.HostTicks == lastTs) continue;
                lastTs = timestamp.HostTicks;

                // Get tracking transform -> This seems quite faulty. There is not even a constant transform from this to the Unity pose
                /*Windows::Perception::Spatial::SpatialLocation t2World = nullptr;
                auto ts = PerceptionTimestampHelper::FromSystemRelativeTargetTime(HundredsOfNanoseconds(checkAndConvertUnsigned(timestamp.HostTicks)));
                t2World = pMarkerTracker->m_locator.TryLocateAtTimestamp(ts, pMarkerTracker->m_refFrame);
                if (t2World == nullptr) continue;
                auto p = t2World.Position(); Vector3d pos(p.x, p.y, p.z);
                auto q = t2World.Orientation(); Quaterniond rot(q.w, q.x, q.y, q.z);
                Isometry3d t2w; t2w.fromPositionOrientationScale(pos, rot, Vector3d::Ones());*/

                // Get the current pose from Unity
                Isometry3d transToWorld;
                {
                    std::scoped_lock<std::mutex> lck(pMarkerTracker->m_poseMutex);
                    transToWorld = Isometry3d(pMarkerTracker->m_devicePose);
                }

                // Make unique pointers and pass them for pose computation. Can we do without copying?
                auto irIm = std::make_unique<std::vector<uint16_t>>(outAbBufferCount);
                memcpy(irIm->data(), pAbImage, outAbBufferCount);
                auto depthIm = std::make_unique<std::vector<uint16_t>>(outBufferCount);
                memcpy(depthIm->data(), pDepth, outBufferCount);
                pMarkerTracker->m_poseTracker->update(transToWorld, std::move(irIm), std::move(depthIm));
                
                // Process and save the depth images
                if (pMarkerTracker->m_saveDepthImages) {
                    // Allocate space for the processed buffers
                    auto pDepthTexture = std::make_unique<uint8_t[]>(outBufferCount);

                    // Save raw depth map
                    if (pMarkerTracker->m_saveRaw) 
                    {
                        auto pRawDepthTexture = std::make_unique<uint16_t[]>(outBufferCount);
                        for (UINT i = 0; i < resolution.Height; i++)
                        {
                            for (UINT j = 0; j < resolution.Width; j++)
                            {
                                auto idx = resolution.Width * i + j;

                                // Ignore pixels outside camera ROI
                                if (i > pMarkerTracker->depthCamRoi.kRowUpper * 512 || i < pMarkerTracker->depthCamRoi.kRowLower * 512 || j > pMarkerTracker->depthCamRoi.kColUpper * 512 || j < pMarkerTracker->depthCamRoi.kColLower * 512) {
                                    pRawDepthTexture.get()[idx] = 0;
                                    continue;
                                }

                                // Save depth map as grayscale texture pixel into temp buffer
                                UINT16 depth = pDepth[idx];
                                if (depth > 4090) {
                                    pRawDepthTexture.get()[idx] = 0;
                                    continue;
                                }
                                else if (depth > pMarkerTracker->depthCamRoi.depthFarClip) depth = pMarkerTracker->depthCamRoi.depthFarClip;
                                else if (depth < pMarkerTracker->depthCamRoi.depthNearClip) depth = pMarkerTracker->depthCamRoi.depthNearClip;
                                pRawDepthTexture.get()[idx] = depth - pMarkerTracker->depthCamRoi.depthNearClip;
                            }
                        }

                        std::scoped_lock<std::mutex> l(pMarkerTracker->m_depthImageMutex);

                        if (!pMarkerTracker->m_depthMap)
                            pMarkerTracker->m_depthMap = new UINT16[outBufferCount];
                        memcpy(pMarkerTracker->m_depthMap, pRawDepthTexture.get(), outBufferCount * sizeof(UINT16));
                        pRawDepthTexture.reset();
                    }
                    // Save processed depth map texture
                    else
                    {
                        // Process the data to an actual useful value (loses some information)
                        for (UINT i = 0; i < resolution.Height; i++)
                        {
                            for (UINT j = 0; j < resolution.Width; j++)
                            {
                                auto idx = resolution.Width * i + j;

                                // save depth map as grayscale texture pixel into temp buffer
                                UINT16 depth = pDepth[idx];
                                depth = (depth > 4090) ? 0 : depth - pMarkerTracker->m_depthOffset;
                                if (depth == 0) { pDepthTexture.get()[idx] = 0; }
                                else { pDepthTexture.get()[idx] = (uint8_t)((float)depth / 1000 * 255); }
                            }
                        }

                        // lock the mutex after processing to avoid holding it for long
                        std::scoped_lock<std::mutex> l(pMarkerTracker->m_depthImageMutex);

                        if (!pMarkerTracker->m_depthMapTexture)
                            pMarkerTracker->m_depthMapTexture = new UINT8[outBufferCount];
                        memcpy(pMarkerTracker->m_depthMapTexture, pDepthTexture.get(), outBufferCount * sizeof(UINT8));
                    }
                    
                    pMarkerTracker->m_depthMapTextureUpdated = true;
                    pDepthTexture.reset();
                }

                // Process and save the images
                if (pMarkerTracker->m_saveIrImages) {
                    auto pAbTexture = std::make_unique<uint8_t[]>(outAbBufferCount);

                    // Save raw IR image
                    if (pMarkerTracker->m_saveRaw) {
                        // Reclaim the same depth mutex. Locking a different mutex here would open the potential for deadlocks
                        std::scoped_lock<std::mutex> l(pMarkerTracker->m_depthImageMutex);

                        if (!pMarkerTracker->m_shortAbImage)
                            pMarkerTracker->m_shortAbImage = new UINT16[outBufferCount];
                        memcpy(pMarkerTracker->m_shortAbImage, pAbImage, outBufferCount * sizeof(UINT16));
                    }
                    // Save processed IR image
                    else
                    {
                        // Process the data to a usable value (loses some information)
                        for (UINT i = 0; i < resolution.Height; i++)
                        {
                            for (UINT j = 0; j < resolution.Width; j++)
                            {
                                auto idx = resolution.Width * i + j;

                                // Convert AbImage to grayscale texture pixel in temp buffer
                                UINT16 abValue = pAbImage[idx];
                                uint8_t processedAbValue = 0;
                                if (abValue > 1000) { processedAbValue = 0xFF; }
                                else { processedAbValue = (uint8_t)((float)abValue / 1000 * 255); }

                                pAbTexture.get()[idx] = processedAbValue;
                            }
                        }

                        // Acquire the mutex after processing to avoid holding it for long
                        // Reclaim the same depth mutex. Locking a different mutex here would open the potential for deadlocks
                        std::scoped_lock<std::mutex> l(pMarkerTracker->m_depthImageMutex);

                        if (!pMarkerTracker->m_shortAbImageTexture)
                            pMarkerTracker->m_shortAbImageTexture = new UINT8[outBufferCount];
                        memcpy(pMarkerTracker->m_shortAbImageTexture, pAbTexture.get(), outBufferCount * sizeof(UINT8));
                    }
                    
                    pMarkerTracker->m_shortAbImageTextureUpdated = true;
                }

                // release space
                if (pDepthFrame)
                    pDepthFrame->Release();
                if (pDepthSensorFrame)
                    pDepthSensorFrame->Release();
            }
        }
        catch (std::exception e) {
            debugLog(std::string(e.what()));
        }
        pMarkerTracker->m_depthSensor->CloseStream();
        pMarkerTracker->m_depthSensor->Release();
        pMarkerTracker->m_depthSensor = nullptr;
    }

    void MarkerTracker::StartStereoCamerasFrontLoop()
    {
        if (m_refFrame == nullptr)
        {
            m_refFrame = m_locator.GetDefault().CreateStationaryFrameOfReferenceAtCurrentLocation().CoordinateSystem();
        }

        m_pSpatialCamerasFrontUpdateThread = new std::thread(MarkerTracker::SpatialCamerasFrontLoop, this);
    }

    void MarkerTracker::SpatialCamerasFrontLoop(MarkerTracker* pMarkerTracker)
    {
        // prevent starting loop for multiple times
        if (!pMarkerTracker->m_spatialCamerasFrontLoopStarted)
        {
            pMarkerTracker->m_spatialCamerasFrontLoopStarted = true;
        }
        else {
            return;
        }

        pMarkerTracker->m_LFSensor->OpenStream();
        pMarkerTracker->m_RFSensor->OpenStream();

        try
        {
            while (pMarkerTracker->m_spatialCamerasFrontLoopStarted)
            {
                IResearchModeSensorFrame* pLFCameraFrame = nullptr;
                IResearchModeSensorFrame* pRFCameraFrame = nullptr;
                ResearchModeSensorResolution LFResolution;
                ResearchModeSensorResolution RFResolution;
                pMarkerTracker->m_LFSensor->GetNextBuffer(&pLFCameraFrame);
                pMarkerTracker->m_RFSensor->GetNextBuffer(&pRFCameraFrame);

                // process sensor frame
                pLFCameraFrame->GetResolution(&LFResolution);
                pMarkerTracker->m_LFResolution = LFResolution;
                pRFCameraFrame->GetResolution(&RFResolution);
                pMarkerTracker->m_RFResolution = RFResolution;

                IResearchModeSensorVLCFrame* pLFFrame = nullptr;
                winrt::check_hresult(pLFCameraFrame->QueryInterface(IID_PPV_ARGS(&pLFFrame)));
                IResearchModeSensorVLCFrame* pRFFrame = nullptr;
                winrt::check_hresult(pRFCameraFrame->QueryInterface(IID_PPV_ARGS(&pRFFrame)));

                size_t LFOutBufferCount = 0;
                const BYTE* pLFImage = nullptr;
                pLFFrame->GetBuffer(&pLFImage, &LFOutBufferCount);
                pMarkerTracker->m_LFbufferSize = LFOutBufferCount;
                size_t RFOutBufferCount = 0;
                const BYTE* pRFImage = nullptr;
                pRFFrame->GetBuffer(&pRFImage, &RFOutBufferCount);
                pMarkerTracker->m_RFbufferSize = RFOutBufferCount;

                // get tracking transform
                ResearchModeSensorTimestamp timestamp_left, timestamp_right;
                pLFCameraFrame->GetTimeStamp(&timestamp_left);
                pRFCameraFrame->GetTimeStamp(&timestamp_right);

                auto ts_left = PerceptionTimestampHelper::FromSystemRelativeTargetTime(HundredsOfNanoseconds(checkAndConvertUnsigned(timestamp_left.HostTicks)));
                auto ts_right = PerceptionTimestampHelper::FromSystemRelativeTargetTime(HundredsOfNanoseconds(checkAndConvertUnsigned(timestamp_right.HostTicks)));

                // uncomment the block below if their transform is needed
                /*auto rigToWorld_l = pMarkerTracker->m_locator.TryLocateAtTimestamp(ts_left, pMarkerTracker->m_refFrame);
                auto rigToWorld_r = rigToWorld_l;
                if (ts_left.TargetTime() != ts_right.TargetTime()) {
                    rigToWorld_r = pMarkerTracker->m_locator.TryLocateAtTimestamp(ts_right, pMarkerTracker->m_refFrame);
                }

                if (rigToWorld_l == nullptr || rigToWorld_r == nullptr)
                {
                    continue;
                }

                auto LfToWorld = pMarkerTracker->m_LFCameraPoseInvMatrix * SpatialLocationToDxMatrix(rigToWorld_l);
                auto RfToWorld = pMarkerTracker->m_RFCameraPoseInvMatrix * SpatialLocationToDxMatrix(rigToWorld_r);*/

                if (pMarkerTracker->m_saveLeftImages || pMarkerTracker->m_saveRightImages) {
                    std::scoped_lock<std::mutex> l(pMarkerTracker->m_stereoImageMutex);

                    if (pMarkerTracker->m_saveLeftImages) {
                        pMarkerTracker->m_lastSpatialFrame.LFFrame.timestamp = timestamp_left.HostTicks;
                        pMarkerTracker->m_lastSpatialFrame.LFFrame.timestamp_ft = ts_left.TargetTime().time_since_epoch().count();

                        if (!pMarkerTracker->m_lastSpatialFrame.LFFrame.image)
                            pMarkerTracker->m_lastSpatialFrame.LFFrame.image = new UINT8[LFOutBufferCount];
                        memcpy(pMarkerTracker->m_lastSpatialFrame.LFFrame.image, pLFImage, LFOutBufferCount * sizeof(UINT8));

                        pMarkerTracker->m_LFImageUpdated = true;
                    }

                    if (pMarkerTracker->m_saveRightImages)
                    {
                        pMarkerTracker->m_lastSpatialFrame.RFFrame.timestamp = timestamp_right.HostTicks;    
                        pMarkerTracker->m_lastSpatialFrame.RFFrame.timestamp_ft = ts_right.TargetTime().time_since_epoch().count();
                        if (!pMarkerTracker->m_lastSpatialFrame.RFFrame.image)
                            pMarkerTracker->m_lastSpatialFrame.RFFrame.image = new UINT8[RFOutBufferCount];
                        memcpy(pMarkerTracker->m_lastSpatialFrame.RFFrame.image, pRFImage, RFOutBufferCount * sizeof(UINT8));

                        pMarkerTracker->m_RFImageUpdated = true;
                    }
                }

                // release space
                if (pLFFrame) pLFFrame->Release();
                if (pRFFrame) pRFFrame->Release();
                if (pLFCameraFrame) pLFCameraFrame->Release();
                if (pRFCameraFrame) pRFCameraFrame->Release();
            }
        }
        catch (...) {}
        pMarkerTracker->m_LFSensor->CloseStream();
        pMarkerTracker->m_LFSensor->Release();
        pMarkerTracker->m_LFSensor = nullptr;

        pMarkerTracker->m_RFSensor->CloseStream();
        pMarkerTracker->m_RFSensor->Release();
        pMarkerTracker->m_RFSensor = nullptr;
    }

    void MarkerTracker::CamAccessOnComplete(ResearchModeSensorConsent consent)
    {
        camAccessCheck = consent;
        SetEvent(camConsentGiven);
    }

    inline bool MarkerTracker::DepthMapUpdated() { return m_depthMapTextureUpdated; }

    inline bool MarkerTracker::IrImageUpdated() { return m_shortAbImageTextureUpdated; }

    inline bool MarkerTracker::LfImageUpdated() { return m_LFImageUpdated; }

    inline bool MarkerTracker::RfImageUpdated() { return m_RFImageUpdated; }

    // Stop the sensor loop and release buffer space.
    // Sensor object should be released at the end of the loop function
    void MarkerTracker::StopAllSensorDevice()
    {
        m_depthSensorLoopStarted = false;
        //m_pDepthUpdateThread->join();
        if (m_depthMap)
        {
            delete[] m_depthMap;
            m_depthMap = nullptr;
        }
        if (m_depthMapTexture)
        {
            delete[] m_depthMapTexture;
            m_depthMapTexture = nullptr;
        }
        if (m_shortAbImage)
        {
            delete[] m_shortAbImage;
            m_shortAbImage = nullptr;
        }
        if (m_shortAbImageTexture)
        {
            delete[] m_shortAbImageTexture;
            m_shortAbImageTexture = nullptr;
        }

        if (m_lastSpatialFrame.LFFrame.image)
        {
            delete[] m_lastSpatialFrame.LFFrame.image;
            m_lastSpatialFrame.LFFrame.image = nullptr;
        }
        if (m_lastSpatialFrame.RFFrame.image)
        {
            delete[] m_lastSpatialFrame.RFFrame.image;
            m_lastSpatialFrame.RFFrame.image = nullptr;
        }

        m_pSensorDevice->Release();
        m_pSensorDevice = nullptr;
        m_pSensorDeviceConsent->Release();
        m_pSensorDeviceConsent = nullptr;
    }

    com_array<uint16_t> MarkerTracker::GetRawDepthMap()
    {
        std::lock_guard<std::mutex> l(m_depthImageMutex);
        if (!m_depthMap)
        {
            return com_array<uint16_t>();
        }
        com_array<UINT16> tempBuffer = com_array<UINT16>(m_depthMap, m_depthMap + m_depthBufferSize);

        return tempBuffer;
    }

    com_array<uint16_t> MarkerTracker::GetRawIrImage()
    {
        std::lock_guard<std::mutex> l(m_depthImageMutex);
        if (!m_shortAbImage)
        {
            return com_array<uint16_t>();
        }
        com_array<UINT16> tempBuffer = com_array<UINT16>(m_shortAbImage, m_shortAbImage + m_depthBufferSize);

        return tempBuffer;
    }

    // Get depth map texture buffer. (For visualization purpose)
    com_array<uint8_t> MarkerTracker::GetProcessedDepthMap()
    {
        std::lock_guard<std::mutex> l(m_depthImageMutex);
        if (!m_depthMapTexture)
        {
            return com_array<UINT8>();
        }
        com_array<UINT8> tempBuffer = com_array<UINT8>(std::move_iterator(m_depthMapTexture), std::move_iterator(m_depthMapTexture + m_depthBufferSize));

        m_depthMapTextureUpdated = false;
        return tempBuffer;
    }

    // Get depth map texture buffer. (For visualization purpose)
    com_array<uint8_t> MarkerTracker::GetProcessedIrImage()
    {
        std::lock_guard<std::mutex> l(m_depthImageMutex);
        if (!m_shortAbImageTexture)
        {
            return com_array<UINT8>();
        }
        com_array<UINT8> tempBuffer = com_array<UINT8>(std::move_iterator(m_shortAbImageTexture), std::move_iterator(m_shortAbImageTexture + m_depthBufferSize));

        m_shortAbImageTextureUpdated = false;
        return tempBuffer;
    }

    com_array<uint8_t> MarkerTracker::GetLfImage(int64_t& ts)
    {
        std::lock_guard<std::mutex> l(m_stereoImageMutex);
        if (!m_lastSpatialFrame.LFFrame.image)
        {
            return com_array<UINT8>();
        }
        com_array<UINT8> tempBuffer = com_array<UINT8>(std::move_iterator(m_lastSpatialFrame.LFFrame.image), std::move_iterator(m_lastSpatialFrame.LFFrame.image + m_LFbufferSize));
        ts = m_lastSpatialFrame.LFFrame.timestamp_ft;
        m_LFImageUpdated = false;
        return tempBuffer;
    }

    com_array<uint8_t> MarkerTracker::GetRfImage(int64_t& ts)
    {
        std::lock_guard<std::mutex> l(m_stereoImageMutex);
        if (!m_lastSpatialFrame.RFFrame.image)
        {
            return com_array<UINT8>();
        }
        com_array<UINT8> tempBuffer = com_array<UINT8>(std::move_iterator(m_lastSpatialFrame.RFFrame.image), std::move_iterator(m_lastSpatialFrame.RFFrame.image + m_RFbufferSize));
        ts = m_lastSpatialFrame.RFFrame.timestamp_ft;
        m_RFImageUpdated = false;
        return tempBuffer;
    }

    com_array<uint8_t> MarkerTracker::GetLrfImages(int64_t& ts_left, int64_t& ts_right)
    {
        std::lock_guard<std::mutex> l(m_stereoImageMutex);
        if (!m_lastSpatialFrame.LFFrame.image || !m_lastSpatialFrame.RFFrame.image)
        {
            return com_array<UINT8>();
        }


        UINT8* rawTempBuffer = new UINT8[m_LFbufferSize + m_RFbufferSize];
        memcpy(rawTempBuffer, m_lastSpatialFrame.LFFrame.image, m_LFbufferSize);
        memcpy(rawTempBuffer + m_LFbufferSize, m_lastSpatialFrame.RFFrame.image, m_RFbufferSize);

        com_array<UINT8> tempBuffer = com_array<UINT8>(std::move_iterator(rawTempBuffer), std::move_iterator(rawTempBuffer + m_LFbufferSize + m_RFbufferSize));
        ts_left = m_lastSpatialFrame.LFFrame.timestamp_ft;
        ts_right = m_lastSpatialFrame.RFFrame.timestamp_ft;

        m_LFImageUpdated = false;
        m_RFImageUpdated = false;
        return tempBuffer;
    }

    // Set the reference coordinate system. Need to be set before the sensor loop starts; otherwise, default coordinate will be used.
    void MarkerTracker::SetReferenceCoordinateSystem(winrt::Windows::Perception::Spatial::SpatialCoordinateSystem refCoord)
    {
        m_refFrame = refCoord;
    }

    long long MarkerTracker::checkAndConvertUnsigned(UINT64 val)
    {
        assert(val <= kMaxLongLong);
        return static_cast<long long>(val);
    }

    XMMATRIX MarkerTracker::SpatialLocationToDxMatrix(SpatialLocation location) {
        auto rot = location.Orientation();
        auto quatInDx = XMFLOAT4(rot.x, rot.y, rot.z, rot.w);
        auto rotMat = XMMatrixRotationQuaternion(XMLoadFloat4(&quatInDx));
        auto pos = location.Position();
        auto posMat = XMMatrixTranslation(pos.x, pos.y, pos.z);
        return rotMat * posMat;
    }
}
