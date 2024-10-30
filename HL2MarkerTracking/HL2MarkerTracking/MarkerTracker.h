#pragma once

#include "MarkerTracker.g.h"
#include "TimeConverter.h"
#include "IrMarkerTracking.h" // For non-ARM, replace with IrTracker.h
#include "PoseTracker.h"

#include "ResearchModeAPI.h"

namespace winrt::HL2MarkerTracking::implementation
{
    struct MarkerTracker : MarkerTrackerT<MarkerTracker>
    {
        /// Create the MarkerTracker object
        /// @param geometry the marker locations in local coordinates, stored as [x1, y1, z1, x2, y2, z2, ...]
        /// @param extrinsicsCorrection a homogeneous matrix for correcting the extrinsics of the ResearchMode API. Applied from the left to the extrinsics matrix. Pass as 16-element array in row-major form
        /// @param markerDiameter the diameter (in metres) of the IR reflective spheres
        /// @param verbose tru to print a lot of debug information or false to print only errors
        MarkerTracker(array_view<float const> geometry, array_view<float const> extrinsicsCorrection, float markerDiameter, bool verbose = false);
        
        /// Set ROI within which to search for the markers
        /// @param x the column of the centre point of the ROI
        /// @param y the row of the centre points of the ROI
        /// @param w the width and height of the ROI (in pixels)
        void SetROI(int x, int y, int w);

        /// Set computer vision parameters
        /// @param minArea the minimum area blob to consider as a potential marker (units of pixels)
        /// @param maxArea the maximum area blob to consider as a potential marker (units of pixels)
        /// @param binThreshold the threshold for image binarization (values > binThreshold are set to 255 and others are set to 0)
        /// @param convexity the minimum convexity blob to consider as a potential marker, measured as (area) / (area of convex hull)
        /// @param circularity the minimum circularity blob to consider as a potential marker, measured as (area) / (area of minimum enclosing circle)
        /// @param smoothing the low-pass filtering factor. 0 does no filtering and 1 always keeps only the old value. Chose a value between
        /// @param contours use contours if true or blob detection if false. Blob detection is much slower but has fewer outliers
        /// @param saveIrImages store the IR reflectivity image so it is accessible from Unity
        /// @param saveDepthImages store the depth map buffer so it is accessible from Unity
        /// @param saveLeftImages store the left front-facing stereo image so it is accessible from Unity
        /// @param saveRightImages store the right front-facing stereo image so it is accessible from Unity
        /// @param saveRaw saves the raw 16-bit depth and IR images if true. Otherwise compresses to 8-bit
        void SetParams(int minArea, int maxArea, int binThreshold, float convexity, float circularity, float smoothing, bool contours, bool saveIrImages, bool saveDepthImages, bool saveLeftImages, bool saveRightImages, bool saveRaw);
        
        /// Turn on/off filtering for larger jumps in marker position, and adjust the settings
        /// @param doFilter activate/deactiveate the filter
        /// @param threhsold the distance (metres) above which an inter-frame change in position is considered a jump. If filtering is on, this measurement will be ignored
        /// @param nFrames when the position has jumped but stays in the new position for nFrames frames, the new position is considered correct
        void SetJumpSettings(bool doFilter, float threshold, int nFrames);

        /// @param ext a homogeneous matrix for correcting the extrinsics of the ResearchMode API. Applied from the left to the extrinsics matrix. Pass as 16-element array in row-major form
        void SetExtrinsicsOffset(array_view<float const> ext);

        bool HasNewPose();

        /// Get the most recent pose of the tracked object
        /// @returns 16-element array containing the homogeneous matrix in row-major order
        com_array<double> GetObjectPose();

        /// Get the most recent pose of the tracked object as well the raw positions of the measured markers
        /// @returns 16-element array containing the homogeneous matrix in row-major order, followed by the marker positions as 3-vectors
        com_array<double> GetObjectPoseAndMarkers();

        /// Update the device (eg. MR headset) pose relative to the world.
        /// Used to compute the points in world coordinates and should be called every frame.
        /// @param pose the homogeneous transform as a 16-element array in row-major form
        void SetDevicePose(array_view<float const> pose);

        /// Call this before StartDepthSensorLoop()
        void InitializeDepthSensor();
        /// Call this before StartStereoCamerasFrontLoop()
        void InitializeStereoCamerasFront();

        /// Call this after InitializeDepthSensor
        void StartDepthSensorLoop();
        /// Call this after InitializeStereoCamerasFront()
        void StartStereoCamerasFrontLoop();

        void StopAllSensorDevice();

        /// Query if the depth image buffer has been updated with a new image
        bool DepthMapUpdated();
        /// Query if the IR reflectivity image buffer has been updated with a new image
        bool IrImageUpdated();
        /// Query if the left front stereo image buffer has been updated with a new image
        bool LfImageUpdated();
        /// Query if the right front stereo image buffer has been updated with a new image
        bool RfImageUpdated();

        void SetReferenceCoordinateSystem(Windows::Perception::Spatial::SpatialCoordinateSystem refCoord);

        /// Get the buffer containing the raw depth map
        com_array<uint16_t> GetRawDepthMap();
        /// Get the buffer containing the processed depth map
        com_array<uint8_t> GetProcessedDepthMap();
        /// Get the buffer containing the raw IR reflectivity image
        com_array<uint16_t> GetRawIrImage();
        /// Get the buffer containing the processed IR reflectivity image
        com_array<uint8_t> GetProcessedIrImage();
        /// Get the buffer containing the left front stereo image
        com_array<uint8_t> GetLfImage(int64_t& ts);
        /// Get the buffer containing the right front stereo image
        com_array<uint8_t> GetRfImage(int64_t& ts);
        /// Get the buffer containing the right and left front stereo images (first left, then right in the same array)
        com_array<uint8_t> GetLrfImages(int64_t& ts_left, int64_t& ts_right);
    private:
        UINT16* m_depthMap = nullptr;
        UINT8* m_depthMapTexture = nullptr;
        UINT16* m_shortAbImage = nullptr;
        UINT8* m_shortAbImageTexture = nullptr;

        UINT8* m_LFImage = nullptr;
        UINT8* m_RFImage = nullptr;

        std::mutex m_depthImageMutex;
        std::mutex m_poseMutex;
        std::mutex m_stereoImageMutex;
        bool m_saveIrImages = false;
        bool m_saveDepthImages = false;
        bool m_saveLeftImages = false;
        bool m_saveRightImages = false;
        bool m_saveRaw = false;
        bool m_contours = true;
        int m_binThreshold = 200;
        int m_maxArea = 1000;
        int m_minArea = 2;
        float m_convexity = 0.8f;
        float m_circularity = 0.6f;

        std::shared_ptr<IrTracker> m_irTracker;
        std::shared_ptr<PoseTracker> m_poseTracker;
        Eigen::Isometry3d m_devicePose;
        Eigen::Isometry3d m_extrinsicsCorrection;

        long GetTimeUs();

        IResearchModeSensor* m_depthSensor = nullptr;
        IResearchModeCameraSensor* m_pDepthCameraSensor = nullptr;
        IResearchModeSensor* m_LFSensor = nullptr;
        IResearchModeCameraSensor* m_LFCameraSensor = nullptr;
        IResearchModeSensor* m_RFSensor = nullptr;
        IResearchModeCameraSensor* m_RFCameraSensor = nullptr;
        ResearchModeSensorResolution m_depthResolution;
        ResearchModeSensorResolution m_LFResolution;
        ResearchModeSensorResolution m_RFResolution;
        IResearchModeSensorDevice* m_pSensorDevice = nullptr;
        std::vector<ResearchModeSensorDescriptor> m_sensorDescriptors;
        IResearchModeSensorDeviceConsent* m_pSensorDeviceConsent = nullptr;
        Windows::Perception::Spatial::SpatialLocator m_locator = 0;
        Windows::Perception::Spatial::SpatialCoordinateSystem m_refFrame = nullptr;
        std::atomic_int m_depthBufferSize = 0;
        std::atomic_int m_LFbufferSize = 0;
        std::atomic_int m_RFbufferSize = 0;
        std::atomic_uint16_t m_centerDepth = 0;
        float m_centerPoint[3]{ 0,0,0 };

        std::atomic_bool m_depthSensorLoopStarted = false;
        std::atomic_bool m_spatialCamerasFrontLoopStarted = false;

        std::atomic_bool m_depthMapTextureUpdated = false;
        std::atomic_bool m_shortAbImageTextureUpdated = false;
        std::atomic_bool m_useRoiFilter = false;
        std::atomic_bool m_LFImageUpdated = false;
        std::atomic_bool m_RFImageUpdated = false;

        //float m_roiBound[3]{ 0,0,0 };
        //float m_roiCenter[3]{ 0,0,0 };
        static void DepthSensorLoop(MarkerTracker* poseTrack);
        static void SpatialCamerasFrontLoop(MarkerTracker* poseTrack);
        static void CamAccessOnComplete(ResearchModeSensorConsent consent);
        DirectX::XMFLOAT4X4 m_depthCameraPose;
        DirectX::XMMATRIX m_depthCameraPoseInvMatrix;
        DirectX::XMMATRIX m_depthCameraPoseMatrix;
        DirectX::XMFLOAT4X4 m_LFCameraPose;
        DirectX::XMMATRIX m_LFCameraPoseInvMatrix;
        DirectX::XMFLOAT4X4 m_RFCameraPose;
        DirectX::XMMATRIX m_RFCameraPoseInvMatrix;
        std::thread* m_pDepthUpdateThread;
        std::thread* m_pSpatialCamerasFrontUpdateThread;
        static long long checkAndConvertUnsigned(UINT64 val);
        static DirectX::XMMATRIX SpatialLocationToDxMatrix(Windows::Perception::Spatial::SpatialLocation location);
        struct DepthCamRoi {
            float kRowLower = 0.2;
            float kRowUpper = 0.55;
            float kColLower = 0.3;
            float kColUpper = 0.7;
            UINT16 depthNearClip = 200; // Unit: mm
            UINT16 depthFarClip = 800;
        } depthCamRoi;
        UINT16 m_depthOffset = 0;

        TimeConverter m_converter;

        struct Frame {
            UINT64 timestamp; // QPC 
            int64_t timestamp_ft; // FileTime
            UINT8* image = nullptr;
        };

        struct SpatialCameraFrame {
            Frame LFFrame;
            Frame RFFrame;
        } m_lastSpatialFrame;

    };
}
namespace winrt::HL2MarkerTracking::factory_implementation
{
    struct MarkerTracker : MarkerTrackerT<MarkerTracker, implementation::MarkerTracker>
    {
    };
}
