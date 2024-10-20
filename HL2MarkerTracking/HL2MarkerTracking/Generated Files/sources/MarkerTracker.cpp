#include "pch.h"
#include "MarkerTracker.h"
#include "MarkerTracker.g.cpp"

// WARNING: This file is automatically generated by a tool. Do not directly
// add this file to your project, as any changes you make will be lost.
// This file is a stub you can use as a starting point for your implementation.
//
// To add a copy of this file to your project:
//   1. Copy this file from its original location to the location where you store 
//      your other source files (e.g. the project root). 
//   2. Add the copied file to your project. In Visual Studio, you can use 
//      Project -> Add Existing Item.
//   3. Delete this comment and the 'static_assert' (below) from the copied file.
//      Do not modify the original file.
//
// To update an existing file in your project:
//   1. Copy the relevant changes from this file and merge them into the copy 
//      you made previously.
//    
// This assertion helps prevent accidental modification of generated files.
static_assert(false, "This file is generated by a tool and will be overwritten. Open this error and view the comment for assistance.");

namespace winrt::HL2MarkerTracking::implementation
{
    MarkerTracker::MarkerTracker(array_view<float const> geometry, array_view<float const> extrinsicsCorrection, bool verbose)
    {
        throw hresult_not_implemented();
    }
    com_array<uint16_t> MarkerTracker::GetRawDepthMap()
    {
        throw hresult_not_implemented();
    }
    com_array<uint8_t> MarkerTracker::GetProcessedDepthMap()
    {
        throw hresult_not_implemented();
    }
    com_array<uint16_t> MarkerTracker::GetRawIrImage()
    {
        throw hresult_not_implemented();
    }
    com_array<uint8_t> MarkerTracker::GetProcessedIrImage()
    {
        throw hresult_not_implemented();
    }
    com_array<uint8_t> MarkerTracker::GetLfImage(int64_t& ts)
    {
        throw hresult_not_implemented();
    }
    com_array<uint8_t> MarkerTracker::GetRfImage(int64_t& ts)
    {
        throw hresult_not_implemented();
    }
    com_array<uint8_t> MarkerTracker::GetLrfImages(int64_t& ts_left, int64_t& ts_right)
    {
        throw hresult_not_implemented();
    }
    void MarkerTracker::SetROI(int32_t x, int32_t y, int32_t w)
    {
        throw hresult_not_implemented();
    }
    void MarkerTracker::SetDevicePose(array_view<float const> pose)
    {
        throw hresult_not_implemented();
    }
    com_array<double> MarkerTracker::GetObjectPose()
    {
        throw hresult_not_implemented();
    }
    void MarkerTracker::SetParams(int32_t minArea, int32_t maxArea, int32_t binThreshold, float convexity, float circularity, float smoothing, bool contours, bool m_saveIrImages, bool saveDepthImages, bool saveLeftImages, bool saveRightImages, bool saveRaw)
    {
        throw hresult_not_implemented();
    }
    void MarkerTracker::SetExtrinsicsOffset(array_view<float const> ext)
    {
        throw hresult_not_implemented();
    }
    bool MarkerTracker::DepthMapUpdated()
    {
        throw hresult_not_implemented();
    }
    bool MarkerTracker::IrImageUpdated()
    {
        throw hresult_not_implemented();
    }
    bool MarkerTracker::LfImageUpdated()
    {
        throw hresult_not_implemented();
    }
    bool MarkerTracker::RfImageUpdated()
    {
        throw hresult_not_implemented();
    }
    void MarkerTracker::InitializeDepthSensor()
    {
        throw hresult_not_implemented();
    }
    void MarkerTracker::InitializeStereoCamerasFront()
    {
        throw hresult_not_implemented();
    }
    void MarkerTracker::StartDepthSensorLoop()
    {
        throw hresult_not_implemented();
    }
    void MarkerTracker::StartStereoCamerasFrontLoop()
    {
        throw hresult_not_implemented();
    }
    void MarkerTracker::StopAllSensorDevice()
    {
        throw hresult_not_implemented();
    }
    void MarkerTracker::SetReferenceCoordinateSystem(winrt::Windows::Perception::Spatial::SpatialCoordinateSystem const& refCoord)
    {
        throw hresult_not_implemented();
    }
}