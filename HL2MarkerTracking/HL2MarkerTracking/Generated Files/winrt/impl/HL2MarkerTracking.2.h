// WARNING: Please don't edit this file. It was generated by C++/WinRT v2.0.220531.1

#pragma once
#ifndef WINRT_HL2MarkerTracking_2_H
#define WINRT_HL2MarkerTracking_2_H
#include "winrt/impl/HL2MarkerTracking.1.h"
WINRT_EXPORT namespace winrt::HL2MarkerTracking
{
    struct __declspec(empty_bases) MarkerTracker : winrt::HL2MarkerTracking::IMarkerTracker
    {
        MarkerTracker(std::nullptr_t) noexcept {}
        MarkerTracker(void* ptr, take_ownership_from_abi_t) noexcept : winrt::HL2MarkerTracking::IMarkerTracker(ptr, take_ownership_from_abi) {}
        MarkerTracker(array_view<float const> geometry, array_view<float const> extrinsicsCorrection, bool verbose);
    };
}
#endif