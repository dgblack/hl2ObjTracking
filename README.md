# hl2ObjTracking
6-DOF, IR marker-based object pose tracking for the Microsoft HoloLens 2 with a square-root unscented Kalman filter from:

*David Black and Septimiu Salcudean*. Robust Object Pose Tracking for Augmented Reality Guidance and Teleoperation. IEEE Transactions on Instrumentation and Measurement. [DOI: 10.1109/TIM.2024.3398108](https://doi.org/10.1109/TIM.2024.3398108)

I have not had time to port the Kalman filter yet, but the rest is here. I did everything in Visual Studio 2022 Community. The repo is organized as follows:
* *Deps* contains the external dependencies. For OpenCV I currently have UWP ARM and ARM64 builds for the HoloLens 2, as well as a Windows x64 build. All of these are Release. To build for other platforms or configurations, you can just add the respective DLLs/libs.
* *IrMarkerTracking* is a Visual Studio 2022 project that builds a DLL. This contains all the actual image processing and pose tracking code.
* *IrMarkerTrackingUwp* is the same thing but with some added details that make the DLL compatible with Universal Windows Platform (UWP). 
* *IrDemo* shows how you might consume the DLL in a project
* *HL2MarkerTracking* is a C++/WinRT runtime component that wraps the IrMarkerTrackingUwp DLL together with the HoloLens 2 Research Mode API into a new DLL that can be called directly form C# and Unity. Simply build this for ARM or ARM64 and drag/drop it into Unity under Assets/Plugins/WSA/[ARM or ARM64]
* *IrTracker.cs* is an example Unity C# script that shows how to consume the HL2MarkerTracking DLL
* *updateManifest.py* is a convenience script. To build a HoloLens 2 application with the HL2MarkerTracking, you will have to enable research mode and also add a couple lines to the Package.appxmanifest file before opening and building the Visual Studio solution. This Python script just adds the lines for you. You'll have to hard-code the path of the appxmanifest file in the script.

Some of the constants and settings probably have to be adjusted to work well in different setups
