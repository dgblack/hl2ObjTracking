#include <IrMarkerTracking/IrTracker.h>
#include <IrMarkerTracking/PoseTracker.h>
#include <Eigen/Dense>

#include <fstream>

int main(void) {
	// ---- Set up the marker geometry ----
	Vector3d marker1(0.03455, 0.01295, 0.01797);
	Vector3d marker2(0.01721, -0.01669, 0.00796);
	Vector3d marker3(-0.01569, -0.01049, -0.01109);
	Vector3d marker4(-0.02375, 0.0304, -0.01573);
	auto geom = std::vector<Vector3d>{marker1, marker2, marker3, marker4};

	// ---- Start the IR tracker ----
	auto irTracker = std::make_shared<IrTracker>(512, 512, IrTracker::LogLevel::Verbose);
	irTracker->setSearchParams(2, 1000, 222, 0.75f, 0.6f, IrTracker::DetectionMode::Contour);
	
	// Set the camera intrinsics function (this is just a made-up example)
	irTracker->setCameraIntrinsics([](const std::array<double, 2>& uv, std::array<double, 2>& xy) {
		xy[0] = uv[0] / 512;
		xy[1] = uv[1] / 512;
	});

	// Set the camera extrinsics
	irTracker->setCameraExtrinsics(Isometry3d::Identity());

	// Set the camera region from which pixels should be used. Pixels outside are ignored
	irTracker->setCameraBoundaries(0, 511, 0, 511, 100, 900);

	// ---- Start the pose calculator ----
	PoseTracker poseTracker(irTracker, geom, false);
	poseTracker.setJumpSettings(true, 0.1, 4);
	poseTracker.setSmoothing(0);

	// ---- Read in images ----
	// Just for this example. Otherwise access the camera hardware
	int i = 0;
	std::string path = "./";
	auto irFile = path + "data/ir_" + std::to_string(i) + ".bin";
	auto depthFile = path + "data/dp_" + std::to_string(i) + ".bin";
	std::ifstream irIn(irFile.c_str(), std::ios::binary);
	std::ifstream depthIn(depthFile.c_str(), std::ios::binary);
	std::vector<uint8_t> irIm(std::istreambuf_iterator<char>(irIn), {});
	std::vector<uint8_t> depthIm(std::istreambuf_iterator<char>(depthIn), {});

	// Convert the images to the format they would have had when raw
	auto irIm16 = std::make_unique<std::vector<uint16_t>>(irIm.size());
	auto depthIm16 = std::make_unique<std::vector<uint16_t>>(depthIm.size());
	for (size_t i = 0; i < irIm.size(); i++) {
		(*irIm16)[i] = irIm[i] * 1000.0 / 255;
		(*depthIm16)[i] = depthIm[i] * 1000.0 / 255;
	}

	std::cout << "loaded images" << std::endl;

	// ---- Pass to the pose tracker ----
	// Usually do this in a thread that accesses the camera sensor
	poseTracker.update(Isometry3d::Identity(), std::move(irIm16), std::move(depthIm16));

	std::cout << "waiting for computed pose" << std::endl;

	// ---- Request the last measured pose ----
	// Usually call getPose in a separate thread and just continue if no pose is available. Eg. in the graphics or haptics loop on can use the measured pose
	Matrix4d T = poseTracker.getPose();
	while (T.isIdentity()) {
		T = poseTracker.getPose();
	}

	std::cout << "Computed pose" << std::endl;
	std::cout << T << std::endl;

	std::getchar();

	return 0;
}