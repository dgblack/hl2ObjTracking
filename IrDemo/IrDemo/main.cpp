#include <IrMarkerTracking/IrTracker.h>
#include <IrMarkerTracking/PoseTracker.h>
#include <Eigen/Dense>

#include <fstream>
#include <istream>

std::vector<double> parseRow(std::string row) {
	std::vector<double> out;
	while (true) {
		int idx = row.find(",");
		if (idx == std::string::npos) {
			out.push_back(stod(row));
			break;
		}
		out.push_back(stod(row.substr(0, idx)));
		row = row.substr(idx + 1);
	}
	return out;
}

int main(void) {
	std::string path = "D:/TeleSonics/Instrumentation/IrTracking/IrDemo/"; // CHANGE THIS!

	// ---- Set up the marker geometry ----
	Vector3d marker1(0.03455, 0.01295, 0.01797);
	Vector3d marker2(0.01721, -0.01669, 0.00796);
	Vector3d marker3(-0.01569, -0.01049, -0.01109);
	Vector3d marker4(-0.02375, 0.0304, -0.01573);
	auto geom = std::vector<Vector3d>{marker1, marker2, marker3, marker4};

	// ---- Read in HoloLens intrinsics look-up table ----
	std::vector<std::vector<double>> lutX;
	std::vector<std::vector<double>> lutY;
	std::ifstream intX(path + "/data/xIntrinsics.csv");
	std::ifstream intY(path + "/data/yIntrinsics.csv");
	std::string row;
	while (!intX.eof()) {
		std::getline(intX, row);
		if (intX.bad() || intX.fail()) {
			break;
		}
		lutX.push_back(parseRow(row));
	}
	while (!intY.eof()) {
		std::getline(intY, row);
		if (intY.bad() || intY.fail()) {
			break;
		}
		lutY.push_back(parseRow(row));
	}

	// ---- Start the IR tracker ----
	auto irTracker = std::make_shared<IrTracker>(512, 512, IrTracker::LogLevel::VeryVerbose);
	irTracker->setSearchParams(2, 1000, 222, 0.75f, 0.6f, IrTracker::DetectionMode::Blob);
	
	// Set the camera intrinsics function (this is just a made-up example)
	irTracker->setCameraIntrinsics([&](const std::array<double, 2>& uv, std::array<double, 2>& xy) {
		xy[0] = lutX[uv[0]][uv[1]];
		xy[1] = lutY[uv[0]][uv[1]];
	});

	// Set the camera extrinsics
	irTracker->setCameraExtrinsics(Isometry3d::Identity());

	// Set the camera region from which pixels should be used. Pixels outside are ignored
	irTracker->setCameraBoundaries(0, 511, 0, 511, 100, 900);

	// ---- Start the pose calculator ----
	PoseTracker poseTracker(irTracker, geom, 0.011, false);
	poseTracker.setJumpSettings(false, 0.1, 4);
	poseTracker.setSmoothing(0);

	// ---- Read in images ----
	// Just for this example. Otherwise access the camera hardware
	for (int i = 0; i < 11; i++) {
		auto irFile = path + "data/ir_" + std::to_string(i) + ".bin";
		auto depthFile = path + "data/dp_" + std::to_string(i) + ".bin";
		std::ifstream irIn(irFile.c_str(), std::ios::binary);
		std::ifstream depthIn(depthFile.c_str(), std::ios::binary);
		std::vector<uint8_t> irIm(std::istreambuf_iterator<char>(irIn), {});
		std::vector<uint8_t> depthIm(std::istreambuf_iterator<char>(depthIn), {});

		if (irIm.size() != 512 * 512) {
			std::cout << "Read in incorrect number of bytes: " << irIm.size() << "bytes. Skipping image." << std::endl;
			std::cout << "Did you change the path at the start of the main function?" << std::endl;
			continue;
		}
		
		// Convert the images to the format they would have had when raw
		auto irIm16 = std::make_unique<std::vector<uint16_t>>(irIm.size());
		auto depthIm16 = std::make_unique<std::vector<uint16_t>>(depthIm.size());
		for (size_t i = 0; i < irIm.size(); i++) {
			(*irIm16)[i] = irIm[i] * 1000.0 / 255;
			(*depthIm16)[i] = depthIm[i] * 1000.0 / 255;
		}

		irIn.close();
		depthIn.close();
		std::cout << "loaded image " << i << std::endl;

		// ---- Pass to the pose tracker ----
		// Usually do this in a thread that accesses the camera sensor
		poseTracker.update(Isometry3d::Identity(), std::move(irIm16), std::move(depthIm16));

		std::cout << "Press enter to get pose" << std::endl;
		std::getchar();

		// ---- Request the last measured pose ----
		// Usually call getPose in a separate thread and just continue if no pose is available. Eg. in the graphics or haptics loop on can use the measured pose
		/*PoseTracker::IRPose mes = poseTracker.getLastMeasurement();
		while (mes.pose.matrix().isIdentity()) {
			mes = poseTracker.getLastMeasurement();
		}
		std::cout << "Computed pose" << std::endl;
		std::cout << mes.pose.matrix() << std::endl;*/

		for (int i = 0; i < 10; i++) {
			if (poseTracker.hasNewPose()) break;
			using namespace std::chrono_literals;
			std::this_thread::sleep_for(10ms);
		}
		if (i == 10) { 
			std::cout << "Failed to compute pose" << std::endl; 
		}
		else {
			Matrix4d T = poseTracker.getPose();
			//PoseTracker::IRPose mes = poseTracker.getLastMeasurement();
			std::cout << "Computed pose" << std::endl;
			std::cout << T << std::endl;
			//std::cout << mes.pose.matrix() << std::endl;
		}
	}

	std::cout << "Done.";
	std::getchar();

	return 0;
}
