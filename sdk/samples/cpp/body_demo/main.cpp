#include <map>
#include <fstream>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <api/Service.h>

#include "ConsoleArgumentsParser.h"

using Context = api::Context;
std::vector<std::pair<std::string, std::string>> bone_map = {
	{"right_ankle","right_knee"},
	{"right_knee","right_hip"},
	{"left_hip","right_hip"},
	{"left_shoulder","left_hip"},
	{"right_shoulder","right_hip"},
	{"left_shoulder","right_shoulder"},
	{"left_shoulder","left_elbow"},
	{"right_shoulder","right_elbow"},
	{"left_elbow","left_wrist"},
	{"right_elbow","right_wrist"},
	{"left_eye","right_eye"},
	{"nose","left_eye"},
	{"left_knee", "left_hip"},
	{"right_ear", "right_shoulder"},
	{"left_ear", "left_shoulder"},
	{"right_eye", "right_ear"},
	{"left_eye", "left_ear"},
	{"nose", "right_eye"},
	{"left_ankle", "left_knee"}
};

void cvMatToBSM(Context& bsmCtx, const cv::Mat& image, bool copy = false);

void demoBody(api::Service& service, const std::string& input_image_path, const std::string& mode, const std::string& output);

void displayResultInWindow(Context& ioData, cv::Mat& image, std::string mode, bool output, const std::string& input_image_path);

int main(int argc, char** argv)
{

	std::cout << "usage: " << argv[0] <<
		" [--mode detection | pose | reidentification]"
		" [--input_image <path to image>]"
		" [--sdk_path ..]"
		" [--output <yes/no>]"
		<< std::endl;

	ConsoleArgumentsParser parser(argc, argv);
	const std::string mode 					= parser.get<std::string>("--mode", "pose");
	const std::string input_image_path 		= parser.get<std::string>("--input_image");
	const std::string sdk_dir            	= parser.get<std::string>("--sdk_path", "..");
	const std::string output 				= parser.get<std::string>("--output", "yes");

	api::Service service = api::Service::createService(sdk_dir);

	try {
		if (mode == "detection" || mode == "reidentification" || mode == "pose") {
			demoBody(service, input_image_path, mode, output);
		}
		else {
			std::cout << "Incorrect mode\n";
		}
	}
	catch (const std::exception& e) {
		std::cout << "! exception catched: '" << e.what() << "' ... exiting" << std::endl;
		return 1;
	}

	return 0;
}

static const std::map<int, std::string> CvTypeToStr{ {CV_8U,"uint8_t"}, {CV_8S,"int8_t"}, {CV_16U,"uint16_t"}, {CV_16S,"int16_t"},
													{CV_32S,"int32_t"}, {CV_32F,"float"}, {CV_64F,"double"} };

void cvMatToBSM(Context& bsmCtx, const cv::Mat& image, bool copy)
{
	const cv::Mat& input_img = image.isContinuous() ? image : image.clone(); // setDataPtr requires continuous data
	size_t copy_sz = (copy || !image.isContinuous()) ? input_img.total() * input_img.elemSize() : 0;
	bsmCtx["format"] = "NDARRAY";
	bsmCtx["blob"].setDataPtr(input_img.data, copy_sz);
	bsmCtx["dtype"] = CvTypeToStr.at(input_img.depth());
	for (int i = 0; i < input_img.dims; ++i)
		bsmCtx["shape"].push_back(input_img.size[i]);
	bsmCtx["shape"].push_back(input_img.channels());
}

void demoBody(api::Service& service, const std::string& input_image_path, const std::string& mode, const std::string& output) {

	Context modelCtx = service.createContext();
	modelCtx["unit_type"] = "HUMAN_BODY_DETECTOR";

	api::ProcessingBlock bodyDetector = service.createProcessingBlock(modelCtx);

	cv::Mat image = cv::imread(input_image_path, cv::IMREAD_COLOR);
	cv::Mat input_image;
	if (image.channels() == 3)
		cv::cvtColor(image, input_image, cv::COLOR_BGR2RGB);
	else
		input_image = image.clone();

	Context imgCtx = service.createContext();
	cvMatToBSM(imgCtx, input_image);
	Context ioData = service.createContext();
	ioData["image"] = imgCtx;

	///////////////////////////
	bodyDetector(ioData);    //
	if (mode == "reidentification") {
		Context bodyReidCtx = service.createContext();
		bodyReidCtx["unit_type"] = "BODY_RE_IDENTIFICATION";

		api::ProcessingBlock bodyReidentification = service.createProcessingBlock(bodyReidCtx);
		bodyReidentification(ioData);
	}
	else if (mode == "pose") {
		Context poseCtx = service.createContext();
		poseCtx["unit_type"] = "POSE_ESTIMATOR";

		api::ProcessingBlock poseEstimator = service.createProcessingBlock(poseCtx);

		poseEstimator(ioData);
	}
	///////////////////////////
	displayResultInWindow(ioData, image, mode, (output == "yes"), input_image_path);
}

void displayResultInWindow(Context& ioData, cv::Mat& image, std::string mode, bool output, const std::string& input_image_path) {
	std::cout << "Bbox coordinates: (x1, y1, x2, y2)" << std::endl;

	for (const Context& obj : ioData["objects"])
	{
		if (obj["class"].getString().compare("body"))
			continue;

		const Context& rectCtx = obj.at("bbox");
		cv::Point topLeft = { static_cast<int>(rectCtx[0].getDouble() * image.size[1]), static_cast<int>(rectCtx[1].getDouble() * image.size[0]) };
		cv::Point bottomRight = { static_cast<int>(rectCtx[2].getDouble() * image.size[1]), static_cast<int>(rectCtx[3].getDouble() * image.size[0]) };
		cv::Rect rect(topLeft, bottomRight);
		
		cv::rectangle(image, rect, { 0, 255, 0 }, 1);

		if (output) 
		{
			std::cout << "Bbox coordinates: (" << topLeft.x << ", " << topLeft.y << ", " << bottomRight.x << ", " << bottomRight.y << ")" << std::endl;
		}

		if (mode == "pose") {
			const Context& posesCtx = obj["keypoints"];

			if (output)
			{
				std::cout << "Keypoints: ";
			}

			for (auto& bone : bone_map) {
				std::string key1 = bone.first;
				std::string key2 = bone.second;
				int x1 = posesCtx[key1]["proj"][0].getDouble() * image.size[1];
				int y1 = posesCtx[key1]["proj"][1].getDouble() * image.size[0];
				int x2 = posesCtx[key2]["proj"][0].getDouble() * image.size[1];
				int y2 = posesCtx[key2]["proj"][1].getDouble() * image.size[0];

				cv::line(image, cv::Point{ x1, y1 }, cv::Point{ x2,y2 }, cv::Scalar(0, 255, 0), 1, cv::LINE_4);
				
				if (output)
				{
					std::cout << '(' << x1 << ", " << y1 << ", " << x2 << ", " << y2 << ") ";
				}
			}

			if (output)
			{
				std::cout << std::endl;
			}

			for (auto ptr = posesCtx.begin(); ptr != posesCtx.end(); ++ptr) {
				auto proj = (*ptr)["proj"];
				cv::circle(image, cv::Point{
							static_cast<int>(proj[0].getDouble() * image.size[1]),
							static_cast<int>(proj[1].getDouble() * image.size[0]) },
							3, cv::Scalar(0, 0, 255), -1, false);
			}

		}
	}

	if (mode == "reidentification")
	{
		Context outputData = ioData["output_data"];
		size_t index = input_image_path.find_last_of("/\\");
		size_t extension = input_image_path.rfind('.');

		if (index != std::string::npos)
		{
			index += 1;
		}
		else
		{
			index = 0;
		}

		Context templateData = outputData["template"];
		uint64_t templateSize = outputData["template_size"].getUnsignedLong();
		std::string templateName = std::string(input_image_path.begin() + index, input_image_path.begin() + extension) + ".txt";
		std::ofstream templateFile(templateName);

		templateFile << templateSize << std::endl;
		
		for (size_t i = 0; i < templateSize; i++)
		{
			templateFile << templateData[i].getDouble();

			if (i + 1 != templateSize)
			{
				templateFile << ' ';
			}
		}

		if (output)
		{
			std::cout << "Template saved in " << templateName << " file" << std::endl;
		}
	}

	cv::imshow("result", image);
	cv::waitKey();
	cv::destroyAllWindows();
}