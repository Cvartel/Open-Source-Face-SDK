#if defined(_WIN32)
#define NOMINMAX
#endif

#include <map>
#include <fstream>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <api/Service.h>

using Context = api::Context;

#include "ConsoleArgumentsParser.h"

template <typename T>
T clip(const T& n, const T& lower, const T& upper) {
	return std::max(lower, std::min(n, upper));
}

void recognitionSample(std::string sdk_path, std::string input_image_path1, std::string input_image_path2, std::string window, std::string output);

void detectorFitterSample(std::string sdk_path, std::string input_image_path, std::string mode, std::string window, std::string output);

int main(int argc, char **argv)
{
	std::cout << "usage: " << argv[0] <<
		" [--mode detection | landmarks | recognition]"
		" [--input_image <path to image>]"
		" [--input_image2 <path to image>]"
		" [--sdk_path ..]"
		" [--window <yes/no>]"
		" [--output <yes/no>]"
		<< std::endl;

	ConsoleArgumentsParser parser(argc, argv);
	const std::string mode          = parser.get<std::string>("--mode", "detection");
	const std::string input_image_path   = parser.get<std::string>("--input_image");
	const std::string sdk_dir            = parser.get<std::string>("--sdk_path", "..");
	const std::string window			 = parser.get<std::string>("--window", "yes");
	const std::string output              = parser.get<std::string>("--output", "yes");

	try{
		if (mode == "recognition"){
			const std::string input_image_path2   = parser.get<std::string>("--input_image2");
			recognitionSample(sdk_dir, input_image_path, input_image_path2, window, output);
		}else if(mode == "detection" || mode == "landmarks"){
			detectorFitterSample(sdk_dir, input_image_path, mode, window, output);
		}else{
			std::cout << "Incorrect mode\n";
		}
	}catch(const std::exception &e){
		std::cout << "! exception catched: '" << e.what() << "' ... exiting" << std::endl;
		return 1;
	}

	return 0;
}

static const std::map<int, std::string> CvTypeToStr{{CV_8U,"uint8_t"}, {CV_8S,"int8_t"}, {CV_16U,"uint16_t"}, {CV_16S,"int16_t"},
													{CV_32S,"int32_t"}, {CV_32F,"float"}, {CV_64F,"double"}};

void cvMatToBSM(api::Context& bsmCtx, const cv::Mat& img, bool copy=false)
{
	const cv::Mat& input_img = img.isContinuous() ? img : img.clone(); // setDataPtr requires continuous data
	size_t copy_sz = (copy || !img.isContinuous()) ? input_img.total()*input_img.elemSize() : 0;
	bsmCtx["format"] = "NDARRAY";
	bsmCtx["blob"].setDataPtr(input_img.data, copy_sz);
	bsmCtx["dtype"] = CvTypeToStr.at(input_img.depth());
	for(int i = 0; i < input_img.dims; ++i)
		bsmCtx["shape"].push_back(input_img.size[i]);
	bsmCtx["shape"].push_back(input_img.channels());
}

void drawBBox(const api::Context& obj, cv::Mat& image, std::string output, cv::Scalar color=cv::Scalar(0, 255, 0)){
	const api::Context& rectCtx = obj.at("bbox");
	cv::Rect rect(cv::Point{static_cast<int>(rectCtx[0].getDouble()*image.cols), static_cast<int>(rectCtx[1].getDouble()*image.rows)},
				  cv::Point{static_cast<int>(rectCtx[2].getDouble()*image.cols), static_cast<int>(rectCtx[3].getDouble()*image.rows)});
	cv::rectangle(image, rect, color, 2);
	if (output == "yes")
		std::cout << "BBox coordinates: " << static_cast<int>(rectCtx[0].getDouble()*image.cols) << ", " << static_cast<int>(rectCtx[1].getDouble()*image.rows) << ", " << static_cast<int>(rectCtx[2].getDouble()*image.cols) << ", " << static_cast<int>(rectCtx[3].getDouble()*image.rows) << "\n";
}

void drawPoints(const api::Context& obj, cv::Mat& img, std::string output){
	const api::Context& rectCtx = obj.at("bbox");
	double width = rectCtx[2].getDouble()*img.cols - rectCtx[0].getDouble()*img.cols;
	double height = rectCtx[3].getDouble()*img.rows - rectCtx[1].getDouble()*img.rows;

	int point_size = width * height > 320 * 480 ? 3 : 1;

	const api::Context& keypoints = obj["keypoints"];
	for(const api::Context& point : keypoints["points"])
		cv::circle(img, cv::Point2f(point["x"].getDouble() * img.cols, point["y"].getDouble() * img.rows), 1, {0, 255, 0}, point_size);
	cv::circle(img, cv::Point2f(keypoints["left_eye"]["proj"][0].getDouble() * img.cols, keypoints["left_eye"]["proj"][1].getDouble() * img.rows), 1, {0, 0, 255}, point_size);
	cv::circle(img, cv::Point2f(keypoints["right_eye"]["proj"][0].getDouble() * img.cols, keypoints["right_eye"]["proj"][1].getDouble() * img.rows), 1, {0, 0, 255}, point_size);
	cv::circle(img, cv::Point2f(keypoints["mouth"]["proj"][0].getDouble() * img.cols, keypoints["mouth"]["proj"][1].getDouble() * img.rows), 1, {0, 0, 255}, point_size);
	if (output == "yes")
		std::cout << "left eye(" << keypoints["left_eye"]["proj"][0].getDouble() * img.cols << ", " << keypoints["left_eye"]["proj"][1].getDouble() * img.rows << "), right eye(" << keypoints["right_eye"]["proj"][0].getDouble() * img.cols << ", " << keypoints["right_eye"]["proj"][1].getDouble() * img.rows << "), mouth(" << keypoints["mouth"]["proj"][0].getDouble() * img.cols << ", " << keypoints["mouth"]["proj"][1].getDouble() * img.rows << ")\n";
}

cv::Mat getCrop(const api::Context& obj, cv::Mat &image)
{
	const api::Context& rectCtx = obj.at("bbox");
	int x1 = rectCtx[0].getDouble() * image.cols;
	int y1 = rectCtx[1].getDouble() * image.rows;
	int x2 = rectCtx[2].getDouble() * image.cols;
	int y2 = rectCtx[3].getDouble() * image.rows;

	cv::Point bboxTopLeft = { clip(x1, 0, image.cols), clip(y1, 0 , image.rows) };
	cv::Point bboxBottomRight = { clip(x2, 0, image.cols), clip(y2, 0 , image.rows) };

	return image(cv::Rect(bboxTopLeft, bboxBottomRight));
}

void getObjectsWithMaxConfidence(api::Context& data, api::Context& result){
	double max_confidence = 0;
	int index_max_confidence = 0;
	for (int i = 0; i < data["objects"].size(); i++)
	{
		if(data["objects"][i]["class"].getString().compare("face"))
			continue;

		double confidence = data["objects"][i]["confidence"].getDouble();
		if (confidence > max_confidence)
			index_max_confidence = i;
	}

	result["objects"].push_back(data["objects"][index_max_confidence]);
	result["image"] = data["image"];
}

void checkFileExist(std::string path)
{
	std::ifstream file(path.c_str());
	if (!file.is_open())
		throw std::runtime_error("file " + path + "  not open");
	file.close();
}

void recognitionSample(std::string sdk_path, std::string input_image_path1, std::string input_image_path2, std::string window, std::string output)
{
	api::Service service = api::Service::createService(sdk_path);

	Context detectorCtx = service.createContext();
	Context fitterCtx = service.createContext();
	Context recognizerCtx = service.createContext();
	Context matcherCtx = service.createContext();

	detectorCtx["unit_type"] = "FACE_DETECTOR";
	fitterCtx["unit_type"] = "FITTER";
	recognizerCtx["unit_type"] = "FACE_RECOGNIZER";
	matcherCtx["unit_type"] = "MATCHER_MODULE";

	api::ProcessingBlock faceDetector = service.createProcessingBlock(detectorCtx);
	api::ProcessingBlock meshFitter = service.createProcessingBlock(fitterCtx);
	api::ProcessingBlock recognizerModule = service.createProcessingBlock(recognizerCtx);
	api::ProcessingBlock matcherModule = service.createProcessingBlock(matcherCtx);

	checkFileExist(input_image_path1);
	checkFileExist(input_image_path2);

	cv::Mat image = cv::imread(input_image_path1, cv::IMREAD_COLOR);
	cv::Mat image2 = cv::imread(input_image_path2, cv::IMREAD_COLOR);

	cv::Mat input_image;
	cv::Mat input_image2;

	cv::cvtColor(image, input_image, cv::COLOR_BGR2RGB);
	cv::cvtColor(image2, input_image2, cv::COLOR_BGR2RGB);

	Context ioData = service.createContext();
	Context ioData2 = service.createContext();

	Context imgCtx = service.createContext();
	Context imgCtx2 = service.createContext();

	cvMatToBSM(imgCtx, input_image);
	cvMatToBSM(imgCtx2, input_image2);

	ioData["image"] = imgCtx;
	ioData2["image"] = imgCtx2;

	///////////Detector////////////////
	faceDetector(ioData);
	faceDetector(ioData2);
	///////////////////////////////////

	if (!ioData["objects"].size())
		throw std::runtime_error("no face detected on " + input_image_path1 + " image");

	if (!ioData2["objects"].size())
		throw std::runtime_error("no face detected on " + input_image_path2 + " image");


	Context obj1 = service.createContext();
	Context obj2 = service.createContext();
	getObjectsWithMaxConfidence(ioData, obj1);
	getObjectsWithMaxConfidence(ioData2, obj2);

	///////////Fitter////////////////
	meshFitter(obj1);
	meshFitter(obj2);
	/////////////////////////////////

	///////////Recognizer////////////////
	recognizerModule(obj1);
	recognizerModule(obj2);
	/////////////////////////////////////

	Context matcherData = service.createContext();
	matcherData["verification"]["objects"].push_back(obj1["objects"][0]);
	matcherData["verification"]["objects"].push_back(obj2["objects"][0]);

	///////////Matcher////////////////
	matcherModule(matcherData);
	//////////////////////////////////

	double distance = matcherData["verification"]["result"]["distance"].getDouble();
	bool verdict = matcherData["verification"]["result"]["verdict"].getBool();

	cv::Scalar color = verdict ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255);
	drawBBox(obj1["objects"][0], image, output, color);
	drawBBox(obj2["objects"][0], image2, output, color);

	cv::Mat crop1 = getCrop(obj1["objects"][0], image);
	cv::Mat crop2 = getCrop(obj2["objects"][0], image2);

	cv::resize(crop1, crop1, cv::Size(320, 480));
	cv::resize(crop2, crop2, cv::Size(320, 480));
	cv::Mat3b result(480, 640, cv::Vec3b(0,0,0));

	crop1.copyTo(result(cv::Rect(0, 0, crop1.cols, crop1.rows)));
	crop2.copyTo(result(cv::Rect(crop1.cols, 0, crop2.cols, crop2.rows)));

	std::cout << "distance = " << distance << "\n";
	std::cout << "verdict = " << (verdict ? "True" : "False") << "\n";

	if (window == "yes"){
		cv::imshow("result", result);
		cv::waitKey();
		cv::destroyWindow("result");
	}
}

void detectorFitterSample(std::string sdk_path, std::string input_image_path, std::string mode, std::string window, std::string output)
{
	api::Service service = api::Service::createService(sdk_path);

	Context detectorCtx = service.createContext();
	detectorCtx["unit_type"] = "FACE_DETECTOR";

	api::ProcessingBlock faceDetector = service.createProcessingBlock(detectorCtx);

	checkFileExist(input_image_path);
	cv::Mat image = cv::imread(input_image_path, cv::IMREAD_COLOR);

	Context ioData = service.createContext();
	Context imgCtx = service.createContext();

	cv::Mat input_image;
	cv::cvtColor(image, input_image, cv::COLOR_BGR2RGB);

	cvMatToBSM(imgCtx, input_image);
	ioData["image"] = imgCtx;
	///////////////////////////
	faceDetector(ioData);    //
	///////////////////////////

	if (mode == "landmarks"){
		Context fitterCtx = service.createContext();
		fitterCtx["unit_type"] = "FITTER";

		api::ProcessingBlock meshFitter = service.createProcessingBlock(fitterCtx);

		meshFitter(ioData);
	}

	for(const Context& obj : ioData["objects"])
	{
		if(obj["class"].getString().compare("face"))
			continue;

		drawBBox(obj, image, output);
		if (mode == "landmarks")
			drawPoints(obj, image, output);
	}

	if (window == "yes"){
		cv::imshow("result", image);
		cv::waitKey();
		cv::destroyWindow("result");
	}
}