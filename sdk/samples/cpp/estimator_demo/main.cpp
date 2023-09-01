#if defined(_WIN32)
#define NOMINMAX
#endif

#include <unordered_map>
#include <functional>
#include <fstream>
#include <iostream>
#include <vector>
#include <stdexcept>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <api/Service.h>

#include "ConsoleArgumentsParser.h"

#define ESTIMATE(unitType, outputContext) \
{ \
	api::Context estimatorContext = service.createContext(); \
	estimatorContext["unit_type"] = unitType; \
	service.createProcessingBlock(estimatorContext)(outputContext); \
}

using Context = api::Context;

constexpr int fontFace = cv::FONT_HERSHEY_SIMPLEX;
constexpr int thickness = 1;
constexpr int precision = 5;
constexpr double fontScale = 0.3;
const cv::Scalar color = CV_RGB(0, 255, 0);
const std::vector<std::string> allModes = { "all", "age", "gender", "emotion", "liveness", "mask", "glasses", "eye_openness" };

void detectorFitterSample(const std::string& sdkPath, const std::string& inputImagePath, const std::string& mode, const std::string& window);

void checkFileExist(const std::string& path);

std::string getAllModes(char delimiter);

int main(int argc, char** argv)
{
	std::streambuf* coutBuffer = std::cout.rdbuf();

	try
	{
		std::cout << "usage: " << argv[0] <<
			" [--mode " << getAllModes('|') << "]"
			" [--input_image <path to image>]"
			" [--sdk_path ..]"
			" [--window <yes/no>]"
			" [--output <yes/no>]"
			<< std::endl;

		ConsoleArgumentsParser parser(argc, argv);
		const std::string mode = parser.get<std::string>("--mode", "all");
		const std::string inputImagePath = parser.get<std::string>("--input_image");
		const std::string sdkPath = parser.get<std::string>("--sdk_path", "..");
		const std::string window = parser.get<std::string>("--window", "yes");
		std::ostringstream stream;

		if (std::find(allModes.begin(), allModes.end(), mode) == allModes.end())
		{
			throw std::invalid_argument("--mode should be: " + getAllModes('|'));
		}

		checkFileExist(inputImagePath);

		if (parser.get<std::string>("--output", "yes") == "no")
		{
			std::cout.rdbuf(stream.rdbuf());
		}

		detectorFitterSample(sdkPath, inputImagePath, mode, window);
	}
	catch (const std::exception& e)
	{
		std::cerr << e.what() << std::endl;
	}

	std::cout.rdbuf(coutBuffer);

	return 0;
}

void cvMatToBSM(api::Context& bsmContex, const cv::Mat& image, bool copy = false)
{
	static const std::unordered_map<int, std::string> cvTypeToStr =
	{
		{ CV_8U, "uint8_t" },
		{ CV_8S, "int8_t" },
		{ CV_16U, "uint16_t" },
		{ CV_16S, "int16_t" },
		{ CV_32S, "int32_t" },
		{ CV_32F, "float" },
		{ CV_64F, "double" }
	};

	const cv::Mat& inputImage = image.isContinuous() ? image : image.clone();
	size_t copySize = (copy || !image.isContinuous()) ? inputImage.total() * inputImage.elemSize() : 0;

	bsmContex["format"] = "NDARRAY";
	bsmContex["blob"].setDataPtr(inputImage.data, copySize);
	bsmContex["dtype"] = cvTypeToStr.at(inputImage.depth());

	for (int i = 0; i < inputImage.dims; i++)
	{
		bsmContex["shape"].push_back(inputImage.size[i]);
	}

	bsmContex["shape"].push_back(inputImage.channels());
}

std::pair<cv::Point, cv::Point> drawBBox(const api::Context& object, cv::Mat& image)
{
	const api::Context& rectContex = object.at("bbox");
	cv::Point first = cv::Point(static_cast<int>(rectContex[0].getDouble() * image.cols), static_cast<int>(rectContex[1].getDouble() * image.rows));
	cv::Point second = cv::Point(static_cast<int>(rectContex[2].getDouble() * image.cols), static_cast<int>(rectContex[3].getDouble() * image.rows));

	cv::rectangle
	(
		image,
		cv::Rect
		(
			first,
			second
		),
		color,
		thickness
	);

	return { first, second };
}

void checkFileExist(const std::string& path)
{
	if (!std::ifstream(path.data()).is_open())
	{
		throw std::runtime_error("file " + path + "  not open");
	}
}

std::string getAllModes(char delimiter)
{
	std::string result;

	for (size_t i = 0; i < allModes.size(); i++)
	{
		result += allModes[i];

		if (i + 1 != allModes.size())
		{
			result.
				append(1, ' ').
				append(1, delimiter).
				append(1, ' ');
		}
	}

	return result;
}

std::string applyPrecision(double value)
{
	std::ostringstream os;

	os << std::fixed << std::setprecision(precision) << value;

	std::string result = os.str();

	return os.str();
}

std::string boolToString(bool value)
{
	return value ? "true" : "false";
}

void ageEvaluateParse(const Context& data, std::vector<std::string>& estimationText)
{
	estimationText.emplace_back
	(
		"Age: " + std::to_string(data["age"].getLong())
	);
}

void genderEvaluateParse(const Context& data, std::vector<std::string>& estimationText)
{
	estimationText.emplace_back
	(
		"Gender: " + data["gender"].getString()
	);
}

void emotionEvaluateParse(const Context& data, std::vector<std::string>& estimationText)
{
	const Context& emotions = data.at("emotions");

	for (size_t i = 0; i < emotions.size(); i++)
	{
		estimationText.emplace_back
		(
			emotions[i]["emotion"].getString() + ": " + applyPrecision(emotions[i]["confidence"].getDouble())
		);
	}
}

void maskEvaluateParse(const Context& data, std::vector<std::string>& estimationText)
{
	estimationText.emplace_back
	(
		"Has Mask: " + boolToString(data["has_medical_mask"]["value"].getBool())
	);
}

void glassesEvaluateParse(const Context& data, std::vector<std::string>& estimationText)
{
	estimationText.emplace_back
	(
		"Has glasses: " + boolToString(data["has_glasses"].getBool())
	);

	estimationText.emplace_back
	(
		"Glasses confidence: " + applyPrecision(data["glasses_confidence"].getDouble())
	);
}

void eyeOpennessEvaluateParse(const Context& data, std::vector<std::string>& estimationText)
{
	const Context& is_left_eye_open = data["is_left_eye_open"];
	const Context& is_right_eye_open = data["is_right_eye_open"];

	estimationText.emplace_back
	(
		"Is left eye open: " + boolToString(is_left_eye_open["value"].getBool())
	);

	estimationText.emplace_back
	(
		"Is right eye open: " + boolToString(is_right_eye_open["value"].getBool())
	);

	estimationText.emplace_back
	(
		"Left eye openness: " + applyPrecision(is_left_eye_open["confidence"].getDouble())
	);

	estimationText.emplace_back
	(
		"Right eye openness: " + applyPrecision(is_right_eye_open["confidence"].getDouble())
	);
}

void livenessEvaluateParse(const Context& data, std::vector<std::string>& estimationText)
{
	const Context& liveness = data["liveness"];

	estimationText.emplace_back
	(
		"Liveness confidence: " + applyPrecision(liveness["confidence"].getDouble())
	);

	estimationText.emplace_back
	(
		"Liveness value: " + liveness["value"].getString()
	);
}

void parseAll(const Context& data, std::vector<std::string>& estimationText)
{
	ageEvaluateParse(data, estimationText);
	genderEvaluateParse(data, estimationText);
	emotionEvaluateParse(data, estimationText);
	maskEvaluateParse(data, estimationText);
	glassesEvaluateParse(data, estimationText);
	eyeOpennessEvaluateParse(data, estimationText);
	livenessEvaluateParse(data, estimationText);
}

Context imageToSDKForm(const cv::Mat& image, api::Service& service)
{
	Context data = service.createContext();
	Context imageContext = service.createContext();

	cvMatToBSM(imageContext, image);

	data["image"] = imageContext;

	return data;
}

void estimate(api::Service& service, std::string mode, Context& data)
{
	std::for_each(mode.begin(), mode.end(), [](char& c) { c = toupper(c); });

	ESTIMATE(mode + "_ESTIMATOR", data)
}

void detectorFitterSample(const std::string& sdkPath, const std::string& inputImagePath, const std::string& mode, const std::string& window)
{
	const std::unordered_map<std::string, std::function<void(const Context&, std::vector<std::string>&)>> parsers =
	{
		{ "all", parseAll },
		{ "age", ageEvaluateParse },
		{ "gender", genderEvaluateParse },
		{ "emotion", emotionEvaluateParse },
		{ "liveness", livenessEvaluateParse },
		{ "mask", maskEvaluateParse },
		{ "glasses", glassesEvaluateParse },
		{ "eye_openness", eyeOpennessEvaluateParse }
	};
	api::Service service = api::Service::createService(sdkPath);
	cv::Mat image = cv::imread(inputImagePath);
	cv::Mat inputImage;

	cv::cvtColor(image, inputImage, cv::COLOR_BGR2RGB);

	Context data = imageToSDKForm(inputImage, service);

	ESTIMATE("FACE_DETECTOR", data);

	if (mode == "eye_openness" || mode == "all")
	{
		ESTIMATE("FITTER", data);

		if (mode == "all")
		{
			for (const std::string& value : allModes)
			{
				if (value == "all")
				{
					continue;
				}

				estimate(service, value, data);
			}
		}
	}

	if (mode != "all")
	{
		estimate(service, mode, data);
	}

	std::vector<std::vector<std::string>> estimationText;
	std::vector<std::pair<cv::Point, cv::Point>> bboxes;
	int maxWidth = 0;
	int maxHeight = 0;

	for (const Context& object : data["objects"])
	{
		if (object["class"].getString().compare("face"))
		{
			continue;
		}

		std::vector<std::string> tem;
		int height = 0;

		bboxes.push_back(drawBBox(object, image));

		parsers.at(mode)(object, tem);

		for (const std::string& text : tem)
		{
			cv::Size textSize = cv::getTextSize(text, fontFace, fontScale, thickness, nullptr);

			height += textSize.height;

			maxWidth = std::max(maxWidth, textSize.width);
		}

		maxHeight = std::max(maxHeight, height);

		estimationText.push_back(std::move(tem));
	}

	cv::copyMakeBorder(image, image, maxHeight, maxHeight, maxWidth, maxWidth, cv::BORDER_CONSTANT);

	for (size_t i = 0; i < estimationText.size(); i++)
	{
		int previousHeight = 0;
		const cv::Point& first = bboxes[i].first;
		const cv::Point& second = bboxes[i].second;

		std::cout << "BBox coordinates: (" << first.x << ", " << first.y << "), (" << second.x << ", " << second.y << ')' << std::endl;

		for (const std::string& text : estimationText[i])
		{
			std::cout << text << std::endl;

			if (window == "yes")
			{
				static constexpr int offset = 2;

				cv::putText
				(
					image,
					text,
					{ second.x + maxWidth, first.y + maxHeight + previousHeight },
					fontFace,
					fontScale,
					color,
					thickness
				);

				previousHeight += offset + cv::getTextSize(text, fontFace, fontScale, thickness, nullptr).height;
			}
		}
	}

	if (window == "yes")
	{
		cv::imshow("result", image);
		cv::waitKey();
		cv::destroyWindow("result");
	}
}
