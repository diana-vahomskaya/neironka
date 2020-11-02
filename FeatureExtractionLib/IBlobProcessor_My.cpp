#define FEATURE_DLL_EXPORT
#define MIDDLE_LEVEL 127
#define MAX_LEVEL 255

#include "IBlobProcessor.h"
#include "FeatureExtraction.h"
#include <iostream>
#include <vector>

#define M_PI 3.1415926535897932
using namespace fe;
using namespace cv;
using namespace std;

class BlobProcessor :public IBlobProcessor
{
	virtual string GetType() override
	{
		return "Simple Blob Processor by Vahomskaya Diana";

	}
	virtual std::vector<cv::Mat> DetectBlobs(cv::Mat image) override
	{
		//throw std::runtime_error("ќшибка");

		vector <Mat> res;
		Mat binary(image.rows, image.cols, CV_8UC1);
		threshold(image, binary, 127, 255, THRESH_BINARY_INV);
		imshow("binary", binary);
		//waitKey(0);
		vector<vector<Point>> contours;
		vector<Vec4i> hierarchy;
		findContours(binary, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);
		Mat drawing = Mat::zeros(binary.size(), CV_8UC3);
		RNG rng(0xFFFFFFFF);
		for (int idx = 0; idx >= 0; idx = hierarchy[idx][0])
		{
			Scalar colour = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
			drawContours(drawing, contours, idx, colour, CV_FILLED, LineTypes::LINE_8, hierarchy);
		}

		for (int idx = 0; idx >= 0; idx = hierarchy[idx][0])
		{
			Point2f center;
			float radius;
			minEnclosingCircle(contours[idx], center, radius);
			res.push_back(Mat::zeros(round((double)radius) * 2, round((double)radius) * 2, CV_8UC1));
			Scalar white = Scalar(255, 255, 255);
			Point2f offset = -center + Point2f(radius, radius);
			drawContours(res.back(), contours, idx, white, CV_FILLED, cv::LineTypes::LINE_8, hierarchy, 255, offset);
			//	imshow("Symbol", res.back());
			//	waitKey(0);
		}
		return res;
	}
	virtual std::vector<cv::Mat> NormalizeBlobs(
		std::vector<cv::Mat>& blobs, int side) override
	{
		//	throw std::runtime_error("ќшибка");
		vector<Mat> normalized;
		for (auto& blob : blobs)
		{
			Mat m;
			resize(blob, m, Size(side, side));
			normalized.push_back(m);
			//imshow("normalized", m);
			//waitKey(0);

		}
		return normalized;
	}
	
};

std::shared_ptr<fe::IBlobProcessor> fe::CreateBlobProcessor()
{
	return make_shared<BlobProcessor>();
}