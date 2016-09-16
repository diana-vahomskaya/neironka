#pragma once

#include <opencv2/opencv.hpp>
#include <vector>

namespace fe {
	class BlobProcessor
	{
	public:
		virtual void DetectBlobs(cv::Mat image, std::vector<cv::Mat> blobs) = 0;

		virtual void NormalizeBlobs(
			std::vector<cv::Mat> & blobs,
			std::vector<cv::Mat> & normalized_blobs
			) = 0;

		virtual std::string GetType() = 0;
	};
};