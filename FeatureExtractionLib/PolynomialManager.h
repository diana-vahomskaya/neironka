#pragma once

#include <opencv2/opencv.hpp>
#include <vector>

namespace fe {
	typedef std::vector<std::vector<cv::Mat>> OrthoBasis;

	class PolynomialManager
	{
	protected:
		OrthoBasis polynomials;
	public:
		virtual void Decompose() = 0;
		virtual void Recovery() = 0;
		virtual OrthoBasis GetBasis();
	};
};

