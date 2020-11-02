#include "PolynomialManager.h"
#include "RadialFunctions.h"
#include "FeatureExtraction.h"
#define M_PI 3.1415926535897932
#include <limits>
#include <string>

using namespace fe;
using namespace cv;

class Walsh : public fe::PolynomialManager
{
    virtual std::string GetType()
    {
        return "Walsh Polynom by Diana Vahomskaya";
    }

    virtual void InitBasis(int n_max, int diameter) override
    {
        double delta = 2. / diameter;
        this->polynomials.resize(n_max + 1);
        for (size_t n = 0; n <= n_max; ++n)
        {
            this->polynomials[n].resize(n_max + 1);
            for (size_t i = 0; i <= n_max; ++i)
            {
                this->polynomials[n][i] = std::make_pair(cv::Mat(diameter, diameter, CV_64FC1), cv::Mat(diameter, diameter, CV_64FC1));
                this->polynomials[n][i].first.setTo(cv::Scalar(0));
                this->polynomials[n][i].second.setTo(cv::Scalar(0));
            }
            for (size_t r = 0; r < diameter / 2; ++r)
            {
                double radial = rf::RadialFunctions::Walsh(r * delta, n, n_max);
                size_t rot_count = (size_t)(2 * M_PI * r * delta * diameter) * 2;
                for (size_t th = 0; th < rot_count; ++th)
                {
                    double theta = 2 * th * M_PI / rot_count;
                    for (size_t i = 0; i <= n_max; ++i)
                    {
                        double sine = std::sin(theta * i) * radial;
                        double cosine = std::cos(theta * i) * radial;
                        double& color_re = this->polynomials[n][i].first.at<double>(r * std::cos(theta) + diameter / 2, r * std::sin(theta) + diameter / 2);
                        color_re = cosine;
                        double& color_im = this->polynomials[n][i].second.at<double>(r * std::cos(theta) + diameter / 2, r * std::sin(theta) + diameter / 2);
                        color_im = sine;
                    }
                }
            }
        }
    }
    virtual ComplexMoments Decompose(cv::Mat blob) override
    {
        ComplexMoments decomposition;
        size_t basis_mat_count = 0;
        for (size_t i = 0; i < this->polynomials.size(); ++i)
        {
            for (size_t j = 0; j < this->polynomials[i].size(); ++j)
            {
                ++basis_mat_count;
            }
        }
        decomposition.re = cv::Mat::zeros(cv::Size(1, basis_mat_count), CV_64FC1);
        decomposition.im = cv::Mat::zeros(cv::Size(1, basis_mat_count), CV_64FC1);

        cv::Mat other;
        blob.convertTo(other, CV_64FC1, 2.0 / 255.0, -1.0);

        size_t basis_idx = 0;
        for (size_t i = 0; i < this->polynomials.size(); ++i)
        {
            for (size_t j = 0; j < this->polynomials[i].size(); ++j)
            {
                double base_norm_re = this->polynomials[i][j].first.dot(this->polynomials[i][j].first);
                double base_norm_im = this->polynomials[i][j].second.dot(this->polynomials[i][j].second);
                if (abs(base_norm_re) > std::numeric_limits<double>::epsilon())
                {
                    decomposition.re.at<double>(basis_idx) = other.dot(this->polynomials[i][j].first) / base_norm_re;
                }
                if (abs(base_norm_im) > std::numeric_limits<double>::epsilon())
                {
                    decomposition.im.at<double>(basis_idx) = other.dot(this->polynomials[i][j].second) / base_norm_im;
                }
                ++basis_idx;
            }
        }
        return decomposition;
    }

    virtual cv::Mat Recovery(ComplexMoments& decomposition) override
    {
        cv::Mat recovery = cv::Mat(this->polynomials[0][0].first.rows, this->polynomials[0][0].first.cols, CV_64FC1);
        recovery.setTo(cv::Scalar(0));
        size_t basis_idx = 0;
        for (size_t i = 0; i < this->polynomials.size(); ++i)
        {
            for (size_t j = 0; j < this->polynomials[i].size(); ++j)
            {
                recovery += this->polynomials[i][j].first * decomposition.re.at<double>(basis_idx)
                    + this->polynomials[i][j].second * decomposition.im.at<double>(basis_idx);
                ++basis_idx;
            }
        }
        return recovery;
    }
};

std::shared_ptr<fe::PolynomialManager> fe::CreatePolynomialManager()
{
    return std::make_shared<Walsh>();
}