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
    virtual std::string GetType()//Получить описание объекта для работы с полиномами.
    {
        return "Walsh Polynom by Diana Vahomskaya";
    }

    virtual void InitBasis(int n_max, int diameter) override // Проинициализировать базис ортогональных полиномов ~exp(jm* fi).
     //n_max - максимальный радиальный порядок полиномов.
     //diameter - диаметр окружности, на которой будут сгенерированы полиномы, пиксели.
    {
        double delta = 2. / diameter; //коэффициент
        this->polynomials.resize(n_max + 1); //выделяем память
        for (size_t n = 0; n <= n_max; ++n) //по степеням радиальной функции
        {
            this->polynomials[n].resize(n_max + 1); //выделяем память для n степени полинома
            for (size_t i = 0; i <= n_max; ++i)
            {
                this->polynomials[n][i] = std::make_pair(cv::Mat(diameter, diameter, CV_64FC1), cv::Mat(diameter, diameter, CV_64FC1));// освобождается память под матрицу полинома
                this->polynomials[n][i].first.setTo(cv::Scalar(0));// реальная часть степени полинома
                this->polynomials[n][i].second.setTo(cv::Scalar(0));// мнимая часть степени полинома
            }
            for (size_t r = 0; r < diameter / 2; ++r)//diametr/2 из-за радиальной функции Уолша, ниже
            {
                double radial = rf::RadialFunctions::Walsh(r * delta, n, n_max);// вычисляет значение сдвинутого полинома уолша
                //r*delta т.к. радиус на котором вычисляется значение лежит в промежутке от [0;1]
                //n - степень полинома
                //n_max - максим.кол-во полиномов
                size_t rot_count = (size_t)(2 * M_PI * r * delta * diameter) * 2;//нужно чтобы выполнить "закручивание" фазового множителя)
                for (size_t th = 0; th < rot_count; ++th)//
                {
                    double theta = 2 * th * M_PI / rot_count;// угол "закрутки")
                    for (size_t i = 0; i <= n_max; ++i)
                    {
                        double sine = std::sin(theta * i) * radial;//сдвиг 
                        double cosine = std::cos(theta * i) * radial;//сдвиг
                        double& color_re = this->polynomials[n][i].first.at<double>(r * std::cos(theta) + diameter / 2, r * std::sin(theta) + diameter / 2); //(x' = x + x0, y' = y + y0)
                        color_re = cosine;// указываем что закрашивать (какой пиксель), их нужно приравнять
                        double& color_im = this->polynomials[n][i].second.at<double>(r * std::cos(theta) + diameter / 2, r * std::sin(theta) + diameter / 2); //(x' = x + x0, y' = y + y0)
                        color_im = sine;// указываем что закрашивать (какой пиксель), их нужно приравнять
                    }
                }
            }
        }
    }
    virtual ComplexMoments Decompose(cv::Mat blob) override//Разложить картинку в ряд по полиномам,
        //blob - картинка(смежная область), должна быть типа CV_8UC1.
    {
        ComplexMoments decomposition;//разложение.
        size_t basis_mat_count = 0;// кол-во базисов
        for (size_t i = 0; i < this->polynomials.size(); ++i)//
        {
            for (size_t j = 0; j < this->polynomials[i].size(); ++j)
            {
                ++basis_mat_count;// кол-во базисов в строчке
            }
        }
        decomposition.re = cv::Mat::zeros(cv::Size(1, basis_mat_count), CV_64FC1);// выделяем память
        decomposition.im = cv::Mat::zeros(cv::Size(1, basis_mat_count), CV_64FC1);// выделяем память

        cv::Mat other;//переменная
        blob.convertTo(other, CV_64FC1, 2.0 / 255.0, -1.0);//приводим к другому типу картинка в формат с даблом

        size_t basis_idx = 0;////скалярное произведение 
        for (size_t i = 0; i < this->polynomials.size(); ++i)// степень разложения, i - место куда записываем
        {
            for (size_t j = 0; j < this->polynomials[i].size(); ++j)// по коэффициентам
            {
                double base_norm_re = this->polynomials[i][j].first.dot(this->polynomials[i][j].first);//модуль
                double base_norm_im = this->polynomials[i][j].second.dot(this->polynomials[i][j].second);//модуль
                if (abs(base_norm_re) > std::numeric_limits<double>::epsilon())//ограничение на модуль действит части, т.к. иногда получаем 0:0
                {
                    decomposition.re.at<double>(basis_idx) = other.dot(this->polynomials[i][j].first) / base_norm_re;//разложение на действительную часть
                }
                if (abs(base_norm_im) > std::numeric_limits<double>::epsilon())//ограничение на модуль реальной частия
                {
                    decomposition.im.at<double>(basis_idx) = other.dot(this->polynomials[i][j].second) / base_norm_im;//разложение на мнимую часть
                }
                ++basis_idx;
            }
        }
        return decomposition;
    }

    virtual cv::Mat Recovery(ComplexMoments& decomposition) override//Восстановить картинку из разложения.
        //decomposition - разложение картинки в ряд.
    {
        cv::Mat recovery = cv::Mat(this->polynomials[0][0].first.rows, this->polynomials[0][0].first.cols, CV_64FC1);// восстановленное изображение, имеет тип CV_64FC1.
        recovery.setTo(cv::Scalar(0));//скаляр
        size_t basis_idx = 0;
        for (size_t i = 0; i < this->polynomials.size(); ++i)//
        {
            for (size_t j = 0; j < this->polynomials[i].size(); ++j)//
            {
                recovery += this->polynomials[i][j].first * decomposition.re.at<double>(basis_idx)
                    + this->polynomials[i][j].second * decomposition.im.at<double>(basis_idx);//соединявкаем мнимую и действительную часть для восстановления
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