#pragma once
#include <FeatureExtraction.h>
#include "MomentsRecognizer.h"
#include <opencv2/ml.hpp>

class MomentsRecognizer_My :public MomentsRecognizer
{

public:

	MomentsRecognizer_My()
	{
	}

	virtual ~MomentsRecognizer_My()
	{
	}

protected:

	virtual cv::Mat MomentsToInput(fe::ComplexMoments& moments); //преобразование моментов к матрице(cv::Mat), подаваемой на вход нейронной сети

	virtual std::string OutputToValue(cv::Mat output); // преобразование выхода нейронной сети в символ, соответствующий цифре

public:

	virtual bool Train(                                                 //Обучение нейронной сети 
		std::map<std::string, std::vector<fe::ComplexMoments>> moments,
		std::vector<int> layers,
		int max_iters,
		float eps,
		float speed);

	virtual std::string Recognize(fe::ComplexMoments& moments); //Распознать символ по моментам  moments - моменты по которым проводится распознавание.
	 

	virtual bool Save(std::string filename);

	virtual bool Read(std::string filename);

};
