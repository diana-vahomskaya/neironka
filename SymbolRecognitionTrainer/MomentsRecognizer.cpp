#include "MomentsRecognizer.h"

using namespace cv;
using namespace std;
using namespace fe;

MomentsRecognizer::MomentsRecognizer()
{
}


MomentsRecognizer::~MomentsRecognizer()
{
}

bool MomentsRecognizer::Train(std::map<std::string, std::vector<fe::ComplexMoments>> moments, std::vector<int> layers, int max_iters /*= 100000*/, float eps /*= 0.1*/, float speed /*= 0.1*/)
{
	int count = moments.begin()->second.begin()->abs.cols;

	//Сформируем упорядоченный массив примеров
	vector<ComplexMoments*> data;
	vector<int> output_index;
	int local_out_index = 0;
	values.clear();
	for (auto primer = moments.begin(); primer != moments.end(); primer++)
	{
		values.push_back(primer->first);
		for (int j = 0; j < (int)primer->second.size(); j++)
		{
			data.push_back(&(primer->second[j]));
			output_index.push_back(local_out_index);
		}
		local_out_index++;
	}

	pAnn = ml::ANN_MLP::create();
	vector<int> full_layers(layers.size()+2);
	full_layers[0] = count;
	for (size_t i = 0; i < layers.size(); i++) {
		full_layers[i + 1] = layers[i];
	}
	full_layers[full_layers.size() - 1] = moments.size();
	pAnn->setLayerSizes(full_layers);
	pAnn->setBackpropMomentumScale(0.1);
	pAnn->setBackpropWeightScale(0.1);
	pAnn->setActivationFunction(ml::ANN_MLP::SIGMOID_SYM, 0., 0.);
	// Создадим критерий остановки обучения.
	TermCriteria term_criteria(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, max_iters, eps);
	pAnn->setTermCriteria(term_criteria);
	pAnn->setTrainMethod(ml::ANN_MLP::RPROP, 0.001);

	Mat in_data(data.size(), count, CV_32FC1);
	Mat out_data(data.size(), moments.size(), CV_32FC1);
	out_data.setTo(-1.0);

	for (int i = 0; i < in_data.rows; i++)
	{
		int index = 0;
		out_data.ptr<float>(i)[output_index[i]] = (float)1./*3.141592*/;
		Mat input = MomentsToInput(*data[i]);
		memcpy(in_data.ptr<float>(i), input.data, sizeof(float)*input.cols);
	}
	auto tdata = ml::TrainData::create(in_data, ml::ROW_SAMPLE, out_data);
	bool iter = pAnn->train(tdata);
	return iter;
}

cv::Mat MomentsRecognizer::MomentsToInput(fe::ComplexMoments& moments)
{
	Mat res;
	moments.abs.convertTo(res, CV_32FC1);
	return res;
}

std::string MomentsRecognizer::OutputToValue(cv::Mat output)
{
	Point min_loc, max_loc;
	double min, max;
	minMaxLoc(output, &min, &max, &min_loc, &max_loc);
	return values[max_loc.x];
}

std::string MomentsRecognizer::Recognize(fe::ComplexMoments & moments)
{
	Mat output;
	pAnn->predict(MomentsToInput(moments), output);
	return OutputToValue(output);
}

double MomentsRecognizer::PrecisionTest(std::map<std::string, std::vector<fe::ComplexMoments>> moments)
{
	int right_answers = 0;
	int tests = 0;
	for (auto primer = moments.begin(); primer != moments.end(); primer++) {
		for (int j = 0; j < (int)primer->second.size(); j++) {
			auto recognized = Recognize(primer->second[j]);
			if (recognized == primer->first) {
				right_answers++;
			}
			tests++;
		}
	}
	return (double)right_answers / (double)tests;
}

bool MomentsRecognizer::Save(std::string filename)
{
	FileStorage fs(filename, FileStorage::WRITE);
	if (!fs.isOpened()) {
		return false;
	}
	pAnn->write(fs);
	fs << "values" << values;
	fs.release();
	return true;
}

bool MomentsRecognizer::Read(std::string filename)
{
	FileStorage fs(filename, FileStorage::READ);
	if (!fs.isOpened()) {
		return false;
	}
	pAnn = ml::ANN_MLP::create();
	pAnn->read(fs.root());
	values.clear();
	for (auto iter = fs["values"].begin(); iter != fs["values"].end(); iter++) {
		values.push_back(*iter);
	}
	fs.release();
	return true;
}

