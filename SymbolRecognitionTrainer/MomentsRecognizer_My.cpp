#include "MomentsRecognizer_My.h"

bool MomentsRecognizer_My::Train(
	std::map<std::string, std::vector<fe::ComplexMoments>> moments, //моменты
	std::vector<int> layers, //кол-во слоев
	int max_iters, //кол-во итераций
	float eps,//точность
	float speed)//скорость
{
	this->pAnn = cv::ml::ANN_MLP::create();//создаем нейронную сеть

	size_t num_of_inputs = moments.begin()->second.front().re.rows * 2;//задаем кол-во входов
	size_t num_of_outputs = moments.size();//задаем кол-во выходов

	std::vector < int > all_layers = { (int)num_of_inputs }; //кол-во слоев
	all_layers.insert(all_layers.end(), layers.begin(), layers.end());//вставляем layers.begin() элементов layers.end() начиная с позиции, на которую указывает all_layers.end()
	all_layers.push_back(num_of_outputs); //добавляем кол-во выходов 

	pAnn->setLayerSizes(all_layers); //задаем размер нейронной сети

	// Конфигурация алгоритма обучения.
	this->pAnn->setBackpropMomentumScale(0.1);
	this->pAnn->setBackpropWeightScale(0.1);
	this->pAnn->setActivationFunction(cv::ml::ANN_MLP::SIGMOID_SYM, 0., 0.);

	// Критерий остановки обучения.
	cv::TermCriteria term_criteria(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, max_iters, eps);
	this->pAnn->setTermCriteria(term_criteria);
	this->pAnn->setTrainMethod(cv::ml::ANN_MLP::RPROP, speed);

	//словарь, для конвертации выхода сети в значение цифры
	this->values.resize(num_of_outputs);

	size_t num_of_samples = 0;
	for (auto it = moments.begin(); it != moments.end(); ++it)
	{
		num_of_samples += it->second.size();//кол-во картинок используемых примером
	}

	cv::Mat input(num_of_samples, num_of_inputs, CV_32FC1);
	cv::Mat output = cv::Mat::zeros(num_of_samples, num_of_outputs, CV_32FC1);

	int sample_idx = 0;
	int class_idx = 0;
	for (auto it = moments.begin(); it != moments.end(); ++it, ++class_idx)
	{
		for (int i = 0; i < it->second.size(); ++i, ++sample_idx)
		{
			MomentsToInput(it->second[i]).copyTo(input.rowRange(sample_idx, sample_idx + 1));//преобразуем то что харнит комплекс моментс
			output.at<float>(sample_idx, class_idx) = 1.0f;//какая должна быть правильная цифра 00 где не правильная, и где правильная
		}
		this->values[class_idx] = it->first;
	}

	return this->pAnn->train(input, cv::ml::SampleTypes::ROW_SAMPLE, output);//преобразует веса и возвращает бул
}

std::string MomentsRecognizer_My::Recognize(fe::ComplexMoments& moments) //создается объект для распознавания определенной конфигурации
{
	cv::Mat output;
	pAnn->predict(MomentsToInput(moments), output);//предсказываем выход по вектору моментов

	return OutputToValue(output);//получаем нашу циферку
}
//Преобразовать моменты изображения ко входу нейронной сети.
cv::Mat MomentsRecognizer_My::MomentsToInput(fe::ComplexMoments& moments)//преобразование выхода нейронной сети (cv::Mat) в символ, соответствующий цифре (std::string).
{
	cv::Mat mat(1, moments.re.rows * 2, CV_32FC1);
	
	for (size_t i = 0; i < moments.re.rows; ++i)
	{
		mat.at<float>(i) = moments.re.at<double>(i) + 1;//первая часть действительная
		mat.at<float>(i + moments.re.rows) = moments.im.at<double>(i) + 1;//вторая мнимая
	}
	return mat;
}

std::string MomentsRecognizer_My::OutputToValue(cv::Mat output) //преобразование моментов(de::ComplexMoments) к символу(cv::Mat), подаваемой на вход нейронной сети.
{
	size_t  max_pos = 0; //позциция момента
	for (size_t i = 0; i < output.cols; i++) //преобразование
	{
		if (output.at<float>(i) > output.at<float>(max_pos))
		{
			max_pos = i;
		}
	}
	return this->values[max_pos]; //получаем циферку с положения момента
}

bool MomentsRecognizer_My::Save(std::string filename)//нейронная сеть сохраняется в файл
{
	cv::FileStorage fs(filename, cv::FileStorage::WRITE);
	if (!fs.isOpened())
	{
		return false;
	}

	pAnn->write(fs);

	fs << "values" << values;
	fs.release();
	return true;
}

bool MomentsRecognizer_My::Read(std::string filename)//считываются обучающие данные
{
	cv::FileStorage fs(filename, cv::FileStorage::READ);
	if (!fs.isOpened())
	{
		return false;
	}

	pAnn = cv::ml::ANN_MLP::create();
	pAnn->read(fs.root());
	values.clear();
	for (auto iter = fs["values"].begin(); iter != fs["values"].end(); iter++)
	{
		values.push_back(*iter);
	}
	fs.release();
	return true;
}