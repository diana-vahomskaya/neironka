#pragma once
#define ANNDLL_EXPORTS
#include <iostream>
#include <ANN.h>

namespace ANN
{
	class Network : public ANeuralNetwork // класс наших функций

	{

	protected: // в классах наследниках



	public: // везде

		Network(
			std::vector<int>& configuration, // чтобы можно было занести изменени¤ в конфигурацию
			ANeuralNetwork::ActivationType activation_type) // функци¤ активации
		{
			this->configuration = configuration; // передаем конфигурацию в наш класс от класса ANN

			for (int i = 0; i < this->configuration.size(); i++)
			{
				++this->configuration[i];
			}


			this->activation_type = activation_type; // передаем функцию активации в наш класс от класса ANN
			this->scale = 1;
			//this->moment = 0.1;
		}

		ANNDLL_API std::string GetType(); // получение строки описани¤ сети

		ANNDLL_API std::vector<float> Predict(std::vector<float>& input);//прогнозирование выхода по известному входу

		ANNDLL_API float BackPropTrain( //обучение сети методом обратного распространени¤ ошибки
			std::shared_ptr<ANN::ANeuralNetwork>ann,
			std::vector<std::vector<float>>& inputs, // входы дл¤ обучени¤
			std::vector<std::vector<float>>& outputs,// выходы дл¤ обучени¤

			int max_iters = 10000, // максимальное количество итераций при обучении
			float eps = 0.1, // средн¤¤ ошибка по всем примерам при которой происходит остановка обучени¤
			float speed = 0.1, // скорость обучени¤
			bool std_dump = false // выводить ли инфу на экран 
		);
	};
}
