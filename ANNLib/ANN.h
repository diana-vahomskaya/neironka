#pragma once

#ifdef ANNDLL_EXPORTS
#define ANNDLL_API __declspec(dllexport) 
#else
#define ANNDLL_API __declspec(dllimport) 
#endif

#include <vector>
#include <memory>

namespace ANN
{
	class NeuralNetwork
	{
	public:
		enum ActivationType
		{
			POSITIVE_SYGMOID,
			BIPOLAR_SYGMOID
		};
		/**Прочитать нейронную сеть из файла
		 *@param
		 */
		virtual bool Read(std::string filepath);
		virtual bool Save(std::string filepath);
		std::vector<int> GetConfiguration();

		/**********************Это вам надо реализовать самим**********************/
		virtual std::string GetType() = 0;
		virtual std::vector<float> Predict(std::vector<float> & input) = 0;
		virtual float MakeTrain(
			std::vector<std::vector<float>> & inputs,
			std::vector<std::vector<float>> & outputs,
			int max_iters = 10000,
			float eps = 0.1,
			float speed = 0.1
		) = 0;

	protected:
		/**Веса сети*/
		std::vector<std::vector<std::vector<float> > > weights;
		/**Конфигурация сети.
		 * номер элемета в массиве соответсвует номеру слоя
		 * значение - количеству нейронов
		 */
		std::vector<int> configuration;
		/**Обучена ли сеть?*/
		bool is_trained;
		/**Масштабирующий коэффициент аргумента сигмоиды*/
		float scale;
		/**Тип активационной функции*/
		ActivationType activation_type;
		/**Вычислить значение активационной функции
		 *@param neuron_input - входное значение нейрона
		 *@return - значение активационной фунции
		 */
		float Activation(float neuron_input);
		/**Вычислить значение производной активационной функции
		 *@param activation - значение активационной фнункции, для которой хотим вычислить производную
		 *@return - значение производной активационной фунции
		 */
		float ActivationDerivative(float activation);
	};

	/**Создать нейронную сеть
	 *@param configuration - конфигурация нейронной сети
	 */
	ANNDLL_API std::shared_ptr<ANN::NeuralNetwork> CreateNeuralNetwork(
		std::vector<int> & configuration = std::vector<int>(),
		NeuralNetwork::ActivationType activation_type = NeuralNetwork::POSITIVE_SYGMOID
	);
}