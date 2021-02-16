#pragma once
#include <random>
#include <ANN.h>
#include "IIndividual.h"

namespace ga
{
	class Individual : public IIndividual, public ANN::ANeuralNetwork
	{
	public:
		Individual(
			std::vector<size_t>& configuration,
			ANN::ANeuralNetwork::ActivationType activation_type,
			float scale,
			std::string data_source
		);
		~Individual();

		/**
		 * Выполнить мутацию особи.
		 * @return мутировавшая особь.
		 */
		std::shared_ptr<IIndividual> Mutation();

		/**
		 * Выполнить скрещивание текущей особи с другой особью.
		 * @param individual - особь с которой будет проведено скрещивание.
		 * @return дочерняя особь после скрещивания.
		 */
		std::shared_ptr<IIndividual> Crossover(std::shared_ptr<IIndividual> individual);

		/**
		 * Провести соревнование между текущей и другой особью.
		 * @param individual - другая особь.
		 * @return пара цифр. Первое значение - количество очков набранное текущей особью.
		 *					  Второе значение - количество очков, набранное второй особью.
		 */
		std::pair<int, int> Spare(std::shared_ptr<IIndividual> individual);

		/**
		 * Принять решение.
		 * В процессе соревнований особи поочередно принимают решения,
		 * от этого зависит процесс развития соревнования.
		 * @param input - входные данные
		 * @return выходные данные.
		 */
		std::vector<float> MakeDecision(std::vector<float>& input);

		/**
		 * Скопировать текущую особь.
		 * @return копия текущей особи.
		 */
		std::shared_ptr<IIndividual> Clone();

		/**
		* Получить строку с типом сети.
		* @return описание сети, содержит запись о типе нейронной сети и авторе библиотеки.
		*/
		virtual std::string GetType() override;

		/**
		* Спрогнозировать выход по заданному входу.
		* @param input - вход, длина должна соответствовать количеству нейронов во входном слое.
		* @return выход сети, длина соответствует количеству нейронов в выходном слое.
		*/
		virtual std::vector<float> Predict(std::vector<float>& input) override;

	protected:
		// Данные для обучения:
		std::vector<std::vector<float>> inputs, outputs;
	};
};
