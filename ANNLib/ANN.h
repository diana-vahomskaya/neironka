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
	class ANeuralNetwork
	{
	public:
		/*
		 * Доступные типы активационных функций.
		 */
		enum ActivationType
		{
			POSITIVE_SYGMOID,   // Положительная униполярная сигнмоида.
			BIPOLAR_SYGMOID		// Биполярная сигмоида.
		};

		/**
		* Прочитать нейронную сеть из файла. Сеть сохраняется вызовом метода Save.
		* @param filepath - имя и путь до файла с сеткой.
		* @return - успешность считывания.
		*/
		ANNDLL_API virtual bool Load(std::string filepath);

		/**
		* Сохранить нейронную сеть в файл. Сеть загружается вызовом метода Load.
		* @param filepath - имя и путь до файла с сеткой.
		* @return - успешность сохранения.
		*/
		ANNDLL_API virtual bool Save(std::string filepath);

		/**
		* Получить конфигурацию сети.
		* @return конфигурация сети - массив - в каждом элементе хранится количество нейронов в слое.
		*			Номер элемента соответствует номеру слоя.
		*/
		ANNDLL_API virtual std::vector<int> GetConfiguration();

		/**
		* Проинициализирвать веса сети случайным образом.
		*/
		ANNDLL_API void RandomInit();

		/**************************************************************************/
		/**********************ЭТО ВАМ НАДО РЕАЛИЗОВАТЬ САМИМ**********************/
		/**************************************************************************/

		/**
		* Получить строку с типом сети.
		* @return описание сети, содержит запись о типе нейронной сети и авторе библиотеки.
		*/
		ANNDLL_API virtual std::string GetType() = 0;

		/**
		* Спрогнозировать выход по заданному входу.
		* @param input - вход, длина должна соответствовать количеству нейронов во входном слое.
		* @return выход сети, длина соответствует количеству нейронов в выходном слое.
		*/
		ANNDLL_API virtual std::vector<float> Predict(std::vector<float> & input) = 0;

		/**
		* Создать нейронную сеть
		* @param configuration - конфигурация нейронной сети.
		*   Каждый элемент представляет собой количество нейронов в очередном слое.
		* @param activation_type - тип активационной функции (униполярная, биполярная).
		*/
		friend ANNDLL_API std::shared_ptr<ANN::ANeuralNetwork> CreateNeuralNetwork(
			std::vector<int> & configuration = std::vector<int>(),
			ANeuralNetwork::ActivationType activation_type = ANeuralNetwork::POSITIVE_SYGMOID
		);

		/**
		* Обучить сеть методом обратного распространения ошибки.
		* В ходе работы метода, после выполнения обучения флаг is_trained должен устанавливаться в true.
		* @param ann - нейронная сеть, которую необходимо обучить.
		* @param inputs - входы для обучения.
		* @param outputs - выходы для обучения.
		* @param max_iters - максимальное количество итераций при обучении.
		* @param eps - средняя ошибка по всем примерам при которой происходит остановка обучения.
		* @param speed - скорость обучения.
		* @param std_dump - сбрасывать ли информацию о процессе обучения в стандартный поток вывода?
		*/
		friend ANNDLL_API float BackPropTrain(
			std::shared_ptr<ANN::ANeuralNetwork> ann,
			std::vector<std::vector<float>> & inputs,
			std::vector<std::vector<float>> & outputs,
			int maxIters = 10000,
			float eps = 0.1,
			float speed = 0.1,
			bool std_dump = false
		);

		/***************************************************************************/
		/***************************************************************************/
		/**
		* Деструктор.
		*/
		ANNDLL_API virtual ~ANeuralNetwork();

	protected:
		/** 
		 * Веса сети. 
		 * Первый индекс - номер слоя, 
		 * второй индекс - номер нейрона от которого идёт связь, 
		 * третий индекс - номер нейрона к которому идёт связь. 
		 */
		std::vector<std::vector<std::vector<float> > > weights;

		/**
		* Конфигурация сети.
		* номер элемета в массиве соответсвует номеру слоя.
		* значение - количеству нейронов.
		*/
		std::vector<int> configuration;

		/** Обучена ли сеть? */
		bool is_trained;

		/** Масштабирующий коэффициент аргумента сигмоиды. */
		float scale;

		/** Тип активационной функции. */
		ActivationType activation_type;

		/**
		* Вычислить значение активационной функции.
		* @param neuron_input - входное значение нейрона.
		* @return - значение активационной фунции.
		*/
		ANNDLL_API float Activation(float neuron_input);

		/**
		* Вычислить значение производной активационной функции.
		* @param activation - значение активационной фнункции, для которой хотим вычислить производную.
		* @return - значение производной активационной фунции.
		*/
		ANNDLL_API float ActivationDerivative(float activation);
	};

	/**
	* Тестовая функция для проверки подключения библиотеки.
	* @return строка с поздравлениями.
	*/
	ANNDLL_API std::string GetTestString();

	/**
	* Считать данные из файла.
	* @param filepath - путь и имя к файлу с данными.
	* @param inputs - буфер для записи входов.
	* @param outputs - буфер для записи выходов.
	* @return - успешность чтения.
	*/
	ANNDLL_API bool LoadData(
		std::string filepath,
		std::vector<std::vector<float>> & inputs,
		std::vector<std::vector<float>> & outputs
	);

	/**
	* Записать данные в файл.
	* @param filepath - путь и имя к файлу с данными.
	* @param inputs - входы для записи.
	* @param outputs - выходы для записи.
	* @return - успешность записи.
	*/
	ANNDLL_API bool SaveData(
		std::string filepath,
		std::vector<std::vector<float>> & inputs,
		std::vector<std::vector<float>> & outputs
	);
}