#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include "ComplexMoments.h"

namespace fe {
	/**
	 * Переопределение типа набора полиномов. Для краткости.
	 * Комплексные полиномы хранятся в двухмерном массиве. Первый индекс - радиальный порядок, второй индекс - угловой порядок.
	 * Полином представлен типом std::pair - в поле first хранится реальная часть полинома, в поле second мнимая.
	 * Каждая часть представлена в cv::Mat типа CV_64FC1.
	 */
	typedef std::vector<std::vector<std::pair<cv::Mat, cv::Mat>>> OrthoBasis;

	/**
	 * Класс, отвечающий за взаимодействие с полиномами ~ exp(jm*fi).
	 */
	class PolynomialManager
	{
	protected:

		/** Полиномы. */
		OrthoBasis polynomials;

	public:
		/**
		 * Разложить картинку в ряд по полиномам.
		 * @param blob - картинка (смежная область), должна быть типа CV_8UC1.
		 * @return decomposition разложение.
		 */
		virtual ComplexMoments Decompose(cv::Mat blob) = 0;

		/**
		 * Восстановить картинку из разложения.
		 * @param decomposition - разложение картинки в ряд.
		 * @return восстановленное изображение, имеет тип CV_64FC1.
		 */
		virtual cv::Mat Recovery(ComplexMoments & decomposition) = 0;

		/**
		 * Проинициализировать базис ортогональных полиномов ~ exp(jm*fi).
		 * @param n_max - максимальный радиальный порядок полиномов.
		 * @param diameter - диаметр окружности, на которой будут сгенерированы полиномы, пиксели.
		 */
		virtual void InitBasis(int n_max, int diameter) = 0;

		/**
		 * Получить описание объекта для работы с полиномами.
		 * @return - описание объекта.
		 */
		virtual std::string GetType() = 0;

		/**
		 * Получить базис ортогональных полиномов. 
		 * @return базис ортогональных полиномов. Каждый полином представлен std::pair<cv::Mat, cv::Mat>.
		 *		   в поле first хранится реальная часть полинома, в поле second мнимая. Каждая часть имеет тип CV_64FC1.
		 */
		virtual OrthoBasis GetBasis();

		/** Деструктор. */
		virtual ~PolynomialManager();
	};
};

