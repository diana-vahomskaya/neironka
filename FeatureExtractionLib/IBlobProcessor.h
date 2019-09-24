#pragma once

#include <opencv2/opencv.hpp>
#include <vector>

namespace fe {
	/*
	 * Интерфейс обработчика смежных областей.
	 */
	class IBlobProcessor
	{
	public:
		/**
		 * Найти смежные области на изображении. Светлые области представляют собой фон, тёмные области объекты.
		 * @param image - изображение для поиска смежных областей,
		 *				  должно иметь тип CV_8UC1.
		 * @param blobs - буфер для записи неотмасштабированных смежных областей.
		 */
		virtual std::vector<cv::Mat> DetectBlobs(cv::Mat image) = 0;

		/**
		 * Привести размер смежных областей к единому масштабу.
		 * @param blobs - смежные области.
		 * @param normilized_blobs - буфер для записи смежных областей единого размера.
		 * @param side - сторона квадрата на котором будет отрисована нормализованная смежная область.
		 */
		virtual void NormalizeBlobs(
			std::vector<cv::Mat> & blobs,
			std::vector<cv::Mat> & normalized_blobs,
			int side
		) = 0;

		/**
		 * Получить описание используемого обработчика смежных областей.
		 * @return описание, содержит название метода выделения смежных областей и фамилию разработчика.
		 */
		virtual std::string GetType() = 0;
	};
};