﻿#include <iostream>
#include <FeatureExtraction.h>
#include <opencv2/opencv.hpp>
#include "Visualisation.h"
#define NMAX 12
#define TSIDE 30
using namespace cv;
using namespace fe;
using namespace std;

int main() 
{
	/*cout << "hello world!" << endl;
	cout << fe::GetTestString().c_str() << endl;
	
	int max;
	int diametr;

	cout << "Max polinom:";
	cin >> max;
	cout << endl;
	cout << "Diametr:";
	cin >> diametr;
	cout << endl;

	auto CPM = fe::CreatePolynomialManager();
	auto CBP = fe::CreateBlobProcessor();

	CPM->InitBasis(max, diametr); //генерируем базис комплексных полиномов
	ShowPolynomials("Basic:", CPM->GetBasis()); //показываем базис на экране

	cv::Mat image = cv::imread("..\\DecompositionSample\\Picture.png", CV_8UC1); //читаем картинку из файла
	imshow("Picture:", image); //Установка входного изображения на экран

	cout << CPM->GetType() << endl << endl;

	vector<cv::Mat>blobs;
	vector<cv::Mat> normalized_blobs;

	blobs = CBP->DetectBlobs(image); // выделяет на изображение смежные области (цифры)
	normalized_blobs = CBP->NormalizeBlobs(blobs, diametr); // приводим смежные области к единому масштабу

	cout << CBP->GetType() << endl;

	vector<fe::ComplexMoments> blobs_decompos; // вектор коэффициентов
	blobs_decompos.resize(normalized_blobs.size()); // выделяем память под кол-во картинок
	vector<cv::Mat> Recovery_blobs; // восстановление изображения
	Recovery_blobs.resize(normalized_blobs.size()); // память под кол-во картинок

	for (int i = 0; i < Recovery_blobs.size(); i++) // цикл по картинкам
	{
		Recovery_blobs[i] = cv::Mat::zeros(diametr, diametr, CV_64FC1); // выделение памяти под картинку (соответсвует нормированному изобраению)
	}

	for (int i = 0; i < blobs.size(); i++) // цикл по картинкам
	{
		blobs_decompos[i] = CPM->Decompose(normalized_blobs[i]); // перебирает все цифры по очередию раскладывая их в ряд
		Recovery_blobs[i] = CPM->Recovery(blobs_decompos[i]); // восстанавливает изображение цифр
		ShowBlobDecomposition("Восстановленные цифры:", normalized_blobs[i], Recovery_blobs[i]);
		cv::waitKey(0);
	}
	waitKey(0);*/
	cout << fe::GetTestString().c_str() << endl;
	shared_ptr<fe::PolynomialManager> pPolyMan = fe::CreatePolynomialManager();
	cout << pPolyMan->GetType() << endl;

	// Формирование базиса:
	int nmax = 12; int dia = 64;
	pPolyMan->InitBasis(nmax, dia);
	fe::OrthoBasis OBS = pPolyMan->GetBasis();
	ShowPolynomials("Polynomial orthogonal basis", OBS);
	waitKey();

	// Загрузка изображения:
	Mat numage = imread("..\\DecompositionSample\\Picture.png", CV_LOAD_IMAGE_GRAYSCALE);	//CV_8UC1
	Mat newmage;
	threshold(numage, newmage, 127.0, 255.0, THRESH_BINARY);
	cv::imshow("New image", newmage);
	cv::waitKey();

	// Обнаружение символов:
	shared_ptr<fe::IBlobProcessor> pIBP = fe::CreateBlobProcessor();
	cout << pIBP->GetType() << endl;
	vector<Mat> blums = pIBP->DetectBlobs(newmage);
	vector<Mat> nblums = pIBP->NormalizeBlobs(blums, dia);

	for (int i = 0; i < nblums.size(); i++)
	{
		// Разложение и восстановление символа:
		fe::ComplexMoments cm = pPolyMan->Decompose(nblums[i]);
		Mat blum = pPolyMan->Recovery(cm);
		ShowBlobDecomposition("Number " + to_string(i), nblums[i], blum);
	}
	waitKey();
	return 0;
}