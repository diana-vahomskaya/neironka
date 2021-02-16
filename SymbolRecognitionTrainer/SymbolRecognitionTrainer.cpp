#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"

#include "FeatureExtraction.h"
#include "MomentsHelper.h"
#include "MomentsRecognizer_My.h"

using namespace cv;
using namespace std;
using namespace fe;


static const string labeled_data = "..\\data\\labeled_data"; //размеченные данные 
static const string ground_data = "..\\data\\ground_data"; //данные для обучения
static const string test_data = "..\\data\\test_data"; //данные для тестирования  
static const string ground_data_moments = "..\\data\\ground_data_moments.dat"; // сохраненные данные для обучения
static const string test_data_moments = "..\\data\\test_data_moment.dat"; // сохраненные данные для тестирования  
static const string train_Network = "..\\data\\train_network.dat"; // сохр нейронная сеть




void generateData()
{
	cout << "===Generate data!===" << endl;

	bool result;

	map < string, vector < ComplexMoments > > moments_test;

	cout << "  Split data onto ground and test ";

	result = MomentsHelper::DistributeData(labeled_data, ground_data, test_data, 80); // Разложить размеченные данные по папкам с обучающими и тестовыми данными
	
	if (result) cout << "success" << endl;
	else        cout << "failed" << endl;

	auto CBP = CreateBlobProcessor();
	auto CPM = CreatePolynomialManager();

	int max;
	int diametr;

	cout << "Max polynomial: "; // макс степень полинома
	cin >> max;
	cout << endl;
	cout << "Diametr: "; // диаметр
	cin >> diametr;
	cout << endl;

	CPM->InitBasis(max, diametr); // генерируем базис комплексных полиномов 

	MomentsHelper::GenerateMoments(ground_data, CBP, CPM, moments_test); // Сгенерировать моменты по примерам
	MomentsHelper::SaveMoments(ground_data_moments, moments_test); // Сохранить моменты

	moments_test.clear();

	MomentsHelper::GenerateMoments(test_data, CBP, CPM, moments_test); // Сгенерировать моменты по примерам
	MomentsHelper::SaveMoments(test_data_moments, moments_test); // Сохранить моменты
	

}

void trainNetwork()
{
	cout << "===Train network!===" << endl;

	map < string, vector < ComplexMoments > > moments_test;

	bool result;

	MomentsRecognizer_My mr;

	cout << "  Read training data moments ";

	result = MomentsHelper::ReadMoments(ground_data_moments, moments_test); // Считать моменты из файла

	if (result) cout << "success" << endl;
	else        cout << "failed" << endl;

	vector < int > ann_config; // конфигурация сети

	size_t num_of_hiddne_layers; // кол-во скрытых слоев

	float ann_precision; // точность

	float ann_speed; // скорость обучения

	int ann_max_iters; // макс кол-во итераций

	cout << " Enter the number of hidden layers: "; //Введите кол-во скрытых слоев
	cin >> num_of_hiddne_layers; // кол-во скрытых слоев
	ann_config.resize(num_of_hiddne_layers);

	for (size_t i = 0; i < num_of_hiddne_layers; ++i)
	{
		cout << " Enter the number of neurons in layer " << i + 1 << ": " ; // Кол-во нейронов в каждом слое:
		cin >> ann_config[i];
	}

	cout << " Max kol iter: ";
	cin >> ann_max_iters;
	cout << " Precision: "; // точность
	cin >> ann_precision;

	cout << " Speed: "; // скорость обучения
	cin >> ann_speed;

	cout << " Network is training. Wait " << endl; // сеть обучается 

	mr.Train(moments_test, ann_config, ann_max_iters, ann_precision, ann_speed); //обучается нейронная сеть

	cout << " Network trained! " << endl; // сеть обучилась

	cout << " Network is save. Wait " << endl; // Сеть сохраняется в файл

	mr.Save(train_Network);  // нейронная сеть сохраняется в файл 

	cout << " Network saved! " << endl; // Сеть сохранилась в файл

}

void precisionTest()
{
	cout << "===Precision test!===" << endl;

	map < string, vector < ComplexMoments > > moments_test;
	bool result;
	MomentsRecognizer_My mr;

	cout << " Network is reading. Wait " << endl; // Читаем данные

	result = mr.Read(train_Network); // обученная нейронная сеть считывается из файла
	if (result) cout << "success" << endl;
	else        cout << "failed" << endl;

	cout << " Network has been read! " << endl; // считали


	cout << " Moments is reading. Wait " << endl; // Читаем моменты

	result = MomentsHelper::ReadMoments(test_data_moments, moments_test); //Считать моменты из файла

	if (result) cout << "success" << endl;
	else        cout << "failed" << endl;


	cout << " Moments has been read! " << endl; // считали

	cout << " Evaluated precision. Wait " << endl; // Выполняется оценка точности

	double precision = mr.PrecisionTest(moments_test); //выполняется оценка точности алгоритма распознавания

	cout << " Precision was evaluated! " << endl;

	cout << " Result: " << precision << endl;
}

void recognizeImage()
{
	cout << "===Recognize single image!===" << endl;

	map < string, vector < ComplexMoments > > moments_test;

	MomentsRecognizer_My mr;

	vector < Mat > blobs, nblobs; // цифры, цифры одного размера
	vector < ComplexMoments > moments;

	string path = "..\\data\\picture.png"; // путь

	auto bp = CreateBlobProcessor();
	auto pm = CreatePolynomialManager();

	cout << " Network is reading. Wait " << endl; // Читаем данные

	mr.Read(train_Network); // обученная нейронная сеть считывается из файла

	cout << " Network has been read! " << endl; // считали

	//cout << " Enter image path: ";
	//cin >> path;

	cout << " Image is reading. Wait " << endl; // Читаем изображение

	Mat img = imread(path, CV_LOAD_IMAGE_GRAYSCALE);



	cout << " Image has been read! " << endl; // считали

	threshold(img, img, 127, 255, CV_THRESH_BINARY_INV); // бинаризация

	int max;
	int diametr;

	cout << "Max polynomial: "; // макс степень полинома
	cin >> max;
	cout << endl;
	cout << "Diametr: "; // диаметр
	cin >> diametr;
	cout << endl;

	pm->InitBasis(max, diametr); // генерируем базис компл полиномов

	cout << "  Decomposing image. Wait " << endl;

	blobs = bp->DetectBlobs(img); // выделяет на изображении смежные области (цифры)
	nblobs = bp->NormalizeBlobs(blobs, diametr); // приводит смежные области к единому масштабу

	moments.resize(blobs.size());

	cout << "    Detected " << blobs.size() << " images" << endl; // обнаружено столько-то картинок
	cout << "    Decomposing ";

	for (int i = 0; i < blobs.size(); i++)
	{
		moments[i] = pm->Decompose(nblobs[i]); // перебирает все цифры по очереди, раскладывая их в ряд
		cout << (i + 1) << " "; // номер картинки 
	}

	cout << endl << " End " << endl;

	cout << "  Recognizing images. Wait " << endl;
	cout << "    Recognized: ";

	for (int i = 0; i < blobs.size(); i++)
	{
		string recognition_result = mr.Recognize(moments[i]); //создается объект для распознавания
		cout << recognition_result << " "; // увиденная цифра

		Mat blob;
		resize(nblobs[i], blob, cv::Size(400, 400));

		imshow("Recognized as: " + recognition_result, blob); // выводи результат на экран
		
		waitKey();
		destroyAllWindows();

	}

	cout << endl << " Ok " << endl;
}

int main(int argc, char** argv)
{
	string key;
	do
	{
		cout << "===Enter next walues to do something:===" << endl;
		cout << "  '1' - to generate data." << endl;
		cout << "  '2' - to train network." << endl;
		cout << "  '3' - to check recognizing precision." << endl;
		cout << "  '4' - to recognize single image." << endl;
		cout << "  'exit' - to close the application." << endl;
		cin >> key;
		cout << endl;
		if (key == "1") {
			generateData();
		}
		else if (key == "2") {
			trainNetwork();
		}
		else if (key == "3") {
			precisionTest();
		}
		else if (key == "4") {
			recognizeImage();
		}
		cout << endl;
	} while (key != "exit");
	return 0;
}