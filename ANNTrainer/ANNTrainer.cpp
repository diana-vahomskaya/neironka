#include <iostream>
#include <ANN.h>
#include <string>
using namespace std;
using namespace ANN;

const string input_file = "xor.data"; // файл с обучающими данными
const string output_file = "xor.ann"; // нейронная сеть

int main()
{
	cout << "hello ANN! ANNTrainer!" << endl;
	cout << GetTestString().c_str() << endl;

	std::vector<std::vector<float>> in; // входы
	std::vector<std::vector<float>> out; // выходы

	if (LoadData(input_file, in, out)) // считываем обучающие данные из файла
	{
		vector < int > config({ 2, 10, 10, 1 });

		auto network = CreateNeuralNetwork(config, ANeuralNetwork::POSITIVE_SYGMOID);// создаем нейронную сеть

		BackPropTrain(network, in, out, 100000, 0.1, 0.1, true); // обучаем нейронную сеть

		cout << "Information about network:" << endl;
		cout << network->GetType() << endl; // выводим информацию о типе нейронной сети

		if (network->Save(output_file))// сохраняем обученную нейронную сеть в текстовый файл xor.data
		{
			cout << "Neural network saved." << endl;
		}
	}
	else cout << "Error! Neural network is not saved" << endl;

	system("pause");
	return 0;

}