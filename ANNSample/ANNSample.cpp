#include <iostream>
#include <ANN.h>
#include <string>
using namespace std;
using namespace ANN;

const string input_file = "xor.data";
const string network_file = "..\\ANNTrainer\\xor.ann";

template < typename T > // шаблон
std::ostream& operator << (std::ostream& out, std::vector < T > v) // перегрузка оператора для вывода вектора
{
	for (size_t i = 0; i < v.size(); ++i)
	{
		out << v[i];
		if ((i + 1) != v.size())
		{
			out << ", ";
		}
	}
	return out;
}

int main()
{
	vector<size_t> configuration({ 2, 10, 10, 1 });
	auto network = CreateNeuralNetwork(configuration, ANN::ANeuralNetwork::POSITIVE_SYGMOID); // создаем нейронную сеть

	network->Load(network_file);  // читаем нейронную сеть из файла

	cout << "Information about network:" << endl;
	cout << network->GetType() << endl; // выводим информацию о типе нейронной сети

	std::vector<std::vector<float>> in; // входы
	std::vector<std::vector<float>> out; // выходы

	LoadData(input_file, in, out); // читаем тестовые данные из файла

	for (int i = 0; i < in.size(); ++i)
	{
		auto out_my = network->Predict(in[i]); //подаем на вход сети тестовые данные
		cout << endl;
		cout << "in:     " << in[i] << endl;
		cout << "out:    " << out[i] << endl;
		cout << "out_my: " << out_my << endl;
	}

	system("pause");
	return 0;
}