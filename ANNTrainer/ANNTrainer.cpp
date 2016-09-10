#include <iostream>
#include <ANN.h>
using namespace std;
using namespace ANN;

int main()
{
	cout << "hello ANN!" << endl;
	auto ann = CreateNeuralNetwork();
	cout << ann->GetType().c_str() << " succesfully created!" << endl;
	system("pause");
	return 0;
}