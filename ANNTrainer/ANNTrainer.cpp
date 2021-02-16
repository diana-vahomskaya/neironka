#include <iostream>
#include <time.h>
#include <My_ANN.h>

int main()
{
	printf((ANN::GetTestString() + "\n").c_str());

	std::vector<std::vector<float>> inputs, outputs;

	if (ANN::LoadData("..\\ANNTrainer\\xor.data", inputs, outputs))
	{
		std::vector<size_t> config;
		config.push_back(2); config.push_back(4); config.push_back(1);
		std::shared_ptr<ANN::ANeuralNetwork> pANN = ANN::CreateNeuralNetwork(config);
		srand(time(0));
		pANN->RandomInit();
		ANN::BackPropTraining(pANN, inputs, outputs, 20000, 0.001f, 1.0f, true);
		printf((pANN->GetType() + "\n").c_str());
		if (pANN->Save("..\\ANNTrainer\\xor.ann"))
			printf("Neural network saved.\n");
	}
	else printf("Unable to read the training data!\n");

	system("pause");
	return 0;
}