#define ANNDLL_EXPORTS
#include <ANN.h>
#include <fstream>
#include <iomanip>


bool ANN::NeuralNetwork::Load(std::string filepath)
{
	std::ifstream file(filepath);
	if (!file.is_open()) return false;
	int buffer;
	const int CHAR_BUF_LEN = 100;
	char char_buffer[CHAR_BUF_LEN];
	file.getline(char_buffer, CHAR_BUF_LEN);
	std::string string_buffer = std::string(char_buffer);
	memset(char_buffer, 0, CHAR_BUF_LEN);
	if (string_buffer != std::string("activation type:")) 
		throw "incorrect file format";
	file >> buffer;
	activation_type = (ActivationType)buffer;
	file.getline(char_buffer, CHAR_BUF_LEN);
	file.getline(char_buffer, CHAR_BUF_LEN);
	string_buffer = std::string(char_buffer);
	memset(char_buffer, 0, CHAR_BUF_LEN);
	if (string_buffer != std::string("activation scale:"))
		throw "incorrect file format";
	file >> scale;
	file.getline(char_buffer, CHAR_BUF_LEN);
	file.getline(char_buffer, CHAR_BUF_LEN);
	string_buffer = std::string(char_buffer);
	memset(char_buffer, 0, CHAR_BUF_LEN);
	if (string_buffer != std::string("configuration:")) 
		throw "incorrect file format";
	file >> buffer;
	configuration.resize(buffer);
	for (size_t i = 0; i < configuration.size(); i++) {
		file >> configuration[i];
	}
	file.getline(char_buffer, CHAR_BUF_LEN);
	file.getline(char_buffer, CHAR_BUF_LEN);
	string_buffer = std::string(char_buffer);
	memset(char_buffer, 0, CHAR_BUF_LEN);
	if (string_buffer != std::string("weights:"))
		throw "incorrect file format";
	weights.resize(configuration.size()-1);
	for (size_t i = 1; i < weights.size(); i++) {
		weights[i].resize(configuration[i]);
		for (size_t j = 0; j < weights.size(); j++) {
			weights[i][j].resize(configuration[i-1]);
			for (size_t k = 0; k < weights[i][j].size(); k++) {
				file >> weights[i][j][k];
			}
		}
	}
	file.close();
	is_trained = true;
	return true;
}

bool ANN::NeuralNetwork::Save(std::string filepath)
{
	if (!is_trained) return false;
	std::ofstream file(filepath);
	if (!file.is_open()) return false;
	file << std::setprecision(9);
	file << "activation type:" << std::endl;
	file << (int)activation_type << std::endl;
	file << "activation scale:" << std::endl;
	file << scale << std::endl;
	file << "configuration:" << std::endl;
	file << configuration.size() << "\t";
	for each (int neuron_count in configuration) {
		file << neuron_count << "\t";
	}
	file << std::endl << "weights:" << std::endl;
	for each (auto weight_matrix in weights) {
		for each (auto weight_line in weight_matrix) {
			for each (auto weight in weight_line) {
				file << weight << " ";
			}
			file << std::endl;
		}
	}
	file.close();
	return true;
}

std::vector<int> ANN::NeuralNetwork::GetConfiguration()
{
	return configuration;
}

float ANN::NeuralNetwork::Activation(float neuronInput)
{
	if (activation_type == POSITIVE_SYGMOID) {
		return (1.f / (1.f + expf(-scale * neuronInput)));
	}
	else if (activation_type == BIPOLAR_SYGMOID) {
		return (2.f / (1.f + expf(-scale * neuronInput)) - 1.f);
	}
	return -1.f;
}

float ANN::NeuralNetwork::ActivationDerivative(float activation)
{
	if (activation_type == POSITIVE_SYGMOID) {
		return scale * activation * (1.f - activation);
	}
	else if (activation_type == BIPOLAR_SYGMOID) {
		return scale * 0.5f * (1.f + activation) * (1.f - activation);
	}
	return -1;
}
