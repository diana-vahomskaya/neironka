#include "ANNIndividual.h"
#include "Util.h"
#include <vector>


ga::ANNIndividual::ANNIndividual(std::vector < int > configuration,
	float max_weight)
{
	this->configuration = configuration;
	this->activation_type = ANN::ANeuralNetwork::ActivationType::BIPOLAR_SYGMOID;
	this->is_trained = true;
	this->scale = 1;
	this->total_num_of_weights = 0;
	this->avg_weight = 0;

	weights.resize(configuration.size() - 1);
	for (size_t i = 0; i < weights.size(); ++i)
	{
		weights[i].resize(configuration[i + 1]);
		for (size_t j = 0; j < weights[i].size(); ++j)
		{
			weights[i][j].resize(configuration[i]);
			for (size_t k = 0; k < weights[i][j].size(); ++k)
			{
				weights[i][j][k] = (float)rand() / RAND_MAX * max_weight * 2 - max_weight;
				this->avg_weight += abs(weights[i][j][k]);
				++total_num_of_weights;
			}
		}
	}
	this->avg_weight /= total_num_of_weights;
}

ga::ANNIndividual::ANNIndividual(const ANNIndividual& ref)
{
	this->configuration = ref.configuration;
	this->activation_type = ref.activation_type;
	this->is_trained = true;
	this->scale = 1;
	this->total_num_of_weights = 0;
	this->avg_weight = 0;

	weights.resize(configuration.size() - 1);
	for (size_t i = 0; i < weights.size(); ++i)
	{
		weights[i].resize(configuration[i + 1]);
		for (size_t j = 0; j < weights[i].size(); ++j)
		{
			weights[i][j].resize(configuration[i]);
			for (size_t k = 0; k < weights[i][j].size(); ++k)
			{
				weights[i][j][k] = ref.weights[i][j][k];
				this->avg_weight += abs(weights[i][j][k]);
				++total_num_of_weights;
			}
		}
	}
	this->avg_weight /= total_num_of_weights;
}

ga::pIIndividual ga::ANNIndividual::Mutation()
{
	auto clone = Clone();
	ANNIndividual* self = (ANNIndividual*)clone.get();
	for (size_t i = 0; i < weights.size(); ++i)
	{
		for (size_t j = 0; j < weights[i].size(); ++j)
		{
			for (size_t k = 0; k < weights[i][j].size(); ++k)
			{
				if (RandomBool(0.05))
				{
					self->weights[i][j][k]
						+= (((float)rand() / RAND_MAX) * 2 * avg_weight - avg_weight) / 10;
				}
			}
		}
	}
	return std::move(clone);
}

ga::pIIndividual ga::ANNIndividual::Crossover(pIIndividual individual)
{
	auto clone = Clone();
	ANNIndividual* self = (ANNIndividual*)clone.get();
	ANNIndividual* other = (ANNIndividual*)individual.get();
	size_t crossover_threshold = ((float)rand() / RAND_MAX) * total_num_of_weights;
	for (size_t i = 0; i < weights.size(); ++i)
	{
		for (size_t j = 0; j < weights[i].size(); ++j)
		{
			for (size_t k = 0; k < weights[i][j].size(); ++k)
			{
				// Intermediate recombination
				// see Panchenko T.V. sec.2.4 p.19
				float d = 0.25;
				float alpha = ((float)rand() / RAND_MAX) * (2 * d + 1) - d;
				self->weights[i][j][k] += alpha * (other->weights[i][j][k] - self->weights[i][j][k]);
			}
		}
	}
	return std::move(clone);
}

std::vector<float> ga::ANNIndividual::Predict(std::vector<float>& input)
{
	std::vector < float > in = input;
	std::vector < float > out;
	for (size_t i = 0; i < weights.size(); ++i)
	{
		out.resize(weights[i].size());
		for (size_t j = 0; j < weights[i].size(); ++j)
		{
			float neuron_input = 0;
			for (size_t k = 0; k < in.size(); ++k)
			{
				neuron_input += in[k] * weights[i][j][k];
			}
			out[j] = Activation(neuron_input);
		}
		in.swap(out);
	}
	return in;
}