#pragma once
#include "IIndividual.h"
#include "ANN.h"

namespace ga
{

	class ANNIndividual
		: public IIndividual
		, public ANN::ANeuralNetwork
	{

	private:

		size_t total_num_of_weights;
		float avg_weight;

	public:

		ANNIndividual(std::vector < int > configuration,
			float max_weight);

		ANNIndividual(const ANNIndividual& ref);

		virtual ~ANNIndividual() { }

	public:

		// IIndividual

		pIIndividual Mutation() override;

		pIIndividual Crossover(pIIndividual individual) override;

		std::pair<int, int> Spare(pIIndividual individual) override = 0;

		std::vector<float> MakeDecision(std::vector<float>& input) override
		{
			return Predict(input);
		}

		pIIndividual Clone() override = 0;

	protected:

		// NeuralNetwork

		std::vector<float> Predict(std::vector<float>& input) override;

		std::string GetType() override
		{
			throw "Not implemented";
		}

		float Train(
			std::vector<std::vector<float>>& inputs,
			std::vector<std::vector<float>>& outputs,
			int max_iters,
			float eps,
			float speed,
			bool std_dump
		)
		{
			throw "Not implemented";
		}
	};
}