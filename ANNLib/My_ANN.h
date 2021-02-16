#pragma once
#define ANNDLL_EXPORTS
#include <iostream>
#include <ANN.h>

namespace ANN
{
	class Network : public ANN::ANeuralNetwork
	{
	public:
		ANNDLL_API Network(
			std::vector<size_t>& configuration,
			ANN::ANeuralNetwork::ActivationType activation_type,
			float scale
		);

		ANNDLL_API virtual std::string GetType() override;
		ANNDLL_API virtual std::vector<float> Predict(std::vector<float>& input) override;
	};

}