#pragma once

#ifdef ANNDLL_EXPORTS
#define ANNDLL_API __declspec(dllexport) 
#else
#define ANNDLL_API __declspec(dllimport) 
#endif

#include <vector>
#include <memory>

namespace ANN
{
	class NeuralNetwork
	{
	public:
		enum ActivationType
		{
			POSITIVE_SYGMOID,
			BIPOLAR_SYGMOID
		};
		/**��������� ��������� ���� �� �����
		 *@param
		 */
		virtual bool Read(std::string filepath);
		virtual bool Save(std::string filepath);
		std::vector<int> GetConfiguration();

		/**********************��� ��� ���� ����������� �����**********************/
		virtual std::string GetType() = 0;
		virtual std::vector<float> Predict(std::vector<float> & input) = 0;
		virtual float MakeTrain(
			std::vector<std::vector<float>> & inputs,
			std::vector<std::vector<float>> & outputs,
			int max_iters = 10000,
			float eps = 0.1,
			float speed = 0.1
		) = 0;

	protected:
		/**���� ����*/
		std::vector<std::vector<std::vector<float> > > weights;
		/**������������ ����.
		 * ����� ������� � ������� ������������ ������ ����
		 * �������� - ���������� ��������
		 */
		std::vector<int> configuration;
		/**������� �� ����?*/
		bool is_trained;
		/**�������������� ����������� ��������� ��������*/
		float scale;
		/**��� ������������� �������*/
		ActivationType activation_type;
		/**��������� �������� ������������� �������
		 *@param neuron_input - ������� �������� �������
		 *@return - �������� ������������� ������
		 */
		float Activation(float neuron_input);
		/**��������� �������� ����������� ������������� �������
		 *@param activation - �������� ������������� ��������, ��� ������� ����� ��������� �����������
		 *@return - �������� ����������� ������������� ������
		 */
		float ActivationDerivative(float activation);
	};

	/**������� ��������� ����
	 *@param configuration - ������������ ��������� ����
	 */
	ANNDLL_API std::shared_ptr<ANN::NeuralNetwork> CreateNeuralNetwork(
		std::vector<int> & configuration = std::vector<int>(),
		NeuralNetwork::ActivationType activation_type = NeuralNetwork::POSITIVE_SYGMOID
	);
}