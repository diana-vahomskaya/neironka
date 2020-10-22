#define ANNDLL_EXPORTS
#include <iostream>
#include <ANN.h>
#include <My_ANN.h>


std::shared_ptr<ANN::ANeuralNetwork> ANN::CreateNeuralNetwork( // Создать нейронную сеть
	std::vector<int>& configuration,
	ANeuralNetwork::ActivationType activation_type)
{
	return std::make_shared<ANN::Network>(configuration, activation_type); // освобождаем память
}

std::string ANN::Network::GetType() // выводим информацию о типе нейронной сети
{
	return "Network by Vahomskaya Diana";
}

std::vector<float> ANN::Network::Predict(std::vector<float>& input) //прогнозирование выхода по известному входу (запоминаем выход последнего нейрона)
{
	std::vector < float > in = input; //входы 
	std::vector < float > out; // выходы 
	for (int i = 0; i < weights.size(); i++) // цикл по слоям
	{
		out.resize(weights[i].size() - 1); // -1 из-за нейрона смещения, у него нет выхода (11-1) выход
		for (int j = 0; j < weights[i].size() - 1; j++) // цикл по нейронам (11-1)
		{
			float neuron = weights[i][j][in.size()];// in - количество нейронов на предыд слое (нейрон смещения)
			for (int k = 0; k < in.size(); k++) // по связям
			{
				neuron += in[k] * weights[i][j][k];
			}
			out[j] = Activation(neuron);
		}
		in.swap(out); // выход делаем входом
	}
	return in;
}

float ANN::BackPropTrain //обучение сети методом обратного распространения ошибки
(
	std::shared_ptr<ANN::ANeuralNetwork>ann,//сеть
	std::vector < std::vector < float > >& inputs, // входы выходы
	std::vector < std::vector < float > >& outputs,
	int max_iters,
	float eps,
	float speed,
	bool std_dump
)
{
	std::vector < std::vector < std::vector < float > > > weight_deltas; // дельта омега

	ann->RandomInit(); //задаем веса рандомно

	////////////////////////////////////////////////////////////////////////////////////////////////////////////

	weight_deltas.resize(ann->configuration.size() - 1); // не учитываем первые нейроны потому что у них нет веса (формируем кол-во слоев) (размер 3)

	for (int i = 0; i < ann->weights.size(); i++) // цикл по слоям 
	{
		weight_deltas[i].resize(ann->configuration[i + 1]); // сколько дельта нейронов в каждом слое

		for (int j = 0; j < ann->weights[i].size(); j++) // цикл по нейронам
		{
			weight_deltas[i][j].resize(ann->configuration[i]); // сколько связей дельта нейронов

		}
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////


	float err; // ошибка
	int iterations = 0; // кол-во итераций

	do // начинаем обучать сеть
	{
		err = 0;
		for (int i = 0; i < inputs.size(); i++) // 4 итерации (нач. данные)
		{
			std::vector < std::vector < float > > layer_outputs(ann->weights.size()); // выход каждого нейрона (кол-во слоев)

			// прогонка вперед

			std::vector < float > in = inputs[i];
			std::vector < float > out;

			for (int t = 0; t < ann->weights.size(); t++) // запоминаем все выходы
			{
				out.resize(ann->weights[t].size() - 1);
				layer_outputs[t].resize(ann->weights[t].size() - 1);

				for (int j = 0; j < ann->weights[t].size() - 1; j++)
				{
					float neuron_input = ann->weights[t][j][in.size()];
					for (int k = 0; k < in.size(); k++)
					{
						neuron_input += in[k] * ann->weights[t][j][k];
					}
					layer_outputs[t][j] = ann->Activation(neuron_input);
					out[j] = layer_outputs[t][j];
				}
				in.swap(out);
			}


			std::vector < float >& output = layer_outputs.back(); // передаем выход последнего нейрона

			std::vector < float > delta_0(output.size()); //  первая дельта для выходного нейрона

			for (int j = 0; j < output.size(); j++)
			{
				float diff = outputs[i][j] - output[j]; // выход ист - вход плучен
				err += diff * diff; // квадратичная ошибка
				delta_0[j] = diff * ann->ActivationDerivative(output[j]); // получаем первую дельту
			}

			// прогонка назад 
			std::vector < float > in1 = delta_0; // начало - конец
			std::vector < float > out1;

			//float moment = 0.01; // момент

			for (int p = ann->weights.size() - 1; p-- > 0;) // цикл по слоям (только до предпоследнего слоя) 
			{
				out1.resize(ann->weights[p].size() - 1); // 10-1 у нейрона смещения нет выхода

				for (int j = 0; j < ann->weights[p].size() - 1; j++) //цикл по нейронам
				{
					float delta_neuron = 0;
					for (int k = 0; k < in1.size(); k++) // цикл по связям
					{
						delta_neuron += in1[k] * ann->weights[p + 1][k][j]; // p+1 берем веса с последнего слоя ( сумма произведений весов на дельту)
					}
					delta_neuron *= ann->ActivationDerivative(layer_outputs[p][j]); // умножаем на производную от соотв выхода
					out1[j] = delta_neuron;
				}

				for (int k = 0; k < in1.size(); k++) // по связям
				{
					for (int j = 0; j < ann->weights[p].size() - 1; j++) //10-1 нейроны
					{
						weight_deltas[p + 1][k][j] = speed * layer_outputs[p][j] * in1[k];
						ann->weights[p + 1][k][j] += weight_deltas[p + 1][k][j]; // прибавляем дельту к старому весу и получаем новый
					}
					weight_deltas[p + 1][k][ann->weights[p + 1][k].size() - 1] = speed * in1[k]; // для нейрона смещения
					ann->weights[p + 1][k][ann->weights[p + 1][k].size() - 1] += weight_deltas[p + 1][k][ann->weights[p + 1][k].size() - 1];
				}
				in1.swap(out1);
			}
			//+ moment * weight_deltas[p + 1][k][j]
			//+ moment * weight_deltas[p + 1][k][ann->weights[p + 1][k].size() - 1]
			for (int k = 0; k < in1.size(); k++) // для входных нейронов (по нейронам)
			{
				for (int j = 0; j < inputs[i].size(); ++j) // по связям
				{
					weight_deltas[0][k][j] = speed * inputs[i][j] * in1[k];
					ann->weights[0][k][j] += weight_deltas[0][k][j];
				}
				weight_deltas[0][k][ann->weights[0][k].size() - 1] = speed * in1[k]; //нейрон смещения
				ann->weights[0][k][ann->weights[0][k].size() - 1] += weight_deltas[0][k][ann->weights[0][k].size() - 1];
			}

			//+ moment * weight_deltas[0][k][j]
			//+ moment * weight_deltas[0][k][ann->weights[0][k].size() - 1]
			if (std_dump && ((iterations % (max_iters / 100)) == 0)) // вывод итераций 
			{
				std::cout << iterations << ": " << err << std::endl;
			}
			iterations++;
		}
	} while ((err > eps) && (iterations < max_iters));

	ann->is_trained = true; // сеть обучена (если не обучена сеть, то файл не записывается)

	if (std_dump)
	{
		std::cout << iterations << ": " << err << std::endl; // конечный результат (макс кол-воитераций) и поледняя ошибка на посл итерации
	}

	return err;

}

