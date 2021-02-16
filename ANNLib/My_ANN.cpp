#define ANNDLL_EXPORTS
#include <iostream>
#include <ANN.h>
#include <My_ANN.h>


ANN::Network::Network(
	std::vector<size_t>& configuration,
	ANN::ANeuralNetwork::ActivationType activation_type,
	float scale
)
{
	this->configuration = configuration;
	this->activation_type = activation_type;
	this->scale = scale;
}


std::shared_ptr<ANN::ANeuralNetwork> ANN::CreateNeuralNetwork(
	std::vector<size_t>& configuration,
	ANN::ANeuralNetwork::ActivationType activation_type,
	float scale
)
{
	return std::make_shared<ANN::Network>(configuration, activation_type, scale);
}


std::string ANN::Network::GetType()
{
	return "Neural network implementation by Diana Vahomskaya";
}


std::vector<float> ANN::Network::Predict(std::vector<float>& input)
{
	std::vector<float> result, buffer = input;
	for (size_t layer_idx = 0; layer_idx < (configuration.size() - 1); layer_idx++)
	{
		result.resize(configuration[layer_idx + 1]);
		for (size_t to_idx = 0; to_idx < configuration[layer_idx + 1]; to_idx++)
		{
			result[to_idx] = 0.0f;
			for (size_t from_idx = 0; from_idx < configuration[layer_idx]; from_idx++)
				result[to_idx] += buffer[from_idx] * weights[layer_idx][from_idx][to_idx];
			result[to_idx] = Activation(result[to_idx]);
		}
		buffer = result;
	}
	return result;
}


float ANN::BackPropTraining(
	std::shared_ptr<ANN::ANeuralNetwork> ann,
	std::vector<std::vector<float>>& inputs,
	std::vector<std::vector<float>>& outputs,
	int maxIters,
	float eps,
	float speed,
	bool std_dump
)
{
	float err = 0.0;
	for (size_t j = 0; j < inputs.size(); j++)
	{
		std::vector<float> result = ann->Predict(inputs[j]);
		for (size_t i = 0; i < result.size(); i++)
			err += (result[i] - outputs[j][i]) * (result[i] - outputs[j][i]);
	}
	err *= 0.5f;

	size_t Iters = 0;
	do {
		err = 0.0;
		for (size_t j = 0; j < inputs.size(); j++)
			err += BackPropTrainingIteration(ann, inputs[j], outputs[j], speed);
		err /= (float)(inputs.size());

		Iters++;
		if (std_dump && (Iters % 100 == 0)) printf("\t%i: error %.4f\n", Iters, err);

		if (err < eps) break;
	} while (Iters < maxIters);

	printf("\t%i: error %.4f\n", Iters, err);
	ann->is_trained = true;
	return err;
}


float ANN::BackPropTrainingIteration(
	std::shared_ptr<ANN::ANeuralNetwork> ann,
	const std::vector<float>& input,
	const std::vector<float>& output,
	float speed
)
{
	// Получение входных и выходных значений всех нейронов:
	std::vector<std::vector<float>> buffers = std::vector<std::vector<float>>(ann->configuration.size());
	buffers[0] = input;
	for (size_t layer_idx = 1; layer_idx < buffers.size(); layer_idx++)
	{
		buffers[layer_idx] = std::vector<float>(ann->configuration[layer_idx]);
		for (size_t to_idx = 0; to_idx < buffers[layer_idx].size(); to_idx++)
		{
			buffers[layer_idx][to_idx] = 0.0f;
			for (size_t from_idx = 0; from_idx < buffers[layer_idx - 1].size(); from_idx++)
				buffers[layer_idx][to_idx] += buffers[layer_idx - 1][from_idx] * ann->weights[layer_idx - 1][from_idx][to_idx];
			buffers[layer_idx][to_idx] = ann->Activation(buffers[layer_idx][to_idx]);
		}
	}

	// Корректировка весов нейронов выходного слоя:
	std::vector<float> deltas = std::vector<float>(ann->configuration.back());
	size_t corr_idx = ann->weights.size() - 1;
	for (size_t to_idx = 0; to_idx < ann->configuration[corr_idx + 1]; to_idx++)
	{
		deltas[to_idx] = (buffers[corr_idx + 1][to_idx] - output[to_idx]);
		deltas[to_idx] *= ann->ActivationDerivative(buffers[corr_idx + 1][to_idx]);

		for (size_t from_idx = 0; from_idx < ann->configuration[corr_idx]; from_idx++)
			ann->weights[corr_idx][from_idx][to_idx] -= speed * deltas[to_idx] * buffers[corr_idx][from_idx];
	}

	// Корректировка весов нейронов внутренних слоёв:
	while (corr_idx > 0)
	{
		corr_idx--;
		std::vector<float> deltas_new = std::vector<float>(ann->configuration[corr_idx + 1]);
		for (size_t to_idx = 0; to_idx < ann->configuration[corr_idx + 1]; to_idx++)
		{
			deltas_new[to_idx] = 0.0;
			for (size_t k = 0; k < ann->configuration[corr_idx + 2]; k++)
				deltas_new[to_idx] += deltas[k] * ann->weights[corr_idx + 1][to_idx][k];
			deltas_new[to_idx] *= ann->ActivationDerivative(buffers[corr_idx + 1][to_idx]);

			for (size_t from_idx = 0; from_idx < ann->configuration[corr_idx]; from_idx++)
				ann->weights[corr_idx][from_idx][to_idx] -= speed * deltas_new[to_idx] * buffers[corr_idx][from_idx];
		}
		deltas = deltas_new;
	}

	// Получение нового выхода нейронной сети:
	buffers[0] = input;
	for (size_t layer_idx = 1; layer_idx < buffers.size(); layer_idx++)
	{
		for (size_t to_idx = 0; to_idx < buffers[layer_idx].size(); to_idx++)
		{
			buffers[layer_idx][to_idx] = 0.0f;
			for (size_t from_idx = 0; from_idx < buffers[layer_idx - 1].size(); from_idx++)
				buffers[layer_idx][to_idx] += buffers[layer_idx - 1][from_idx] * ann->weights[layer_idx - 1][from_idx][to_idx];
			buffers[layer_idx][to_idx] = ann->Activation(buffers[layer_idx][to_idx]);
		}
	}
	// Сравнение полученного выхода с исходными данными:
	float err = 0.0f;
	for (size_t i = 0; i < buffers.back().size(); i++)
		err += (buffers.back()[i] - output[i]) * (buffers.back()[i] - output[i]);
	return 0.5f * err;
}
/*std::shared_ptr<ANN::ANeuralNetwork> ANN::CreateNeuralNetwork( // Создать нейронную сеть
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

			// делаем прогонку вперед

			std::vector < float > in = inputs[i];
			std::vector < float > out;

			for (int t = 0; t < ann->weights.size(); t++) // запоминаем все выходы
			{
				out.resize(ann->weights[t].size() - 1); //изменяем размер выходов 
				layer_outputs[t].resize(ann->weights[t].size() - 1);//изменяем размер кол-ва слоев 

				for (int j = 0; j < ann->weights[t].size() - 1; j++)
				{
					float neuron_input = ann->weights[t][j][in.size()]; //входные нейроны
					for (int k = 0; k < in.size(); k++)// цикл по связям
					{
						neuron_input += in[k] * ann->weights[t][j][k]; //каждый входной нейрон отправляет полученный сигнал всем нейронам в след.слое(скрытом) 
						//- суммирование взвешенных входящих сигналов
					}
					layer_outputs[t][j] = ann->Activation(neuron_input); //применяем активационную функцию
					out[j] = layer_outputs[t][j]; // посылем результат всем элементам след.слоя (выходного)
				}
				in.swap(out);//вход становится выходом
			}


			std::vector < float >& output = layer_outputs.back(); // передаем выход последнего нейрона

			std::vector < float > delta_0(output.size()); //  первая дельта для выходного нейрона

			//Вычисление ошибки
			for (int j = 0; j < output.size(); j++)
			{
				float diff = outputs[i][j] - output[j]; // выход ист - вход плучен
				err += diff * diff; // квадратичная ошибка

				//вычисление корректировки смещения 
				delta_0[j] = diff * ann->ActivationDerivative(output[j]); // получаем первую дельту и посылает нейронам в пред.слое
			}

			// прогонка назад 
			std::vector < float > in1 = delta_0; // начало становится концом
			std::vector < float > out1;

			for (int p = ann->weights.size() - 1; p-- > 0;) // цикл по слоям (только до предпоследнего слоя) 
			{
				out1.resize(ann->weights[p].size() - 1); // 10-1 у нейрона смещения нет выхода

				for (int j = 0; j < ann->weights[p].size() - 1; j++) //цикл по нейронам
				{
					float delta_neuron = 0;
					for (int k = 0; k < in1.size(); k++) // цикл по связям
					{
						//каждый скрытый нейрон суммирует входящие ошибки (от нейронов в предыдущем слое)
						delta_neuron += in1[k] * ann->weights[p + 1][k][j]; // p+1 берем веса с последнего слоя ( сумма произведений весов на дельту)
					}
					//вычисляем величину ошибки умножая полученное значение на производную активационной функции
					delta_neuron *= ann->ActivationDerivative(layer_outputs[p][j]); // умножаем на производную от соотв выхода
					out1[j] = delta_neuron;
				}

				//Так же вычисляем величину на которую изменится вес связи
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

			//Изменение весов
			for (int k = 0; k < in1.size(); k++) // цикл для входных нейронов (по нейронам)
			{
				for (int j = 0; j < inputs[i].size(); ++j) // цикл по связям
				{
					weight_deltas[0][k][j] = speed * inputs[i][j] * in1[k];//
					ann->weights[0][k][j] += weight_deltas[0][k][j];
				}
				weight_deltas[0][k][ann->weights[0][k].size() - 1] = speed * in1[k]; // рассчет веса скрытых нейроновб каждый скрытый нейрон изменяет 
				//веса своих связей с элементам смещения и выходными нейронами
				ann->weights[0][k][ann->weights[0][k].size() - 1] += weight_deltas[0][k][ann->weights[0][k].size() - 1];//получение нового веса
			}

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

*/