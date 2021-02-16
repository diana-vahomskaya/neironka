#include <iostream>
#include "GeneticAlgorithm.h"
#include <string>
#include <ANN.h>
#include <algorithm>
using  namespace std;
using namespace ga;

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


void xor_demo()
{
	cout << "Population size: "; int population_size = 20; cin >> population_size;
	if (population_size > 0)
	{
		vector<size_t> config;
		// Neural network configuration:
		config.push_back(2); config.push_back(4); config.push_back(1);
		GeneticAlgorithm GAl(config, population_size);

		cout << "Percent of survivors:   "; float unchange_perc = 0.5;  cin >> unchange_perc;
		cout << "Percent of descendants: "; float crossover_perc = 0.2; cin >> crossover_perc;
		cout << "Percent of permutation: "; float mutation_perc = 0.05; cin >> mutation_perc;
		unchange_perc *= 0.01; crossover_perc *= 0.01; mutation_perc *= 0.01;
		cout << "Maximum number of epochs to pass: "; int lineage = 4;  cin >> lineage;
		cout << endl << "Evolution start!" << endl;
		while ((lineage > 0) && (population_size > 1))
		{
			GAl.epoch->EpochBattle();
			cout << endl << "Scoreboard:" << endl;
			for each (auto unit in GAl.epoch->population) { cout << unit.first << '\t'; }

			GAl.Selection(unchange_perc, mutation_perc, crossover_perc);
			population_size = GAl.epoch->population.size();
			lineage--;
		}
		GAl.epoch->EpochBattle();
		sort(GAl.epoch->population.begin(), GAl.epoch->population.end(),
			[](pair<int, pIIndividual> a, pair<int, pIIndividual> b)
			{ return a.first > b.first; });

		vector<vector<float>> inputs, outputs;
		ANN::LoadData("xor.data", inputs, outputs);
		cout << endl << "Total evolution results:" << endl << "Score\t";
		for (int k = 0; k < inputs.size(); k++)
		{
			cout << inputs[k] << "\t\t";
		}
		for (int i = 0; i < population_size; i++)
		{
			cout << endl << GAl.epoch->population[i].first;
			float eps = 0.0f;
			for (int k = 0; k < inputs.size(); k++)
			{
				auto prediction = GAl.epoch->population[i].second->MakeDecision(inputs[k]);
				for (int l = 0; l < outputs[k].size(); l++) {
					cout << '\t' << prediction[l];
					eps += (outputs[k][l] - prediction[l]) * (outputs[k][l] - prediction[l]);
				}
			}
			cout << "\tError=" << eps << endl;
		}
	}
	cout << endl; system("pause");
}

int main()
{
    cout << "Hello!" << endl;

    while (1)
    {
        cout << "1 - XOR" << endl;
        cout << "0 - Exit" << endl;
        cout << endl;

        int option; cin >> option;

        switch (option)
        {
        case 0:
            return 0;
        case 1:
            xor_demo();
            break;
        default:
            break;
        }
    }

    return 0;
}


//if (lineage > 0) { cout << "Extinction reached! " << lineage << " more epochs remained." << endl; }