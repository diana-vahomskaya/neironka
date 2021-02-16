#include <algorithm>
#include "GeneticAlgorithm.h"

using namespace ga;
using namespace std;

GeneticAlgorithm::GeneticAlgorithm(vector<size_t> config, int population_size)
{
	// —оздание новой попул¤ции:
	vector<pair<int, pIIndividual>> new_population;
	for (int i = 0; i < population_size; i++)
	{
		Individual newIn(config, ANN::ANeuralNetwork::ActivationType::POSITIVE_SYGMOID, 1.0f, "xor.data");
		pIIndividual new_individual = std::make_shared<Individual>(newIn);
		new_population.push_back(pair<int, pIIndividual>(0, new_individual));
	}
	Epoch newEpoch;
	this->epoch = make_shared<Epoch>(newEpoch);
	this->epoch->population = new_population;
}


GeneticAlgorithm::~GeneticAlgorithm()
{
	this->epoch->population.clear();
}

pEpoch GeneticAlgorithm::Selection(double unchange_perc, double mutation_perc, double crossover_perc)
{
	// —ортировка особей по набранным очкам.
	sort(epoch->population.begin(), epoch->population.end(),
		[](pair<int, pIIndividual> a, pair<int, pIIndividual> b)
		{ return a.first > b.first; });

	// —оздание новой попул¤ции:
	vector<pair<int, pIIndividual>> new_population;
	// ѕеренос сильнейших особей без изменений:
	int survivors_count = (int)(epoch->population.size() * unchange_perc);
	for (int i = 0; i < survivors_count; i++)
	{
		new_population.push_back(epoch->population[i]);
	}
	// —крещивание случайно выбранных сильнейших особей:
	std::random_device rd; std::mt19937 randm(rd());
	for (int j = 0; j < (int)(epoch->population.size() * crossover_perc); j++)
	{
		int p1 = randm() % survivors_count;
		int p2 = randm() % survivors_count;
		pIIndividual childe = new_population[p1].second->Crossover(new_population[p2].second);
		new_population.push_back(pair<int, pIIndividual>(0, childe));
	}
	// ƒополнение мутировавшими особ¤ми:
	for (int k = 0; k < (int)(epoch->population.size() * mutation_perc); k++)
	{
		int r = randm() % survivors_count;
		pIIndividual mutant = new_population[r].second->Mutation();
		new_population.push_back(pair<int, pIIndividual>(0, mutant));
	}

	// ѕереход в новую эпоху:
	this->epoch->population = new_population;
	return this->epoch;
}
