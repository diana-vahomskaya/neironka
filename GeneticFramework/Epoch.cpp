#include "Epoch.h"


ga::Epoch::Epoch()
{
}


ga::Epoch::~Epoch()
{
}

void ga::Epoch::EpochBattle()
{
	for each (auto individual1 in population) {
		for each (auto individual2 in population) {
			auto score = individual1.second->Spare(individual2.second);
			individual1.first += score.first;
			individual2.first += score.second;
		}
	}
}