#define ANNDLL_EXPORTS
#include "FeatureExtraction.h"

std::string fe::GetTestString()
{
	return "You successfuly plug feature extraction library!";
}

std::shared_ptr<fe::BlobProcessor> fe::CreateBlobProcessor()
{
	return NULL;
}

std::shared_ptr<fe::BlobProcessor> fe::CreatePolynomialManager()
{
	return NULL;
}
