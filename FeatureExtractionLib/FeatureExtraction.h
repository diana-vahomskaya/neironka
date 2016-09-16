#pragma once

#ifdef ANNDLL_EXPORTS
#define ANNDLL_API __declspec(dllexport) 
#else
#define ANNDLL_API __declspec(dllimport) 
#endif

#include <vector>
#include <memory>

#include "BlobProcessor.h"
#include "PolynomialManager.h"

namespace fe {
	ANNDLL_API std::string GetTestString();
	ANNDLL_API std::shared_ptr<BlobProcessor> CreateBlobProcessor();
	ANNDLL_API std::shared_ptr<BlobProcessor> CreatePolynomialManager();
};

