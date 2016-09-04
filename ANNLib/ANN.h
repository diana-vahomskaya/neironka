#pragma once

#ifdef ANNDLL_EXPORTS
#define ANNDDLL_API __declspec(dllexport) 
#else
#define ANNDSDLL_API __declspec(dllimport) 
#endif

ANNDSDLL_API int func();