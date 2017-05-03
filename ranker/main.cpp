#include "database.h"
#include "mytimer.h"
#include <algorithm>
#include <iostream>
#include <numeric>
#include <vector>
#include <thread>

#include <memory>

#define PRINTVAL(e) std::cout << #e << "=" << (e) << std::endl;

int main(int nArgCnt, char *ppArgs[])
{
	typedef float _DTYPE;

	CDatabase::DATATYPE dt;
	float fFactor = 1.0f;
	if (typeid(short) == typeid(_DTYPE))
	{
		dt = CDatabase::DT_SHORT;
		fFactor = 32767.0f;
	}
	else if(typeid(float) == typeid(_DTYPE))
	{
		dt = CDatabase::DT_FLOAT;
		fFactor = 1.0f;
	}
	
	int nCap = 500000;
	int nLen = 128;
	int nAvgCnt = 100;
	int nTops = 2;
	int nGpus = -1;
	
	for (int i = 1; i < nArgCnt; ++i)
	{
		switch (ppArgs[i][0])
		{
		case 'n':
			nCap = atoi(ppArgs[i] + 1);
			break;
		case 'l':
			nLen = atoi(ppArgs[i] + 1);
			break;
		case 'a':
			nAvgCnt = atoi(ppArgs[i] + 1);
			break;
		case 't':
			nTops = atoi(ppArgs[i] + 1);
			break;
		case 'g':
			nGpus = atoi(ppArgs[i] + 1);
			break;
		default:
			std::cout << "Bad argument: " << ppArgs[i] << std::endl;
		}
	}
	if (nCap <= 0 || nLen <= 0 || nArgCnt <= 0 || nCap < nTops)
	{
		std::cout << "Bad argument!" << std::endl;
		return -1;
	}
	std::cout << "\n\tExecuting command: ";
	std::cout << ppArgs[0] << " n" << nCap << " l" << nLen;
	std::cout << " a" << nAvgCnt << " t" << nTops;
	std::cout << " g" << nGpus << std::endl << std::endl;
	
	std::cout << "(n)umber of items: " << nCap << std::endl;
	std::cout << "(l)ength of vector: " << nLen << std::endl;
	std::cout << "(a)verage count: " << nAvgCnt << std::endl;
	std::cout << "(t)ops: " << nTops << std::endl;
	std::cout << "(g)pus: " << nGpus << std::endl;
	
	std::cout << "\nInitializing and loading data..." << std::endl;
	
	CDatabase db(dt);
	db.Initialize(nCap + 4, nLen, 10, nGpus);

	std::vector<_DTYPE> items(nCap * nLen);
	std::vector<float> tmp(nLen);
	for (int i = 0; i < nCap; ++i)
	{
		float fDist = 0.0f;
		for (int l = 0; l < nLen; ++l)
		{
			tmp[l] = (float)std::rand() / (float)RAND_MAX;
			fDist += tmp[l] * tmp[l];
		}
		fDist = fFactor / std::sqrt(fDist);
		for (int l = 0; l < nLen; ++l)
		{
			items[i * nLen + l] = (_DTYPE)(tmp[l] * fDist);
		}
	}
	std::vector<CDatabase::_ITYPE> ids(nCap);
	std::iota(ids.begin(), ids.end(), 0);
	
	//Create a thread to add items
	std::thread addTest([&](){
			db.AddItems(items.data(), ids.data(), nCap);
		});
	addTest.join();

	std::cout << nCap << " itmes loaded." << std::endl;
	
	std::cout << "\nTesting..." << std::endl;
	double dTime = 0.0;
	
	//Create a thread to perform query
	auto QueryTestFunc = [&](){
		std::vector<CDatabase::DIST> dists(nTops);
		std::vector<_DTYPE> query(nLen);
		for (int k = 0;; ++k)
		{
			int nQueryId = std::rand() % nCap;
			std::copy(
				items.begin() + nQueryId * nLen,
				items.begin() + (nQueryId + 1) * nLen,
				query.begin()
			);
			db.QueryTopN(query.data(), dists.size(), dists.data());
			if (nQueryId != dists[0].id)
			{
				std::cout << "QUERY ERROR!" << std::endl;
				exit(-1);
			}
		}
	};
	std::vector<std::thread> queryThreads;
	for (int i = 0; i < 10; ++i)
	{
		queryThreads.push_back(std::thread(QueryTestFunc));
	}

	for (int i = 0; i < 10; ++i)
	{
		queryThreads[i].join();
	}
	std::cout << "YOU SHOULDN'T SEE THIS!" << std::endl;
	return 0;
}
