
#include <iostream>
#include <vector>
#include <thread>

#include "database.h"
#include "mytimer.h"

//pdwLabel: dwNum * 2
//pfFeats: dwNum * dwFeatDim
//pPaths: dwNum
extern "C" int save_feature_info_c(int *pdwLabel, float *pfFeats, 
                         char *pPaths[], int dwNum, int dwFeatDim, char *pFolder)
{
  
  return 0;
}


CDatabase db(CDatabase::DT_FLOAT, CDatabase::DF_L2);

//pfQueryOne: dwFeatDim
//pfDistractorSet: dwDSetSize * dwFeatDim
//pdwIDs: dwDSetSize
//pdwTopNIdx: dwTopN
//pdwTopNScore: dwTopN
extern "C" int init_ranker(float *pfDistractorSet, int *pdwIDs, int dwDSetSize, int dwFeatDim)
{
  db.Initialize(dwDSetSize + 4, dwFeatDim, 1, 1);
  db.AddItems(pfDistractorSet, pdwIDs, dwDSetSize);
  return 0;
}

//pfQueryOne: dwFeatDim
//pdwTopNIdx: dwTopN
//pdwTopNScore: dwTopN
extern "C" int do_ranker(float *pfQueryOne, int dwFeatDim,
                         int *pdwTopNIdx, float *pfTopNScore, int dwTopN)
{
  std::vector<CDatabase::DIST> dists(dwTopN);
  db.QueryTopN(pfQueryOne, dists.size(), dists.data());

  for (int i = 0; i < dwTopN; i++)
  {
    pdwTopNIdx[i] = dists[i].id;
    pfTopNScore[i] = dists[i].dist;
  }
 
  return 0;
}
