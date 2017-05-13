
#include "simple_thread_pool.h"
#include "random"

#include "cv.h"
#include "highgui.h"


struct Aug_Params
{
  string strfn;
  cv::Mat matOut;
  condition_variable *p_cv;
  mutex *p_countmt;
  int *p_FinishCount;
  int needNum;
  int stdsize[2];
  float *pfImgOut;
};

int showimage()
{
  cout << "hello world\n";
  cv::Mat img = cv::imread("/Users/starimpact/work/2.jpg");
  if (img.cols==0)
  {
    cout << "can not read image.\n";
    return 0;
  }
  cv::imshow("hi", img);
  cv::waitKey(0);
  return 0;
}

void do_augment_onethread(void *p);
extern "C" int do_augment_threads(char *pfns[], int num, 
                                  int stdH, int stdW, float *pfImgOut)
{
  const int cdwMaxTNum = 24;
  static dg::ThreadPool *psPool = NULL;
  if (psPool == NULL)
  {
    printf("Max Thread Number:%d\n", cdwMaxTNum);
    psPool = new dg::ThreadPool(cdwMaxTNum);
  }

  srand(static_cast<unsigned>(time(0)));

  vector<string> vecfn;
  for (int i = 0; i < num; i++)
  {
    vecfn.push_back(pfns[i]);
  }
  condition_variable cv;
  mutex countmt;
  int dwFinishCount = 0;
  int fnum = vecfn.size();
  Aug_Params *pParams = new Aug_Params[fnum];

  for (int fi=0; fi < fnum; fi++) 
  {
    string strfn = vecfn[fi];
    pParams[fi].strfn = strfn;
    pParams[fi].p_cv = &cv;
    pParams[fi].p_countmt = &countmt;
    pParams[fi].p_FinishCount = &dwFinishCount;
    pParams[fi].needNum = fnum;
    pParams[fi].stdsize[0] = stdH;
    pParams[fi].stdsize[1] = stdW;
    pParams[fi].pfImgOut = pfImgOut + fi * stdH * stdW * 3;
    psPool->enqueue(do_augment_onethread, (void*)&pParams[fi]);
  }

  unique_lock<mutex> waitlc(countmt);
  cv.wait(waitlc, [&dwFinishCount, &fnum](){return dwFinishCount==fnum;});

//  for (int fi = 0; fi < fnum; fi++)
//  {
//    cv::Mat &img = pParams[fi].matOut;
//    memcpy(pfImgOut + fi * stdH * stdW * 3,  
//           img.data, sizeof(float) * stdH * stdW * 3);
////    cv::imshow("hi", img);
////    cv::waitKey(0);
//  }

  delete []pParams;
  return 0;
}



int rnd_crop(cv::Mat &matIn);
int rnd_rotate(cv::Mat &matIn);
int normalize_img(cv::Mat &matIn);
int rnd_mask(cv::Mat &matIn);

void do_augment_onethread(void *p)
{
  Aug_Params *pParam = (Aug_Params*)p;
  string &strfn = pParam->strfn; 
  mutex *p_countmt = pParam->p_countmt;
  condition_variable *p_cv = pParam->p_cv;
  int *p_FinishCount = pParam->p_FinishCount;
  int needNum = pParam->needNum;
  int stdH = pParam->stdsize[0];
  int stdW = pParam->stdsize[1];
  cv::Mat &matOut = pParam->matOut;
  float *pfImgOut = pParam->pfImgOut;

  cv::Mat img = cv::imread(strfn);
  if (img.cols==0)
  {
    printf("Can not read image %s\n", strfn.c_str());

    unique_lock<mutex> countlc(*p_countmt);
    if (*p_FinishCount < needNum)
    {
      (*p_FinishCount)++;
    }
    if ((*p_FinishCount) == needNum)
    {
      p_cv->notify_all();
    } 
    countlc.unlock();

    return;
  }
 
//  printf("%s\n", strfn.c_str()); 
  //mask rows
//  int rnd0 = rand();
//  if (rnd0 < (RAND_MAX / 4) * 3)
  {
//    rnd_mask(img);
  }
  
  //crop
//  rnd_crop(img);
  //reisze
  cv::resize(img, img, cv::Size(stdW, stdH));
  //normalize
  normalize_img(img);
  //rotate
//  rnd_rotate(img);
  //flip
  int rnd = rand();
  if (rnd < RAND_MAX / 2)
  {
//    cv::flip(img, img, 1);
  }
//  img.copyTo(matOut);
  float *pfImg = (float*)img.data;

  float *pfOutR = pfImgOut;
  float *pfOutG = pfImgOut + stdH * stdW;
  float *pfOutB = pfImgOut + stdH * stdW * 2;
  for (int ri = 0; ri < stdH; ri++)
  {
    for (int ci = 0; ci < stdW; ci++)
    {
       int dwOft = ri * stdW + ci;
       pfOutR[dwOft] = pfImg[dwOft * 3 + 0];
       pfOutG[dwOft] = pfImg[dwOft * 3 + 1];
       pfOutB[dwOft] = pfImg[dwOft * 3 + 2];
    }
  }

  unique_lock<mutex> countlc(*p_countmt);
  if (*p_FinishCount < needNum)
  {
    (*p_FinishCount)++;
  }
  if ((*p_FinishCount) == needNum)
  {
    p_cv->notify_all();
  } 
  countlc.unlock();

  return;
}

float randMToN(float M, float N)
{
  return M + (rand() / (RAND_MAX/(N-M))) ;  
}

int rnd_crop(cv::Mat &matIn)
{
  int dwH = matIn.rows;
  int dwW = matIn.cols;
  int adwHWs[4] = {dwH, dwH, dwW, dwW};
  for (int dwI = 0; dwI < 4; dwI++)
  {
    float frndv = randMToN(0, 10) / 100;
//    cout << frndv << endl; 
    adwHWs[dwI] *= frndv;
  }
  cv::Rect roi(adwHWs[2], adwHWs[0], 
               dwW-adwHWs[3]-adwHWs[2], 
               dwH-adwHWs[1]-adwHWs[0]);
  cv::Mat matROI(matIn, roi);
  matROI.copyTo(matIn);
  return 0;
}


int rnd_rotate(cv::Mat &matIn)
{
  int dwH = matIn.rows;
  int dwW = matIn.cols;
  
  float rndv = randMToN(0, 60) - 30;
  cv::Mat matRot = cv::getRotationMatrix2D(cv::Point(dwW/2, dwH/2), rndv, 1.0);
  cv::warpAffine(matIn, matIn, matRot, cv::Size(dwW, dwH));

  return 0;
}


int normalize_img(cv::Mat &matIn)
{
  matIn.convertTo(matIn, CV_32FC3, 1.0, 0);
  cv::Mat matmean, matstdv;
  cv::meanStdDev(matIn, matmean, matstdv);
//  cout << matmean.total() << "," << matstdv.total() << endl;
  float fmean = matmean.at<double>(0) + matmean.at<double>(1) + matmean.at<double>(2);
  fmean /= 3.0f;
  float fstdv = matstdv.at<double>(0) + matstdv.at<double>(1) + matstdv.at<double>(2);
  fstdv /= 3.0f;
//  cout << fmean << "," << fstdv << endl;
  matIn.convertTo(matIn, -1, 1.0f/fstdv, -fmean/fstdv);
//  cv::meanStdDev(matIn, matmean, matstdv);
//  cout << matmean << matstdv << endl;

  return 0;
}


int rnd_mask(cv::Mat &matIn)
{
  assert(matIn.type()==CV_8UC3);
  int dwH = matIn.rows;
  int dwW = matIn.cols;
  
  int rndH = (int)randMToN(dwH/8, dwH/4);
  int rndRI = (int)randMToN(0, dwH-1);
  if (rndH + rndRI >= dwH)
  {
    rndRI = dwH - rndH - 1;
  }
  
  memset(matIn.data + rndRI * dwW * 3, 0, rndH * dwW * 3);

  return 0;
}










