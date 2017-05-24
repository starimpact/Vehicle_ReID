
#ifndef __VREID_H__
#define __VREID_H__

#include "iostream"
#include "string"
#include "vector"
#include <sys/time.h>

#include "opencv/cv.h"
#include "opencv/highgui.h"

namespace dgVReID {

struct ImageInfo {
  int dwID;
  int dwType; //0:front, 1:back
  cv::Mat Image;
  std::vector<float> vecFeat;
};

class VehicleReID {
 public:
  //dwDevType: 1 cpu, 2 gpu
  //dwPoseType: 0 front, 1 back
  static VehicleReID *Create(const std::string &strModelPath, int dwGpuID, int dwPoseType, bool bEncrypted=false, int dwDevType=2);
  virtual ~VehicleReID() {};
  virtual int Process(std::vector<ImageInfo> &vecImageInfo) = 0;
};

}


#endif


