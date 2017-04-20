
#include "simple_thread_pool.h"
#include "cv.h"
#include "highgui.h"

int showimage();
extern "C" int do_augment_threads(char *pfns[], int num, int stdH, int stdW, float *pfImgOut);
int main()
{
//  showimage();
  char *pfns[] = {(char*)"1", (char*)"2", (char*)"3"};
  do_augment_threads(pfns, 3, 10, 10, 0);
}

