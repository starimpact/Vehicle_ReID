
#ifndef MYTIMER_H_
#define MYTIMER_H_
#include <time.h>

class CMyTimer
{
protected:
	double m_dBeg;
public:
	CMyTimer()
	{
		Start();
	}
	inline void Start()
	{
		timespec cur;
		clock_gettime(CLOCK_REALTIME, &cur);
		m_dBeg = cur.tv_sec + (double)cur.tv_nsec / 1.0e9;
	}
	inline double Now()
	{
		timespec cur;
		clock_gettime(CLOCK_REALTIME, &cur);
		return cur.tv_sec + (double)cur.tv_nsec / 1.0e9 - m_dBeg;
	}
	inline double Reset()
	{
		double dNow = Now();
		Start();
		return dNow;
	}
};

#endif /* MYTIMER_H_ */
