

//*****************************************************************************
// A great source code just likes a poem that any comment shall profanes it.
//														-- devymex@gmail.com
//*****************************************************************************


#ifndef DATABASE_H_
#define DATABASE_H_

#include <boost/thread/mutex.hpp>
#include <boost/thread/shared_mutex.hpp>
#include <boost/thread/condition_variable.hpp>

#include <vector>
#include <stdint.h>

#define OUTPUT_TIME


class CDatabase
{
public:
	typedef int32_t		_ITYPE;
	typedef float		_FTYPE;

	enum DATATYPE
	{
		DT_UNKNOWN,
		DT_FLOAT,
		DT_SHORT
	};

	enum DISTFUNC
	{
		DF_UNKNOWN,
		DF_COS,
		DF_L2
	};

	struct DIST
	{
		_ITYPE	id;			// Item index of one device
		_FTYPE	dist;		// Squared distance
	};


protected:
	DATATYPE				m_DataType;
	DISTFUNC				m_DistFunc;
	int32_t					m_nElemSize;

	_ITYPE					m_nCapacity;	// Capacity per gpu
	int32_t					m_nItemLen;		// Item length;
	std::vector<_ITYPE>		m_ItemCnts;		// Current items
	std::vector<void*>		m_ItemSets;

	std::vector<class CQuery*>	m_Queries;

	mutable boost::shared_mutex			m_OpMutex;
	mutable boost::mutex				m_QrMutex;
	mutable boost::mutex				m_CvMutex;
	mutable boost::condition_variable	m_CondVar;


public:
	//-------------------------------------------------------------------------
	//--------CLASS LIFETIME--------
	// Constructor
	CDatabase(DATATYPE dataType = DT_FLOAT, DISTFUNC distFunc = DF_COS);
	// Destructor
	~CDatabase();
	// Initialization
	bool	Initialize(
			int64_t nCapacity,
			int32_t nItemLen,
			int32_t nQueryCnt,
			int32_t nGpuMask = -1
			);
	// Uninitalization
	void	Destroy();

	//-------------------------------------------------------------------------
	//--------DEVICE INFORMATIONS--------
	// Get total number of installed GPUs
	int32_t	GetGpuCount() const;
	//
	bool	UseGpu(int32_t iGpu) const;
	//
	DISTFUNC GetDistFunc() const;

	//-------------------------------------------------------------------------
	//--------DATA INFORMATION--------
	//
	DATATYPE GetDataType() const;
	// Get total number of items added to all GPUs
	int32_t	GetElemSize() const;
	//
	_ITYPE	GetTotalItems() const;
	// Get item length
	_ITYPE	GetGpuItems(int32_t iGpu) const;
	//
	int32_t	GetItemLength() const;
	//
	_ITYPE	GetCapacity() const;
	//
	bool	RetrieveItemById(_ITYPE nId, void *pItem) const;
	//
	void	DumpItems(void *pOut);

	//-------------------------------------------------------------------------
	//--------DATA MANAGEMENT--------
	void	AddItems(const void *pItems, const _ITYPE *pIds, _ITYPE nCnt);
	//
	void	ResetItems();

	//-------------------------------------------------------------------------
	//--------TOP-N QUERY---------
	void	QueryTopN(const void *pItem, _ITYPE N, DIST *pResults);

private:
	CDatabase(const CDatabase&) = delete;

	CDatabase& operator = (const CDatabase&) = delete;

protected:
	friend class CQuery;
};

#endif /* DATABASE_H_ */
