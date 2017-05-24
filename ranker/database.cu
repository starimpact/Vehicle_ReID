#include "database.h"
#include "mytimer.h"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>

#include <cmath>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <thread>

typedef unsigned char byte;
typedef thrust::device_vector<CDatabase::DIST> 	DIST_ARY_GPU;

//***************************************************************************//
//**  Error Handling Macros and Function									 //
//***************************************************************************//

//-----------------------------------------------------------------------------
#define CUABORT(msg)						\
{											\
	cuPrintError(msg, __FILE__, __LINE__);	\
	exit(-1);								\
}

//-----------------------------------------------------------------------------
#define CUASSERT(exp)						\
{											\
	if (!(exp))								\
	{										\
		CUABORT(#exp);						\
	}										\
}

//-----------------------------------------------------------------------------
#define CUCHECK(exp)						\
{											\
	cudaError_t err = (exp);				\
	if (err != cudaSuccess)					\
	{										\
		CUABORT(cudaGetErrorString(err));	\
	}										\
}

//-----------------------------------------------------------------------------
inline void cuPrintFileLine(const char *pFile, int nLine)
{
	std::cout << "\"" << pFile << "\"(" << nLine << ")";
}

//-----------------------------------------------------------------------------
inline void cuPrintError(const char *pMsg, const char *pFile, int nLine)
{
	std::cerr << "An error occured at ";
	cuPrintFileLine(pFile, nLine);
	std::cerr << ": " << pMsg << std::endl;
}


//***************************************************************************//
//**  Device Kernal Functions												 //
//***************************************************************************//

//-----------------------------------------------------------------------------
// Kernel Functor -- compare two DIST structs by their `dist'
struct DIST_CMP
{
	__host__ __device__ bool operator()(
		const CDatabase::DIST &d1,
		const CDatabase::DIST &d2) const
	{
		return d1.dist < d2.dist;
	}
};

//-----------------------------------------------------------------------------
// Kernel function -- compute euclidean distance of two vectors
template<typename _DTYPE, typename _FTYPE>
__device__ _FTYPE L2Dist(const _DTYPE *p1, const _DTYPE *p2, int32_t nLen)
{
	_FTYPE fSum = _FTYPE(0);
	for (int32_t i = 0; i < nLen; ++i)
	{
		_FTYPE fDiff = (_FTYPE)*p1++ - (_FTYPE)*p2++;
		fSum += fDiff * fDiff;
	}
	return fSum;
}

//-----------------------------------------------------------------------------
// Kernel function -- compute cos of angle of two vectors
template<typename _DTYPE, typename _FTYPE>
__device__ _FTYPE CosDist(const _DTYPE *p1, const _DTYPE *p2, int32_t nLen)
{
	_FTYPE fSum = _FTYPE(0);
	for (int32_t i = 0; i < nLen; ++i)
	{
		fSum -= (_FTYPE)*p1++ * (_FTYPE)*p2++;
	}
	return fSum;
}

//-----------------------------------------------------------------------------
// Kernel function -- compute cos of angle of two vectors
template<typename _DTYPE, typename _FTYPE>
__device__ _FTYPE Dist(
	CDatabase::DISTFUNC distFunc,
	const _DTYPE *p1,
	const _DTYPE *p2,
	int32_t nLen
	)
{
	if (distFunc == CDatabase::DF_COS)
	{
		return CosDist<_DTYPE, _FTYPE>(p1, p2, nLen);
	}
	else if (distFunc == CDatabase::DF_L2)
	{
		return L2Dist<_DTYPE, _FTYPE>(p1, p2, nLen);
	}
	return _FTYPE(0);
}

//-----------------------------------------------------------------------------
// TODO: Comments for cuDistances
__global__ void cuDistances(
	CDatabase::DATATYPE dataType,
	CDatabase::DISTFUNC distFunc,
	const void *pDatabase,
	const void *pItem,
	CDatabase::DIST *pResults,
	int32_t nItemLen,
	CDatabase::_ITYPE nBaseIdx
	)
{
	int64_t iItem = nBaseIdx + blockIdx.x * blockDim.x + threadIdx.x;
	switch (dataType)
	{
		case CDatabase::DT_FLOAT:
			pResults[iItem].dist = Dist<float, CDatabase::_FTYPE>(
				distFunc,
				((float*)pItem),
				((float*)pDatabase) + iItem * nItemLen,
				nItemLen
				);
			break;
		case CDatabase::DT_SHORT:
			pResults[iItem].dist = Dist<short, CDatabase::_FTYPE>(
				distFunc,
				((short*)pItem),
				((short*)pDatabase) + iItem * nItemLen,
				nItemLen
				);
			break;
	}
}

//-----------------------------------------------------------------------------
// TODO: Comments for cuGetSamples
__global__ void cuGetSamples(
	const CDatabase::DIST *pResults,
	CDatabase::_ITYPE nSampleInterval,
	CDatabase::DIST *pSamples,
	CDatabase::_ITYPE nBaseIdx)
{
	int64_t iDstItem = (int64_t)nBaseIdx + blockIdx.x * blockDim.x + threadIdx.x;
	int64_t iSrcItem = iDstItem * (int64_t)nSampleInterval;
	pSamples[iDstItem].id = pResults[iSrcItem].id;
	pSamples[iDstItem].dist = pResults[iSrcItem].dist;
}

//-----------------------------------------------------------------------------
// TODO: Comments for cuMalloc0
template<typename _Ty>
_Ty* cuMalloc0(int64_t nUnitCnt)
{
	_Ty *pDevMem = 0;
	CUCHECK(cudaMalloc((void**)&pDevMem,  sizeof(_Ty) * nUnitCnt));
	CUCHECK(cudaMemset((void*)pDevMem, 0, sizeof(_Ty) * nUnitCnt));
	return pDevMem;
}

//-----------------------------------------------------------------------------
// TODO: Comments for cuMalloc0
__device__ bool operator == (
	const CDatabase::DIST &d1,
	const CDatabase::DIST &d2
	)
{
	return d1.id == d2.id;
}

//***************************************************************************//
//**  CQuery Declaration													//
//***************************************************************************//

class CQuery
{
public:
	typedef CDatabase::_ITYPE	_ITYPE;
	typedef CDatabase::_FTYPE	_FTYPE;
	typedef CDatabase::DIST		DIST;

	CDatabase				*m_pDb;

protected:
	const int32_t			m_nCuThreads;
	const int32_t			m_nCuBlocks;


	// Pointers of Device Memory
	std::vector<void*>		m_QueryItem;
	std::vector<DIST*>		m_QueryResults;

	std::vector<DIST>		m_Results;
	std::vector<void*>		m_Samples;

public:
	// Constructor
	CQuery(CDatabase *pDb);

	// Deconstructor
	~CQuery();

	//
	void	AddItems(int32_t iGpu, DIST *pDists, _ITYPE nCnt);

	//
	bool	NearestN(const void *pItem, _ITYPE N, DIST *pResults);

protected:
	void	_UploadQueryItem(const void *pItem);

	void	_DoQuery();

	float	_SampleMaxDist(_ITYPE N, _ITYPE nSampleInterval);

	void	_DownloadResults(_FTYPE fMaxDist, std::vector<DIST> &results);
	
	void	_SynchronizeAll();

private:
	CQuery(const CQuery&) = delete;
	CQuery& operator = (const CQuery&) = delete;
};

//***************************************************************************//
//**  CQuery Implementation													//
//***************************************************************************//

//-----------------------------------------------------------------------------
// Constructor
CQuery::CQuery(CDatabase *pDb)
	: m_pDb(pDb)
	, m_nCuThreads(0)
	, m_nCuBlocks(0)
{
	CUASSERT(m_pDb != 0);
	CUASSERT(m_pDb->m_nCapacity > 0);

	int32_t nGpuCnt = m_pDb->GetGpuCount();
	CUASSERT(nGpuCnt > 0);

	for (int32_t i = 0; i < nGpuCnt; ++i)
	{
		cudaDeviceProp devProp;
		CUCHECK(cudaGetDeviceProperties(&devProp, i));
		if (m_nCuThreads < devProp.maxThreadsPerBlock || m_nCuThreads == 0)
		{
			const_cast<int32_t&>(m_nCuThreads) = devProp.maxThreadsPerBlock;
		}
		if (m_nCuBlocks < devProp.maxGridSize[2] || m_nCuBlocks == 0)
		{
			const_cast<int32_t&>(m_nCuBlocks) = devProp.maxGridSize[2];
		}
	}

	m_QueryItem.resize(nGpuCnt, 0);
	m_QueryResults.resize(nGpuCnt, 0);
	m_Samples.resize(nGpuCnt, 0);
	
	int32_t nItemLen = m_pDb->m_nItemLen;
	_ITYPE nCapacity = m_pDb->m_nCapacity;

	for (int32_t iGpu = 0; iGpu < nGpuCnt; ++iGpu)
	{
		if (m_pDb->UseGpu(iGpu))
		{
			switch (m_pDb->GetDataType())
			{
			case CDatabase::DT_FLOAT:
				m_QueryItem[iGpu]	= cuMalloc0<float>(nItemLen);
				break;
			case CDatabase::DT_SHORT:
				m_QueryItem[iGpu]	= cuMalloc0<short>(nItemLen);
				break;
			default:
				CUABORT("Unknown data type!");
				break;
			}
			m_QueryResults[iGpu] = cuMalloc0<DIST>(nCapacity);
			m_Samples[iGpu] = new DIST_ARY_GPU;
		}
	}
}

//-----------------------------------------------------------------------------
// Deconstructor
CQuery::~CQuery()
{
	int32_t nGpuCnt = m_pDb->GetGpuCount();
	for (int32_t iGpu = 0; iGpu < nGpuCnt; ++iGpu)
	{
		if (m_pDb->UseGpu(iGpu))
		{
			CUCHECK(cudaDeviceSynchronize());
			CUCHECK(cudaFree(m_QueryItem[iGpu]));
			CUCHECK(cudaFree(m_QueryResults[iGpu]));
		}
	}
	m_QueryItem.clear();
	m_QueryResults.clear();
	m_Results.clear();
	
	for (int32_t i = 0; i < nGpuCnt; ++i)
	{
		delete (DIST_ARY_GPU*)m_Samples[i];
	}
}

//-----------------------------------------------------------------------------
void CQuery::AddItems(int32_t iGpu, DIST *pDists, _ITYPE nCnt)
{
	m_pDb->UseGpu(iGpu);
	CUCHECK(cudaMemcpyAsync(
		m_QueryResults[iGpu] + m_pDb->m_ItemCnts[iGpu],
		pDists,
		(int64_t)nCnt * sizeof(DIST),
		cudaMemcpyHostToDevice
		));
}

//-----------------------------------------------------------------------------
// TODO: Comment for NearestN
bool CQuery::NearestN(const void *pItem, _ITYPE N, DIST *pResults)
{
	// Checking parameters
	CUASSERT(pItem != 0);
	CUASSERT(N > 0);
	CUASSERT(pResults != 0);

	_ITYPE nTotalItems = m_pDb->GetTotalItems();
	if (nTotalItems <= 0)
	{
		return false;
	}
	CUASSERT(N <= nTotalItems);

	_UploadQueryItem(pItem);
	_DoQuery();

	// Compute costs: fSamples / nTotalItems,
	//		the smaller fSamples, the lower costs;
	// Precision of estimation: fSamples / N,
	//		the larger fSamples, the higher estimation;
	// So, Let fSamples = sqrt(nTotalItems * N) * factor
	_FTYPE fMaxDist = std::numeric_limits<_FTYPE>::max();
	_FTYPE fSamples = std::sqrt((_FTYPE)nTotalItems * N) * 2.0f;
	_ITYPE nSampleInterval = (_ITYPE)(nTotalItems / fSamples);
	if (nSampleInterval > 2)
	{
		fMaxDist = _SampleMaxDist(N, nSampleInterval);
	}

	m_Results.clear();
	m_Results.resize(nTotalItems);

	_DownloadResults(fMaxDist, m_Results);

	// Sorting results
	thrust::device_vector<DIST> devBuf(m_Results.begin(), m_Results.end());
	thrust::sort(devBuf.begin(), devBuf.end(), DIST_CMP());
	devBuf.resize(N);
	thrust::copy(devBuf.begin(), devBuf.end(), pResults);

	return cudaGetLastError() == cudaSuccess;
}

//-----------------------------------------------------------------------------
// TODO: Comment for _UploadQueryItem
void CQuery::_UploadQueryItem(const void *pItem)
{
	int64_t nItemBytes = m_pDb->m_nItemLen * m_pDb->m_nElemSize;
	int32_t nGpuCnt = m_pDb->GetGpuCount();
	for (int32_t iGpu = 0; iGpu < nGpuCnt; ++iGpu)
	{
		if (m_pDb->UseGpu(iGpu))
		{
			//CUCHECK(cudaDeviceSynchronize());
			CUCHECK(cudaMemcpyAsync(
				(byte*)(m_QueryItem[iGpu]),
				(byte*)pItem,
				nItemBytes,
				cudaMemcpyHostToDevice
				));
		}
	}
}

//-----------------------------------------------------------------------------
// TODO: Comment for _DoQuery
void CQuery::_DoQuery()
{
	_ITYPE nItemsPerQuery = m_nCuBlocks * m_nCuThreads;
	
	int32_t nItemLen = m_pDb->m_nItemLen;
	int32_t nGpuCnt = m_pDb->GetGpuCount();
	for (int32_t iGpu = 0; iGpu < nGpuCnt; ++iGpu)
	{
		if (m_pDb->UseGpu(iGpu))
		{
			//CUCHECK(cudaDeviceSynchronize());
			_ITYPE nQueryCnt = m_pDb->m_ItemCnts[iGpu] / nItemsPerQuery;
			for (_ITYPE iQuery = 0; iQuery < nQueryCnt; ++iQuery)
			{
				cuDistances<<<m_nCuBlocks, m_nCuThreads>>>(
					m_pDb->GetDataType(),
					m_pDb->GetDistFunc(),
					m_pDb->m_ItemSets[iGpu],
					m_QueryItem[iGpu],
					m_QueryResults[iGpu],
					nItemLen,
					iQuery * nItemsPerQuery
					);
			}
			_ITYPE nRemainBlocks = (m_pDb->m_ItemCnts[iGpu] % nItemsPerQuery)
				/ m_nCuThreads;
			if (nRemainBlocks > 0)
			{
				cuDistances<<<nRemainBlocks, m_nCuThreads>>>(
					m_pDb->GetDataType(),
					m_pDb->GetDistFunc(),
					m_pDb->m_ItemSets[iGpu],
					m_QueryItem[iGpu],
					m_QueryResults[iGpu],
					nItemLen,
					nQueryCnt * nItemsPerQuery
					);
			}
			_ITYPE nItemsRemains = m_pDb->m_ItemCnts[iGpu] % m_nCuThreads;
			if (nItemsRemains > 0)
			{
				cuDistances<<<1, nItemsRemains>>>(
					m_pDb->GetDataType(),
					m_pDb->GetDistFunc(),
					m_pDb->m_ItemSets[iGpu],
					m_QueryItem[iGpu],
					m_QueryResults[iGpu],
					nItemLen,
					m_pDb->m_ItemCnts[iGpu] - nItemsRemains
					);
			}
		}
	}
}

//-----------------------------------------------------------------------------
// TODO: Comment for _SampleMaxDist
float CQuery::_SampleMaxDist(_ITYPE N, _ITYPE nSampleInterval)
{
	// Alloc buffer for store samples
	int32_t nGpuCnt = m_pDb->GetGpuCount();
	for (int32_t iGpu = 0; iGpu < nGpuCnt; ++iGpu)
	{
		if (m_pDb->UseGpu(iGpu))
		{
			_ITYPE nSamples = m_pDb->m_ItemCnts[iGpu] / nSampleInterval;
			if (nSamples >= m_nCuBlocks * m_nCuThreads)
			{
				return std::numeric_limits<_FTYPE>::max();
			}
			DIST_ARY_GPU &smpl = *(DIST_ARY_GPU*)m_Samples[iGpu];
			smpl.clear();
			smpl.resize(nSamples);
			CUCHECK(cudaDeviceSynchronize());
		}
	}

	// Retrieve samples
	for (int32_t iGpu = 0; iGpu < nGpuCnt; ++iGpu)
	{
		if (m_pDb->UseGpu(iGpu))
		{
			DIST_ARY_GPU &smpl = *(DIST_ARY_GPU*)m_Samples[iGpu];
			DIST* pSamples = thrust::raw_pointer_cast(smpl.data());
			int32_t nCudaBlks = (int32_t)smpl.size() / m_nCuThreads;
			if (nCudaBlks != 0)
			{
				cuGetSamples<<<nCudaBlks, m_nCuThreads>>>(
					m_QueryResults[iGpu],
					nSampleInterval,
					pSamples,
					0
					);
			}
			int32_t nRemains = (int32_t)(smpl.size() % m_nCuThreads);
			if (nRemains > 0)
			{
				cuGetSamples<<<1, nRemains>>>(
					m_QueryResults[iGpu],
					nSampleInterval,
					pSamples,
					smpl.size() - nRemains
					);
			}
			//CUCHECK(cudaDeviceSynchronize());
		}
	}


	thrust::host_vector<DIST> maxN;
	thrust::host_vector<DIST> merged;

	for (int32_t iGpu = 0; iGpu < nGpuCnt; ++iGpu)
	{
		if (m_pDb->UseGpu(iGpu))
		{
			DIST_ARY_GPU &smpl = *(DIST_ARY_GPU*)m_Samples[iGpu];
			thrust::sort(smpl.begin(), smpl.end(), DIST_CMP());

			_ITYPE nTop = std::min((_ITYPE)smpl.size(), N);
			smpl.resize(nTop);
			thrust::host_vector<DIST> src(smpl.begin(), smpl.end());

			merged.resize(maxN.size() + src.size());
			thrust::merge(
				maxN.begin(),
				maxN.end(),
				src.begin(),
				src.end(),
				merged.begin(),
				DIST_CMP()
				);
			merged.resize(N);
			merged.swap(maxN);
		}
	}
	return maxN[N - 1].dist;
}

//-----------------------------------------------------------------------------
void CQuery::_DownloadResults(_FTYPE fMaxDist, std::vector<DIST> &results)
{
	int64_t nCopied = 0;
	int32_t nGpuCnt = m_pDb->GetGpuCount();
	for (int32_t iGpu = 0; iGpu < nGpuCnt; ++iGpu)
	{
		if (m_pDb->UseGpu(iGpu))
		{
			auto iSrcBeg = thrust::device_pointer_cast(m_QueryResults[iGpu]);
			auto iSrcEnd = iSrcBeg + m_pDb->m_ItemCnts[iGpu];
			thrust::copy(iSrcBeg, iSrcEnd, results.data() + nCopied);

			auto iDstBeg = results.begin() + nCopied;
			auto iNewEnd = std::remove_if(
				iDstBeg,
				iDstBeg + m_pDb->m_ItemCnts[iGpu],
				[&fMaxDist](const DIST &d1) -> bool
				{
					return d1.dist > fMaxDist;
				});

			nCopied += (int64_t)(iNewEnd - iDstBeg);
		}
	}
	results.resize(nCopied);
}

//-----------------------------------------------------------------------------
// TODO: Comment for DumpItems
void CQuery::_SynchronizeAll()
{
	int32_t nGpuCnt = m_pDb->GetGpuCount();
	for (int32_t iGpu = 0; iGpu < nGpuCnt; ++iGpu)
	{
		if (m_pDb->UseGpu(iGpu))
		{
			CUCHECK(cudaDeviceSynchronize());
		}
	}
}


//***************************************************************************//
//**  CDatabase Implementation												 //
//***************************************************************************//

//-----------------------------------------------------------------------------
// Constructor
CDatabase::CDatabase(DATATYPE dataType, DISTFUNC distFunc)
	: m_DataType(dataType)
	, m_DistFunc(distFunc)
	, m_nElemSize(0)
	, m_nCapacity(0)
	, m_nItemLen(0)
{
	int32_t nGpuCnt = (int32_t)GetGpuCount();
	CUASSERT(nGpuCnt > 0);

	m_ItemCnts.resize(nGpuCnt, 0);
	
	switch (m_DataType)
	{
	case CDatabase::DT_FLOAT:
		m_nElemSize = sizeof(float);
		break;
	case CDatabase::DT_SHORT:
		m_nElemSize = sizeof(short);
		break;
	default:
		CUABORT("Unknown data type!");
		break;
	}

}

//-----------------------------------------------------------------------------
// Destructor
CDatabase::~CDatabase()
{
	Destroy();
}

//-----------------------------------------------------------------------------
// Initialization
bool CDatabase::Initialize(
		int64_t nCapacity,
		int32_t nItemLen,
		int32_t nQueryCnt,
		int32_t nGpuMask
		)
{
	// Lock for write
	boost::unique_lock<boost::shared_mutex> lock(m_OpMutex);

	CUASSERT(m_Queries.empty());

	// Checking Parameters
	CUASSERT(nCapacity > 0);
	CUASSERT(nItemLen > 0);

	int32_t nGpuCnt = (int32_t)m_ItemCnts.size();

	if (nGpuMask < 0)
	{
		std::fill(m_ItemCnts.begin(), m_ItemCnts.end(), 0);
	}
	else
	{
		CUASSERT((nGpuMask >> nGpuCnt) == 0);

		for (int32_t iGpu = 0; iGpu < nGpuCnt; ++iGpu)
		{
			if (nGpuMask & (1 << iGpu))
			{
				m_ItemCnts[iGpu] = 0;
			}
			else
			{
				m_ItemCnts[iGpu] = -1;
			}
		}
	}

	m_ItemSets.resize(nGpuCnt, 0);

	int64_t nElems = nCapacity * nItemLen;
	for (int32_t iGpu = 0; iGpu < nGpuCnt; ++iGpu)
	{
		if (UseGpu(iGpu))
		{
			switch (m_DataType)
			{
			case CDatabase::DT_FLOAT:
				m_ItemSets[iGpu]	= cuMalloc0<float>(nElems);
				break;
			case CDatabase::DT_SHORT:
				m_ItemSets[iGpu]	= cuMalloc0<short>(nElems);
				break;
			default:
				CUABORT("Unknown data type!");
				break;
			}
		}
	}

	m_nCapacity	= nCapacity;
	m_nItemLen	= nItemLen;
	
	for (int i = 0; i < (int)nQueryCnt; ++i)
	{
		CQuery *q = new CQuery(this);
		m_Queries.push_back(q);
	}

	return true;
}

//-----------------------------------------------------------------------------
// Destroy
void CDatabase::Destroy()
{
	boost::unique_lock<boost::mutex> qrLock(m_QrMutex);
	for (auto q : m_Queries)
	{
		delete q;
	}
	m_Queries.clear();
	qrLock.unlock();

	// Lock for write
	boost::unique_lock<boost::shared_mutex> lock(m_OpMutex);

	int32_t nGpuCnt = GetGpuCount();
	if (m_nCapacity > 0)
	{
		for (int32_t iGpu = 0; iGpu < nGpuCnt; ++iGpu)
		{
			if (UseGpu(iGpu))
			{
				CUCHECK(cudaDeviceSynchronize());
				CUCHECK(cudaFree(m_ItemSets[iGpu]));
			}
		}
		m_ItemSets.clear();

		m_nCapacity = 0;
		m_nItemLen = 0;
	}
}

//-----------------------------------------------------------------------------
// Get total number of installed GPUs
int32_t CDatabase::GetGpuCount() const
{
	int32_t nGpuCnt = 0;
	CUCHECK(cudaGetDeviceCount(&nGpuCnt));

	return nGpuCnt;
}

//-----------------------------------------------------------------------------
bool CDatabase::UseGpu(int32_t iGpu) const
{
	if(iGpu < (int32_t)m_ItemCnts.size())
	{
		if (m_ItemCnts[iGpu] >= 0)
		{
			CUCHECK(cudaSetDevice(iGpu));
			return true;
		}
	}
	return false;
}

//-----------------------------------------------------------------------------
CDatabase::DISTFUNC CDatabase::GetDistFunc() const
{
	return m_DistFunc;
}

//-----------------------------------------------------------------------------
// Get item length
CDatabase::DATATYPE CDatabase::GetDataType() const
{
	return m_DataType;
}

//-----------------------------------------------------------------------------
// Get item length
int32_t	CDatabase::GetElemSize() const
{
	return m_nElemSize;
}

//-----------------------------------------------------------------------------
// Get item length
CDatabase::_ITYPE CDatabase::GetTotalItems() const
{
	// Lock for read
	boost::shared_lock<boost::shared_mutex> lock(m_OpMutex);

	_ITYPE nTotalItems = 0;
	int32_t nGpuCnt = GetGpuCount();
	for (int32_t iGpu = 0; iGpu < nGpuCnt; ++iGpu)
	{
		if (UseGpu(iGpu))
		{
			nTotalItems += m_ItemCnts[iGpu];
		}
	}
	return nTotalItems;
}

//-----------------------------------------------------------------------------
CDatabase::_ITYPE CDatabase::GetGpuItems(int32_t iGpu) const
{
	// Lock for read
	boost::shared_lock<boost::shared_mutex> lock(m_OpMutex);

	CUASSERT(iGpu < (int32_t)m_ItemCnts.size());
	return m_ItemCnts[iGpu];
}

//-----------------------------------------------------------------------------
// Get item length
int32_t CDatabase::GetItemLength() const
{
	boost::shared_lock<boost::shared_mutex> lock(m_OpMutex);

	return m_nItemLen;
}

//-----------------------------------------------------------------------------
CDatabase::_ITYPE CDatabase::GetCapacity() const
{
	// Lock for read
	boost::shared_lock<boost::shared_mutex> lock(m_OpMutex);
	
	return m_nCapacity;
}

//-----------------------------------------------------------------------------
// TODO: Comment for RetrieveItemById
bool CDatabase::RetrieveItemById(_ITYPE nId, void *pItem) const
{
	// NOT IMPLEMENTED
	return false;
}

//-----------------------------------------------------------------------------
// TODO: Comment for DumpItems
void CDatabase::DumpItems(void *pOut)
{
	// Lock for read
	boost::shared_lock<boost::shared_mutex> lock(m_OpMutex);

	CUASSERT(m_nCapacity > 0);

	int64_t nBaseIdx = 0;
	int64_t nItemBytes = m_nItemLen * m_nElemSize;
	int32_t nGpuCnt = GetGpuCount();
	for (int32_t iGpu = 0; iGpu < nGpuCnt; ++iGpu)
	{
		if (UseGpu(iGpu))
		{
			CUCHECK(cudaDeviceSynchronize());
			CUCHECK(cudaMemcpy(
					(byte*)pOut + nBaseIdx,
					(byte*)(m_ItemSets[iGpu]),
					(int64_t)m_ItemCnts[iGpu] * nItemBytes,
					cudaMemcpyDeviceToHost
					));
			nBaseIdx += (int64_t)m_ItemCnts[iGpu] * nItemBytes;
		}
	}
}

//-----------------------------------------------------------------------------
// Constructor
void CDatabase::AddItems(const void *pItems, const _ITYPE *pIds, _ITYPE nCnt)
{
	boost::unique_lock<boost::shared_mutex> lock(m_OpMutex);

	//Checking Parameter list
	if (nCnt == 0)
	{
		return;
	}
	CUASSERT(pItems != 0);
	CUASSERT(pIds != 0);

	std::vector<_ITYPE> gpuAddItems(m_ItemCnts);

	int32_t nGpuCnt = GetGpuCount();
	// Determin the number of items add for each gpu
	{
		_ITYPE nUsedGpuCnt = 0;
		_ITYPE nTotalCap = 0;
		_ITYPE nTotalItems = 0;

		for (auto &nGpuItemCnt : m_ItemCnts)
		{
			if (nGpuItemCnt >= 0)
			{
				++nUsedGpuCnt;
				nTotalItems += nGpuItemCnt;
				nTotalCap += m_nCapacity;
			}
		}
		CUASSERT(nTotalItems + nCnt <= nTotalCap);

		for (int32_t iGpu = 0; iGpu < nGpuCnt; ++iGpu)
		{
			if (m_ItemCnts[iGpu] >= 0)
			{
				gpuAddItems[iGpu] += nCnt / nUsedGpuCnt;
			}
			else
			{
				gpuAddItems[iGpu] = nTotalCap;
			}
		}
		for (_ITYPE iRem = 0; iRem < nCnt % nUsedGpuCnt; ++iRem)
		{
			int32_t iMinGpu = std::min_element(
				gpuAddItems.begin(),
				gpuAddItems.end()
				) - gpuAddItems.begin();
			++gpuAddItems[iMinGpu];
		}
		for (int32_t iGpu = 0; iGpu < nGpuCnt; ++iGpu)
		{
			if (m_ItemCnts[iGpu] >= 0)
			{
				gpuAddItems[iGpu] -= m_ItemCnts[iGpu];
			}
		}
	}

	std::vector<DIST> resBuf;
	int64_t nItemBytes = m_nItemLen * m_nElemSize;
	int64_t nInputBaseIdx = 0;
	for (int32_t iGpu = 0; iGpu < nGpuCnt; ++iGpu)
	{
		if (UseGpu(iGpu))
		{
			CUCHECK(cudaDeviceSynchronize());

			CUCHECK(cudaMemcpyAsync(
				(byte*)(m_ItemSets[iGpu]) + (int64_t)m_ItemCnts[iGpu] * nItemBytes,
				(byte*)pItems + nInputBaseIdx * nItemBytes,
				gpuAddItems[iGpu] * nItemBytes, //In bytes
				cudaMemcpyHostToDevice
				));

			resBuf.resize(gpuAddItems[iGpu]);
			for (int64_t iItem = 0; iItem < (int64_t)resBuf.size(); ++iItem)
			{
				resBuf[iItem].id = pIds[nInputBaseIdx + iItem];
			}
			for (auto &q : m_Queries)
			{
				q->AddItems(iGpu, resBuf.data(), gpuAddItems[iGpu]);
			}

			nInputBaseIdx += gpuAddItems[iGpu];
			m_ItemCnts[iGpu] += gpuAddItems[iGpu];
		}
	}
	CUCHECK(cudaGetLastError());
}

//-----------------------------------------------------------------------------
// TODO: Comment for ResetItems
void CDatabase::ResetItems()
{
	boost::unique_lock<boost::shared_mutex> lock(m_OpMutex);

	for (auto &iGpuItemCnt : m_ItemCnts)
	{
		if (iGpuItemCnt > 0)
		{
			iGpuItemCnt = 0;
		}
	}
}

//-----------------------------------------------------------------------------
// KNN Query function
void CDatabase::QueryTopN(const void *pItem, _ITYPE N, DIST *pResults)
{
	CQuery *pQuery = 0;
	for ( ; pQuery == 0;)
	{
		boost::unique_lock<boost::mutex> cvLock(m_CvMutex);
		m_CondVar.wait(cvLock, [&](){
				boost::unique_lock<boost::mutex> qrLock(m_QrMutex);
				if (!m_Queries.empty())
				{
					pQuery = m_Queries.back();
					m_Queries.pop_back();
				}
				return (pQuery != 0);
			});
	};

	boost::shared_lock<boost::shared_mutex> opLock(m_OpMutex);
	pQuery->NearestN(pItem, N, pResults);
	opLock.unlock();

	boost::unique_lock<boost::mutex> qrLock(m_QrMutex);
	m_Queries.push_back(pQuery);
	qrLock.unlock();
	m_CondVar.notify_all();

#ifdef OUTPUT_TIME
	static CMyTimer t;
	static int nQueryies = 0;
	if (++nQueryies > 100)
	{
		double dTime = t.Reset();
//		std::cout << "AvgTime=" << dTime / nQueryies << std::endl;
		nQueryies = 0;
	}
#endif
}
//===========================================================================//

