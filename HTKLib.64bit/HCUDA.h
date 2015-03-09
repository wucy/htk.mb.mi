/* ----------------------------------------------------------- */
/*                                                             */
/*                          ___                                */
/*                       |_| | |_/   SPEECH                    */
/*                       | | | | \   RECOGNITION               */
/*                       =========   SOFTWARE                  */
/*                                                             */
/*                                                             */
/* ----------------------------------------------------------- */
/* developed at:                                               */
/*                                                             */
/*      Machine Intelligence Laboratory                        */
/*      Cambridge University Engineering Department            */
/*      http://mil.eng.cam.ac.uk/                              */
/*                                                             */
/*                                                             */
/* ----------------------------------------------------------- */
/*         Copyright: Microsoft Corporation                    */
/*          1995-2000 Redmond, Washington USA                  */
/*                    http://www.microsoft.com                 */
/*                                                             */
/*              2002  Cambridge University                     */
/*                    Engineering Department                   */
/*                                                             */
/*   Use of this software is governed by a License Agreement   */
/*    ** See the file License for the Conditions of Use  **    */
/*    **     This banner notice must not be removed      **    */
/*                                                             */
/* ----------------------------------------------------------- */
/*              File: HCUDA.h   CUDA Utilities                 */
/* ----------------------------------------------------------- */

/* !HVER!HCUDA.h:   3.4.1 [CUED 29/11/13] */

#ifndef _HCUDA_H_
#define _HCUDA_H_

#ifdef __cplusplus
extern "C" {
#endif

/*#include "HShell.h"*/
#include "HMem.h"

#define THREADPERBLOCK 256                      /*  */
#define MAXBLOCKNUM 2147483647                  /*  */

#define MINCUDAVER 4000                         /*  */
#define MINMAJORSMARCH 2                        /*  */
#define MINMINORSMARCH 0                        /*  */

/*
#define MINCUDAVER 6000
#define MINMAJORSMARCH 3
#define MINMINORSMARCH 0
*/

/* cz277 - cuda fblat */
/* a simplified structure to the struct Acoustic in HArc.h */
#ifdef CUDA
typedef struct _AcousticDev{
    int Nq;
    int t_start;
    int t_end;
    NFloat aclike;
    NFloat locc;
    Boolean SP;

    int *indexes;       /* [1 ... Nq] */
    NFloat *transp;	/* [1 ... Nq][1 ... Nq] */
    /*NFloat *alphat;*/     /* [1 ... Nq] */
    /*NFloat *alphat1;*/    /* [1 ... Nq] */
    NFloat *betaPlus;   /* [t_start ... t_end][1 ... Nq] */
    NFloat *alphaPlus;	/* [t_start ... t_end][1 ... Nq] */
    NFloat *otprob;	/* [t_start ... t_end][1 ... Nq] */
} AcousticDev;
#endif

void InitCUDA(void);					/* use to initialize CUDA */
void StartCUDA(void);					/*  */
void StopCUDA(void);					/* use to shutdown CUDA */

void SyncDev2Host(void *devPtr, void *hostPtr, size_t size);
void SyncHost2Dev(void *hostPtr, void *devPtr, size_t size);
void DevDispose(void *devPtr, size_t size);
void DevNew(void **devAddr, size_t size);
void ShowGPUMemUsage(void);

/*void SetNSegment(NFloat val, NFloat *seg, int len);*/
void CopyNSegmentCUDA(NFloat *srcPtr, int segLen, NFloat *dstPtr);
void AddNSegmentCUDA(NFloat *srcPtr, int segLen, NFloat *dstPtr);
void ScaleNSegmentCUDA(int segLen, NFloat scale, NFloat *valPtr);
void ScaledSelfAddNSegmentCUDA(NFloat *rhPtr, int segLen, NFloat scale, NFloat *lhPtr);
void DupNSegmentCUDA(NFloat *srcPtr, int segLen, NFloat *dstPtr, int times);
void SubNSegmentCUDA(NFloat *lhPtr, NFloat *rhPtr, int segLen, NFloat *resPtr);
void MulNSegmentCUDA(NFloat *lhPtr, NFloat *rhPtr, int segLen, NFloat *resPtr);
void ApplyReLActCUDA(NFloat *srcPtr, int len, NFloat *dstPtr);
void ApplyDReLActCUDA(NFloat *srcPtr, int len, NFloat *dstPtr);
void ApplyDLinearActCUDA(NFloat *srcPtr, int len, NFloat *dstPtr);
void ApplySigmoidActCUDA(NFloat *srcPtr, int len, NFloat *dstPtr);
void ApplyDSigmoidActCUDA(NFloat *srcPtr, int len, NFloat *dstPtr);
void ApplyTanHActCUDA(NFloat *srcPtr, int len, NFloat *dstPtr);
void ApplyDTanHActCUDA(NFloat *srcPtr, int len, NFloat *dstPtr);
void ApplySoftmaxActCUDA(NFloat *srcPtr, int row, int col, NFloat *dstPtr);
void ApplyRedSoftmaxActCUDA(NFloat *srcPtr, int row, int col, NFloat *dstPtr);
void ApplySoftReLActCUDA(NFloat *srcPtr, int len, NFloat *dstPtr);
void ApplyDSoftReLActCUDA(NFloat *srcPtr, int len, NFloat *dstPtr);
void ApplySoftSignActCUDA(NFloat *srcPtr, int len, NFloat *dstPtr);
void ApplyTanHActCUDA(NFloat *srcPtr, int len, NFloat *dstPtr);
void ApplyLogTransCUDA(NFloat *srcPtr, int len, NFloat *dstPtr);
void SumNMatrixByColCUDA(NFloat *srcPtr, int row, int col, NFloat *dstPtr);
void RedSumNMatrixByColCUDA(NFloat *srcPtr, int row, int col, Boolean accFlag, NFloat *dstPtr);
void SquaredNSegmentCUDA(NFloat *srcPtr, int segLen, NFloat *dstPtr);
void CompAdaGradNSegmentCUDA(NFloat eta, int K, int segLen, NFloat *ssgSeg, NFloat *nlrSeg);
void HNBlasNNgemmCUDA(int m, int n, int k, NFloat alpha, NFloat *A, NFloat *B, NFloat beta, NFloat *C);
void HNBlasNTgemmCUDA(int m, int n, int k, NFloat alpha, NFloat *A, NFloat *B, NFloat beta, NFloat *C);
void HNBlasTNgemmCUDA(int m, int n, int k, NFloat alpha, NFloat *A, NFloat *B, NFloat beta, NFloat *C);
void FindMaxElementCUDA(NFloat *srcPtr, int row, int col, NFloat *dstPtr);

void CalXENTCriterionCUDA(NFloat *refPtr, NFloat *hypPtr, int segLen, NFloat *crtPtr);
void CalMMSECriterionCUDA(NFloat *refPtr, NFloat *hypPtr, int segLen, NFloat *crtPtr);
void AddNSegmentTargetPenCUDA(NFloat *srcSeg, NFloat *penSeg, int row, int col, NFloat *dstSeg);
/*void SubNSegmentByConstCUDA(NFloat *srcSeg, int segLen, NFloat constVal, NFloat *dstSeg);*/
/* cz277 - semi */
void ShiftNSegmentValsCUDA(NFloat *srcSeg, int segLen, NFloat shiftVal, NFloat *dstSeg);
void SetNSegmentCUDA(NFloat val, NFloat *segPtr, int segLen);
void ClearNSegmentCUDA(NFloat *segPtr, int segLen);
void CopyPartialNSegmentCUDA(int minRow, int minCol, NFloat *srcPtr, int srcCol, NFloat *dstPtr, int dstCol);
/* cz277 - l2 fix */
void AddScaledNSegmentCUDA(NFloat *srcPtr, int segLen, NFloat scale, NFloat *dstPtr);

/* cz277 - cuda fblat */
void SetModelBetaPlusCUDA(int T, NMatrix *llhMat, int *qLo, int *qHi, int Q, float probScale, AcousticDev *acList);

/* cz277 - cuda fblat */
void ZeroAlphasCUDA(int T, int Q, AcousticDev *acList);
void StepAlphaCUDA(int Q, AcousticDev *acList);

/* cz277 - gradlim */
void ClipNSegmentValsCUDA(NFloat* srcSeg, int len, NFloat upperLim, NFloat lowerLim, NFloat *dstSeg);

/* cz277 - max norm */
void CalExtNMatrixL2NormCUDA(NFloat *matPtr, NFloat *vecPtr, int row, int col, NFloat *alphas);


#ifdef __cplusplus
}
#endif

#endif

/* ----------------------- End of HCUDA.h --------------------------- */

