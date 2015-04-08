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
/*         File: HANNet.c  ANN Model Definition Data Type      */
/* ----------------------------------------------------------- */

char *hannet_version = "!HVER!HANNet:   3.4.1 [CUED 30/11/13]";
char *hannet_vc_id = "$Id: HANNet.c,v 1.1.1.1 2013/11/13 09:54:58 cz277 Exp $";

#include "cfgs.h"
#include <time.h>
#include "HShell.h"
#include "HMem.h"
#include "HMath.h"
#include "HSigP.h"
#include "HWave.h"
#include "HAudio.h"
#include "HParm.h"
#include "HLabel.h"
#include "HANNet.h"
#include "HModel.h"
#include "HTrain.h"
#include "HNet.h"
#include "HArc.h"
#include "HFBLat.h"
#include "HDict.h"
#include "HAdapt.h"
#include <math.h>

/* ------------------------------ Trace Flags ------------------------------ */

static int trace = 0;

#define T_TOP 0001
#define T_CCH 0002

/* --------------------------- Memory Management --------------------------- */


/* ----------------------------- Configuration ------------------------------*/

static ConfParam *cParm[MAXGLOBS];      /* config parameters */
static int nParm = 0;
static size_t batchSamples = 1;                 /* the number of samples in batch; 1 sample by default */
static char *updtFlagStr = NULL;                /* the string pointer indicating the layers to update */
static int updtIdx = 0;                         /* the index of current update*/
static Boolean hasShownUpdtFlag = FALSE;
/* cz277 - 1007 */
static int batIdx = 0;


/* get the batch size */
int GetNBatchSamples(void) {
    return batchSamples;
}

/* set the batch size */
void SetNBatchSamples(int userBatchSamples) {
    batchSamples = userBatchSamples;
#ifdef CUDA
    RegisterTmpNMat(1, batchSamples);
#endif
}

/* set the index of current update */
void SetUpdateIndex(int curUpdtIdx) {
    updtIdx = curUpdtIdx;
}

/* get the index of current update */
int GetUpdateIndex(void) {
    return updtIdx;
}

/*  */
void SetBatchIndex(int curBatIdx) {
    batIdx = curBatIdx;
}

/*  */
int GetBatchIndex(void) {
    return batIdx;
}

/*  */
void InitANNet(void)
{
    int intVal;
    char buf[MAXSTRLEN];

    Register(hannet_version, hannet_vc_id);
    nParm = GetConfig("HANNET", TRUE, cParm, MAXGLOBS);

    if (nParm > 0) {
        if (GetConfInt(cParm, nParm, "TRACE", &intVal)) { 
            trace = intVal;
        }
        if (GetConfInt(cParm, nParm, "BATCHSAMP", &intVal)) {
            if (intVal <= 0) {
                HError(9999, "InitANNet: Fail to set batch size");
            }
            /*batchSamples = intVal;*/
            SetNBatchSamples(intVal);
        }
        if (GetConfStr(cParm, nParm, "UPDATEFLAGS", buf)) {
            /*updtFlagStr = (char *) New(&gcheap, strlen(buf));
            strcpy(updtFlagStr, buf);*/
            updtFlagStr = CopyString(&gcheap, buf);
        }
    }

    if (TRUE) {
            /* GPU/MKL/CPU */              /* discard: should be set when compiling */
            /* THREADS */
            /* SGD/HF */
            /* LEARNING RATE SCHEDULE */
            /*     RELATED STUFFS */
    }
}

/* set the update flag for each ANN layer */
void SetUpdateFlags(ANNSet *annSet) {
    int i;
    AILink curAI;
    ADLink annDef;
    LELink layerElem;
    char *charPtr = NULL;
    char buf[256];
    
    if ((trace & T_TOP) && (hasShownUpdtFlag == FALSE)) {
        printf("SetUpdateFlags: Updating ");
    }

    if (updtFlagStr != NULL) {
        strcpy(buf, updtFlagStr);
        charPtr = strtok(buf, ",");
        /*charPtr = strtok(updtFlagStr, ",");*/
    }
    curAI = annSet->defsHead;
    while (curAI != NULL) {
        annDef = curAI->annDef;
        for (i = 0; i < annDef->layerNum; ++i) {
            layerElem = annDef->layerList[i];
            if (charPtr != NULL) {
                layerElem->trainInfo->updtFlag = atoi(charPtr);
                charPtr = strtok(NULL, ",");
            }
            else {
                layerElem->trainInfo->updtFlag = ACTFUNUK | BIASUK | WEIGHTUK;
            }
            if ((trace & T_TOP) && (hasShownUpdtFlag == FALSE)) {
                if (!(layerElem->trainInfo->updtFlag & (ACTFUNUK | BIASUK | WEIGHTUK))) {
                    printf(", NoParam");
                }
                else {
                    printf(", ");
                    if (layerElem->trainInfo->updtFlag & ACTFUNUK) { 
                        if (layerElem->actfunKind == HERMITEAF) {
                            printf("+ActFun");
                        }
                    }
                    if (layerElem->trainInfo->updtFlag & BIASUK) { 
                        printf("+Bias");
                    }
                    if (layerElem->trainInfo->updtFlag & WEIGHTUK) {
                        printf("+Weight");
                    }
                }
            }
        }
        curAI = curAI->next;
    }

    if ((trace & T_TOP) && (hasShownUpdtFlag == FALSE)) {
        printf("\n");
        hasShownUpdtFlag = TRUE;
    }
}

static inline void FillBatchFromFeaMix(FeaMix *feaMix, int batLen, int *CMDVecPL) {
    int i, j, k, srcOff = 0, curOff = 0, dstOff, hisOff, hisDim;
    FELink feaElem;

    /* if it is the shared */
    if (feaMix->feaList[0]->feaMat == feaMix->mixMat) {
        return;
    }
    /* cz277 - 1007 */
    if (feaMix->batIdx > batIdx + 1 || feaMix->batIdx < batIdx) {
        HError(9999, "FillBatchFromFeaMix: batIdx of this feature mix does not match the global index");
    }
    else if (feaMix->batIdx == batIdx + 1) {
        return;
    }
    else {
        ++feaMix->batIdx;
    }

    /* otherwise, fill the batch with a mixture of the FeaElem */
    for (i = 0; i < feaMix->elemNum; ++i) {
        feaElem = feaMix->feaList[i];

        if (feaElem->inputKind == INPFEAIK || feaElem->inputKind == AUGFEAIK) {
            for (j = 0, srcOff = 0, dstOff = curOff; j < batLen; ++j, srcOff += feaElem->extDim, dstOff += feaMix->mixDim) {
                CopyNSegment(feaElem->feaMat, srcOff, feaElem->extDim, feaMix->mixMat, dstOff);
            }
        }
        else if (feaElem->inputKind == ANNFEAIK) {  /* ANNFEAIK, left context is consecutive */
            for (j = 0; j < batLen; ++j) {

                /* cz277 - gap */
                hisDim = feaElem->hisLen * feaElem->feaDim;
                hisOff = j * hisDim;
                if (CMDVecPL != NULL && feaElem->hisMat != NULL) {
                    if (CMDVecPL[j] == 0) {	/* reset the history */
			            ClearNMatrixSegment(feaElem->hisMat, hisOff, hisDim);
                    }
                    else if (CMDVecPL[j] > 0) {	/* shift the history */
                        CopyNSegment(feaElem->hisMat, CMDVecPL[j] * hisDim, hisDim, feaElem->hisMat, hisOff);
                    }
                }
                /* standard operations */
                dstOff = j * feaMix->mixDim + curOff;
                for (k = 1; k <= feaElem->ctxMap[0]; ++k, dstOff += feaElem->feaDim) { 
                    if (feaElem->ctxMap[k] < 0) {
                        /* first, previous segments from hisMat to feaMix->mixMat */
                        srcOff = ((j + 1) * feaElem->hisLen + feaElem->ctxMap[k]) * feaElem->feaDim;
                        CopyNSegment(feaElem->hisMat, srcOff, feaElem->feaDim, feaMix->mixMat, dstOff);
                    }
                    else if (feaElem->ctxMap[k] == 0) {
                        /* second, copy current segment from feaMat to feaMix->mixMat */
                        srcOff = j * feaElem->srcDim + feaElem->dimOff;
                        CopyNSegment(feaElem->feaMat, srcOff, feaElem->feaDim, feaMix->mixMat, dstOff);
                    }
                    else {
                        HError(9999, "FillBatchFromFeaMix: The future of ANN features are not applicable");
                    }
                }
                /* shift history info in hisMat and copy current segment from feaMat to hisMat */
                if (feaElem->hisMat != NULL) {
                    dstOff = hisOff;
                    srcOff = dstOff + feaElem->feaDim;
                    for (k = 0; k < feaElem->hisLen - 1; ++k, srcOff += feaElem->feaDim, dstOff += feaElem->feaDim) {
                        CopyNSegment(feaElem->hisMat, srcOff, feaElem->feaDim, feaElem->hisMat, dstOff);    
                    }
                    srcOff = j * feaElem->srcDim + feaElem->dimOff;
                    CopyNSegment(feaElem->feaMat, srcOff, feaElem->feaDim, feaElem->hisMat, dstOff);
                }
            }
        }
        curOff += feaElem->extDim;
    }
}


/* fill a batch with error signal */
static inline void FillBatchFromErrMix(FeaMix *errMix, int batLen, NMatrix *mixMat) {
    int i, j, srcOff, dstOff, segLen;
    FELink errElem;

    /* if it is the shared */
    if (errMix->feaList[0]->feaMat == mixMat) {
        return;
    }

    /* otherwise, fill the batch with a mixture of the FeaElem */
    dstOff = 0;
    /* reset mixMat to 0 */
    /*SetNMatrix(0.0, mixMat, batLen);*/
    ClearNMatrix(mixMat, batLen);
    /* accumulate the error signals from each source */
    for (i = 0; i < batLen; ++i) {
        for (j = 0; j < errMix->elemNum; ++j) {
            errElem = errMix->feaList[j];
            srcOff = i * errElem->srcDim + errElem->dimOff;
            segLen = errElem->extDim;
            AddNSegment(errElem->feaMat, srcOff, segLen, mixMat, dstOff);
            dstOff += segLen;
        }
    }
}

/* temp function */
void ShowAddress(ANNSet *annSet) {
    int i;
    AILink curAI;
    ADLink annDef;
    LELink layerElem;

    curAI = annSet->defsHead;
    while (curAI != NULL) {
        annDef = curAI->annDef;
        printf("ANNInfo = %p. ANNDef = %p: \n", curAI, annDef);
        for (i = 0; i < annDef->layerNum; ++i) {
            layerElem = annDef->layerList[i];
            printf("layerElem = %p, feaMix[0]->feaMat = %p, xFeaMat = %p, yFeaMat = %p, trainInfo = %p, dxFeaMat = %p, dyFeaMat = %p, labMat = %p\n", layerElem, layerElem->feaMix->feaList[0]->feaMat, layerElem->xFeaMat, layerElem->yFeaMat, layerElem->trainInfo, layerElem->trainInfo->dxFeaMat, layerElem->trainInfo->dyFeaMat, layerElem->trainInfo->labMat);
        }
        printf("\n");
        curAI = curAI->next;
    }
}

/* update the map sum matrix for outputs */
void UpdateOutMatMapSum(ANNSet *annSet, int batLen, int streamIdx) {

    HNBlasTNgemm(annSet->mapStruct->mappedTargetNum, batLen, annSet->outLayers[streamIdx]->nodeNum, 1.0, annSet->mapStruct->maskMatMapSum[streamIdx], annSet->outLayers[streamIdx]->yFeaMat, 0.0, annSet->mapStruct->outMatMapSum[streamIdx]);
}

/* update the map sum matrix for labels */
void UpdateLabMatMapSum(ANNSet *annSet, int batLen, int streamIdx) {

    HNBlasTNgemm(annSet->mapStruct->mappedTargetNum, batLen, annSet->outLayers[streamIdx]->nodeNum, 1.0, annSet->mapStruct->maskMatMapSum[streamIdx], annSet->outLayers[streamIdx]->trainInfo->labMat, 0.0, annSet->mapStruct->labMatMapSum[streamIdx]);
}

/* the batch with input features are assumed to be filled */
void ForwardPropBatch(ANNSet *annSet, int batLen, int *CMDVecPL) {
    int i;
    AILink curAI;
    ADLink annDef;
    LELink layerElem;

    /* init the ANNInfo pointer */
    curAI = annSet->defsHead;
    /* proceed in the forward fashion */
    while (curAI != NULL) {
        /* fetch current ANNDef */
        annDef = curAI->annDef;
        /* proceed layer by layer */
        for (i = 0; i < annDef->layerNum; ++i) {
            /* get current LayerElem */
            layerElem = annDef->layerList[i];
            /* at least the batch (feaMat) for each FeaElem is already */
            FillBatchFromFeaMix(layerElem->feaMix, batLen, CMDVecPL);
            /* do the operation of current layer */
            switch (layerElem->operKind) {
                case MAXOK:

                    break;
                case SUMOK: 
                    /* y = b, B^T should be row major matrix, duplicate the bias vectors */ 
                    DupNVector(layerElem->biasVec, layerElem->yFeaMat, batLen);
                    /* y += w * b, X^T is row major, W^T is column major, Y^T = X^T * W^T + B^T */
                    
                    
                    //cw564 - mb -- begin
                    if (i == annDef->layerNum - 1)
                    {
                        SetNSegmentCUDA(0, layerElem->mb_bases_yFeaMat, batLen * MBP()->num_basis * layerElem->nodeNum);

                        HNBlasTNgemm(layerElem->nodeNum, 
                                batLen * MBP()->num_basis, 
                                layerElem->inputDim, 1.0, 
                                layerElem->wghtMat, layerElem->xFeaMat, 1.0, 
                                layerElem->mb_bases_yFeaMat);
                        printf("%d %d %d\n", layerElem->nodeNum, batLen, layerElem->inputDim);
                        exit(0);
                    }
                    else
                    {
                        HNBlasTNgemm(layerElem->nodeNum, batLen, layerElem->inputDim, 1.0, layerElem->wghtMat, layerElem->xFeaMat, 1.0, layerElem->yFeaMat);
                    }
                    //cw564 - mb -- end
                    
                    
                    break;
                case PRODOK:

                    break;
                default:
                    HError(9999, "ForwardPropBatch: Unknown layer operation kind");
            }
            /* apply activation transformation */
            switch (layerElem->actfunKind) {
                case HERMITEAF:
                    ApplyHermiteAct(layerElem->yFeaMat, batLen, layerElem->nodeNum, layerElem->actParmVec, layerElem->yFeaMat);
                    break;
                case LINEARAF:
                    break;
                case RELAF:
                    ApplyReLAct(layerElem->yFeaMat, batLen, layerElem->nodeNum, layerElem->yFeaMat);
                    break;
                case SIGMOIDAF:
                    ApplySigmoidAct(layerElem->yFeaMat, batLen, layerElem->nodeNum, layerElem->yFeaMat);
                    break;
                case SOFTMAXAF:
                    ApplySoftmaxAct(layerElem->yFeaMat, batLen, layerElem->nodeNum, layerElem->yFeaMat);
                    break;
                case SOFTRELAF:
                    ApplySoftReLAct(layerElem->yFeaMat, batLen, layerElem->nodeNum, layerElem->yFeaMat);
                    break;
                case SOFTSIGNAF:
                    ApplySoftSignAct(layerElem->yFeaMat, batLen, layerElem->nodeNum, layerElem->yFeaMat);
                    break;
                case TANHAF:
                    ApplyTanHAct(layerElem->yFeaMat, batLen, layerElem->nodeNum, layerElem->yFeaMat);
                    break;
                default:
                    HError(9999, "ForwardPropBatch: Unknown activation function kind");
            }
        }

        /* get the next ANNDef */
        curAI = curAI->next;
    }

    SetBatchIndex(GetBatchIndex() + 1);
}

/* function to compute the error signal for frame level criteria (for sequence level, do nothing) */
void CalcOutLayerBackwardSignal(LELink layerElem, int batLen, ObjFunKind objfunKind) {

    if (layerElem->roleKind != OUTRK) {
        HError(9999, "CalcOutLayerBackwardSignal: Function only valid for output layers");
    }

    switch (objfunKind) {
        case MMSEOF:
            /* proceed for MMSE objective function */
            switch (layerElem->actfunKind) {
                case HERMITEAF:

                    break;
                case LINEARAF:

                    break;
                case RELAF:

                    break;
                case SIGMOIDAF:

                    break;
                case SOFTMAXAF:

                    break;
                case SOFTRELAF:

                    break;
                case SOFTSIGNAF:

                    break;
                case TANHAF:

                    break;
                default:
                    HError(9999, "CalcOutLayerBackwardSignal: Unknown output activation function");
            }
            break;
        case XENTOF:
            /* proceed for XENT objective function */
            switch (layerElem->actfunKind) {
                case HERMITEAF:

                    break;
                case LINEARAF:

                    break;
                case RELAF:

                    break;
                case SIGMOIDAF:

                    break;
                case SOFTMAXAF:
                    SubNMatrix(layerElem->yFeaMat, layerElem->trainInfo->labMat, batLen, layerElem->nodeNum, layerElem->yFeaMat);
                    break;
                case SOFTRELAF:

                    break;
                case SOFTSIGNAF:

                    break;
                case TANHAF:

                    break;
                default:
                    HError(9999, "CalcOutLayerBackwardSignal: Unknown output activation function");
            }
            break;
        case MLOF:
        case MMIOF:
        case MPEOF:
        case MWEOF:
        case SMBROF:
            break;
        default:
            HError(9999, "CalcOutLayerBackwardSignal: Unknown objective function kind");

    }
}

/* cz277 - gradprobe */
#ifdef GRADPROBE
void AccGradProbeWeight(LayerElem *layerElem) {
    int i, j, k, size;
    NFloat *wghtMat;

#ifdef CUDA
    SyncNMatrixDev2Host(layerElem->trainInfo->gradInfo->wghtMat);
#endif
    wghtMat = layerElem->trainInfo->gradInfo->wghtMat->matElems;
    /* weights */
    size = layerElem->nodeNum * layerElem->inputDim;
    j = DVectorSize(layerElem->wghtGradInfoVec);
    for (i = 0; i < size; ++i) {
        if (wghtMat[i] > layerElem->maxWghtGrad) 
            layerElem->maxWghtGrad = wghtMat[i];
        if (wghtMat[i] < layerElem->minWghtGrad)
            layerElem->minWghtGrad = wghtMat[i];
        layerElem->meanWghtGrad += wghtMat[i];
        k = wghtMat[i] / PROBERESOLUTE + j / 2;
        layerElem->wghtGradInfoVec[k + 1] += 1;
    }
}
#endif

/* cz277 - gradprobe */
#ifdef GRADPROBE
void AccGradProbeBias(LayerElem *layerElem) {
    int i, j, k, size;
    NFloat *biasVec;

#ifdef CUDA
    SyncNVectorDev2Host(layerElem->trainInfo->gradInfo->biasVec);
#endif
    biasVec = layerElem->trainInfo->gradInfo->biasVec->vecElems;
    /* biases */
    size = layerElem->nodeNum;
    j = DVectorSize(layerElem->biasGradInfoVec);
    for (i = 0; i < size; ++i) {
        if (biasVec[i] > layerElem->maxBiasGrad)
            layerElem->maxBiasGrad = biasVec[i];
        if (biasVec[i] < layerElem->minBiasGrad)
            layerElem->minBiasGrad = biasVec[i];
        layerElem->meanBiasGrad += biasVec[i];
        k = biasVec[i] / PROBERESOLUTE + j / 2;
        layerElem->biasGradInfoVec[k + 1] += 1;
    }
}
#endif

/* backward propagation algorithm */
void BackwardPropBatch(ANNSet *annSet, int batLen, Boolean accFlag) {
    int i;
    AILink curAI;
    ADLink annDef;
    LELink layerElem;
    NMatrix *dyFeaMat;

    /* init the ANNInfo pointer */
    curAI = annSet->defsTail;
    /* proceed in the backward fashion */
    while (curAI != NULL) {
        /* fetch current ANNDef */
        annDef = curAI->annDef;
        /* proceed layer by layer */
        for (i = annDef->layerNum - 1; i >= 0; --i) {
            /* get current LayerElem */
            layerElem = annDef->layerList[i];
            /* proceed different types of layers */
            if (layerElem->roleKind == OUTRK) {
                /* set dyFeaMat */
                dyFeaMat = layerElem->yFeaMat;
                CalcOutLayerBackwardSignal(layerElem, batLen, annDef->objfunKind);
            }
            else {
                /* set dyFeaMat */
                dyFeaMat = layerElem->trainInfo->dyFeaMat;
                /* at least the batch (feaMat) for each FeaElem is already */
                FillBatchFromErrMix(layerElem->errMix, batLen, dyFeaMat);
                /* apply activation transformation */
                switch (layerElem->actfunKind) {
                    case HERMITEAF:
                    
                        break;
                    case LINEARAF:
                        ApplyDLinearAct(layerElem->yFeaMat, batLen, layerElem->nodeNum, layerElem->yFeaMat);
                        /*MulNMatrix(layerElem->yFeaMat, dyFeaMat, batLen, layerElem->nodeNum, dyFeaMat);*/
                        break;
                    case RELAF:
			ApplyDReLAct(layerElem->yFeaMat, batLen, layerElem->nodeNum, layerElem->yFeaMat);
                        /*MulNMatrix(layerElem->yFeaMat, dyFeaMat, batLen, layerElem->nodeNum, dyFeaMat);*/
                        break;
                    case SIGMOIDAF:
                        ApplyDSigmoidAct(layerElem->yFeaMat, batLen, layerElem->nodeNum, layerElem->yFeaMat);
                        /*MulNMatrix(layerElem->yFeaMat, dyFeaMat, batLen, layerElem->nodeNum, dyFeaMat);*/
                        break;
                    case SOFTMAXAF:

                        break;
                    case SOFTRELAF:
                        ApplyDSoftReLAct(layerElem->yFeaMat, batLen, layerElem->nodeNum, layerElem->yFeaMat);
                        /*MulNMatrix(layerElem->yFeaMat, dyFeaMat, batLen, layerElem->nodeNum, dyFeaMat);*/
                        break;
                    case SOFTSIGNAF:

                        break;
                    case TANHAF:
                        ApplyDTanHAct(layerElem->yFeaMat, batLen, layerElem->nodeNum, layerElem->yFeaMat);
                        break;
                    default:
                        HError(9999, "BackwardPropBatch: Unknown hidden activation function kind");
                }
                /* times sigma_k (dyFeaMat, from the next layer) */
                MulNMatrix(layerElem->yFeaMat, dyFeaMat, batLen, layerElem->nodeNum, dyFeaMat);
            }
            /* do current layer operation */
            switch (layerElem->operKind) {
                case MAXOK:

                    break;
                case SUMOK:
                    /* Y^T is row major, W^T is column major, X^T = Y^T * W^T */
                    HNBlasNNgemm(layerElem->inputDim, batLen, layerElem->nodeNum, 1.0, layerElem->wghtMat, dyFeaMat, 0.0, layerElem->trainInfo->dxFeaMat);
                    break;
                case PRODOK:

                    break;
                default:
                    HError(9999, "BackwardPropBatch: Unknown layer operation kind");
            }
            /* compute and accumulate the updates */
            /* {layerElem->xFeaMat[n_frames * inputDim]}^T * dyFeaMat[n_frames * nodeNum] = deltaWeights[inputDim * nodeNum] */
            if (layerElem->trainInfo->updtFlag & WEIGHTUK) { 
                HNBlasNTgemm(layerElem->inputDim, layerElem->nodeNum, batLen, 1.0, layerElem->xFeaMat, dyFeaMat, accFlag, layerElem->trainInfo->gradInfo->wghtMat);
                /* cz277 - gradprobe */
#ifdef GRADPROBE
                AccGradProbeWeight(layerElem);    
#endif
            }
            /* graidents for biases */
            if (layerElem->trainInfo->updtFlag & BIASUK) {
                SumNMatrixByCol(dyFeaMat, batLen, layerElem->nodeNum, accFlag, layerElem->trainInfo->gradInfo->biasVec);
                /* cz277 - gradprobe */
#ifdef GRADPROBE
                AccGradProbeBias(layerElem);
#endif
            }

            if (layerElem->trainInfo->ssgInfo != NULL) {
                /* attention: these two operations are gonna to change dyFeaMat elements to their square */
                SquaredNMatrix(layerElem->xFeaMat, batLen, layerElem->inputDim, GetTmpNMat());
                SquaredNMatrix(dyFeaMat, batLen, layerElem->nodeNum, dyFeaMat);
                if (layerElem->trainInfo->updtFlag & WEIGHTUK)
                    HNBlasNTgemm(layerElem->inputDim, layerElem->nodeNum, batLen, 1.0, GetTmpNMat(), dyFeaMat, 1.0, layerElem->trainInfo->ssgInfo->wghtMat);
                if (layerElem->trainInfo->updtFlag & BIASUK) 
                    SumNMatrixByCol(dyFeaMat, batLen, layerElem->nodeNum, TRUE, layerElem->trainInfo->ssgInfo->biasVec);
            }
        }

        /* get the previous ANNDef */
        curAI = curAI->prev;
    }
}

/* randomise an ANN layer */
void RandANNLayer(LELink layerElem, int seed, float scale) {
    float r;
 
    /*r = 4 * sqrt(6.0 / (float) (layerElem->nodeNum + layerElem->inputDim));*/
    /*r = sqrt(6.0 / (float) (layerElem->nodeNum + layerElem->inputDim));*/
    r = 0.25 * sqrt(6.0 / (float) (layerElem->nodeNum + layerElem->inputDim));
    r *= scale;

    RandInit(seed); 
    RandNSegment(-1.0 * r, r, layerElem->wghtMat->rowNum * layerElem->wghtMat->colNum, layerElem->wghtMat->matElems);

    if (layerElem->actfunKind == RELAF || layerElem->actfunKind == SOFTRELAF) {
        RandMaskNSegment(0.25, 0.0, layerElem->wghtMat->rowNum * layerElem->wghtMat->colNum, layerElem->wghtMat->matElems);
    }

    ClearNVector(layerElem->biasVec);
    /* TODO: if HERMITEAF */
#ifdef CUDA
    SyncNMatrixHost2Dev(layerElem->wghtMat);
    SyncNVectorHost2Dev(layerElem->biasVec);
#endif
}

/* generate a new ANN layer and randomise it */
/*LELink GenRandLayer(MemHeap *heap, int nodeNum, int inputDim, int seed) {*/
LELink GenNewLayer(MemHeap *heap, int nodeNum, int inputDim) {
     LELink layerElem;

     /*layerElem = (LELink) New(heap, sizeof(LayerElem));*/
     layerElem = GenBlankLayer(heap);
     /*layerElem->operKind = operKind;
     layerElem->actfunKind = actfunKind;*/
     layerElem->nodeNum = nodeNum;
     layerElem->inputDim = inputDim;
     layerElem->wghtMat = CreateNMatrix(heap, nodeNum, inputDim);
     layerElem->biasVec = CreateNVector(heap, nodeNum);

     /*RandANNLayer(layerElem, seed);*/

     return layerElem;     
}

void SetFeaMixBatchIdxes(ANNSet *annSet, int newIdx) {
    int i;
    AILink curAI;
    ADLink annDef;
    LELink layerElem;
    NMatrix *dyFeaMat;

    /* init the ANNInfo pointer */
    curAI = annSet->defsTail;
    while (curAI != NULL) {
        annDef = curAI->annDef;
        for (i = annDef->layerNum - 1; i >= 0; --i) {
            layerElem = annDef->layerList[i];
	    if (layerElem->feaMix->batIdx == 0) {
                layerElem->feaMix->batIdx = newIdx; 
            }
        }
        curAI = curAI->next;
    }
}

/* cz277 - max norm2 */
Boolean IsLinearActFun(ActFunKind actfunKind) {
    switch (actfunKind) {
        case HERMITEAF:
        case LINEARAF:
        case RELAF:
        case SOFTRELAF:
            return TRUE;
        default:
            return FALSE;
    }
}

/* cz277 - max norm2 */
Boolean IsNonLinearActFun(ActFunKind actfunKind) {
    switch (actfunKind) {
        case SIGMOIDAF:
        case SOFTMAXAF:
        case SOFTSIGNAF:
        case TANHAF:
            return TRUE;
        default:
            return FALSE;
    }
}



//cw564 - mb -- begin

static int nMBParm = 0;
static ConfParam * cMBParm[MAXGLOBS];
static MBParam mbp;

static void ParseLam(char * lamfn)
{
    FILE * file = fopen(lamfn, "r");
    if (!file)
    {
        HError(9999, "HNMB: Lambda file does not exist.");    
    }

    int my_num_basis, my_num_rgc;

    fscanf(file, "%d %d %d", &(mbp.num_spkr), &(my_num_basis), &(my_num_rgc));
    if (my_num_basis != mbp.num_basis || my_num_rgc != mbp.num_rgc)
    {
        HError(9999, "HNMB: Num_basis or Num_rgc mismatch.");
    }
    

    int per_spkr_dim = mbp.num_basis * mbp.num_rgc;
    for (int i = 0; i < mbp.num_spkr; ++ i)
    {
        char * buf = malloc(sizeof(char) * MAXARRAYLEN);
        mbp.lam[i] = malloc(sizeof(float) * per_spkr_dim);

        fscanf(file, "%s", buf);
        mbp.spkrid2spkrname[i] = buf;
        
        /*
        char tmp[256];
        MaskMatch(mbp.adaptmask, tmp, buf);
        printf("%s\n", tmp);
        exit(0);
        */
        for (int j = 0; j < per_spkr_dim; ++ j)
        {
            fscanf(file, "%f", &(mbp.lam[i][j]));
        }
    }
    


    /*
    printf("NININININI%d\n", mbp.num_spkr);
    
    for (int i = 0; i < mbp.num_spkr; ++ i)
    {
        printf("%s", mbp.spkrid2spkrname[i]);
        for (int j = 0; j < per_spkr_dim; ++ j)
        {
            printf(" %f", mbp.lam[i][j]);
        }
        printf("\n");
    }
    exit(0); 
    */

    fclose(file);
}


static void ParseSta2Rgc(char * s2cfn)
{
    FILE * file = fopen(s2cfn, "r");
    
    if (!file)
    {
        HError(9999, "HNMB: STA2RGC file does not exist.");    
    }

    int num_sta;
    int local_num_rgc;

    fscanf(file, "%d %d", &num_sta, &local_num_rgc);

    if (local_num_rgc != mbp.num_rgc)
    {
        HError(9999, "HNMB: Number of regression class mismatches in LAM and STA2RGC files.");
    }

    int now_sta, now_rgc;
    for (int i = 0; i < num_sta; ++ i)
    {
        fscanf(file, "%d %d", &now_sta, &now_rgc);
        mbp.sta2rgc[now_sta] = now_rgc;
    }

    /*
    printf("WOWOWOWOWO %d\n", num_sta);
    for (int i = 0; i < num_sta; ++ i)
    {
        printf("%d->%d ", i, mbp.sta2rgc[i]);
    }
    exit(0);
    */

    fclose(file);
}

void InitMB(void)
{
    int intVal, tmpInt;
    double doubleVal;
    Boolean boolVal;
    char buf[MAXSTRLEN], buf2[MAXSTRLEN];
    char *charPtr, *charPtr2;
    ConfParam *cpVal;

    char lamfn[MAXSTRLEN], sta2rgcfn[MAXSTRLEN], adaptmask[MAXSTRLEN];

    nMBParm = GetConfig("HNMB", TRUE, cMBParm, MAXGLOBS);
    if (nMBParm > 0)
    {
        if (GetConfInt(cMBParm, nMBParm, "NUMBASIS", &intVal))
        {
            mbp.num_basis = intVal;
        }
        if (GetConfInt(cMBParm, nMBParm, "NUMRGC", &intVal))
        {
            mbp.num_rgc = intVal;
        }
        if (GetConfStr(cMBParm, nMBParm, "LAMFN", buf))
        {
            strcpy(lamfn, buf);
        }
        if (GetConfStr(cMBParm, nMBParm, "STA2RGCMAPFN", buf))
        {
            strcpy(sta2rgcfn, buf);
        }
        if (GetConfStr(cMBParm, nMBParm, "ADAPTMASK", buf))
        {
            strcpy(mbp.adaptmask, buf);
        }
    }
    
    ParseLam(lamfn);
    ParseSta2Rgc(sta2rgcfn);
}

MBParam * MBP(void)
{
    return &mbp;
}

//cw564 - mb -- end
