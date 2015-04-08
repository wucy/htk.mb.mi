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
/*         File: HANNet.h  ANN Model Definition Data Type      */
/* ----------------------------------------------------------- */

/* !HVER!HANNet:   3.4.1 [CUED 29/11/13] */

#ifndef _HANNET_H_
#define _HANNET_H_

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include "HMem.h"

/* ------------------------- Trace Flags ------------------------- */

/*
 The following types define the in-memory representation of a ANN
*/

/* ------------------------- Predefined Types ------------------------- */

#define MAPTARGETPREF "_MAPPED"	/* the prefix to convert the mapped target */
/* cz277 - 1007 */
#define MAXFEAMIXOWNER 20	/* the maximum number of layers or hset streams that could own a feature mixture */

/* cz277 - gradprobe */
#ifdef GRADPROBE
#define PROBERESOLUTE 50
#define PROBEBOUNDARY 1000
#endif

/* ------------------------- ANN Definition ------------------------- */

enum _ANNKind {AUTOENCAK, CNNAK, FNNAK, MAXOUTAK, RBMAK, RNNAK, SPNAK, USERAK}; /* regard RBM as a ANN for the ease of programming */
typedef enum _ANNKind ANNKind;

enum _OperKind {MAXOK, SUMOK, PRODOK};
typedef enum _OperKind OperKind;

enum _InputKind {INPFEAIK, ANNFEAIK, ERRFEAIK, AUGFEAIK}; /* cz277 - aug */
typedef enum _InputKind InputKind;

enum _ActFunKind {HERMITEAF, LINEARAF, RELAF, SIGMOIDAF, SOFTMAXAF, SOFTRELAF, SOFTSIGNAF, TANHAF};
typedef enum _ActFunKind ActFunKind;

enum _LayerRoleKind {HIDRK, INPRK, OUTRK};
typedef enum _LayerRoleKind LayerRoleKind;

/*enum _ObjFunKind {MLOF, MMIOF, MMSEOF, MPEOF, MWEOF, SMBROF, UNKOF, XENTOF};*/
enum _ObjFunKind {UNKOF = 0, MLOF = 1, MMIOF = 2, MMSEOF = 4, MPEOF = 8, MWEOF = 16, SMBROF = 32, XENTOF = 64};
typedef enum _ObjFunKind ObjFunKind;

enum _ANNUpdtKind {ACTFUNUK = 1, BIASUK = 2, WEIGHTUK = 4};
typedef enum _ANNUpdtKind ANNUpdtKind;

typedef struct _LayerElem *LELink;
typedef struct _FeaElem *FELink;
typedef struct _ANNDef *ADLink;
typedef struct _ANNInfo *AILink;
typedef struct _LayerInfo *LILink;

//cw564 - mb -- begin
#define MAXARRAYLEN 1000000
typedef struct _MBParam {
    int num_basis;
    int num_rgc;
    int num_spkr;

    char adaptmask[MAXSTRLEN];

    float * lam[MAXARRAYLEN];
    int sta2rgc[MAXARRAYLEN];

    char * uttrid2spkrname[MAXARRAYLEN];
    char * uttrid2uttrname[MAXARRAYLEN];
    int uttrid2spkrid[MAXARRAYLEN];
    char * spkrid2spkrname[MAXARRAYLEN];

} MBParam;


extern void InitMB(void);
extern MBParam * MBP(void);

//cw564 - mb -- end


typedef struct _FeaElem {
    int feaDim;                 /* the dimension of this kind of feature (without context expansion) */
    int extDim;                 /* the dimension of this kind of feature (with context expansion and transforms) */
    IntVec ctxMap;              /* the array contains the offset to current for context expansion */
    InputKind inputKind;        /* the kind of the feature */
    LELink feaSrc;              /* the layer pointer to the source of current feature element */
    NMatrix *feaMat;            /* the feature batch of this element; NULL if ANNFEAIK */
    int dimOff;                 /* the offset of the start dimension in feaMat of this FeaElem; useful for backprop */
    int srcDim;                 /* the dimension of the feature in feaMat */
    /* cz277 - aug */
    int augFeaIdx;		/* the index of this (if it was) augmented feature index; default: 0 */
    /* cz277 - split */
    int streamIdx;		/* the index of the associated stream; default: 0 */
    /* cz277 - semi */
    char mName[MAXSTRLEN];      /* the ANN feature source macro name */
    char mType;                 /* the ANN feature source macro type */
    Boolean doBackProp;		/* do back propagation through this ANNFEAIK or not */
    /* cz277 - gap */
    int hisLen;			/* the length of the history */
    NMatrix *hisMat;		/* the matrix for ANN feature history */
    int nUse;                   /* the usage counter */

    char* * frm2scp; //cw564 - mb -- to be removed
    int * frm2spkrIdx; //cw564 - mb

} FeaElem;

typedef struct _FeaMix {
    /* cz277 - 1007 */
    int batIdx;			/* the number of batches been processed */
    int ownerNum;		/* the number of owners of this feature mixture  */
    LELink ownerList[MAXFEAMIXOWNER];		/* the layers which employs this feature mxiture */
    int elemNum;                /* number of different feature components */
    int mixDim;                 /* the total dimension of the input (a mixture of different features) */
    FELink *feaList;            /* the feature information structure */
    /* cz277 - 1007 */
    NMatrix *mixMat;		/* the feature mixture matrix */
    int nUse;                   /* usage counter */
} FeaMix;

typedef struct _LayerInfo {
    NMatrix *wghtMat;           /* the weight matrix for momentum of current update index */
    NVector *biasVec;           /* the bias vector for momentum of current update index */
} LayerInfo;

typedef struct _TrainInfo {
    NMatrix *labMat;            /* the batches for the output targets for all streams */ 
    /*NMatrix *labMatMapSum;*/	/* the mapping label matrix */
    LILink gradInfo;            /* the structure for gradients */
    LILink updtInfo;            /* the structure for updates */
    LILink nlrInfo;             /* the structure for learning rates (for SGD only) */
    LILink ssgInfo;             /* the  structure for the sum of the squared gradients */
    NMatrix *dxFeaMat;          /* de/dx */
    NMatrix *dyFeaMat;          /* de/dy */
    ANNUpdtKind updtFlag;       /* whether update this layer or not */
} TrainInfo;

typedef struct _ANNInfo {
    AILink next;                /* pointer to the next item of the chain */
    AILink prev;                /* pointer to the previous item of the chain */
    ADLink annDef;              /* one owner of this layer */
    int index;                  /* for LayerElem, the index of this layer in that owner ANNDef */
    int fidx;			/* the file index of the associated ANNDef */
} ANNInfo;                      /* for ANNSet, the index of this of this ANNDef in that ANNSet */

typedef struct _LayerElem {
    int ownerCnt;               /* the number of owners in the owner chain */
    ANNInfo *ownerHead;         /* the head of the chain contains all owners */
    ANNInfo *ownerTail;         /* the tail of the chain contains all owners */
    OperKind operKind;          /* the kind of current layer */
    ActFunKind actfunKind;      /* the kind of activation function */
    NVector *actParmVec;        /* the parameters for current activation function */
    FeaMix *feaMix;             /* a list of different features for forward propagation */
    FeaMix *errMix;             /* a list of different error signals for back propagation */
    IntVec shareVec;            /* a vector contains the weight sharing by the rows in the extended matrix */
    int inputDim;               /* the number of inputs to each node in current layer (column number of wgthMat) */
    int nodeNum;                /* the number of nodes in current layer (row number of wghtMat) */
    NMatrix *wghtMat;           /* the weight matrix of current layer (a transposed matrix) */
    NVector *biasVec;           /* the bias vector of current layer */
    NMatrix *xFeaMat;           /* the feature batch for the input signal, could point to another yFeaMat in a different LayerElem */
    NMatrix *yFeaMat;           /* the feature batch for the output signal */
    TrainInfo *trainInfo;       /* the structure for training info, could be NULL (if not training) */
    /*Boolean expInput;*/           /* a boolean flag for whether the input feature of this layer element should be output */
    LayerRoleKind roleKind;     /* the role of current layer */
    int nUse;                   /* usage counter */
    int nDrv;                   /* feature derived counter */
    
    NMatrix *mb_bases_yFeaMat; //cw564 - mb

    /*ANNUpdtKind updtFlag;*/       /* whether update this layer or not */
    /* cz277 - gradprobe */
#ifdef GRADPROBE
    DVector wghtGradInfoVec;
    DVector biasGradInfoVec;
    NFloat maxWghtGrad;
    NFloat minWghtGrad;
    NFloat meanWghtGrad;
    NFloat maxBiasGrad;
    NFloat minBiasGrad;
    NFloat meanBiasGrad;
#endif    
} LayerElem;

typedef struct _ANNDef {
    ANNKind annKind;
    int layerNum;               /* the number of layers */
    LELink *layerList;          /* a list of layers */
    int targetNum;              /* number of targets in this ANN */
    char *annDefId;             /* identifier for the ANNDef */ 
    ObjFunKind objfunKind;      /* the objective function used to train this ANN */
    int nUse;                   /* usage counter */
    int nDrv;
} ANNDef;

typedef struct _TargetMap {
    char *name;                 /* the input name of this target */
    char *mappedName;           /* the modified name of this mapped target (with MAPTARGETPREF) */
    int index;                  /* the index of this mapped target, in MappedList, maxResults, and sumResults */
    IntVec maxResults;          /* mapped confusion list */
    IntVec sumResults;          /* summed confusion list */
    int sampNum;		/* the total number of samples */
    float mappedTargetPen[SMAX];/* the penalty of the mapped target */
} TargetMap;

typedef struct _TargetMapStruct {
    TargetMap *targetMapList;   /* used to convert the index of a mapped target to its structure */
    int mappedTargetNum;        /* the total number of mapped targets */
    IntVec mapVectors[SMAX];    /* the mapping vectors for target map */
    NMatrix *maskMatMapSum[SMAX]; /* the mat matrix generated by extending mapVec to get outMatMapSum*/
    NMatrix *labMatMapSum[SMAX];
    NMatrix *outMatMapSum[SMAX];/* the mapping matrix for the yFeaMat of the output layers */
    NMatrix *llhMatMapSum[SMAX];/* the mapping matrix with llh values */
    NVector *penVecMapSum[SMAX];
} TargetMapStruct;

typedef struct _ANNSet {
    int annNum;                 /* an ANN is a mixture of a set of sub ANNs */
    AILink defsHead;            /* the head of the chain contains all sub ANNs (ANNDefs) */
    AILink defsTail;            /* the tail of the chain contains all sub ANNs (ANNDefs) */
    LELink outLayers[SMAX];     /* pointers to the output layer in each stream */ 
    TargetMapStruct *mapStruct;	/* the structure for target mapping */
    NMatrix *llhMat[SMAX];	/* the llr matrix of the yFeaMat of the output layers */
    NVector *penVec[SMAX];
    
    MBParam mbp;

} ANNSet;

/* ------------------------ Global Settings ------------------------- */

int GetNBatchSamples(void);
void SetNBatchSamples(int userBatchSamples);
void InitANNet(void);

void UpdateOutMatMapSum(ANNSet *annSet, int batLen, int streamIdx);
void UpdateLabMatMapSum(ANNSet *annSet, int batLen, int streamIdx);
void ForwardPropBatch(ANNSet *annSet, int batLen, int *CMDVecPL);
void CalcOutLayerBackwardSignal(LayerElem *layerElem, int batLen, ObjFunKind objfunKind);
void BackwardPropBatch(ANNSet *annSet, int batLen, Boolean accGrad);
void SetUpdateFlags(ANNSet *annSet);
int GetUpdateIndex(void);
void SetUpdateIndex(int curUpdtIdx);
/* cz277 - 1007 */
int GetBatchIndex(void);
void SetBatchIndex(int curBatIdx);

void RandANNLayer(LELink layerElem, int seed, float scale);
/*LELink GenRandLayer(MemHeap *heap, int nodeNum, int inputDim, int seed);*/
LELink GenNewLayer(MemHeap *heap, int nodeNum, int inputDim);
void SetFeaMixBatchIdxes(ANNSet *annSet, int newIdx);
/* cz277 - max norm2 */
Boolean IsLinearActFun(ActFunKind actfunKind);
Boolean IsNonLinearActFun(ActFunKind actfunKind);


#ifdef __cplusplus
}
#endif

#endif  /* _HANNET_H_ */

/* ------------------------- End of HANNet.h ------------------------- */ 
