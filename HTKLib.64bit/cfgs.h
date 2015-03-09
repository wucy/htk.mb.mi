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
/*      Department of Engineering                              */
/*      University of Cambridge                                */
/*      http://mi.eng.cam.ac.uk/                               */
/*                                                             */
/* ----------------------------------------------------------- */
/*         Copyright:                                          */
/*         2000-2003  Cambridge University                     */
/*                    Engineering Department                   */
/*                                                             */
/*   Use of this software is governed by a License Agreement   */
/*    ** See the file License for the Conditions of Use  **    */
/*    **     This banner notice must not be removed      **    */
/*                                                             */
/* ----------------------------------------------------------- */
/*           File: cfgs.h: Global Configuration Options        */
/* ----------------------------------------------------------- */

#ifndef _CFGS_H_
#define _CFGS_H_

/*#define DOUBLEANN*/

#define CUDA

/*#define MKL*/

#ifdef MKL
#include <mkl.h>
#endif

/*#ifdef CUDA
#include "HCUDA.h"
#endif*/

/*#ifdef MKL
#undef CUDA*/

/*#ifdef CUDA 
#undef MKL
*/

/*#define GRADPROBE*/

#endif

