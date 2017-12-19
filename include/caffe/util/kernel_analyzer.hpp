/**
 *       @file  kernel_analyzer.hpp
 *      @brief  Kernel analyzer implementation.
 *
 * The KernelAnalyzer class is used to analyze the kernel records and figure out how many
 * concurrent kernels supported on a single GPU device by utilizing the Hyper-Q tech.
 *
 *     @author  Hao Fu (Hao), haofu@tju.edu.cn
 *
 *     Created  2017-09-12
 *    Compiler  gcc/g++
 *     Company  Tianjin University
 *   Copyright  GNU GPL v3.0
 *
 * This source code is released for free distribution under the terms of the
 * GNU General Public License as published by the Free Software Foundation.
 * =====================================================================================
 */

#ifndef CPU_ONLY
#ifdef USE_PROF

#ifndef CAFFE_ANALYZER_HPP_
#define CAFFE_ANALYZER_HPP_

#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <map>

#include <cmath>

#include <cuda.h>
#include <cuda_runtime.h>

#include "caffe/util/async_tracker.hpp"
#include "caffe/util/gpu_manager.hpp"

// Keywords adopted which are existed in std namespaces.
using std::string;
using std::vector;
using std::map;
using std::ceil;
using std::floor;
using std::sqrt;

namespace caffe {
  enum LABEL {START = 0, END = 1};
  /**
   * @class KernelAnalyzer
   * @brief Class used to analyze kernel runtime configurations and
   *        figure out the maximum kernels that can be launched in
   *        parallel.
   */
  class KernelAnalyzer {
    public:
      /**
       * @brief   KernelAnalyzer deconstructor.
       *
       * Deconstructor used to destroy memory spaces allocated dynamically.
       */
      ~KernelAnalyzer();

      /**
       * @brief   Get a KernelAnalyzer object.
       *
       * Thread local context for KernelAnalyzer.
       */
      static KernelAnalyzer& Get();

      /**
       * @brief Kernel profiler starter.
       *
       * Function used to start the parallel analyzer. If there is a analysis result
       * already, return the parallel_degree value, or start resource tracker to
       * profiling subsequent kernels, and return 1.
       *
       * @param[in]  layer_name        Name of current network layer analyzed.
       * @param[in]  loop_label        Unique label for the current kernel block.
       * @param[out] parallel_degree   Result concurrency supported by block labeled in
       *                               current network layer.
       */
      void AnalyzerStart(const string layer_name, const string loop_label, int& parallel_degree);

      /**
       * @brief Kernel profiler stopper.
       *
       * Function used to stop the parallel analyzer and start the analysis of the current
       * loop recorded.
       */
      void AnalyzerStop();

      /**
       * @brief ParallelDegree analyzer AND getter.
       *
       * Method used to analyze recorded kernels and return the degree of parallelism of current
       * kernel block. The method is based on upper bound of thread blocks on a single
       * multiprocessor.
       *
       * @param[in] t_launch          Time needed to launch a single CUDA kernel.
       * @param[in] kernels           Vector used to store execution configuration of
       *                              recorded kernels.
       *
       * @return Degree of parallelism of the current kernel block.
       */
      int ParallelDegreeUB(uint64_t t_launch, const vector<Kernel_t>* kernels, int device_id);
      int ParallelDegreeLB(uint64_t t_launch, const vector<Kernel_t>* kernels, int device_id);
//      void OptParallelDegree(bool &dop_flag, unsigned int &curr_dop, LABEL label);

      /**
       * @brief Degree of parallelism recorder.
       *
       * Method used to record the degree of parallelism of each execution unit.
       */
      void RecordParallelDegree();
      /**
       * @brief Kernels analyzed recorder.
       *
       * Method used to record the kernels that is adopted to do analysis.
       */
      void RecordKernelsAnalyzed(const vector<Kernel_t>* kernels) const;

      /**
       * @brief  Device setting function.
       *
       * Function used to set the current device that kernels are running on.
       *
       * @param[in] device_id     The device ID that kernels are deployed on right now.
       */
      void SetDevice(int device_id);

    protected:
      string current_key_str_; /**< Key value of the current loop profiled. */
      // Used to manage degrees of parallelism of kernel blocks.
      // Map between a specific loop and the corresponding parallel degree.
      map<string, DopVal_t> pdegree_map_;

      int device_id_; /**< ID of the GPU device that kernels run on. */
      int *k_num_bnd_; /**< Array for recording bounds of degree of parallelism. */
      cudaEvent_t *start_, *end_;

    private:
      // The private constructor to avoid duplicate instantiation.
      KernelAnalyzer();

      DISABLE_COPY_AND_ASSIGN(KernelAnalyzer);
  }; /** class KernelAnalyzer **/
} /** namespace caffe **/

#endif    /** CAFFE_ANALYZER_HPP_ **/
#endif    /** USE_PROF **/
#endif    /** CPU_ONLY definition. **/
