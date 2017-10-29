#ifndef CPU_ONLY
#ifdef USE_PROF

#include <glpk.h>

#include <boost/thread.hpp>

#include "caffe/util/kernel_analyzer.hpp"

#define MIN(a, b) ((std::ceil(a) < std::ceil(b)) ? std::ceil(a) : std::ceil(b))

#define CHECK_GLP_ERROR(val, func) {                                                                      \
  if (val == GLP_EBOUND) {                                                                                \
    LOG(FATAL) << "GLP_EBOUND! Unable to start the search, because some double-bounded variables have " <<\
                  "incorrent bouds or some integer variables have non-integer bounds! @" << func;         \
  } else if (val == GLP_EROOT) {                                                                          \
    LOG(FATAL) << "GLP_EROOT! Unable to start the search, because optimal basis for initial LP " <<       \
                  "relaxation is not provided. @" << func;                                                \
    LOG(FATAL) << "This code may appear only if the presolver is disabled.";                              \
  } else if (val == GLP_ENOPFS) {                                                                         \
    LOG(FATAL) << "GLP_ENPFS! Unable to start the search, because LP relaxation of the MIP problem " <<   \
                  "instance has no primal feasible solution. @" << func;                                  \
    LOG(FATAL) << "This code may appear only if the presolver is enabled.";                               \
  } else if (val == GLP_ENODFS) {                                                                         \
    LOG(FATAL) << "GLP_ENODFS! Unable to start the search, because LP relaxation of the MIP problem " <<  \
                  "instance has no dual feasible solution. @" << func;                                    \
    LOG(FATAL) << "This code may appear only if the presolver is enabled.";                               \
  } else if (val == GLP_EFAIL) {                                                                          \
    LOG(FATAL) << "GLP_EFAIL! The search was prematurely terminated due to the solver failure. @" << func;\
  } else if (val == GLP_EMIPGAP) {                                                                        \
    LOG(FATAL) << "GLP_EMIPGAP! The search was prematurely terminated, because the relative mip gap " <<  \
                  "tolerance has been reached. @" << func;                                                \
  } else if (val == GLP_ETMLIM) {                                                                         \
    LOG(FATAL) << "GLP_ETMLIM! The search was prematurely terminated, because the time limit has been " <<\
                  "exceeded. @" << func;                                                                  \
  } else if (val == GLP_ESTOP) {                                                                          \
    LOG(FATAL) << "GLP_ESTOP! The search was prematurely terminated by application. @" << func;           \
    LOG(FATAL) << "This code may appear only if the advanced solver interface is used.";                  \
  }                                                                                                       \
}

namespace caffe {
  static boost::thread_specific_ptr<KernelAnalyzer> thread_kernel_analyzer_;

  KernelAnalyzer& KernelAnalyzer::Get() {
    // If thread_kAnalyzer_instance_ has not been initialized,
    // then initialize it.
    if (!thread_kernel_analyzer_.get()) {
      thread_kernel_analyzer_.reset(new KernelAnalyzer());
    }

    return *(thread_kernel_analyzer_.get());
  }

  KernelAnalyzer::KernelAnalyzer() {
    this->device_id_ = -1;

    if (!pdegree_map_.empty()) {
      pdegree_map_.clear();
    }
  }

  KernelAnalyzer::~KernelAnalyzer() {
    if (!pdegree_map_.empty()) {
      pdegree_map_.clear();
    }
  }

  void KernelAnalyzer::AnalyzerStart(const string layer_name, const string loop_label, int &parallel_degree) {
    if (this->device_id_ == -1) {
      CHECK_CUDA_ERROR(cudaGetDevice(&this->device_id_), "cudaGetDevice");
      LOG(INFO) << "Current DEVICE@" << this->device_id_ << ".";
    }

    stringstream temp_ss;
    temp_ss << layer_name << "_" << loop_label << "_" << this->device_id_;
    current_key_str_ = temp_ss.str();

    if (pdegree_map_.find(current_key_str_) == pdegree_map_.end()) {
      // If there is only one resource tracker among all threads, mutex can be added in
      // this place.
      AsyncResTracker::Get().ProfilerLock();
      AsyncResTracker::Get().ProfilerStart(this->device_id_);
      parallel_degree = 0;
    } else {
      parallel_degree = pdegree_map_[current_key_str_];
    }

    temp_ss.str("");
    temp_ss.clear();

    return ;
  }

  void KernelAnalyzer::AnalyzerStop() {
    if (pdegree_map_.find(current_key_str_) == pdegree_map_.end()) {
      AsyncResTracker::Get().ProfilerStop(this->device_id_);

      const uint64_t kernel_launch_overhead = AsyncResTracker::Get().GetKernelLaunchOverhead();
      const vector<Kernel_t> *kernels = &AsyncResTracker::Get().GetKernelsRecorded();

      pdegree_map_[current_key_str_] = ParallelDegree(kernel_launch_overhead, kernels, this->device_id_);

      LOG(INFO) << current_key_str_ << ": " << pdegree_map_[current_key_str_];
      AsyncResTracker::Get().ProfilerUnlock();
      GpuStreamPool::Get().SetPoolSize(pdegree_map_[current_key_str_]);
    }

    return ;
  }

  int KernelAnalyzer::ParallelDegree(uint64_t t_launch, const vector<Kernel_t>* kernels, int device_id) {
    cudaDeviceProp gpu_prop;
    CHECK_CUDA_ERROR(cudaGetDeviceProperties(&gpu_prop, this->device_id_), "cudaGetDeviceProperties");

    glp_prob *dop_mip = glp_create_prob();
    glp_set_prob_name(dop_mip, "DegreeOfParallelismSolver");
    glp_set_obj_dir(dop_mip, GLP_MAX);

    if (kernels == NULL || kernels->size() == 0) {
      LOG(FATAL) << "There is no kernels recorded!";
    }

    glp_add_cols(dop_mip, kernels->size());
    if (glp_get_num_cols(dop_mip) == 0 ) {
      LOG(INFO) << "ERROR! There is no kernel recorded.";
    }

    stringstream temp_ss;
    // Bounds settings.
    double launch_bnd = 0.0, sm_bnd = 0.0, regs_bnd = 0.0, threads_bnd = 0.0;
    double k_num_bnd = 0.0;
    double coef_a = 0.0, coef_b = 0.0, coef_c = 0.0;
    double coef_k = 0.0, constant_term = 0.0;
    double blocks_k = 0.0, threads_k = 0.0;
    for (int i = 0; i < kernels->size(); ++ i) {
      blocks_k = static_cast<double>(kernels->at(i).gridX * kernels->at(i).gridY * kernels->at(i).gridZ);
      threads_k = static_cast<double>(kernels->at(i).blockX * kernels->at(i).blockY * kernels->at(i).blockZ);

      launch_bnd = static_cast<double>((kernels->at(i).duration + t_launch - 1) / t_launch);

      coef_a = blocks_k;
      coef_b = static_cast<double>(gpu_prop.multiProcessorCount - 1);
      if (kernels->at(i).smPerBlock != 0) {
        coef_c = static_cast<double>(-1 * (gpu_prop.sharedMemPerMultiprocessor * gpu_prop.multiProcessorCount)) / static_cast<double>(kernels->at(i).smPerBlock);
        sm_bnd = (-1 * coef_b + sqrt(coef_b * coef_b - 4 * coef_a * coef_c)) / (2 * coef_a);
        k_num_bnd = MIN(launch_bnd, sm_bnd);
      } else {
        k_num_bnd = launch_bnd;
      }

      coef_c = static_cast<double>(-1 * gpu_prop.regsPerMultiprocessor * gpu_prop.multiProcessorCount) / static_cast<double>(kernels->at(i).regPerThread * threads_k);
      regs_bnd = (-1 * coef_b + sqrt(coef_b * coef_b - 4 * coef_a * coef_c)) / (2 * coef_a);
      k_num_bnd = MIN(k_num_bnd, regs_bnd);

      coef_c = static_cast<double>(-1 * gpu_prop.maxThreadsPerMultiProcessor * gpu_prop.multiProcessorCount) / static_cast<double>(threads_k);
      threads_bnd = (-1 * coef_b + sqrt(coef_b * coef_b - 4 * coef_a * coef_c)) / (2 * coef_a);
      k_num_bnd = MIN(k_num_bnd, threads_bnd);

      coef_k = static_cast<double>(blocks_k * threads_k);
      constant_term += static_cast<double>(threads_k * (gpu_prop.multiProcessorCount - 1)) / static_cast<double>(gpu_prop.multiProcessorCount);

      temp_ss << kernels->at(i).name << ": " << k_num_bnd << "; ";

      glp_set_col_name(dop_mip, i + 1, kernels->at(i).name.c_str());
      glp_set_col_bnds(dop_mip, i + 1, GLP_DB, 0.0, k_num_bnd);
      glp_set_obj_coef(dop_mip, i + 1, coef_k);
    }
    glp_set_obj_coef(dop_mip, 0, constant_term);
    // End of bounds settings.

    temp_ss.str("");
    temp_ss.clear();

    // Constraints to the goal.
    glp_add_rows(dop_mip, 4);
    if (glp_get_num_rows(dop_mip) == 0 ) {
      LOG(INFO) << "ERROR! There is no kernel recorded.";
    }

    double regs_bias = 0.0, sm_bias = 0.0, threads_bias = 0.0;
    double coef_bias = static_cast<double>(gpu_prop.multiProcessorCount - 1) / static_cast<double>(gpu_prop.multiProcessorCount);
    double coef_regs = 0.0, coef_sm = 0.0, coef_threads = 0.0;
    int total_kernel_kinds = kernels->size();
    int *row_idx = new int[1 + 4 * total_kernel_kinds],
        *col_idx = new int[1 + 4 * total_kernel_kinds];
    double *coef_k_arr = new double[1 + 4 * total_kernel_kinds];
    for (int i = 0; i < total_kernel_kinds; ++ i) {
      blocks_k = static_cast<double>(kernels->at(i).gridX * kernels->at(i).gridY * kernels->at(i).gridZ);
      threads_k = static_cast<double>(kernels->at(i).blockX * kernels->at(i).blockY * kernels->at(i).blockZ);

      regs_bias += static_cast<double>(kernels->at(i).regPerThread) * threads_k * coef_bias;
      sm_bias += static_cast<double>(kernels->at(i).smPerBlock) * coef_bias;
      threads_bias += threads_k * coef_bias;

      coef_regs = static_cast<double>(kernels->at(i).regPerThread) * threads_k * blocks_k
                  / static_cast<double>(gpu_prop.multiProcessorCount);
      coef_sm = static_cast<double>(kernels->at(i).smPerBlock) * blocks_k /
                  static_cast<double>(gpu_prop.multiProcessorCount);
      coef_threads = threads_k * blocks_k / static_cast<double>(gpu_prop.multiProcessorCount);

      row_idx[i + 1] = static_cast<int>(ceil((i + 1) / static_cast<double>(total_kernel_kinds)));
      col_idx[i + 1] = i + 1;
      coef_k_arr[i + 1] = coef_regs;
      row_idx[i + 1 + total_kernel_kinds] = static_cast<int>(ceil((i + 1 + total_kernel_kinds) / static_cast<double>(total_kernel_kinds)));
      col_idx[i + 1 + total_kernel_kinds] = i + 1;
      coef_k_arr[i + 1 + total_kernel_kinds] = coef_sm;
      row_idx[i + 1 + total_kernel_kinds * 2] = static_cast<int>(ceil((i + 1 + 2 * total_kernel_kinds) / static_cast<double>(total_kernel_kinds)));
      col_idx[i + 1 + total_kernel_kinds * 2] = i + 1;
      coef_k_arr[i + 1 + total_kernel_kinds * 2] = coef_threads;
      row_idx[i + 1 + total_kernel_kinds * 3] = static_cast<int>(ceil((i + 1 + 3 * total_kernel_kinds) / static_cast<double>(total_kernel_kinds)));
      col_idx[i + 1 + total_kernel_kinds * 3] = i + 1;
      coef_k_arr[i + 1 + total_kernel_kinds * 3] = 1;
    }
    glp_set_row_name(dop_mip, 1, "Regs");
    glp_set_row_bnds(dop_mip, 1, GLP_UP, 0.0, static_cast<double>(gpu_prop.regsPerMultiprocessor - regs_bias));
    glp_set_row_name(dop_mip, 2, "SMs");
    glp_set_row_bnds(dop_mip, 2, GLP_UP, 0.0, static_cast<double>(gpu_prop.sharedMemPerMultiprocessor - sm_bias));
    glp_set_row_name(dop_mip, 3, "Threads");
    glp_set_row_bnds(dop_mip, 3, GLP_UP, 0.0, static_cast<double>(gpu_prop.maxThreadsPerMultiProcessor - threads_bias));
    glp_set_row_name(dop_mip, 4, "Concurrency");
    glp_set_row_bnds(dop_mip, 4, GLP_UP, 0.0, static_cast<double>(GpuStreamPool::Get().GetMaxNumOfStreams()));

    glp_load_matrix(dop_mip, total_kernel_kinds * 4, row_idx, col_idx, coef_k_arr);
    // End of constraints settings and the initialization of MIP parameter matrix.

    glp_iocp dop_param;
    glp_init_iocp(&dop_param);
    dop_param.presolve = GLP_ON;
    CHECK_GLP_ERROR(glp_intopt(dop_mip, &dop_param), "glp_intopt");

    int max_degree_of_parallelism = 0;
    // double obj_val = glp_mip_obj_val(dop_mip);
    // LOG(INFO) << "OBJECTIVE VALUE: " << obj_val;
    int *obj_k_val = new int[total_kernel_kinds];
    for (int i = 0; i < total_kernel_kinds; ++ i) {
      obj_k_val[i] = glp_mip_col_val(dop_mip, i + 1);
      max_degree_of_parallelism += obj_k_val[i];
      temp_ss << "[" << kernels->at(i).name << " = " << obj_k_val[i];
      if (i != (total_kernel_kinds - 1)) {
        temp_ss << ", ";
      } else {
        temp_ss << "]; ";
      }
    }
    LOG(INFO) << "Kernel concurrency settings: " << temp_ss.str();
    temp_ss.str("");
    temp_ss.clear();

    delete[] obj_k_val;
    delete[] row_idx;
    delete[] col_idx;
    delete[] coef_k_arr;

    return max_degree_of_parallelism;
  }

  void KernelAnalyzer::SetDevice(int device_id) {
    if (this->device_id_ != device_id) {
      this->device_id_ = device_id;

      if (!pdegree_map_.empty()) {
        pdegree_map_.clear();
      }
    }

    return ;
  }
} /** namespace caffe **/

#endif /** USE_PROF **/
#endif /** CPU_ONLY settings. **/
