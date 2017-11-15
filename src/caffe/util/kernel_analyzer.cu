#ifndef CPU_ONLY
#ifdef USE_PROF

#include <glpk.h>

#include <boost/thread.hpp>

#include "caffe/util/kernel_analyzer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/info_log.hpp"

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

  __global__ void sync() {}

  string GetCurrentTime() {
    time_t curr_time = time(NULL);
    tm *local_time = localtime(&curr_time);
    stringstream temp_ss;
    temp_ss << local_time->tm_year << "-" << local_time->tm_mon << "-" << local_time->tm_mday;
    string result = temp_ss.str();
    temp_ss.str("");
    temp_ss.clear();

    return result;
  }

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

      Timer analyzer_timer;
      stringstream temp_ss;
      analyzer_timer.Start();

      const uint64_t kernel_launch_overhead = AsyncResTracker::Get().GetKernelLaunchOverhead();
      const vector<Kernel_t> *kernels = &AsyncResTracker::Get().GetKernelsRecorded();

      pdegree_map_[current_key_str_] = ParallelDegree(kernel_launch_overhead, kernels, this->device_id_);

      LOG(INFO) << current_key_str_ << ": " << pdegree_map_[current_key_str_];
      GpuStreamPool::Get().SetPoolSize(pdegree_map_[current_key_str_]);
      AsyncResTracker::Get().ProfilerUnlock();

      LOG(INFO) << "Asynchronous resource tracker stop!";

      double analyzer_overhead = analyzer_timer.MicroSeconds();
      temp_ss << current_key_str_ << "," << analyzer_overhead << "us";
      InfoLog::Get().RecordInfoLog("analyzer_overhead", GetCurrentTime() + "-ANALYZER", temp_ss.str());

      temp_ss.str("");
      temp_ss.clear();
    }
    sync<<<1,1>>>();

    return ;
  }

  void KernelAnalyzer::RecordParallelDegree() {
    stringstream temp_ss;
    for (auto& pd_record : pdegree_map_) {
      temp_ss << pd_record.first << "," << pd_record.second << std::endl;
    }

    InfoLog::Get().RecordInfoLog("", "ParallelDegree_Record", temp_ss.str());
    temp_ss.str("");
    temp_ss.clear();
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

    // Bounds settings.
    // launch_bnd: The maximum number of kernels that can be launched concurrently subject to the execution time.
    // sm_bnd: The maximum number of kernels that can be launched concurrently subject to the shared memory.
    // threads_bnd: The maximum number of kernels that can be launched concurrently subject to the maxThreadsPerMultiProcessor.
    // k_num_bnd: The final upper bounds of the number of concurrent kernels.
    unsigned int launch_bnd = 0, sm_bnd = 0, threads_bnd = 0;
    double k_num_bnd = 0.0;
    double coef_k = 0.0, constant_term = 0.0;
    double blocks_k = 0.0, threads_k = 0.0;
    for (int i = 0; i < kernels->size(); ++ i) {
      launch_bnd = sm_bnd = threads_bnd = 0.0;
      blocks_k = static_cast<double>(kernels->at(i).gridX * kernels->at(i).gridY * kernels->at(i).gridZ);
      threads_k = static_cast<double>(kernels->at(i).blockX * kernels->at(i).blockY * kernels->at(i).blockZ);

      launch_bnd = (kernels->at(i).duration + t_launch - 1) / t_launch;

      if (kernels->at(i).smPerBlock != 0) {
        sm_bnd = (gpu_prop.sharedMemPerMultiprocessor * gpu_prop.multiProcessorCount) / (kernels->at(i).smPerBlock * blocks_k);
        k_num_bnd = MIN(launch_bnd, static_cast<int>(sm_bnd));
      } else {
        k_num_bnd = launch_bnd;
      }

      threads_bnd = (gpu_prop.maxThreadsPerMultiProcessor * gpu_prop.multiProcessorCount) / (threads_k * blocks_k);
      k_num_bnd = MIN(k_num_bnd, static_cast<int>(threads_bnd));

      coef_k = static_cast<double>(blocks_k * threads_k) / gpu_prop.multiProcessorCount;
      constant_term += static_cast<double>(threads_k * (gpu_prop.multiProcessorCount - 1)) / gpu_prop.multiProcessorCount;

      LOG(INFO) << kernels->at(i).name << " ---> " << "theads_k: " << threads_k << ", max_thread: " << gpu_prop.maxThreadsPerMultiProcessor << ", blcoks_k: " << blocks_k << ", k_num_bnd: " << k_num_bnd;
      if (kernels->at(i).smPerBlock != 0) {
        LOG(INFO) << "sm_bnd: " << sm_bnd;
      }

      if (std::ceil(blocks_k / gpu_prop.multiProcessorCount) * threads_k > gpu_prop.maxThreadsPerMultiProcessor) {
        k_num_bnd = 1;
      }

      glp_set_col_name(dop_mip, i + 1, kernels->at(i).name.c_str());
      glp_set_col_bnds(dop_mip, i + 1, GLP_DB, 0.0, k_num_bnd);
      glp_set_col_kind(dop_mip, i + 1, GLP_IV);
      glp_set_obj_coef(dop_mip, i + 1, coef_k);
    }
    // Set the constant part of the objective function to constant_term.
    glp_set_obj_coef(dop_mip, 0, constant_term);
    // End of bounds settings.

    // Constraints to the goal.
    const int total_constraints = 3;
    glp_add_rows(dop_mip, total_constraints);
    if (glp_get_num_rows(dop_mip) == 0 ) {
      LOG(INFO) << "ERROR! There is no kernel recorded.";
    }

    // regs_bias: The bias term of the register constraint formula. DEPRECATED!
    // sm_bias: The bias term of the shared memory constraint formula.
    // thread_bias: The bias term of the thread constraint formula.
    // coef_bias: The bias term used to compute the above three bias terms.
    double sm_bias = 0.0, threads_bias = 0.0;
    double coef_bias = static_cast<double>(gpu_prop.multiProcessorCount - 1) / static_cast<double>(gpu_prop.multiProcessorCount);
    double coef_sm = 0.0, coef_threads = 0.0;
    int total_kernel_kinds = kernels->size();
    int *row_idx = new int[1 + total_constraints * total_kernel_kinds],
        *col_idx = new int[1 + total_constraints * total_kernel_kinds];
    double *coef_k_arr = new double[1 + total_constraints * total_kernel_kinds];
    for (int i = 0; i < total_kernel_kinds; ++ i) {
      blocks_k = static_cast<double>(kernels->at(i).gridX * kernels->at(i).gridY * kernels->at(i).gridZ);
      threads_k = static_cast<double>(kernels->at(i).blockX * kernels->at(i).blockY * kernels->at(i).blockZ);

      sm_bias += static_cast<double>(kernels->at(i).smPerBlock) * coef_bias;
      threads_bias += static_cast<double>(threads_k) * coef_bias;

      coef_sm = static_cast<double>(kernels->at(i).smPerBlock) * blocks_k /
                  static_cast<double>(gpu_prop.multiProcessorCount);
      coef_threads = static_cast<double>(threads_k * blocks_k) / static_cast<double>(gpu_prop.multiProcessorCount);

      row_idx[i + 1] = static_cast<int>(ceil((i + 1) / static_cast<double>(total_kernel_kinds)));
      col_idx[i + 1] = i + 1;
      coef_k_arr[i + 1] = coef_sm;
      row_idx[i + 1 + total_kernel_kinds * 1] = 1 + static_cast<int>(ceil((i + 1) / static_cast<double>(total_kernel_kinds)));
      col_idx[i + 1 + total_kernel_kinds * 1] = i + 1;
      coef_k_arr[i + 1 + total_kernel_kinds * 1] = coef_threads;
      row_idx[i + 1 + total_kernel_kinds * 2] = 2 + static_cast<int>(ceil((i + 1) / static_cast<double>(total_kernel_kinds)));
      col_idx[i + 1 + total_kernel_kinds * 2] = i + 1;
      coef_k_arr[i + 1 + total_kernel_kinds * 2] = 1;
    }
    glp_set_row_name(dop_mip, 1, "SMs");
    glp_set_row_bnds(dop_mip, 1, GLP_DB, 0.0, static_cast<double>(gpu_prop.sharedMemPerMultiprocessor - sm_bias));
    glp_set_row_name(dop_mip, 2, "Threads");
    glp_set_row_bnds(dop_mip, 2, GLP_DB, 0.0, static_cast<double>(gpu_prop.maxThreadsPerMultiProcessor - threads_bias));
    glp_set_row_name(dop_mip, 3, "Concurrency");
    glp_set_row_bnds(dop_mip, 3, GLP_DB, 1.0, static_cast<double>(GpuStreamPool::Get().GetMaxNumOfStreams()));

    glp_load_matrix(dop_mip, total_kernel_kinds * total_constraints, row_idx, col_idx, coef_k_arr);
    // End of constraints settings and the initialization of MIP parameter matrix.

    glp_iocp dop_param;
    glp_init_iocp(&dop_param);
    dop_param.presolve = GLP_ON;
    CHECK_GLP_ERROR(glp_intopt(dop_mip, &dop_param), "glp_intopt");

    stringstream temp_ss;
    int max_degree_of_parallelism = 0;
    double obj_val = glp_mip_obj_val(dop_mip);
    LOG(INFO) << "OBJECTIVE VALUE: " << obj_val;
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

    if (max_degree_of_parallelism == 0) {
      LOG(INFO) << "CANNOT LAUNCH KERNELS CONCURRENTLY!";
      max_degree_of_parallelism = 1;
    }

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
