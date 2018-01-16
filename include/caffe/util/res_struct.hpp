#ifndef CPU_ONLY
#ifdef USE_PROF

#ifndef CAFFE_RES_STRUCT_HPP_
#define CAFFE_RES_STRUCT_HPP_

#include <string>

using std::string;

namespace caffe {
  /**
   * @brief Structure used to record the start and end timestamp of a kernel.
   */
  typedef struct K_Timestamp {
    string name;
    uint64_t start;  /**< The start time of a kernel. */
    uint64_t end;  /**< The end time of a kernel. */
    uint32_t streamId;  /**< ID of the stream running this kernel. */

    struct K_Timestamp& operator=(const struct K_Timestamp& other) {
      if (this != &other) {
        this->start = other.start;
        this->end = other.end;
        this->streamId = other.streamId;
      }

      return *this;
    }

    bool operator<(/*const K_Timestamp& l_val, */const K_Timestamp& r_val) {
      return (start < r_val.start);
    }
  } Timestamp_t;

  /**
   * @brief Structure designed to store kernel execution configuration
   *        and info.
   */
  typedef struct Kernel {
    string name;  /**< Kernel's name in C++ standard. */
    unsigned int gridX, gridY, gridZ;  /**< Grid configuration of a kernel. */
    unsigned int blockX, blockY, blockZ;  /**< Thread block configuration of a kernel. */
    unsigned int smPerBlock;  /**< Amount of shared memory occupied by a block. */
    unsigned int regPerThread;  /**< Amount of registers occupied by a thread. */
    unsigned int invocations;  /**< Total times that this kernel was invoked. */
    uint64_t duration;  /**< The kernel's execution time. */
    uint64_t average_exec_time;
    // Timestamp_t timestamp;  /**< Kernel's timestamp. */

    /**
     * @brief  Constructor.
     */
    Kernel () {
      this->name = "";
      this->gridX = this->gridY = this->gridZ = 0;
      this->blockX = this->blockY = this->blockZ = 0;
      this->smPerBlock = 0;
      this->regPerThread = 0;
      this->invocations = 0;
      this->duration = 0;
      this->average_exec_time = 0;
    }

    /**
     * @brief   == Operator.
     * @param[in] other   Second kernel object.
     * @return  Return true if two object is identical.
     */
    inline bool operator==(const Kernel& other) const {
      if (this->name == other.name &&
          this->gridX == other.gridX &&
          this->gridY == other.gridY &&
          this->gridZ == other.gridZ &&
          this->blockX == other.blockX &&
          this->blockY == other.blockY &&
          this->blockZ == other.blockZ &&
          this->smPerBlock == other.smPerBlock &&
          this->regPerThread == other.regPerThread) {
        return true;
      }

      return false;
    }
  } Kernel_t;

  /**
   * @brief Node structure used to construct a segment tree.
   */
  typedef struct node {
    int left, right; /**< The left and right child of a tree node. */
    int start, end; /**< The start and end value that a tree node represents. */
    int covered; /**< Flag used to show whether the current node is covered. */

    node() {
      this->left = this->right = this->start = this->end = this->covered = 0;
    }

    node(int val) {
      if (val != 0) {
        std::cerr << "WRONG VALUE! " << __FILE__ << "@" << __LINE__ << std::endl;
        exit(EXIT_FAILURE);
      }
      this->left = this->right = this->start = this->end = this->covered = 0;
    }

    struct node& operator=(const struct node& other) {
      this->left = other.left;
      this->right = other.right;
      this->start = other.start;
      this->end = other.end;
      this->covered = other.covered;

      return *this;
    }
  } SegTree_t, *SegTree_ptr;
} /** namespace caffe. **/

#endif    /** CAFFE_RES_STRUCT_HPP_ **/
#endif    /** USE_PROF **/
#endif    /** CPU_ONLY settings **/
