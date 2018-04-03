#ifdef GPU_METRIC

#include "caffe/util/metric_func.hpp"

namespace caffe{
  CUpti_SubscriberHandle subscriber;
  CUpti_MetricID metricId;
  CUpti_EventGroupSets *passData;
  MetricData_t metricData;
  CUcontext context = 0;
  CUdevice device = 0;
  char deviceName[32];
  //const char *metricName;
  CUpti_MetricValue metricValue;
  uint64_t kernelDuration;

  void InitMetric(const string metricName) {
    CHECK_DRIVER_ERROR(cuInit(0));
    CHECK_DRIVER_ERROR(cuDeviceGet(&device, 0));
    CHECK_DRIVER_ERROR(cuDeviceGetName(deviceName, 32, device));
    CHECK_DRIVER_ERROR(cuCtxGetCurrent(&context));

    // setup launch callback for event collection.
    CHECK_CUPTI_ERROR(cuptiSubscribe(&subscriber, (CUpti_CallbackFunc)getMetricValueCallback, &metricData), "cuptiSubscribe");
    CHECK_CUPTI_ERROR(cuptiEnableCallback(1, subscriber, CUPTI_CB_DOMAIN_RUNTIME_API, \
          CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_v3020), "cuptiEnableCallback");
    // allocate space to hold all the events needed for the metric.
    CHECK_CUPTI_ERROR(cuptiMetricGetIdFromName(device, metricName.c_str(), &metricId), "cuptiMetricGetIdFromName");
    CHECK_CUPTI_ERROR(cuptiMetricGetNumEvents(metricId, &metricData.numEvents), "cuptiMetricGetNumEvents");
    metricData.device = device;
    metricData.eventIdArray = reinterpret_cast<CUpti_EventID *>(malloc(metricData.numEvents * sizeof(CUpti_EventID)));
    metricData.eventValueArray = reinterpret_cast<uint64_t *>(malloc(metricData.numEvents * sizeof(uint64_t)));
    metricData.eventIdx = 0;

    CHECK_CUPTI_ERROR(cuptiMetricCreateEventGroupSets(context, sizeof(metricId), &metricId, &passData), "cuptiMetricCreateEventGroupSets");
    for (unsigned int pass = 0; pass < passData->numSets; ++ pass) {
      LOG(INFO) << "Pass " << pass;
      metricData.eventGroups = passData->sets + pass;
    }
  }

  void EndMetric(const string metricName) {
    //if (metricData.eventIdx != metricData.numEvents) {
    //  LOG(INFO) << "Error: expected " << metricData.numEvents << " metric events, got " << metricData.eventIdx;
    //  exit(-1);
    //}

    // use all the collected events to calculate the metric value.
    CHECK_CUPTI_ERROR(cuptiMetricGetValue(device, metricId,
          metricData.numEvents * sizeof(CUpti_EventID), metricData.eventIdArray,
          metricData.numEvents * sizeof(uint64_t), metricData.eventValueArray,
          kernelDuration, &metricValue), "cuptiMetricGetValue");

    CUpti_MetricValueKind valueKind;
    size_t valueKindSize = sizeof(valueKind);
    CHECK_CUPTI_ERROR(cuptiMetricGetAttribute(metricId, CUPTI_METRIC_ATTR_VALUE_KIND,
          &valueKindSize, &valueKind), "cuptiMetricGetAttribute");
    switch(valueKind) {
      case CUPTI_METRIC_VALUE_KIND_DOUBLE:
        LOG(INFO) << "Metric(double) " << metricName << " = " << metricValue.metricValueDouble;
        break;
      case CUPTI_METRIC_VALUE_KIND_UINT64:
        LOG(INFO) << "Metric(uint64) " << metricName << " = " << \
          static_cast<unsigned long long>(metricValue.metricValueUint64);
        break;
      case CUPTI_METRIC_VALUE_KIND_INT64:
        LOG(INFO) << "Metric(int64) " << metricName << " = " << \
          static_cast<long long>(metricValue.metricValueInt64);
        break;
      case CUPTI_METRIC_VALUE_KIND_PERCENT:
        LOG(INFO) << "Metric(percent) " << metricName << " = " << \
          metricValue.metricValuePercent;
        break;
      case CUPTI_METRIC_VALUE_KIND_THROUGHPUT:
        LOG(INFO) << "Metric(throughput) " << metricName << " = " << \
          static_cast<unsigned long long>(metricValue.metricValueThroughput) << " bytes/sec";
        break;
      case CUPTI_METRIC_VALUE_KIND_UTILIZATION_LEVEL:
        LOG(INFO) << "Metric(utilization level) " << metricName << " = utilization level " << \
          static_cast<unsigned int>(metricValue.metricValueUtilizationLevel);
        break;
      default:
        LOG(INFO) << "ERROR: Unknown value kind";
        exit(-1);
    }

    CHECK_CUPTI_ERROR(cuptiUnsubscribe(subscriber), "cuptiUnsubscribe");
  }

  void CUPTIAPI getMetricValueCallback(void *userdata, CUpti_CallbackDomain domain,
      CUpti_CallbackId cbid, const CUpti_CallbackData *cbInfo) {
    MetricData_t *metricData = reinterpret_cast<MetricData_t*>(userdata);
    unsigned int i = 0, j = 0, k = 0;

    // This callback is enabled only for launch so we shouldn't see anything else.
    if (cbid != CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_v3020) {
      std::cerr << __FILE__ << ":" << __LINE__ << ": unexpected cbid " << cbid << std::endl;
      exit(-1);
    }

    // on entry, enable all the event groups being collected this pass,
    // for metrics we collect for all instances of the event.
    if (cbInfo->callbackSite == CUPTI_API_ENTER) {
      cudaDeviceSynchronize();
      CHECK_CUPTI_ERROR(cuptiSetEventCollectionMode(cbInfo->context, CUPTI_EVENT_COLLECTION_MODE_KERNEL), "cuptiSetEventCollectionMode");

      for (i = 0; i < metricData->eventGroups->numEventGroups; ++ i) {
        uint32_t all = 1;
        CHECK_CUPTI_ERROR(cuptiEventGroupSetAttribute(metricData->eventGroups->eventGroups[i],
              CUPTI_EVENT_GROUP_ATTR_PROFILE_ALL_DOMAIN_INSTANCES, sizeof(all), &all), "cuptiEventGroupSetAttribute");
        CHECK_CUPTI_ERROR(cuptiEventGroupEnable(metricData->eventGroups->eventGroups[i]), "cuptiEventGroupEnable");
      }
    }

    // on exit, read and record event values.
    if (cbInfo->callbackSite == CUPTI_API_EXIT) {
      cudaDeviceSynchronize();

      // for each group, read the event values from the group and record
      // in metricData.
      for (i = 0; i < metricData->eventGroups->numEventGroups; ++ i) {
        CUpti_EventGroup group = metricData->eventGroups->eventGroups[i];
        CUpti_EventDomainID groupDomain;
        uint32_t numEvents, numInstances, numTotalInstances;
        CUpti_EventID *eventIds;
        size_t groupDomainSize = sizeof(groupDomain);
        size_t numEventsSize = sizeof(numEvents);
        size_t numInstancesSize = sizeof(numInstances);
        size_t numTotalInstancesSize = sizeof(numTotalInstances);
        uint64_t *values, normalized, sum;
        size_t valuesSize, eventIdsSize;

        CHECK_CUPTI_ERROR(cuptiEventGroupGetAttribute(group,
              CUPTI_EVENT_GROUP_ATTR_EVENT_DOMAIN_ID,
              &groupDomainSize, &groupDomain), "cuptiEventGroupGetAttribute");
        CHECK_CUPTI_ERROR(cuptiDeviceGetEventDomainAttribute(metricData->device, groupDomain,
              CUPTI_EVENT_DOMAIN_ATTR_TOTAL_INSTANCE_COUNT,
              &numTotalInstancesSize, &numTotalInstances), "cuptiDeviceGetEventDomainAttribute");
        CHECK_CUPTI_ERROR(cuptiEventGroupGetAttribute(group,
              CUPTI_EVENT_GROUP_ATTR_INSTANCE_COUNT,
              &numInstancesSize, &numInstances), "cuptiEventGroupGetAttribute");
        CHECK_CUPTI_ERROR(cuptiEventGroupGetAttribute(group,
              CUPTI_EVENT_GROUP_ATTR_NUM_EVENTS,
              &numEventsSize, &numEvents), "cuptiEventGroupGetAttribute");
        eventIdsSize = numEvents * sizeof(CUpti_EventID);
        eventIds = reinterpret_cast<CUpti_EventID *>(malloc(eventIdsSize));
        CHECK_CUPTI_ERROR(cuptiEventGroupGetAttribute(group,
              CUPTI_EVENT_GROUP_ATTR_EVENTS,
              &eventIdsSize, eventIds), "cuptiEventGroupGetAttribute");

        valuesSize = sizeof(uint64_t) * numInstances;
        values = reinterpret_cast<uint64_t *>(malloc(valuesSize));

        for (j = 0; j < numEvents; j ++) {
          CHECK_CUPTI_ERROR(cuptiEventGroupReadEvent(group,
                CUPTI_EVENT_READ_FLAG_NONE,
                eventIds[j], &valuesSize, values), "cuptiEventGroupReadEvent");
          //LOG(INFO) << "J = " << j << ", EventIdx == " << metricData->eventIdx << ", Number of events: " << metricData->numEvents;
          if (metricData->eventIdx >= metricData->numEvents) {
            std::cerr << "Error: too many events collected, metric expects only " << \
              static_cast<int>(metricData->numEvents) << std::endl;
            exit(-1);
          }

          // sum collect event values from all instances.
          sum = 0;
          for (k = 0; k < numInstances; ++ k) {
            sum += values[k];
          }

          // normalize the event value to represent the total number of
          // domain instances on the device.
          normalized = (sum * numTotalInstances) / numInstances;
          metricData->eventIdArray[metricData->eventIdx] = eventIds[j];
          metricData->eventValueArray[metricData->eventIdx] = normalized;
          metricData->eventIdx ++;

          // print collected value.
          char eventName[128];
          size_t eventNameSize = sizeof(eventName) - 1;
          CHECK_CUPTI_ERROR(cuptiEventGetAttribute(eventIds[j],
                CUPTI_EVENT_ATTR_NAME,
                &eventNameSize, eventName), "cuptiEventGetAttribute");
          eventName[127] = '\0';
          stringstream temp_ss;
          temp_ss << eventName << " = " << static_cast<unsigned long long>(sum) << "\n";
          if (numInstances > 1) {
            for (k = 0; k < numInstances; ++ k) {
              if (k != 0) {
                temp_ss << ", ";
              }
              temp_ss << static_cast<unsigned long long>(values[k]);
            }
          }

          temp_ss << ")\n\t" << eventName << " (normalized) (" << static_cast<unsigned long long>(sum) \
            << " * " << numTotalInstances << ") / " << numInstances << " = " << \
            static_cast<unsigned long long>(normalized) << "\n";
          LOG(INFO) << temp_ss.str();
        }
        free(values);
      }
      for (i = 0; i < metricData->eventGroups->numEventGroups; ++ i) {
        CHECK_CUPTI_ERROR(cuptiEventGroupDisable(metricData->eventGroups->eventGroups[i]), "cuptiEventGroupDisable");
      }
    }
    metricData->eventIdx = 0;
  }
}

#endif
