/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * npu_ze.h - the part of Level Zero that npu.c calls, declared here so that
 * Janas builds without Level Zero's headers and loads the loader at run time.
 *
 * Layouts and values as in the oneAPI Level Zero specification (core API)
 * and Intel's NPU graph extension ("ZE_extension_graph", 1.5 and later):
 * only the fields and the table entries npu.c uses, the others as padding
 * of the same size. Private to npu.c.
 */
#ifndef JANAS_COMMON_NPU_ZE_H
#define JANAS_COMMON_NPU_ZE_H

#include <stddef.h>
#include <stdint.h>

typedef int32_t ze_result_t; /* 0 success */
typedef struct ze_driver *ze_driver_handle_t;
typedef struct ze_device *ze_device_handle_t;
typedef struct ze_context *ze_context_handle_t;
typedef struct ze_queue *ze_command_queue_handle_t;
typedef struct ze_list *ze_command_list_handle_t;
typedef struct ze_fence *ze_fence_handle_t;
typedef struct ze_graph *ze_graph_handle_t;

#define ZE_INIT_FLAG_VPU_ONLY 2u
#define ZE_DEVICE_TYPE_VPU 5

/* structure types of the core API */
#define ZE_STYPE_DRIVER_PROPERTIES 0x1
#define ZE_STYPE_DEVICE_PROPERTIES 0x3
#define ZE_STYPE_CONTEXT_DESC 0xd
#define ZE_STYPE_COMMAND_QUEUE_DESC 0xe
#define ZE_STYPE_COMMAND_LIST_DESC 0xf
#define ZE_STYPE_FENCE_DESC 0x12
#define ZE_STYPE_HOST_MEM_ALLOC_DESC 0x16

/* the graph extension's */
#define ZE_GSTYPE_DEVICE_GRAPH_PROPERTIES 0x1
#define ZE_GSTYPE_GRAPH_PROPERTIES 0x3
#define ZE_GSTYPE_GRAPH_ARGUMENT_PROPERTIES 0x4
#define ZE_GSTYPE_GRAPH_DESC_2 0xe
#define ZE_GRAPH_FORMAT_NATIVE 0x1
#define ZE_GRAPH_FORMAT_NGRAPH_LITE 0x2
#define ZE_GRAPH_FLAG_DISABLE_CACHING 1u
#define ZE_GRAPH_ARGUMENT_TYPE_INPUT 0
#define ZE_GRAPH_EXT_NAME "ZE_extension_graph"

typedef struct {
    int32_t stype;
    void *pNext;
    int32_t type;
    uint32_t vendorId, deviceId, flags, subdeviceId, coreClockRate;
    uint64_t maxMemAllocSize;
    uint32_t maxHardwareContexts, maxCommandQueuePriority, numThreadsPerEU,
        physicalEUSimdWidth, numEUsPerSubslice, numSubslicesPerSlice, numSlices;
    uint64_t timerResolution;
    uint32_t timestampValidBits, kernelTimestampValidBits;
    uint8_t uuid[16];
    char name[256];
} ze_device_properties_t;

typedef struct {
    int32_t stype;
    void *pNext;
    uint8_t uuid[16];
    uint32_t driverVersion;
} ze_driver_properties_t;

typedef struct {
    int32_t stype;
    const void *pNext;
    uint32_t flags;
} ze_simple_desc_t; /* context, fence and host memory descriptors */

typedef struct {
    int32_t stype;
    const void *pNext;
    uint32_t ordinal, index, flags;
    int32_t mode, priority;
} ze_command_queue_desc_t;

typedef struct {
    int32_t stype;
    const void *pNext;
    uint32_t commandQueueGroupOrdinal, flags;
} ze_command_list_desc_t;

/* the NPU's queue extension ("ZE_extension_command_queue_npu" 1.0): turbo,
   chained to a queue's descriptor */
#define ZE_NSTYPE_COMMAND_QUEUE_DESC_NPU_EXT 0x1
typedef struct {
    int32_t stype;
    const void *pNext;
    _Bool turbo;
} ze_command_queue_desc_npu_ext_t;

typedef struct {
    int32_t stype;
    void *pNext;
    uint32_t graphExtensionVersion;
    uint16_t compilerMajor, compilerMinor;
    int32_t graphFormatsSupported;
    uint32_t maxOVOpsetVersionSupported;
} ze_device_graph_properties_t;

typedef struct {
    int32_t stype;
    const void *pNext;
    int32_t format;
    size_t inputSize;
    const uint8_t *pInput;
    const char *pBuildFlags;
    uint32_t flags;
} ze_graph_desc_2_t;

typedef struct {
    int32_t stype;
    void *pNext;
    uint32_t numGraphArgs;
} ze_graph_properties_t;

typedef struct {
    int32_t stype;
    void *pNext;
    char name[256];
    int32_t type;
    uint32_t dims[5];
    int32_t networkPrecision, networkLayout, devicePrecision, deviceLayout;
} ze_graph_argument_properties_t;

/* the extension's function table, as far as pfnCreate2 (version 1.5) */
typedef struct {
    void *pfnCreate;
    ze_result_t (*pfnDestroy)(ze_graph_handle_t);
    ze_result_t (*pfnGetProperties)(ze_graph_handle_t, ze_graph_properties_t *);
    ze_result_t (*pfnGetArgumentProperties)(ze_graph_handle_t, uint32_t,
                                            ze_graph_argument_properties_t *);
    ze_result_t (*pfnSetArgumentValue)(ze_graph_handle_t, uint32_t,
                                       const void *);
    ze_result_t (*pfnAppendGraphInitialize)(ze_command_list_handle_t,
                                            ze_graph_handle_t, void *, uint32_t,
                                            void *);
    ze_result_t (*pfnAppendGraphExecute)(ze_command_list_handle_t,
                                         ze_graph_handle_t, void *, void *,
                                         uint32_t, void *);
    ze_result_t (*pfnGetNativeBinary)(ze_graph_handle_t, size_t *, uint8_t *);
    ze_result_t (*pfnDeviceGetGraphProperties)(ze_device_handle_t,
                                               ze_device_graph_properties_t *);
    void *pfnGraphGetArgumentMetadata, *pfnGetArgumentProperties2,
        *pfnGetArgumentProperties3, *pfnQueryNetworkCreate,
        *pfnQueryNetworkDestroy, *pfnQueryNetworkGetSupportedLayers;
    ze_result_t (*pfnBuildLogGetString)(ze_graph_handle_t, uint32_t *, char *);
    ze_result_t (*pfnCreate2)(ze_context_handle_t, ze_device_handle_t,
                              const ze_graph_desc_2_t *, ze_graph_handle_t *);
} ze_graph_dditable_ext_t;

#endif
