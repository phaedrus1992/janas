/* SPDX-License-Identifier: GPL-3.0-or-later
   Copyright (C) 2026 Maurizio Cammalleri */
/*
 * vkcheck - whether this machine's GPU can run Janas-LLM's GPU backend, and
 * how. Every requirement with its answer, the decision the engine would
 * take (as src/llm/gpu.c takes it), then three tests on that device:
 *
 *   zero-copy  memory of this program imported by the GPU
 *              (VK_EXT_external_memory_host), read and written by a shader,
 *              the CPU seeing the result with no copy: an integrated GPU -
 *              Apple's under MoltenVK too - is taken only so;
 *   dot        the integer dot product of four signed bytes against the
 *              CPU's, and a rough rate: Janas-LLM's quantized products are
 *              made of it, and a driver may emulate it;
 *   float      fma() rounded once, a precise a * b + c twice, a precise sum
 *              in the order written: what keeps the GPU's results the CPU's
 *              to the bit, and what a fast-math compiler breaks; whether
 *              subnormals survive (for information).
 *
 * A single file, which any C compiler builds where the Vulkan headers are,
 * so that a GPU can be checked where Janas does not build yet (macOS with
 * MoltenVK):
 *   cc -O2 tools/vkcheck.c -o vkcheck -ldl -lm  (Linux)
 *   cc -O2 -I/opt/homebrew/include tools/vkcheck.c -o vkcheck   (macOS)
 * The Vulkan library is loaded at run time: the loader, or on macOS MoltenVK
 * itself; vkcheck <library> names another. JANAS_GPU_DEVICE and JANAS_GPU_SG
 * act as on the engine. The shaders are the .comp files in vkcheck/, as SPIR-V
 * in vkcheck/spv.h (vkcheck/gen_spv.sh). Exit 0 when the engine would take a
 * GPU and every test passes.
 */
#if defined(__has_include)
#if __has_include(<vulkan/vulkan.h>)
#define VKCHECK_VULKAN 1
#endif
#endif

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifndef VKCHECK_VULKAN
int main(void)
{
    printf("vkcheck: built without the Vulkan headers\n");
    return 2;
}
#else

#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#include "vkcheck/spv.h"

#define INSTANCE_FNS(X)                                                        \
    X(vkDestroyInstance)                                                       \
    X(vkEnumeratePhysicalDevices)                                              \
    X(vkGetPhysicalDeviceProperties)                                           \
    X(vkGetPhysicalDeviceProperties2)                                          \
    X(vkGetPhysicalDeviceFeatures2)                                            \
    X(vkGetPhysicalDeviceMemoryProperties)                                     \
    X(vkGetPhysicalDeviceQueueFamilyProperties)                                \
    X(vkEnumerateDeviceExtensionProperties)                                    \
    X(vkCreateDevice)                                                          \
    X(vkGetDeviceProcAddr)
#define DEVICE_FNS(X)                                                          \
    X(vkDestroyDevice)                                                         \
    X(vkGetDeviceQueue)                                                        \
    X(vkCreateBuffer)                                                          \
    X(vkDestroyBuffer)                                                         \
    X(vkGetBufferMemoryRequirements)                                           \
    X(vkAllocateMemory)                                                        \
    X(vkFreeMemory)                                                            \
    X(vkBindBufferMemory)                                                      \
    X(vkMapMemory)                                                             \
    X(vkGetBufferDeviceAddress)                                                \
    X(vkCreateShaderModule)                                                    \
    X(vkDestroyShaderModule)                                                   \
    X(vkCreatePipelineLayout)                                                  \
    X(vkDestroyPipelineLayout)                                                 \
    X(vkCreateComputePipelines)                                                \
    X(vkDestroyPipeline)                                                       \
    X(vkCreateCommandPool)                                                     \
    X(vkDestroyCommandPool)                                                    \
    X(vkAllocateCommandBuffers)                                                \
    X(vkBeginCommandBuffer)                                                    \
    X(vkEndCommandBuffer)                                                      \
    X(vkCmdBindPipeline)                                                       \
    X(vkCmdPushConstants)                                                      \
    X(vkCmdDispatch)                                                           \
    X(vkCmdPipelineBarrier)                                                    \
    X(vkCreateFence)                                                           \
    X(vkDestroyFence)                                                          \
    X(vkResetFences)                                                           \
    X(vkQueueSubmit)                                                           \
    X(vkWaitForFences)

#define DECLARE(f) static PFN_##f f;
INSTANCE_FNS(DECLARE)
DEVICE_FNS(DECLARE)
static PFN_vkGetInstanceProcAddr gipa;
static PFN_vkGetMemoryHostPointerPropertiesEXT get_host_pointer_props;

static int load(const char *name)
{
#ifdef _WIN32
    HMODULE h = LoadLibraryA(name);
    if (h)
        gipa = (PFN_vkGetInstanceProcAddr)(void (*)(void))GetProcAddress(
            h, "vkGetInstanceProcAddr");
#else
    void *h = dlopen(name, RTLD_NOW | RTLD_LOCAL);
    if (h)
        gipa = (PFN_vkGetInstanceProcAddr)dlsym(h, "vkGetInstanceProcAddr");
#endif
    return gipa != NULL;
}

static double now(void)
{
#ifdef _WIN32
    LARGE_INTEGER c, f;
    QueryPerformanceCounter(&c);
    QueryPerformanceFrequency(&f);
    return (double)c.QuadPart / (double)f.QuadPart;
#else
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + 1e-9 * (double)t.tv_nsec;
#endif
}

static int has_ext(const VkExtensionProperties *e, uint32_t n, const char *name)
{
    for (uint32_t i = 0; i < n; i++)
        if (!strcmp(e[i].extensionName, name))
            return 1;
    return 0;
}

static const char *yes(int b)
{
    return b ? "yes" : "NO";
}

static const char *type_name(VkPhysicalDeviceType t)
{
    return t == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU     ? "discrete"
           : t == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU ? "integrated"
           : t == VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU    ? "virtual"
           : t == VK_PHYSICAL_DEVICE_TYPE_CPU            ? "CPU"
                                                         : "other";
}

/* What a device offers and what the engine would do with it. */
struct dev {
    VkPhysicalDevice pd;
    VkPhysicalDeviceProperties p;
    int bda, idp, ballot, arith, clustered, shuffle, compute_subgroups;
    int control, host_import, portability;
    uint32_t sg, sg_min, sg_max, sg_fixed;
    VkDeviceSize import_align;
    uint32_t family;
    int taken;
    char why[160];
};

/* As src/llm/gpu.c decides; *d filled and printed. */
static void examine(VkPhysicalDevice pd, int index, struct dev *d)
{
    memset(d, 0, sizeof(*d));
    d->pd = pd;
    vkGetPhysicalDeviceProperties(pd, &d->p);
    uint32_t v = d->p.apiVersion;
    printf("GPU %d: %s (%s), Vulkan %u.%u.%u\n", index, d->p.deviceName,
           type_name(d->p.deviceType), VK_API_VERSION_MAJOR(v),
           VK_API_VERSION_MINOR(v), VK_API_VERSION_PATCH(v));
    int discrete = d->p.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU;
    if (!discrete &&
        d->p.deviceType != VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU) {
        snprintf(d->why, sizeof(d->why), "neither discrete nor integrated");
        printf("  Janas-LLM would not take it: %s\n", d->why);
        return;
    }
    printf("  Vulkan 1.3 or later: %s\n", yes(v >= VK_API_VERSION_1_3));
    if (v < VK_API_VERSION_1_3) {
        snprintf(d->why, sizeof(d->why), "its driver offers Vulkan below 1.3");
        printf("  Janas-LLM would not take it: %s\n", d->why);
        return;
    }
    uint32_t ne = 0;
    vkEnumerateDeviceExtensionProperties(pd, NULL, &ne, NULL);
    VkExtensionProperties *ext = calloc(ne ? ne : 1, sizeof(*ext));
    if (!ext)
        exit(1);
    vkEnumerateDeviceExtensionProperties(pd, NULL, &ne, ext);
    d->host_import =
        has_ext(ext, ne, VK_EXT_EXTERNAL_MEMORY_HOST_EXTENSION_NAME);
    d->portability = has_ext(ext, ne, "VK_KHR_portability_subset");
    free(ext);

    VkPhysicalDeviceExternalMemoryHostPropertiesEXT hp = {
        .sType =
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_MEMORY_HOST_PROPERTIES_EXT};
    VkPhysicalDeviceVulkan13Properties p13 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_PROPERTIES,
        .pNext = d->host_import ? &hp : NULL};
    VkPhysicalDeviceVulkan12Properties p12 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_PROPERTIES,
        .pNext = &p13};
    VkPhysicalDeviceVulkan11Properties p11 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_PROPERTIES,
        .pNext = &p12};
    VkPhysicalDeviceProperties2 pp2 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, .pNext = &p11};
    vkGetPhysicalDeviceProperties2(pd, &pp2);
    VkPhysicalDeviceVulkan13Features f13 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    VkPhysicalDeviceVulkan12Features f12 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
        .pNext = &f13};
    VkPhysicalDeviceFeatures2 f2 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, .pNext = &f12};
    vkGetPhysicalDeviceFeatures2(pd, &f2);
    printf("  driver: %s %s\n", p12.driverName, p12.driverInfo);

    d->bda = f12.bufferDeviceAddress;
    d->idp = f13.shaderIntegerDotProduct;
    d->compute_subgroups =
        (p11.subgroupSupportedStages & VK_SHADER_STAGE_COMPUTE_BIT) != 0;
    VkSubgroupFeatureFlags ops =
        d->compute_subgroups ? p11.subgroupSupportedOperations : 0;
    d->ballot = (ops & VK_SUBGROUP_FEATURE_BALLOT_BIT) != 0;
    d->arith = (ops & VK_SUBGROUP_FEATURE_ARITHMETIC_BIT) != 0;
    d->clustered = (ops & VK_SUBGROUP_FEATURE_CLUSTERED_BIT) != 0;
    d->shuffle = (ops & VK_SUBGROUP_FEATURE_SHUFFLE_BIT) != 0;
    d->control =
        f13.subgroupSizeControl && f13.computeFullSubgroups &&
        (p13.requiredSubgroupSizeStages & VK_SHADER_STAGE_COMPUTE_BIT) != 0;
    d->sg_min = p13.minSubgroupSize;
    d->sg_max = p13.maxSubgroupSize;
    d->sg_fixed = p11.subgroupSize;
    d->import_align = d->host_import ? hp.minImportedHostPointerAlignment : 0;

    printf("  buffer device addresses: %s\n", yes(d->bda));
    printf("  integer dot product: %s\n", yes(d->idp));
    printf("  subgroup operations in compute shaders: basic %s, ballot %s, "
           "arithmetic %s, clustered %s, shuffle %s\n",
           yes(d->compute_subgroups), yes(d->ballot), yes(d->arith),
           yes(d->clustered), yes(d->shuffle));
    if (d->control)
        printf("  subgroup size: %u to %u, selectable (default %u)\n",
               d->sg_min, d->sg_max, d->sg_fixed);
    else
        printf("  subgroup size: %u%s\n", d->sg_fixed,
               d->sg_min == d->sg_max ? ", fixed"
                                      : ", varying, not selectable");
    if (d->host_import)
        printf("  host memory imported in place (zero-copy): yes, aligned to "
               "%llu bytes\n",
               (unsigned long long)d->import_align);
    else
        printf("  host memory imported in place (zero-copy): NO\n");
    if (d->portability)
        printf("  a portability implementation (VK_KHR_portability_subset)\n");
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(pd, &mp);
    VkDeviceSize local = 0, shared = 0;
    for (uint32_t h = 0; h < mp.memoryHeapCount; h++) {
        if (!(mp.memoryHeaps[h].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT))
            continue;
        local += mp.memoryHeaps[h].size;
        for (uint32_t t = 0; t < mp.memoryTypeCount; t++)
            if (mp.memoryTypes[t].heapIndex == h &&
                (mp.memoryTypes[t].propertyFlags &
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)) {
                shared += mp.memoryHeaps[h].size;
                break;
            }
    }
    printf("  device-local memory: %.1f GB, of which the CPU can map %.1f GB\n",
           (double)local / 1e9, (double)shared / 1e9);
    uint32_t nq = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &nq, NULL);
    VkQueueFamilyProperties qf[64];
    if (nq > 64)
        nq = 64;
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &nq, qf);
    d->family = UINT32_MAX;
    for (uint32_t f = 0; f < nq && d->family == UINT32_MAX; f++)
        if (qf[f].queueFlags & VK_QUEUE_COMPUTE_BIT)
            d->family = f;

    /* the engine's decision */
    const char *se = getenv("JANAS_GPU_SG");
    uint32_t want = se ? (uint32_t)atoi(se) : 8;
    if (!d->bda || !d->idp || !d->ballot)
        snprintf(d->why, sizeof(d->why),
                 "no buffer addresses, integer dot product or ballot");
    else if (!d->arith || !d->clustered || !d->shuffle) {
        /* as the engine says it (gpu.c) */
        char miss[48] = "";
        if (!d->arith)
            strcat(miss, ", arithmetic");
        if (!d->clustered)
            strcat(miss, ", clustered");
        if (!d->shuffle)
            strcat(miss, ", shuffle");
        snprintf(d->why, sizeof(d->why),
                 "no %s subgroup operations in compute shaders", miss + 2);
    } else if (!discrete && !d->host_import)
        snprintf(d->why, sizeof(d->why),
                 "integrated, without VK_EXT_external_memory_host");
    else if (d->family == UINT32_MAX)
        snprintf(d->why, sizeof(d->why), "no compute queue");
    else {
        if (d->control && want >= d->sg_min && want <= d->sg_max)
            d->sg = want;
        else if (d->control || d->sg_min == d->sg_max)
            d->sg = d->sg_fixed;
        else
            snprintf(d->why, sizeof(d->why),
                     "subgroup size varying and not selectable");
        if (d->sg && (d->sg < 4 || d->sg > 64 || (d->sg & (d->sg - 1))))
            snprintf(d->why, sizeof(d->why), "subgroup size %u not usable",
                     d->sg);
    }
    if (d->why[0]) {
        printf("  Janas-LLM would not take it: %s\n", d->why);
        return;
    }
    d->taken = 1;
    printf("  Janas-LLM would take it: %s, subgroups of %u\n",
           discrete ? "with its own memory, the weights copied there"
                    : "reading the host's memory in place",
           d->sg);
}

/* The device opened for the tests, and one compute pipeline per shader. */
static VkDevice dev;
static VkQueue queue;
static VkCommandPool pool;
static VkCommandBuffer cb;
static VkFence fence;

struct pipe {
    VkShaderModule mod;
    VkPipelineLayout layout;
    VkPipeline pipe;
    uint32_t push;
};

static int make_pipe(const uint32_t *code, size_t bytes, uint32_t push,
                     struct pipe *p)
{
    VkShaderModuleCreateInfo mi = {
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = bytes,
        .pCode = code};
    VkPushConstantRange pr = {.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
                              .size = push};
    VkPipelineLayoutCreateInfo li = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &pr};
    p->push = push;
    if (vkCreateShaderModule(dev, &mi, NULL, &p->mod) != VK_SUCCESS ||
        vkCreatePipelineLayout(dev, &li, NULL, &p->layout) != VK_SUCCESS)
        return -1;
    VkComputePipelineCreateInfo ci = {
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                  .stage = VK_SHADER_STAGE_COMPUTE_BIT,
                  .module = p->mod,
                  .pName = "main"},
        .layout = p->layout};
    return vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &ci, NULL,
                                    &p->pipe) == VK_SUCCESS
               ? 0
               : -1;
}

static void drop_pipe(struct pipe *p)
{
    vkDestroyPipeline(dev, p->pipe, NULL);
    vkDestroyPipelineLayout(dev, p->layout, NULL);
    vkDestroyShaderModule(dev, p->mod, NULL);
}

/* One dispatch of groups workgroups, waited for: its seconds, or -1. */
static double run(const struct pipe *p, const void *push, uint32_t groups)
{
    VkCommandBufferBeginInfo bi = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
    VkSubmitInfo si = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                       .commandBufferCount = 1,
                       .pCommandBuffers = &cb};
    double t0 = now();
    if (vkBeginCommandBuffer(cb, &bi) != VK_SUCCESS)
        return -1;
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, p->pipe);
    vkCmdPushConstants(cb, p->layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, p->push,
                       push);
    vkCmdDispatch(cb, groups, 1, 1);
    /* the shader's writes made visible to the host's reads */
    VkMemoryBarrier mb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                          .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
                          .dstAccessMask = VK_ACCESS_HOST_READ_BIT};
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &mb, 0, NULL, 0,
                         NULL);
    if (vkEndCommandBuffer(cb) != VK_SUCCESS ||
        vkResetFences(dev, 1, &fence) != VK_SUCCESS ||
        vkQueueSubmit(queue, 1, &si, fence) != VK_SUCCESS ||
        vkWaitForFences(dev, 1, &fence, VK_TRUE, UINT64_MAX) != VK_SUCCESS)
        return -1;
    return now() - t0;
}

/* The tests' memory: the host's own imported where the GPU can (the
   engine's way on an integrated GPU), else memory the CPU maps. */
static uint8_t *host;
static VkBuffer buf;
static VkDeviceMemory mem;
static VkDeviceAddress addr;

static int make_memory(const struct dev *d, size_t bytes, int import)
{
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(d->pd, &mp);
    VkExternalMemoryBufferCreateInfo eb = {
        .sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO,
        .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT};
    VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                             .pNext = import ? &eb : NULL,
                             .size = bytes,
                             .usage =
                                 VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                 VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT};
    if (vkCreateBuffer(dev, &bi, NULL, &buf) != VK_SUCCESS)
        return -1;
    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(dev, buf, &req);
    uint32_t bits = req.memoryTypeBits;
    VkImportMemoryHostPointerInfoEXT ip = {
        .sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_HOST_POINTER_INFO_EXT,
        .handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT,
        .pHostPointer = host};
    if (import) {
        VkMemoryHostPointerPropertiesEXT hp = {
            .sType = VK_STRUCTURE_TYPE_MEMORY_HOST_POINTER_PROPERTIES_EXT};
        if (get_host_pointer_props(
                dev, VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT,
                host, &hp) != VK_SUCCESS)
            return -1;
        bits &= hp.memoryTypeBits;
    }
    VkMemoryPropertyFlags need = import
                                     ? 0
                                     : VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                           VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    uint32_t type = UINT32_MAX;
    for (uint32_t t = 0; t < mp.memoryTypeCount && type == UINT32_MAX; t++)
        if ((bits & (1u << t)) &&
            (mp.memoryTypes[t].propertyFlags & need) == need)
            type = t;
    if (type == UINT32_MAX)
        return -1;
    VkMemoryAllocateFlagsInfo fl = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO,
        .pNext = import ? &ip : NULL,
        .flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT};
    VkMemoryAllocateInfo ai = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                               .pNext = &fl,
                               .allocationSize = import ? bytes : req.size,
                               .memoryTypeIndex = type};
    if (vkAllocateMemory(dev, &ai, NULL, &mem) != VK_SUCCESS ||
        vkBindBufferMemory(dev, buf, mem, 0) != VK_SUCCESS)
        return -1;
    if (!import && vkMapMemory(dev, mem, 0, VK_WHOLE_SIZE, 0, (void **)&host) !=
                       VK_SUCCESS)
        return -1;
    VkBufferDeviceAddressInfo ad = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO, .buffer = buf};
    addr = vkGetBufferDeviceAddress(dev, &ad);
    return addr ? 0 : -1;
}

static int dot4(uint32_t a, uint32_t b)
{
    int s = 0;
    for (int k = 0; k < 4; k++)
        s += (int8_t)(a >> 8 * k) * (int8_t)(b >> 8 * k);
    return s;
}

static int tests(const struct dev *d)
{
    int fails = 0, import = d->host_import;
    /* the device, as the engine creates it */
    float prio = 1.0f;
    VkDeviceQueueCreateInfo qi = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .queueFamilyIndex = d->family,
        .queueCount = 1,
        .pQueuePriorities = &prio};
    VkPhysicalDeviceVulkan13Features f13 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
        .subgroupSizeControl = d->control ? VK_TRUE : VK_FALSE,
        .computeFullSubgroups = d->control ? VK_TRUE : VK_FALSE,
        .shaderIntegerDotProduct = VK_TRUE};
    VkPhysicalDeviceVulkan12Features f12 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
        .pNext = &f13,
        .bufferDeviceAddress = VK_TRUE};
    const char *ext[2];
    uint32_t n_ext = 0;
    if (import)
        ext[n_ext++] = VK_EXT_EXTERNAL_MEMORY_HOST_EXTENSION_NAME;
    if (d->portability)
        ext[n_ext++] = "VK_KHR_portability_subset";
    VkDeviceCreateInfo di = {.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
                             .pNext = &f12,
                             .queueCreateInfoCount = 1,
                             .pQueueCreateInfos = &qi,
                             .enabledExtensionCount = n_ext,
                             .ppEnabledExtensionNames = ext};
    if (vkCreateDevice(d->pd, &di, NULL, &dev) != VK_SUCCESS) {
        printf("  cannot open the device\n");
        return 1;
    }
#define LOAD_DEVICE(f) f = (PFN_##f)vkGetDeviceProcAddr(dev, #f);
    DEVICE_FNS(LOAD_DEVICE)
    get_host_pointer_props =
        (PFN_vkGetMemoryHostPointerPropertiesEXT)vkGetDeviceProcAddr(
            dev, "vkGetMemoryHostPointerPropertiesEXT");
    if (import && !get_host_pointer_props)
        import = 0;
    vkGetDeviceQueue(dev, d->family, 0, &queue);
    VkCommandPoolCreateInfo pi = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
        .queueFamilyIndex = d->family};
    VkCommandBufferAllocateInfo ca = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1};
    VkFenceCreateInfo fi = {.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    struct pipe copy, dot, order;
    if (vkCreateCommandPool(dev, &pi, NULL, &pool) != VK_SUCCESS ||
        (ca.commandPool = pool,
         vkAllocateCommandBuffers(dev, &ca, &cb) != VK_SUCCESS) ||
        vkCreateFence(dev, &fi, NULL, &fence) != VK_SUCCESS ||
        make_pipe(vkcheck_spv_copy, sizeof(vkcheck_spv_copy), 20, &copy) ||
        make_pipe(vkcheck_spv_dot, sizeof(vkcheck_spv_dot), 24, &dot) ||
        make_pipe(vkcheck_spv_order, sizeof(vkcheck_spv_order), 16, &order)) {
        printf("  cannot build the test shaders\n");
        return 1;
    }

    /* 8 MB of memory, in the host's own pages when imported */
    const uint32_t n = 1u << 20;
    size_t bytes = (size_t)2 * n * 4, align = 65536;
    if (d->import_align > align)
        align = (size_t)d->import_align;
    if (import) {
#ifdef _WIN32
        host = _aligned_malloc(bytes, align);
#else
        host = aligned_alloc(align, bytes);
#endif
        if (!host)
            return 1;
    }
    if (make_memory(d, bytes, import) != 0) {
        printf("  zero-copy: FAIL, the host's memory could not be imported "
               "(%zu bytes aligned to %zu)\n",
               bytes, align);
        return 1;
    }
    uint32_t *u = (uint32_t *)host;

    /* zero-copy: y = 3x + 1, both in that memory */
    for (uint32_t i = 0; i < n; i++) {
        u[i] = i ^ 0x9E3779B9u;
        u[n + i] = 0xDEADBEEFu;
    }
    struct {
        uint64_t x, y;
        uint32_t n;
    } __attribute__((packed)) pc = {addr, addr + (uint64_t)n * 4, n};
    uint32_t bad = 0, stale = 0;
    if (run(&copy, &pc, (n + 63) / 64) < 0)
        bad = n;
    else
        for (uint32_t i = 0; i < n; i++) {
            bad += u[n + i] != u[i] * 3u + 1u;
            stale += u[n + i] == 0xDEADBEEFu;
        }
    if (!import)
        printf("  zero-copy: not available; memory the CPU maps instead%s\n",
               bad ? " - and even that FAILED" : ", which works");
    else if (!bad)
        printf("  zero-copy: ok, the GPU read and wrote this program's own "
               "memory (%u values)\n",
               n);
    else
        printf("  zero-copy: FAIL, %u of %u values wrong (%u never written)\n",
               bad, n, stale);
    fails += bad != 0;

    /* the integer dot product: correct, then how fast */
    struct {
        uint64_t a, y;
        uint32_t n, reps;
    } __attribute__((packed)) dc = {addr, addr + (uint64_t)n * 4, n, 64};
    uint32_t seed = 1;
    for (uint32_t i = 0; i < n; i++) {
        seed = seed * 1103515245u + 12345u;
        u[i] = seed ^ (seed >> 15);
    }
    bad = 0;
    if (run(&dot, &dc, (n + 63) / 64) < 0)
        bad = 1;
    for (uint32_t i = 0; i < n && !bad; i += 16) {
        uint32_t x = u[i], w0 = u[(i * 7u + 3u) % n];
        uint32_t w1 = w0 ^ 0x5A5A5A5Au, w2 = w0 + 0x01020304u, w3 = ~w0;
        int acc = 0;
        for (uint32_t r = 0; r < dc.reps; r++) {
            acc += dot4(x, w0) + dot4(x, w1) + dot4(x, w2) + dot4(x, w3);
            x += 0x9E3779B9u;
        }
        bad += (int32_t)u[n + i] != acc;
    }
    dc.reps = 4096;
    double t = bad ? -1 : run(&dot, &dc, (n + 63) / 64);
    if (bad || t <= 0) {
        printf("  integer dot product: FAIL, results differ from the CPU's\n");
        fails++;
    } else
        printf("  integer dot product: ok, as the CPU's; about %.0f billion "
               "int8 multiply-adds a second\n",
               (double)n * dc.reps * 16 / t / 1e9);

    /* float: fma, precise, order, subnormals */
    float *f = (float *)host;
    float a = 1.0f + ldexpf(1.0f, -12), c = -(1.0f + ldexpf(1.0f, -11));
    f[0] = a;
    f[1] = a;
    f[2] = c;
    f[3] = 1e8f;
    f[4] = -1e8f;
    f[5] = 1.0f;
    f[6] = ldexpf(1.0f, -130);
    f[7] = 1.0f;
    struct {
        uint64_t x, y;
    } __attribute__((packed)) oc = {addr, addr + 64};
    for (int i = 0; i < 5; i++)
        f[16 + i] = -999.0f;
    volatile float m = a * a; /* rounded once, not fused with what follows */
    float fused = fmaf(a, a, c), unfused = m + c;
    if (run(&order, &oc, 1) < 0) {
        printf("  float: FAIL, the shader did not run\n");
        fails++;
    } else {
        int ok_fma = f[16] == fused, ok_precise = f[17] == unfused,
            ok_order = f[18] == 1.0f;
        printf("  float: fma() rounded once %s, a precise a * b + c rounded "
               "twice %s, a precise sum in the order written %s\n",
               ok_fma ? "ok" : "FAIL", ok_precise ? "ok" : "FAIL",
               ok_order ? "ok" : "FAIL");
        printf("  float, for information: fma() not marked precise %s; "
               "subnormals %s\n",
               f[19] == fused ? "fused" : "not fused",
               f[20] != 0.0f ? "kept" : "flushed to zero");
        fails += !ok_fma + !ok_precise + !ok_order;
    }
    drop_pipe(&copy);
    drop_pipe(&dot);
    drop_pipe(&order);
    vkDestroyBuffer(dev, buf, NULL);
    vkFreeMemory(dev, mem, NULL);
    vkDestroyFence(dev, fence, NULL);
    vkDestroyCommandPool(dev, pool, NULL);
    vkDestroyDevice(dev, NULL);
    if (import) {
#ifdef _WIN32
        _aligned_free(host);
#else
        free(host);
#endif
    }
    return fails;
}

int main(int argc, char **argv)
{
    static const char *const names[] = {
#if defined(_WIN32)
        "vulkan-1.dll",
#elif defined(__APPLE__)
        "libvulkan.1.dylib",
        "/usr/local/lib/libvulkan.1.dylib",
        "/opt/homebrew/lib/libvulkan.1.dylib",
        "libMoltenVK.dylib",
        "/usr/local/lib/libMoltenVK.dylib",
        "/opt/homebrew/lib/libMoltenVK.dylib",
#else
        "libvulkan.so.1", "libvulkan.so",
#endif
        NULL};
    const char *lib = NULL;
    if (argc > 1) {
        if (load(argv[1]))
            lib = argv[1];
    } else
        for (int i = 0; names[i] && !lib; i++)
            if (load(names[i]))
                lib = names[i];
    if (!lib) {
        printf("vkcheck: no Vulkan library (%s)\n",
               argc > 1 ? argv[1] : "the loader, or on macOS MoltenVK");
        return 1;
    }
    PFN_vkCreateInstance create =
        (PFN_vkCreateInstance)gipa(NULL, "vkCreateInstance");
    PFN_vkEnumerateInstanceExtensionProperties inst_ext =
        (PFN_vkEnumerateInstanceExtensionProperties)gipa(
            NULL, "vkEnumerateInstanceExtensionProperties");
    PFN_vkEnumerateInstanceVersion inst_ver =
        (PFN_vkEnumerateInstanceVersion)gipa(NULL,
                                             "vkEnumerateInstanceVersion");
    uint32_t iv = VK_API_VERSION_1_0;
    if (inst_ver)
        inst_ver(&iv);
    /* MoltenVK through the loader is listed only when asked for */
    uint32_t ne = 0;
    if (inst_ext)
        inst_ext(NULL, &ne, NULL);
    VkExtensionProperties *ie = calloc(ne ? ne : 1, sizeof(*ie));
    if (!ie)
        return 1;
    if (inst_ext)
        inst_ext(NULL, &ne, ie);
    int port = has_ext(ie, ne, "VK_KHR_portability_enumeration");
    free(ie);
    const char *iext[] = {"VK_KHR_portability_enumeration"};
    VkApplicationInfo app = {.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
                             .pApplicationName = "vkcheck",
                             .apiVersion = VK_API_VERSION_1_3};
    VkInstanceCreateInfo ici = {
        .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .flags = port ? VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR : 0,
        .pApplicationInfo = &app,
        .enabledExtensionCount = port ? 1 : 0,
        .ppEnabledExtensionNames = iext};
    VkInstance inst;
    VkResult r;
    if (!create || (r = create(&ici, NULL, &inst)) != VK_SUCCESS) {
        printf("vkcheck: %s: no Vulkan instance\n", lib);
        return 1;
    }
    printf("vkcheck: %s, instance Vulkan %u.%u%s\n", lib,
           VK_API_VERSION_MAJOR(iv), VK_API_VERSION_MINOR(iv),
           port ? ", portability implementations listed" : "");
#define LOAD_INSTANCE(f) f = (PFN_##f)gipa(inst, #f);
    INSTANCE_FNS(LOAD_INSTANCE)
    uint32_t np = 0;
    vkEnumeratePhysicalDevices(inst, &np, NULL);
    VkPhysicalDevice pds[16];
    if (np > 16)
        np = 16;
    vkEnumeratePhysicalDevices(inst, &np, pds);
    if (!np) {
        printf("vkcheck: Vulkan sees no GPU\n");
        return 1;
    }
    struct dev d[16];
    for (uint32_t i = 0; i < np; i++)
        examine(pds[i], (int)i, &d[i]);
    /* the engine's choice: a discrete GPU first, then an integrated one;
       JANAS_GPU_DEVICE=discrete|integrated restricts */
    const char *want = getenv("JANAS_GPU_DEVICE");
    int pick = -1;
    for (int pass = 0; pass < 2 && pick < 0; pass++) {
        VkPhysicalDeviceType ty = pass == 0
                                      ? VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU
                                      : VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU;
        if (want && strcmp(want, pass == 0 ? "discrete" : "integrated"))
            continue;
        for (uint32_t i = 0; i < np && pick < 0; i++)
            if (d[i].taken && d[i].p.deviceType == ty)
                pick = (int)i;
    }
    int rc = 1;
    if (pick < 0)
        printf("\nJanas-LLM would run on the CPU alone: no GPU it takes\n");
    else {
        printf("\nJanas-LLM would use GPU %d; tests on it:\n", pick);
        rc = tests(&d[pick]) ? 1 : 0;
        printf("%s\n", rc ? "some test FAILED: tell the project, with this "
                            "output"
                          : "every test passed");
    }
    vkDestroyInstance(inst, NULL);
    return rc;
}
#endif
