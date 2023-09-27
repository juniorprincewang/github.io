---
title: Vulkan中GPU执行时间
tags:
  - vulkan
categories:
  - - vulkan
date: 2023-09-27 19:17:52
---


本文总结Vulkan中使用QueryPool记录GPU执行时间的方法。

<!--more-->

记录GPU执行时间需要用到 `VkQueryPool` 的 `VK_QUERY_TYPE_TIMESTAMP` 查询类型。
在程序中使用方法：


### 声明变量 `VkQueryPool        m_QueryPool;` 。

### 创建 QueryPool 对象

```c++
const VkQueryPoolCreateInfo queryPoolCreateInfo =
{
    VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,     // VkStructureType                  sType
    NULL,                                         // const void*                      pNext
    (VkQueryPoolCreateFlags)0,                    // VkQueryPoolCreateFlags           flags
    VK_QUERY_TYPE_TIMESTAMP ,                     // VkQueryType                      queryType
    MaxValuesPerFrame * numberOfBackBuffers,      // deUint32                         queryCount
    0,                                            // VkQueryPipelineStatisticFlags    pipelineStatistics
};

VkResult res = vkCreateQueryPool(pDevice->GetDevice(), &queryPoolCreateInfo, NULL, &m_QueryPool);
```

### 记录时间戳

根据不同的 PIPELINE Stage，比如transfer 或者 compute stage 等，在**执行开始**和**执行结束**分别打上时间戳。

```c++
vkCmdWriteTimestamp(cmd_buf, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, m_QueryPool, offset);
```

### 查询时间戳

需要延后查询，如果程序是online需要在下一帧开始查询，有可能GPU未执行完成没有查询结果。如果是offline，则可以在等待GPU操作完成，比如 `vkWaitForFences()` 后查询。
```c++
// timestampPeriod is the number of nanoseconds per timestamp value increment
double microsecondsPerTick = (1e-3f * m_pDevice->GetPhysicalDeviceProperries().limits.timestampPeriod);      
UINT64 TimingsInTicks[256] = {};
VkResult res = vkGetQueryPoolResults(m_pDevice->GetDevice(), m_QueryPool, offset, measurements, measurements * sizeof(UINT64), &TimingsInTicks, sizeof(UINT64), VK_QUERY_RESULT_64_BIT);
if (res == VK_SUCCESS)
{
    for (uint32_t i = 1; i < measurements; i++)
    {
        float ts = float(microsecondsPerTick * (double)(TimingsInTicks[i] - TimingsInTicks[i - 1]));
    }

    // compute total
    float ts = float(microsecondsPerTick * (double)(TimingsInTicks[measurements - 1] - TimingsInTicks[0]));
}

```

获取一组queries的状态和结果:

```c++
// Provided by VK_VERSION_1_0
VkResult vkGetQueryPoolResults(
 VkDevice device,
 VkQueryPool queryPool,
 uint32_t firstQuery,
 uint32_t queryCount,
 size_t dataSize,
 void* pData,
 VkDeviceSize stride,
 VkQueryResultFlags flags);
```


- `device` 持有该query pool的逻辑设备。
- `queryPool` 管理着包含所求结果的queries的query pool。
- `firstQuery` 第一个query的索引。
- `queryCount` 要读取的queries的数量。
- `dataSize` pData所指向的缓冲的字节大小。
- `pData` 指向一个由用户分配的缓冲，结果将写入该缓冲中。
- `stride` 在pData中，queries的每个结果之间的字节跨度。
- `flags` 一个VkQueryResultFlagBits的bitmask，指出了结果将如何与何时返回。

查询返回的值为 microsecond（us），需要转换成 millisecond（ms）。

### 重置QueryPool

记录前或者获取完查询结果后进行重置。必须需要在 `vkBeginCommandBuffer()` 后使用。
```c++
vkCmdResetQueryPool(cmd_buf, m_QueryPool, offset, MaxValuesPerFrame);
```


参考：

+ [Timestamp Queries](https://registry.khronos.org/vulkan/specs/1.1-extensions/html/vkspec.html#queries-timestamps)
+ [How to measure execution time of Vulkan pipeline](https://stackoverflow.com/questions/67358235/how-to-measure-execution-time-of-vulkan-pipeline)