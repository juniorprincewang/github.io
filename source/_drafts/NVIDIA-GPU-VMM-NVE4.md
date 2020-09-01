---
title: NVIDIA GPU虚拟内存（NVE4）
tags:
- virtual memory
- GPU
categories:
- [GPU]
---

本文分析 `NVIDIA GPU` 的虚拟内存（virtual memory），由 novueau源码和 `envytools` 工具和实验得出。分析的GTX 680显卡（NVE4，GK104），仅供参考。  
<!-- more -->

关于 NVE4 的虚拟内存代码在  
<https://github.com/skeggsb/nouveau/blob/master/drm/nouveau/nvkm/subdev/mmu/vmmgk104.c>  
<https://github.com/skeggsb/nouveau/blob/master/drm/nouveau/nvkm/subdev/mmu/vmmgf100.c>  
<https://github.com/skeggsb/nouveau/blob/master/drm/nouveau/nvkm/subdev/mmu/vmm.c>  

# Channel  

关于 channel 的设置要重点看 fifo chan，即  <https://github.com/skeggsb/nouveau/blob/master/drm/nouveau/nvkm/subdev/mmu/gpfifogk104.c> 。  

channel descriptor 的抽象概念对应的结构体是 `PFIFO_CHAN[ID].CHAN` 和 `PFIFO_CHAN[ID].STATE`。  
正如CPU 中虚拟页表一样，不同channel的虚拟地址由页目录根地址记录，而这个PD记录在了 `PFIFO_CHAN` MMIO 寄存器中。  

```c
static void
gk104_fifo_gpfifo_init(struct nvkm_fifo_chan *base)
{
    struct gk104_fifo_chan *chan = gk104_fifo_chan(base);
    struct gk104_fifo *fifo = chan->fifo;
    struct nvkm_device *device = fifo->base.engine.subdev.device;
    u32 addr = chan->base.inst->addr >> 12;
    u32 coff = chan->base.chid * 8;
    
    nvif_debug(&base->object, " func %s: addr 0x%x coff 0x%x\n", __func__, addr, coff);
    // 0x800000 PFIFO_CHAN[ID].STATE
    nvkm_mask(device, 0x800004 + coff, 0x000f0000, chan->runl << 16); // 16~19 ENGINE
    // 0x800004 PFIFO_CHAN[ID].CHAN
    nvkm_wr32(device, 0x800000 + coff, 0x80000000 | addr);  // VRAM | addr>>12

    if (list_empty(&chan->head) && !chan->killed) {
        gk104_fifo_runlist_insert(fifo, chan);
        nvkm_mask(device, 0x800004 + coff, 0x00000400, 0x00000400);
        gk104_fifo_runlist_commit(fifo, chan->runl);
        nvkm_mask(device, 0x800004 + coff, 0x00000400, 0x00000400);
    }
}
```


# Page Tables

<https://github.com/skeggsb/nouveau/blob/master/drm/nouveau/nvkm/subdev/mmu/vmmgk104.c>   

GK104 的虚拟地址寻址是40 bit，页表为二级页表， PD(13) | PT(15) | PAGE(12)。下面将怎么得到的。    

page的设置在 subdev `fb` 中，`mmu->subdev.device->fb->page` 的值在 *nvkm\subdev\fb\gk104.c* 的 `struct nvkm_fb_func gk104_fb` 变量的 `.default_bigpage = 17`，在 `nvkm_fb_ctor()` 构造函数中 gk104的 `fb->page` 会被这个默认值赋值。  

因此，`struct nvkm_vmm_func gk104_vmm_17` 会得到调用。  
```c
// nvkm\subdev\mmu\vmmgk104.c
static const struct nvkm_vmm_func
gk104_vmm_17 = {
    .join = gf100_vmm_join,
    .part = gf100_vmm_part,
    .aper = gf100_vmm_aper,
    .valid = gf100_vmm_valid,
    .flush = gf100_vmm_flush,
    .page = {
        { 17, &gk104_vmm_desc_17_17[0], NVKM_VMM_PAGE_xVxC },
        // the smallest page size: 12bit
        { 12, &gk104_vmm_desc_17_12[0], NVKM_VMM_PAGE_xVHx },
        {}
    }
};
```

这里的 `func->page` 有两个, 在 `nvkm_vmm_ctor()` 中会取 `page[1]`操作，即 `shift=12`(称之为最小page size)，desc为 `gk104_vmm_desc_17_12`，页表15 bit，页目录13 bit。

```c
static const struct nvkm_vmm_desc_func
gk104_vmm_lpt = {
    .invalid = gk104_vmm_lpt_invalid,
    .unmap = gf100_vmm_pgt_unmap,
    .mem = gf100_vmm_pgt_mem,
};

const struct nvkm_vmm_desc
gk104_vmm_desc_17_12[] = {
    { SPT, 15, 8, 0x1000, &gf100_vmm_pgt },
    { PGD, 13, 8, 0x1000, &gf100_vmm_pgd },
    {}
};
```

根据最小页表对应的`gk104_vmm_desc_17_12`页表描述可以，页表有两级 页目录PGD(13 bits) | 小页表SPT(15 bits) ,每个页表项条目8 bytes，并且页表0x1000对齐，因此寻址为 40bit，即 13+15+12。   
页目录的操作的描述符函数 `gf100_vmm_pgd`，页表的操作的描述符函数 `gf100_vmm_pgt` 。  
注意，这里还有个大页表的概念LPT，即每页 17 bits。  

![NVE4页表图](../NVIDIA-GPU-VMM-NVE4/pagetable.jpg)

## Page Directory Update  

```c
void gf100_vmm_pgd_pde(struct nvkm_vmm *vmm, struct nvkm_vmm_pt *pgd, u32 pdei)
{
    struct nvkm_vmm_pt *pgt = pgd->pde[pdei];
    struct nvkm_mmu_pt *pd = pgd->pt[0];
    struct nvkm_mmu_pt *pt;
    u64 data = 0;

    if ((pt = pgt->pt[0])) {
        VMM_DEBUG(vmm, "none SPT\n");
        switch (nvkm_memory_target(pt->memory)) {
        case NVKM_MEM_TARGET_VRAM: data |= 1ULL << 0; break;
        case NVKM_MEM_TARGET_HOST: data |= 2ULL << 0;
            data |= BIT_ULL(35); /* VOL */
            break;
        case NVKM_MEM_TARGET_NCOH: data |= 3ULL << 0; break;
        default:
            WARN_ON(1);
            return;
        }
        data |= pt->addr >> 8;
    }

    if ((pt = pgt->pt[1])) {
        VMM_DEBUG(vmm, "SPT\n");
        switch (nvkm_memory_target(pt->memory)) {
        case NVKM_MEM_TARGET_VRAM: data |= 1ULL << 32; break;
        case NVKM_MEM_TARGET_HOST: data |= 2ULL << 32;
            data |= BIT_ULL(34); /* VOL */
            break;
        case NVKM_MEM_TARGET_NCOH: data |= 3ULL << 32; break;
        default:
            WARN_ON(1);
            return;
        }
        data |= pt->addr << 24;
    }

    nvkm_kmap(pd->memory);
    VMM_WO064(pd, vmm, pdei * 8, data);
    nvkm_done(pd->memory);
}
```

`gf100_vmm_pgd_pde()` 在 vmm.c 的 `nvkm_vmm_ref_hwpt()` 中的 `it->desc[it->lvl].func->pde(it->vmm, pgd, pdei);` 中调用。  
`gf100_vmm_pgd_pde()` 将页目录的第 pdei 个目录项的页表地址写在当前pd inst内存中，由于页目录条目8bytes，因此写入的偏移地址为 `pdei*8`。  
页目录的内存类型为VRAM，在GPU内。但是页就不一定在GPU内了。    


## Page Table Update  

```c
static inline void 
gf100_vmm_pgt_pte(struct nvkm_vmm *vmm, struct nvkm_mmu_pt *pt,
          u32 ptei, u32 ptes, struct nvkm_vmm_map *map, u64 addr)
{
    u64 base = (addr >> 8) | map->type;
    u64 data = base;

    if (map->ctag && !(map->next & (1ULL << 44))) {
        while (ptes--) {
            data = base | ((map->ctag >> 1) << 44);
            if (!(map->ctag++ & 1))
                data |= BIT_ULL(60);

            VMM_WO064(pt, vmm, ptei++ * 8, data);
            base += map->next;
        }
    } else {
        map->type += ptes * map->ctag;

        while (ptes--) {
            VMM_WO064(pt, vmm, ptei++ * 8, data);
            data += map->next;
        }
    }
}
```

`gf100_vmm_pgt_pte()` 是将所有页表项ptes个，逐一在 pt inst memory上偏移地址为 `ptei++ * 8` 的地址写入 `data` 。  

有三种页表内存访问方式：vram，dma，sgl。其中vram代表页表指向的页内存地址在VRAM，dma代表页表指向的内存地址是bus的dma地址，sgl我没有分析。  

# TLB flush  

`struct nvkm_vmm_func gk104_vmm_17` 对象的 `flush` 函数是 `gf100_vmm_flush()`，实现在 <https://github.com/skeggsb/nouveau/blob/master/drm/nouveau/nvkm/subdev/mmu/vmmgf100.c> 中。  

```c
void gf100_vmm_flush_(struct nvkm_vmm *vmm, int depth)
{
    struct nvkm_subdev *subdev = &vmm->mmu->subdev;
    struct nvkm_device *device = subdev->device;
    u32 type = depth << 24;

    type = 0x00000001; /* PAGE_ALL */
    if (atomic_read(&vmm->engref[NVKM_SUBDEV_BAR]))
        type |= 0x00000004; /* HUB_ONLY */

    mutex_lock(&subdev->mutex);
    /* Looks like maybe a "free flush slots" counter, the
     * faster you write to 0x100cbc to more it decreases.
     */
    nvkm_msec(device, 2000,
        if (nvkm_rd32(device, 0x100c80) & 0x00ff0000)
            break;
    );
    // PFFB.VM.TLB_FLUSH_VSPACE
    nvkm_wr32(device, 0x100cb8, vmm->pd->pt[0]->addr >> 8);
    // PFFB.VM.TLB_FLUSH_TRIGGER
    nvkm_wr32(device, 0x100cbc, 0x80000000 | type);

    /* Wait for flush to be queued? */
    nvkm_msec(device, 2000,
        if (nvkm_rd32(device, 0x100c80) & 0x00008000)
            break;
    );
    mutex_unlock(&subdev->mutex);
}

void gf100_vmm_flush(struct nvkm_vmm *vmm, int depth)
{
    gf100_vmm_flush_(vmm, 0);
}
```

这里用到了三个MMIO寄存器：  
+ 0x100cb8
    * `PFFB.VM.TLB_FLUSH_VSPACE`
+ 0x100cbc
    * `PFFB.VM.TLB_FLUSH_TRIGGER`
+ 0x100c80
    * `PFFB.VM.CFG`

页目录的物理地址 写入到了 `PFFB.VM.TLB_FLUSH_VSPACE` MMIO中，并触发 `PFFB.VM.TLB_FLUSH_TRIGGER` 进行刷新。  