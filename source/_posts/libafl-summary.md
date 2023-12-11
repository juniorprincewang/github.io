---
title: libafl summary
tags:
  - fuzz
  - afl
categories:
  - fuzz
date: 2023-03-13 10:08:12
---


本文总结libafl 的学习历程。

<!-- more -->

自从基于路径覆盖反馈的模糊测试工具 AFL 横空出世后，afl思路的衍生工具屡屡创新，而afl已经作古，后继者 [AFLPlusPlus](https://github.com/AFLplusplus/AFLplusplus) 拥有着更快、更多变异方法、更多插桩选择和自定义的模块支持等优势取而代之。但是这里我要讨论的是AFL集大成之作，AFLPlusPlus作者们最新力作 [LibAFL](https://github.com/AFLplusplus/LibAFL)，该fuzzer有着更快、扩展性强、适用性高、多平台选择、多目标等明显优点，也集成了多fuzzer方案，非常值得学习和研究。

以 LibAFL 0.11.1 版本为准。

# forkserver

forkserver 模式是最基础的模式之一，以 `ForkserverExecutor` 为具体实现，Client fuzzer 会fork出子进程去执行目标binary，执行获得的覆盖路径通过共享内存获得。

# inprocess

inprocess 模式也是最基础的模式之一，以 `InProcessExecutor` 实现，client fuzzer 回去执行 harness 闭包函数，harness闭包函数会执行目标函数，这样避免了fork引入的性能开销。

# libfuzzer

- harness 函数，封装在函数 `extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size)` 中，实现在 `harness.cc` 中。
- fuzzer，此处fuzzer 以静态库的形式生成，实现在 lib.rs 中。
	+ libfuzzer 是以 `InProcessExecutor` 当作执行器
	+ 需要调用封装好的 libfuzzer native 初始化函数 `libfuzzer_initialize()`，在 harness 闭包函数中调用 `LLVMFuzzerTestOneInput()` 来处理mutated input buffer。
- ClangWrapper，需要将 harness.cc 编译并链接 fuzzer 静态库，实则是链接 `LLVMFuzzerTestOneInput` 函数实现，实现在 `libafl_cc.rs` 中。

Makefile.toml 中关键一步：

```
[tasks.fuzzer_unix]
command = "${CARGO_TARGET_DIR}/debug/libafl_cxx"
args = ["${PROJECT_DIR}/harness.cc", "-o", "${FUZZER_NAME}"]
dependencies = [ "cxx", "cc" ]
```

# fork_qemu

`fork_qemu` 的目的是以 fork 的形式来替代 restore state，
启动QEMU载入目标 target binary，并分析得到目标函数偏移地址，先对此地址打断点，运行程序到此处，得到运行时函数上下文（函数地址RIP，栈地址RSP，由栈地址得到返回地址），后面将此处当作snapshot，只保留唯一breakpoint，即目标函数的返回地址。每次只从函数入口运行到函数返回即可。
具体实现：

- fuzzer:
	+ 以 `QemuForkExecutor` 做执行器，`QemuForkExecutor` 内部以 `InProcessForkExecutor` 作为具体实现
	+ Client fuzzer 每次执行，会fork当前进程，即每次QEMU状态都是目标函数入口处，在子进程去执行 harness。
	+ harness 需要重新覆盖目标函数输入，见下面 Linux x64 调用约定描述（比如 将当前变异后的输入写入到 QEMU 映射内存中，并内存地址写入寄存器 RDI，写入内存长度写入寄存器RSI）。随后继续运行 QEMU，到断点停止运行，并返回。


Linux 的 x64 下也只有一种函数调用约定，即 `__fastcall` ，其他调用约定的关键字会被忽略。
如果函数参数个数小于等于 6 个，前 6 个参数是从左至右依次存放于 RDI，RSI，RDX，RCX，R8，R9 寄存器里面，剩下的参数通过栈传递，从右至左顺序入栈；
如果参数个数大于 6 个，前 5 个参数是从左至右依次存放于 RDI，RSI，RDX，RCX，RAX 寄存器里面，剩下的参数通过栈传递，从右至左顺序入栈；
对于系统调用，使用 R10 代替 RCX；
XMM0 ~ XMM7 用于传递浮点参数。

# concolic execution

concolic 来源于concrete(具体)和symbolic(符号)的组合，此处是混合执行的意思。

	从一个给定的输入或随机输入开始执行程序，沿着执行的条件语句在输入上收集符号约束，然后使用约束求解推断先前输入的变化，以便引导程序接下来的执行该走向哪一个执行路径。重复此过程，直到探索了所有执行路径，或者满足用户定义的覆盖标准、时间设置到期为止。

LibAFL 的 Concolic Tracing 由 SymCC 实现， SymCC 是 clang 的一款插件，可以替换掉 C/C++ 编译器。
SymCC 会插桩源码用户指定的回调函数，这些回调允许运行时构建一个trace。

使用LibAFL构建一个混合型fuzzer主要有三个步骤:

1. 建立一个运行时间
使用 symcc_runtime 模块构建自定义的runtime，生成 `cdylib` 类型名字为 `SymRuntime` 的libSymRuntime.so 库。

2. 选择一个插桩的方法
有源码就选择 编译时插桩化的目标与SymCC。
设置环境变量 `CC=symcc`、`CXX=sym++`、`SYMCC_RUNTIME_DIR`，使用sym++作为clang的替代品，并对目标进行编译时插桩。

3. 构建 fuzzer

使用 `CommandExecutor` 来执行 target，可以通过实现 `CommandConfigurator` trait，创建并启动 `std::process::Command`。
如果target有输入文件可以由 `SYMCC_INPUT_FILE` 指定。

序列化 Serialization ：
虽然完全可以构建一个自定义运行时，该运行时也可以在目标进程的上下文中执行混合fuzzing，但LibAFL使用 `TracingRuntime` 序列化（过滤和预处理）分支条件。这个序列化的表示可以在fuzzing过程中进行反序列化，以便使用封装在 `ConcolicTracingStage` 中的 `ConcolicObserver` 进行求解，该 `ConcolicTraceingStage` 将向每个TestCase附加一个 `ConcolicMetadata` 。

`ConcolicMetadata` 可用于回放 concolic trace 并使用 SMT求解器 解决条件。然而，大多数涉及一致性追踪的用例都需要定义一些策略，围绕它们想要解决的分支。
`SimpleConcolicMutationalStage` 可用于测试用途，它尝试使用 `Z3` 解决附加到 `crate::corpus::Testcase` 的 concolic约束。

[Concolic Tracing and Hybrid Fuzzing](https://aflplus.plus/libafl-book/advanced_features/concolic.html)