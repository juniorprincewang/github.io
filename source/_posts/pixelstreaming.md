---
title: UE5的PixelStreaming分析
tags:
  - UE
categories:
  - - UE
date: 2023-04-20 09:48:19
---

本文分析UE5.1版本的PixelStreaming的Editor推流原理。 UE5.1相较于UE5.0在PixelStreaming的改动很大，引入了PixelStreamingEditor、PixelStreamingPlayer等功能。

<!-- more -->


# [使用](#使用)

使用可以参考官方教程：<https://docs.unrealengine.com/5.1/en-US/pixel-streaming-in-editor/>。

Editor启动命令：
```
UnrealEditor-Cmd.exe -project Path\To\Your\Project.uproject -RenderOffscreen -EditorPixelStreamingRes=1920x1080 -EditorPixelStreamingStartOnLaunch=true -PixelStreamingURL=ws://127.0.0.1:8888
```
如果指定RenderOffscreen，就需要设置启动开启PixelStreaming选项和URL选项。如果正常启动，可以在Editor的工具栏中的Pixel Streaming中设置URL。Pixel Streaming可以连接远端已启动的信令服务器，也可以自启动/停止信令服务器。
Pixel Streaming Editor提供了两种串流模式，一种是Full Editor，串流整个窗口;一种是Level Editor，串流Level Editor Viewport。两种方法获取Backbuffer方式不同而已。
浏览器访问时需要设置鼠标为HoveringMouse模式，这样才会响应鼠标点击事件，或者在url中加入 hoveringMouse=true 选项。

# [原理](#原理)

## P2P 建立

游戏客户端与信令服务器间的通信由WebSocket建立后，便在信令服务器协调下与Browser建立起P2P连接。建立的P2P连接共建立了音频、视频流通道和输入数据通道，此后PixelStream客户端和Browser之间的数据交互不在经由信令服务器。
P2P连接建立的过程如下图。

1.	PixelStreaming客户端启动会带上启动参数 `-PixelStreamingURL=ws://127.0.0.1:8888` 形式的信令服务器地址,。PS通过WebSocket连接信令服务器地址，信令服务器收到请求后会返回附带iceServers候选项的配置信息供后面使用。
2.	用户在Browser访问WebServer，信令服务器收到用户请求后向PixelStreaming客户端发送playerConnected命令。
3.	PixelStreaming客户端收到playerConnected后，启动专门的线程处理与SignallingServer的操作，为player创建独立的session，创建 PeerConnectionFactory、创建PeerConnection、创建DataChannel、为Audio与Video添加track，创建Offer并设置Local SDP，将SDP发送给信令服务器。
4.	Browser收到经由信令服务器发送来的WebRTC Offer请求后，同样也创建PeerConnectionFactory、创建PeerConnection、创建DataChannel、为Audio与Video添加track，创建Answer并设置Local SDP，将Offer设置成Remote SDP，同时将Answer SDP发送给信令服务器。
5.	PixelStreaming客户端收到信令服务器发送来的Answer指令后设置Remote SDP。
6.	PixelStreaming客户端与Candidate在协商好STUN服务器后，便可开始P2P的通信。


![P2P时序图](/img/pixelstreaming/p2parch.jpg) 

P2P连接过程涉及几个关键类，包括：
+ `FPixelStreamingSignallingConnection` ： 负责与Signalling Server的连接、接收消息、发送消息。该类主要封装了对 `IWebSocket` 的调用，消息以Json格式传递。在 `OnMessage()` 函数中会解析信令服务器发送来的信息，解析出type字段，包括了config、offer、answer、iceCandidate、playerConnected、playerDisconnected、pong心跳信息等，后再去FStreamer对象处理。
+ `FStreamer`：核心类，管理与信令服务器通信、管理用户会话，负责WebRTC的交互处理、UE输入、视频源等。
+ `FPixelStreamingPeerConnection`：负责具体创建和处理PeerConnection一端的监听事件，包括创建 `PeerConnectionFactory`、异步创建offer、answer异步接收offer、answer，异步设置LocalSDP和RemoteSDP，添加Remote Ice Candidate，创建DataChannel。
+ `FPixelStreamingDataChannel`: 负责处理DataChannel一端的监听事件（browser输入）和发送数据。


## Editor的Video推流

Video推流流程分四个步骤：（1）捕获渲染帧，（2）创建编码器，（3）编码，（4）发送。

### 1.	捕获渲染帧

FullEditor模式捕获渲染帧通过UE中的Slate Render的回调函数 `OnBackBufferReadyToPresent()` 完成，传递的参数 `SlateWindow` 与 `FrameBuffer` 即为捕获到的窗口和渲染帧。
由于可视窗口可能会开多个且有叠加情况，该模式对所有可视化窗口帧做了组合。
组合原理：将可视化窗口从底到顶逐一处理，这样后处理的纹理能够覆盖前面的纹理。这里组合纹理大小是屏幕坐标+窗口大小，初始化的组合纹理长宽为1。
首先获取窗口坐标+大小，如果其长或宽有一方超过了组合纹理长或宽，则重新创建大纹理，将组合纹理赋值到此纹理；再将窗口纹理复制到组合纹理上。
这种做法简单粗暴，未作覆盖部分的裁剪，底层纹理仅保留露出来的纹理即可，该过程可在CPU端处理。
 

LevelEditor模式捕获渲染帧由 `UGameViewportClient::OnViewportRendered` 代理完成，该函数在viewport渲染完成后执行，拦截到的FViewport对象的RenderTargetTexture就是渲染帧。
回调函数在渲染线程中调用，捕获后的组合纹理会被拷贝到一个循环队列中缓存起来，等到对纹理编码时再从循环队列中读取。
纹理拷贝被封装到 `FPixelCapture Plugin` 中，由类 `FPixelCaptureCapturer` 完成，拷贝除了使用RHI外，又多了一种RDG方式。
纹理拷贝通过RHI异步拷贝，轮询Fence 来确定GPU拷贝完成；RDG方法不在GPU上立即执行Pass，而是先收集所有需要渲染的Pass，然后按照依赖的顺序对图表进行编译和执行，期间会执行各类裁剪和优化。
纹理缓存由 `FOutputFrameBuffer` 完成，内部实现为RingBuffer的循环队列。

### 2.	创建编码器

PixelStreaming使用 AVEncoder 组件对视频进行编码，选取的编码类型为H264。
`AVEncoder::FVideoEncoderFactory::Get().Create()` 可用于创建视频编码器工厂对象。先配置 VideoConfig，设置宽、高、码率、帧率等，根据 RHI （Vulkan\D3D11\D3D12） 创建 `VideoEncoderInput`，最终调用 `VideoEncoderFactory` 对象创建 encoder，此外还需要注册编码完成后的回调函数 `SetOnEncodedPacket`。

首先是向WebRTC的 `PeerConnectionFactory` 传入自定义的 `VideoEncoderFactory`。
在创建 `PeerConnectionFactory` 的函数 `webrtc::CreatePeerConnectionFactory` 的参数中会传入 `AudioEncoderFactory`、 `AudioDecoderFactory`、 `VideoEncoderFactory` 和 `VideoDecoderFactory` 四个编解码器工厂对象。
P2P创建 `PeerConnectionFactory` 传入的 `AudioEncoderFactory`、 `AudioDecoderFactory` 都是WebRTC内部指定类，而 `VideoEncoderFactory` 传入的是 `FVideoEncoderFactoryLayered` 对象，`VideoDecoderFactory` 传入的是 `FVideoDecoderFactory`对象。
接着，创建自定义的 `VideoEncoder`。编码器的创建由WebRTC内部调用 `FVideoEncoderFactoryLayered` 对象的 `CreateVideoEncoder()` 方法，创建出了 `FVideoEncoderSingleLayerH264` 类型的 `VideoEncoder` 对象。
最后，完成对 AVEncoder 的创建。WebRTC在内部调用 `FVideoEncoderSingleLayerH264` 对象的 `InitEncode` 函数时，创建了封装AVEncoder的 `FVideoEncoderWrapperHardware` 对象，并在 `AVEncoder::FVideoEncoder` 的 `SetOnEncodedPacket`回调函数中注册了 `FVideoEncoderWrapperHardware:: OnEncodedPacket` 以用于发送编码后的FrameBuffer。

### 3.	编码

编码过程发生在WebRTC的内部调用 `FVideoEncoderSingleLayerH264` 对象的 `encode()` 函数中，要将 WebRtc的 `VideoFrame` 转换成 `TextureBuffer` 并再次绑定到AVEncoder可以操作的 `FVideoEncoderInputFrame`，进而调用AVEncoder的 `encode()` 函数进行真正的编码。

### 4.	发送

发送过程发生在 `AVEncoder::FVideoEncoder` 编码完成后的回调函数中，也就是函数 `FVideoEncoderWrapperHardware::OnEncodedPacket()`。
由于发送利用的是WebRTC内部接口，因此需要将AVEncoder的编码后内容构建成WebRTC格式，最终再去调用 `FVideoEncoderSingleLayerH264` 的默认回调函数 `OnEncodedImageCallback->OnEncodedImage()` 来完成发送编码图像。
