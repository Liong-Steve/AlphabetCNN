1 下载Anaconda，并安装

> 在用户目录下配置下载源
>
> [.condarc]
>
> ```
> channels:
>   - defaults
> show_channel_urls: true
> channel_alias: https://mirror.tuna.tsinghua.edu.cn/anaconda
> default_channels:
>   - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
>   - https://mirror.tuna.tsinghua.edu.cn/anaconda/pkgs/free
>   - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r
>   - https://mirror.tuna.tsinghua.edu.cn/anaconda/pkgs/pro
>   - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2
> custom_channels:
>   conda-forge: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
>   msys2: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
>   bioconda: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
>   menpo: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
>   pytorch: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
>   pytorch-lts: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
>   simpleitk: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud1
> ssl_verify: true
> ```

2 创建环境,并进入

> conda create -n TF2.1 python=3.7
> conda activate TF2.1

3 安装cudatoolkit、cudnn

> conda install cudatoolkit=10.1
> conda install cudnn=7.6

4 安装tensorflow，并验证

> pip install tensorflow==2.1
>
> python
>
> ```python
> import tensorflow as tf
> tf.__version__
> ```

5 PyCharm配置环境

>  进入Setting，选择解释器，选择导入创建的conda环境中对应的python.exe文件

6  your generated code is out of date and must be regenerated with protoc >= 3.19.0

```
pip uninstall protobuf
pip install protobuf==3.19.0
```

7 提示无法加载

```
2022-09-17 08:39:04.813958: W tensorflow/stream_executor/platform/default/dso_loader.cc:55] Could not load dynamic library 'cudart64_101.dll'; dlerror: cudart64_101.dll not found
2022-09-17 08:39:04.814333: I tensorflow/stream_executor/cuda/cudart_stub.cc:29] Ignore above cudart dlerror if you do not have a GPU set up on your machine.
WARNING:tensorflow:From D:/Python/AlphabetCNN/util/test.py:5: is_gpu_available (from tensorflow.python.framework.test_util) is deprecated and will be removed in a future version.
Instructions for updating:
Use `tf.config.list_physical_devices('GPU')` instead.
2022-09-17 08:39:07.326487: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2
2022-09-17 08:39:07.328963: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library nvcuda.dll
2022-09-17 08:39:07.359991: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1555] Found device 0 with properties: 
pciBusID: 0000:01:00.0 name: NVIDIA GeForce GTX 1060 computeCapability: 6.1
coreClock: 1.6705GHz coreCount: 10 deviceMemorySize: 6.00GiB deviceMemoryBandwidth: 178.99GiB/s
2022-09-17 08:39:07.370537: W tensorflow/stream_executor/platform/default/dso_loader.cc:55] Could not load dynamic library 'cudart64_101.dll'; dlerror: cudart64_101.dll not found
2022-09-17 08:39:07.380754: W tensorflow/stream_executor/platform/default/dso_loader.cc:55] Could not load dynamic library 'cublas64_10.dll'; dlerror: cublas64_10.dll not found
2022-09-17 08:39:07.391526: W tensorflow/stream_executor/platform/default/dso_loader.cc:55] Could not load dynamic library 'cufft64_10.dll'; dlerror: cufft64_10.dll not found
2022-09-17 08:39:07.402437: W tensorflow/stream_executor/platform/default/dso_loader.cc:55] Could not load dynamic library 'curand64_10.dll'; dlerror: curand64_10.dll not found
2022-09-17 08:39:07.413357: W tensorflow/stream_executor/platform/default/dso_loader.cc:55] Could not load dynamic library 'cusolver64_10.dll'; dlerror: cusolver64_10.dll not found
2022-09-17 08:39:07.423972: W tensorflow/stream_executor/platform/default/dso_loader.cc:55] Could not load dynamic library 'cusparse64_10.dll'; dlerror: cusparse64_10.dll not found
2022-09-17 08:39:07.435057: W tensorflow/stream_executor/platform/default/dso_loader.cc:55] Could not load dynamic library 'cudnn64_7.dll'; dlerror: cudnn64_7.dll not found
2022-09-17 08:39:07.435547: W tensorflow/core/common_runtime/gpu/gpu_device.cc:1592] Cannot dlopen some GPU libraries. Please make sure the missing libraries mentioned above are installed properly if you would like to use GPU. Follow the guide at https://www.tensorflow.org/install/gpu for how to download and setup the required libraries for your platform.
Skipping registering GPU devices...
2022-09-17 08:39:07.515368: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1096] Device interconnect StreamExecutor with strength 1 edge matrix:
```

[安装CUDA](https://developer.nvidia.com/cuda-toolkit-archive)

[下载CuDNN](https://developer.nvidia.com/rdp/cudnn-download)

将CuDNN中的内容分别复制到CUDA路径`C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1`下的文件夹中

[知乎解答](https://zhuanlan.zhihu.com/p/264161757)

[灰信网解答](https://www.freesion.com/article/85561272631/)

- 将`cudart64_101.dll`放在`C:\Windows\System32`
- cublas64_10.dll
- cufft64_10.dll
- curand64_10.dll
- cusolver64_10.dll
- cusparse64_10.dll
- cudnn64_7.dll 同上

链接：https://pan.baidu.com/s/1-P0eMFsyxRSA8erQZutDgQ?pwd=lion 
提取码：lion

![image-20220917111844597](https://raw.githubusercontent.com/Liong-Steve/blogImage/main/img/202209171119333.png)



8  No module named 'PIL'

```
pip install pillow
```



9 No module named 'PyQt5'

```
pip install PyQt5
```



10 No module named 'matplotlib'

```
pip install matplotlib
```



11 WARNING:absl:Found a different version 3.0.0 of dataset emnist in data_dir ../tensorflow_datasets. Using currently defined version 1.0.1.

```
pip install tensorflow-datasets==4.5.2
```

