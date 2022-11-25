# weight_refit
- [weight\_refit](#weight_refit)
  - [1.介绍](#1介绍)
  - [2.前提条件](#2前提条件)
  - [3.环境准备](#3环境准备)
    - [3.1 下载镜像](#31-下载镜像)
    - [3.2 加载镜像](#32-加载镜像)
    - [3.3 运行容器](#33-运行容器)
  - [4.使用方法](#4使用方法)
    - [4.1 生成模型阶段启用权值热更](#41-生成模型阶段启用权值热更)
    - [4.2 运行期调用接口更新权重](#42-运行期调用接口更新权重)
  - [5.运行sample](#5运行sample)
    - [5.1 编译代码](#51-编译代码)
    - [5.2 生成pytorch模型](#52-生成pytorch模型)
    - [5.3 pytorch模型转mm](#53-pytorch模型转mm)
    - [5.4 单线程sample](#54-单线程sample)
    - [5.5 多线程sample](#55-多线程sample)


## 1.介绍
权值热更新，即在runtime期间，支持在不重新加载模型的前提下，更新模型中的权值。   

## 2.前提条件

- Linux 常见操作系统版本(如 Ubuntu16.04，Ubuntu18.04，CentOS7.x 等)，安装 docker(>=v18.00.0)应用程序；
- 服务器装配好寒武纪 300 系列及以上的智能加速卡，并安装好驱动(>=v4.20.6)；
- 若不具备以上软硬件条件，可前往[寒武纪开发者社区](https://developer.cambricon.com/)申请试用;

## 3.环境准备

若基于寒武纪云平台环境可跳过该环节。否则需运行以下步骤：

### 3.1 下载镜像
请前往[寒武纪开发者社区](https://developer.cambricon.com/)下载 MagicMind(version >= 0.13.0)镜像，名字如下：

magicmind_version_os.tar.gz, 例如 magicmind_0.13.1-1_ubuntu18.04.tar.gz

### 3.2 加载镜像

```bash
docker load -i magicmind_version_os.tar.gz
```

### 3.3 运行容器

```bash
docker run -it --name=dockername \
           --network=host --cap-add=sys_ptrace \
           -v /your/host/path/MagicMind:/MagicMind \
           -v /usr/bin/cnmon:/usr/bin/cnmon \
           --device=/dev/cambricon_dev0:/dev/cambricon_dev0 --device=/dev/cambricon_ctl \
           -w /MagicMind/ magicmind_version_image_name:tag_name /bin/bash
```

## 4.使用方法
### 4.1 生成模型阶段启用权值热更            
启用该功能需要在生成模型阶段，在build_config中添加"enable_refit": true    
```bash
auto build_config = CreateIBuilderConfig();
build_config->ParseFromString(R"({"enable_refit": true})");
```

### 4.2 运行期调用接口更新权重
```c++
auto fit = magicmind::CreateIRefitter(engine);
int new_weight[9] = {9,8,7,6,5,4,3,2,1,0};
auto weight_tensor = magicmind::CreateIRTTensor(magicmind::DataType::INT32,
                                             "const_data",
                                             Layout::NONE,
                                             TensorLocation::kHost);
weight_tensor->SetDimensions(Dims({10}));
weight_tensor->SetData(&new_weight);
fit->SetNamedWeights("tensor_name", weight_tensor);
fit->RefitEngine();
```

## 5.运行sample
### 5.1 编译代码   
编译项目中的所有c++代码,其中包含pytorch模型转mm, 单线程sample和多线程sample。
```bash
./build.sh
```
### 5.2 生成pytorch模型    
gen_pytorch_model.py定义了,只有一个卷积层的模型,输入shape为(1,1,3,3),kernel size(3,3),stride(1,1),没有bias,权重初始化为全0，此时卷积的计算过程为，input和kernel对应位置相乘相加。output = input[i,j] * kernel[i,j], i,j = [1,2,3]。
```bash
python gen_pytorch_model.py
```
### 5.3 pytorch模型转mm     
```
./bin/gen_model
```
### 5.4 单线程sample
```bash
./bin/single_thread
```
该demo加载了模型的初始权重,即conv的权重全为0,此时计算结果等于0    
通过权值热更新接口,将卷积的权重设为全1,模型的输出变为9
```bash
input: 1 1 1  conv kernel: 0 0 0  result: 0
       1 1 1               0 0 0 
       1 1 1               0 0 0 
--- weight refit ---
input: 1 1 1  conv kernel: 1 1 1  result: 9
       1 1 1               1 1 1 
       1 1 1               1 1 1 
```
### 5.5 多线程sample
```bash
./bin/multi_thread
```
该demo启动了两个线程,线程1先启动，循环推理并打印结果,线程2通接口替换权值，从全0一直到全9(每秒+1), 控制台可以观察到输出结果的变化
```bash
-------------------------
input: 1 1 1  result: 0
       1 1 1  
       1 1 1  
... ... 
(a lot of log)
... ... 
-------------------------
input: 1 1 1  result: 9
       1 1 1  
       1 1 1  
... ... 
(a lot of log)
... ... 
-------------------------
input: 1 1 1  result: 18
       1 1 1  
       1 1 1  
... ... 
(a lot of log)
(a lot of log)
... ... 
-------------------------
input: 1 1 1  result: 81
       1 1 1  
       1 1 1  
```
