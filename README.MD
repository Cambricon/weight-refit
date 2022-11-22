# weight_refit
- [weight\_refit](#weight_refit)
  - [1.介绍](#1介绍)
  - [2.前提条件](#2前提条件)
  - [3.环境准备](#3环境准备)
  - [4.运行程序](#4运行程序)

## 1.介绍
权值热更新，即在runtime期间，支持在不重新加载模型的前提下，更新模型中的权值。   
启用该功能需要在生成模型时，在build_config中加入，"enable_refit": true， 并获取到权值的名字，运行时期间，调用api接口重新设置权值，具体使用方法参考infer.cpp

## 2.前提条件

- Linux 常见操作系统版本(如 Ubuntu16.04，Ubuntu18.04，CentOS7.x 等)，安装 docker(>=v18.00.0)应用程序；
- 服务器装配好寒武纪 300 系列及以上的智能加速卡，并安装好驱动(>=v4.20.6)；
- 若不具备以上软硬件条件，可前往[寒武纪开发者社区](https://developer.cambricon.com/)申请试用;

## 3.环境准备

若基于寒武纪云平台环境可跳过该环节。否则需运行以下步骤：

1.请前往[寒武纪开发者社区](https://developer.cambricon.com/)下载 MagicMind(version >= 0.13.0)镜像，名字如下：

magicmind_version_os.tar.gz, 例如 magicmind_0.13.1-1_ubuntu18.04.tar.gz

2.加载：

```bash
docker load -i magicmind_version_os.tar.gz
```

3.运行：

```bash
docker run -it --name=dockername \
           --network=host --cap-add=sys_ptrace \
           -v /your/host/path/MagicMind:/MagicMind \
           -v /usr/bin/cnmon:/usr/bin/cnmon \
           --device=/dev/cambricon_dev0:/dev/cambricon_dev0 --device=/dev/cambricon_ctl \
           -w /MagicMind/ magicmind_version_image_name:tag_name /bin/bash
```

## 4.运行程序

1. 运行
```bash
# 下载模型
./download_model.sh
# 编译代码
./build.sh
# 生成模型
./gen_model.sh
# 执行推理
./run.sh
```
