# 权值热更新
1. 介绍   
权值热更新，即在runtime期间，支持在不重新加载模型的前提下，更新模型中的权值。   
启用该功能需要在生成模型时，在build_config中加入，"enable_refit": true， 并获取到权值的名字，运行时期间，调用api接口重新设置权值，具体使用方法参考infer.cpp

2. 模型下载
模型来源于[github](https://github.com/soeaver/caffe-model), 请下载resnet18-priv模型的prototxt文件和caffemodel文件，放置于dada文件夹下

3. 运行
```bash
# 编译代码
./build.sh
# 生成模型
./gen_model.sh
# 执行推理
./run.sh
```
