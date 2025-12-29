# msModelSlim工具安装指南

## 安装说明
本文介绍msModelSlim工具的安装。

## 安装前准备
准备python环境：需要 Python 3.10 或更高版本。

## 安装步骤

注：使用 msmodelslim 命令行工具时，请不要在 msmodelslim 源码目录下执行 msmodelslim 命令，这样做可能会因为 Python 导入模块时源码路径和安装路径冲突，导致命令执行报错。

#### 基于Atlas A2 训练、推理产品，Atlas A3 训练、推理系列产品安装

```shell
# 1.git clone msmodelslim代码
git clone https://gitcode.com/Ascend/msmodelslim.git

# 2.进入到msmodelslim的目录并运行安装脚本
cd msmodelslim
bash install.sh
```

#### 基于Atlas 300I Duo 系列产品安装

前置条件：已安装CANN并设置环境变量

注意：

1.Atlas 300I Duo 卡仅支持单卡单芯片处理器量化。

2.如果需要进行稀疏量化和压缩，则需要安装CANN（8.2.RC1以上版本）：

下载[CANN](https://www.hiascend.com/developer/download/community/result?module=cann)，请根据系统选择aarch64或x86_64对应版本，具体安装方式请参考[CANN安装指南](https://www.hiascend.com/document/detail/zh/canncommercial/82RC1/softwareinst/instg/instg_0008.html?Mode=PmIns&InstallType=local&OS=Debian&Software=cannToolKit)。

```shell
# 1.git clone msmodelslim代码
git clone https://gitcode.com/Ascend/msmodelslim.git

# 2.进入到msmodelslim的目录并运行安装脚本
cd msmodelslim
bash install.sh

# 注：如果需要进行稀疏量化和压缩，则继续以下操作。
# 3.进入python环境下的site_packages包管理路径，其中${python_envs}为Python环境路径。
cd ${python_envs}/site-packages/msmodelslim/pytorch/weight_compression/compress_graph/  
# 以下是以/usr/local/为用户所在目录、Python版本为3.11.10为例
cd /usr/local/lib/python3.11/site-packages/msmodelslim/pytorch/weight_compression/compress_graph/

# 4.编译weight_compression组件，其中${install_path}为CANN软件的安装目录。
sudo bash build.sh ${install_path}/ascend-toolkit/latest

# 5.上一步编译操作会得到build文件夹，给build文件夹相关权限
chmod -R 550 build
```

## 安装后配置

如果是昇腾NPU设备，请参考如下配置

### CANN安装

下载[CANN](https://www.hiascend.com/developer/download/community/result?module=cann)，请根据系统选择aarch64或x86_64对应版本，具体安装方式请参考[CANN安装指南](https://www.hiascend.com/document/detail/zh/canncommercial/82RC1/softwareinst/instg/instg_0008.html?Mode=PmIns&InstallType=local&OS=Debian&Software=cannToolKit)。

### PTA安装

PyTorch安装请参考[Ascend Extension for PyTorch](https://www.hiascend.com/document/detail/zh/Pytorch/710/configandinstg/instg/insg_0004.html)配置与安装。

## 卸载

```
pip uninstall msmodelslim -y
```