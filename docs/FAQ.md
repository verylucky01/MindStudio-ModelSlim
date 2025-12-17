# FAQ
## 1. 为什么我的程序会显示'Killed'并异常退出？

在使用msModelSlim工具运行推理量化时，出现类似以下报错信息：
```
Killed
...
[Error] TBE Subprocess[task_distribute] raise error[], main process disappeared!
...
```
请先确认你的进程没有被其他用户kill或抢占同一个NPU资源。一般而言，如果不存在其他用户抢占系统资源的情况，那么可能就是NPU显存不足或系统内存不足导致。可通过以下命令查看系统日志、看管系统内存情况、清理系统内存。
```shell
# dmesg查看被内核终止的进程或显存不足终止的进程
dmesg | grep -A 3 -B 1 -i "killed process\|oom-kill"

# 看管系统内存
watch free -h

# 清理缓存和内存，部分场景可能需要sudo权限
sync && echo 3 > /proc/sys/vm/drop_caches

# 停止所有python进程，部分场景可能需要sudo权限
pkill python
```


## 2. 为什么安装时提示pydantic版本冲突？

msModelSlim依赖pydantic>=2.10.1，请确保环境中的pydantic版本满足此要求。

### pip安装时的错误信息
```
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
check-wheel-contents 0.6.0 requires pydantic~=2.0, but you have pydantic 1.0 which is incompatible.
```

### 解决方法

请尝试升级pydantic或卸载环境中依赖低版本pydantic的其他软件包，直至环境无版本冲突。