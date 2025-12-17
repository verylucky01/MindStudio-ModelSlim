## 公共接口

### 功能说明
使用msModelSlim工具过程中，过程日志信息以及日志级别，可以通过`set_logger_level`接口设置。其中日志包括打印在屏幕上的日志。

该接口为可选配置，如果不设置，则按照默认日志级别，默认级别为INFO。

### 函数原型
```python
set_logger_level(level="info")
```

### 参数说明
| 信息级别 | 含义 |
| ------ | ------ |
| notset | 不设置日志级别，默认打印所有级别的日志信息。 |
| debug | 打印debug、info、warn/warning、error、fatal和critical级别的日志信息。 |
| info | 打印info、warn/warning、error、fatal和critical级别的日志信息。|
| warn | 打印warn/warning、error、fatal和critical级别的日志信息。 |
| warning | 打印warn/warning、error、fatal和critical级别的日志信息。 |
| error | 打印error、fatal和critical级别的日志信息。 |
| fatal | 打印fatal和critical级别的日志信息。 |
| critical | 打印fatal和critical级别的日志信息。|

信息级别不区分大小写，即Info、info、INFO均为有效取值。

### 调用示例
```python
from msmodelslim import set_logger_level  
set_logger_level("info")  
```