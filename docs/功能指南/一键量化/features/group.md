# GroupProcessor 分组处理器

## 概述

对多个处理器进行分组管理，支持对同一组内的处理器进行统一的配置管理。

## 使用场景

- 当需要对不同类型的层应用不同的量化策略时，可以使用group进行分组管理，可以降低资源消耗。
- 例如：对self_attention层使用静态量化，对mlp层使用动态量化。

## YAML配置示例

### W8A8混合量化配置

```yaml
default_w8a8: &default_w8a8
  act:
    scope: "per_tensor"   
    dtype: "int8"        
    symmetric: False      
    method: "minmax"     
  weight:
    scope: "per_channel"   
    dtype: "int8"       
    symmetric: True       
    method: "minmax"     

default_w8a8_dynamic: &default_w8a8_dynamic
  act:
    scope: "per_token"   
    dtype: "int8"        
    symmetric: True      
    method: "minmax"      
  weight:
    scope: "per_channel"   
    dtype: "int8"        
    symmetric: True       
    method: "minmax"
    
spec:
  process:
    - type: "group"
      configs:
        - type: "linear_quant"
          qconfig: *default_w8a8
          include: ["*self_attn*"]
        - type: "linear_quant"
          qconfig: *default_w8a8_dynamic
          include: ["*mlp*"]
          exclude: ["*gate"]
```

## YAML配置字段详解

| 字段名 | 作用 | 说明 |
|--------|------|------|
| type | 处理器类型标识 | 固定值"group"，用于标识这是一个分组处理器 |
| configs | 组内处理器配置列表 | 包含多个处理器配置，支持不同类型处理器组合 |
