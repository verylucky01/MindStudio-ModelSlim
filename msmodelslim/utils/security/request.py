#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
-------------------------------------------------------------------------
This file is part of the MindStudio project.
Copyright (c) 2025 Huawei Technologies Co.,Ltd.

MindStudio is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:

         http://license.coscl.org.cn/MulanPSL2

THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details.
-------------------------------------------------------------------------
"""

import re
import socket
from urllib.parse import urljoin, urlparse

import requests

from msmodelslim.utils.exception import SecurityError


def validate_safe_host(host: str, field_name: str = "host") -> str:
    """
    验证主机地址，防止 SSRF 攻击。
    只允许 localhost、127.0.0.1 或内网 IP 地址（RFC 1918）。
    
    Args:
        host: 要验证的主机地址
        field_name: 字段名称，用于错误消息，默认为 "host"
        
    Returns:
        验证后的主机地址（标准化）
        
    Raises:
        SecurityError: 如果 host 为空或不在允许的范围内
    """
    if not host:
        raise SecurityError(
            f"{field_name} cannot be empty.",
            action=f"Please provide a non-empty value for {field_name}."
        )
    
    # 允许 localhost 及其变体
    if host.lower() in ('localhost', '127.0.0.1', '::1', '[::1]'):
        return host.lower()
    
    # 验证是否为有效的 IP 地址格式
    try:
        ip = socket.gethostbyname(host)
        # 检查是否为内网地址（RFC 1918）
        # 10.0.0.0/8, 172.16.0.0/12, 192.168.0.0/16
        parts = ip.split('.')
        if len(parts) == 4:
            first_octet = int(parts[0])
            second_octet = int(parts[1])
            if (first_octet == 10 or
                (first_octet == 172 and 16 <= second_octet <= 31) or
                (first_octet == 192 and second_octet == 168)):
                return host
    except (ValueError, OSError):
        pass
    
    # 如果验证失败，抛出安全错误
    raise SecurityError(
        f"{field_name} '{host}' is not allowed. Only localhost or private network addresses are permitted.",
        action=f"Please use 'localhost', '127.0.0.1', or "
               f"a private IP address (10.x.x.x, 172.16-31.x.x, 192.168.x.x) for {field_name}."
    )


def validate_safe_endpoint(endpoint: str, field_name: str = "endpoint") -> str:
    """
    验证端点路径，防止路径遍历攻击。
    
    Args:
        endpoint: 要验证的端点路径
        field_name: 字段名称，用于错误消息，默认为 "endpoint"
        
    Returns:
        验证后的端点路径
        
    Raises:
        SecurityError: 如果 endpoint 为空、格式不正确或包含不安全字符
    """
    if not endpoint:
        raise SecurityError(
            f"{field_name} cannot be empty.",
            action=f"Please provide a non-empty value for {field_name}."
        )
    
    # 确保以 / 开头
    if not endpoint.startswith('/'):
        raise SecurityError(
            f"{field_name} must start with '/'.",
            action=f"Please provide a valid absolute path starting with '/' for {field_name}."
        )
    
    # 防止路径遍历攻击（../）
    if '..' in endpoint or endpoint.startswith('//'):
        raise SecurityError(
            f"{field_name} '{endpoint}' contains invalid path components.",
            action=f"Please use a valid absolute path starting with '/' for {field_name}."
        )
    
    # 验证路径只包含安全字符
    if not re.match(r'^/[a-zA-Z0-9_\-/]*$', endpoint):
        raise SecurityError(
            f"{field_name} '{endpoint}' contains invalid characters.",
            action=f"Please use only alphanumeric characters, hyphens, underscores, and slashes for {field_name}."
        )
    
    return endpoint


def build_safe_url(host: str, port: int, endpoint: str, scheme: str = 'http') -> str:
    """
    安全地构建 URL，防止 URL 注入攻击。
    
    Args:
        host: 主机地址（会被验证）
        port: 端口号
        endpoint: 端点路径（会被验证）
        scheme: URL scheme，默认为 'http'
        
    Returns:
        构建的安全 URL
        
    Raises:
        SecurityError: 如果 URL 构建失败或验证不通过
    """
    # 验证 host 和 endpoint
    validated_host = validate_safe_host(host, field_name="host")
    validated_endpoint = validate_safe_endpoint(endpoint, field_name="endpoint")
    
    # 验证 scheme
    if scheme not in ('http', 'https'):
        raise SecurityError(
            f"Invalid URL scheme: {scheme}",
            action="Only http and https schemes are allowed."
        )
    
    # 使用 urllib.parse 安全地构建 URL
    base_url = f"{scheme}://{validated_host}:{port}"
    # urljoin 会自动处理路径拼接，防止路径注入
    url = urljoin(base_url, validated_endpoint)
    
    # 验证最终 URL 的安全性
    parsed = urlparse(url)
    if parsed.scheme not in ('http', 'https'):
        raise SecurityError(
            f"Invalid URL scheme: {parsed.scheme}",
            action="Only http and https schemes are allowed."
        )
    
    # 确保 host 没有被篡改（双重验证）
    if parsed.hostname and parsed.hostname != validated_host:
        raise SecurityError(
            f"URL hostname mismatch: {parsed.hostname} != {validated_host}",
            action="URL construction validation failed."
        )
    
    return url


def safe_get(
    url: str,
    timeout: float = 3.0,
    allow_redirects: bool = False,
    verify: bool = True,
    stream: bool = False,
    **kwargs
) -> requests.Response:
    """
    执行安全的 GET 请求，防止 SSRF 和其他网络攻击。
    
    此函数会：
    - 验证 URL 的安全性（scheme、hostname）
    - 使用安全配置（禁用重定向、验证 SSL、设置超时）
    - 提供详细的错误处理
    
    Args:
        url: 请求的 URL（应该已经通过 build_safe_url 构建）
        timeout: 请求超时时间（秒），默认 3.0
        allow_redirects: 是否允许重定向，默认 False（防止重定向攻击）
        verify: 是否验证 SSL 证书，默认 True
        stream: 是否流式传输，默认 False（避免大响应）
        **kwargs: 其他 requests.get() 的参数
        
    Returns:
        requests.Response 对象
        
    Raises:
        SecurityError: 如果 URL 不安全
        requests.RequestException: 请求相关的异常
    """
    # 验证 URL 的基本安全性
    parsed = urlparse(url)
    
    # 验证 scheme
    if parsed.scheme not in ('http', 'https'):
        raise SecurityError(
            f"Invalid URL scheme: {parsed.scheme}",
            action="Only http and https schemes are allowed."
        )
    
    # 验证 hostname（如果 URL 不是通过 build_safe_url 构建的）
    if parsed.hostname:
        try:
            validate_safe_host(parsed.hostname, field_name="URL hostname")
        except SecurityError as e:
            raise SecurityError(
                f"URL hostname '{parsed.hostname}' is not safe: {e}",
                action="Please use build_safe_url() to construct URLs."
            ) from e
    
    # 执行请求，使用安全配置
    return requests.get(
        url,
        timeout=timeout,
        allow_redirects=allow_redirects,
        verify=verify,
        stream=stream,
        **kwargs
    )


def safe_post(
    url: str,
    json: dict = None,
    timeout: float = 10.0,
    allow_redirects: bool = False,
    verify: bool = True,
    **kwargs
) -> requests.Response:
    """
    执行安全的 POST 请求，防止 SSRF 和其他网络攻击。
    
    此函数会：
    - 验证 URL 的安全性（scheme、hostname）
    - 使用安全配置（禁用重定向、验证 SSL、设置超时）
    - 提供详细的错误处理
    
    Args:
        url: 请求的 URL（应该已经通过 build_safe_url 构建）
        json: 要发送的 JSON 数据
        timeout: 请求超时时间（秒），默认 10.0
        allow_redirects: 是否允许重定向，默认 False（防止重定向攻击）
        verify: 是否验证 SSL 证书，默认 True
        **kwargs: 其他 requests.post() 的参数
        
    Returns:
        requests.Response 对象
        
    Raises:
        SecurityError: 如果 URL 不安全
        requests.RequestException: 请求相关的异常
    """
    # 验证 URL 的基本安全性
    parsed = urlparse(url)
    
    # 验证 scheme
    if parsed.scheme not in ('http', 'https'):
        raise SecurityError(
            f"Invalid URL scheme: {parsed.scheme}",
            action="Only http and https schemes are allowed."
        )
    
    # 验证 hostname（如果 URL 不是通过 build_safe_url 构建的）
    if parsed.hostname:
        try:
            validate_safe_host(parsed.hostname, field_name="URL hostname")
        except SecurityError as e:
            raise SecurityError(
                f"URL hostname '{parsed.hostname}' is not safe: {e}",
                action="Please use build_safe_url() to construct URLs."
            ) from e
    
    # 执行请求，使用安全配置
    return requests.post(
        url,
        json=json,
        timeout=timeout,
        allow_redirects=allow_redirects,
        verify=verify,
        **kwargs
    )
