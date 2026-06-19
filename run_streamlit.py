"""
Streamlit 启动器 — 修复 Python 3.9 + Windows 证书存储损坏问题

问题：tornado/netutil.py 在模块导入时调用
ssl.create_default_context() → load_default_certs()
加载 Windows 证书存储时遇到损坏数据抛出 SSLError。

解决：在导入 streamlit 前 monkey-patch SSLContext.load_default_certs
捕获异常静默处理。
"""

import ssl

_original_load_default_certs = ssl.SSLContext.load_default_certs


def _patched_load_default_certs(self, purpose=ssl.Purpose.SERVER_AUTH):
    try:
        _original_load_default_certs(self, purpose)
    except ssl.SSLError:
        # Windows 证书存储损坏，静默跳过
        # 使用 certifi 的证书包作为替代
        try:
            import certifi

            self.load_verify_locations(cafile=certifi.where())
        except Exception:
            pass


ssl.SSLContext.load_default_certs = _patched_load_default_certs

# 同时修补已创建默认 context（tornado 在 netutil 模块顶层调用）
_original_create_default_context = ssl.create_default_context


def _patched_create_default_context(purpose=ssl.Purpose.SERVER_AUTH, *, cafile=None, capath=None, cadata=None):
    try:
        return _original_create_default_context(purpose, cafile=cafile, capath=capath, cadata=cadata)
    except ssl.SSLError:
        # 回退：创建一个不使用 Windows 证书存储的 context
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        try:
            import certifi

            ctx.load_verify_locations(cafile=certifi.where())
        except Exception:
            pass
        return ctx


ssl.create_default_context = _patched_create_default_context

if __name__ == "__main__":
    import sys
    from streamlit.web.cli import main

    sys.argv[0] = "streamlit"
    main()
