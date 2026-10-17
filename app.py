#!/usr/bin/env python3
"""
半导体晶圆缺陷检测系统 - 启动入口
统一启动后端服务和前端界面
"""
import sys
import os
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

# 禁止写入 __pycache__（避免对环境目录的写入需求）
sys.dont_write_bytecode = True

# 项目自带的后端依赖目录（Web 依赖随项目分发，无需污染 Python 环境）
DEPS_DIR = ROOT / ".deps"
if DEPS_DIR.exists():
    sys.path.insert(0, str(DEPS_DIR))

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
import uvicorn


def create_app():
    """创建完整应用（后端API + 前端静态文件）"""
    # 导入后端主应用
    from backend.main import app as backend_app
    
    # 挂载前端静态文件（根路径由 backend/main.py 重定向到前端）
    backend_app.mount(
        "/frontend",
        StaticFiles(directory=str(ROOT / "frontend"), html=True),
        name="frontend"
    )

    # 前端页面禁用浏览器缓存，保证更新即时生效
    @backend_app.middleware("http")
    async def no_cache_frontend(request, call_next):
        response = await call_next(request)
        if request.url.path == "/frontend" or request.url.path.endswith(".html"):
            response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        return response

    return backend_app


def main():
    """主入口"""
    print("=" * 60)
    print("  半导体晶圆缺陷检测系统")
    print("  Semiconductor Wafer Defect Detection System")
    print("=" * 60)
    print()
    print("正在启动服务...")
    print()
    
    app = create_app()
    
    print("服务地址:")
    print("  - 前端界面:  http://localhost:8000/")
    print("  - API 文档:  http://localhost:8000/docs")
    print("  - 健康检查:  http://localhost:8000/api/health")
    print()
    print("按 Ctrl+C 停止服务")
    print("-" * 60)
    print()
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")


if __name__ == "__main__":
    main()
