"""
实现异步操作
"""
import logging; logging.basicConfig(level=logging.INFO)  # 导入日志库，并设置日志显示的层是信息层

import asyncio, os, json, time
from datetime import datetime

from aiohttp import web


def index(request):
    return web.Response(body=b'<h1>Hello,xiaonan</h1>')


@asyncio.coroutine
def init(loop):
    app =web.Application(loop=loop)
    app.router.add_route('Get', '/', index)  # 设置路由请求方式get,响应路径
    srv = yield from loop.create_server(app.make_handler(), '127.0.0.1', 9001)  # 设置服务路径和端口
    logging.info('server started at http://127.0.0.1:9000')
    return srv


loop = asyncio.get_event_loop()  # 获得异步活动循环
loop.run_until_complete(init(loop))
loop.run_forever()