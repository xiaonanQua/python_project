"""
创建持久层
"""
import logging
import aiomysql
import asyncio
logging.basicConfig(level=logging.INFO)


@asyncio.coroutine
def create_pool(loop, **kw):
    """
    创建http连接池。
    创建一个全局的连接池，每个HTTP请求都可以从连接池中直接获取数据库连接。
    使用连接池的好处是不必频繁地打开和关闭数据库，而是能复用则复用。
    :param loop:循环
    :param kw:超参数
    :return:
    """
    logging.info("create database connection pool...")
    global __pool  # 定义全局连接池
    __pool = yield from aiomysql.create_pool(
        host=kw.get('host', 'localhost'),
        port=kw.get('port', 3306),
        user=kw['root'],
        password=kw['password'],
        db=kw['db'],
        charset=kw.get('charset', 'utf-8'),
        autocommit=kw.get('autocommit', True),
        maxsize=kw.get('maxsize', 10),
        minsize=kw.get('minsize', 1),
        loop=loop
        )


@asyncio.coroutine
def select(sql, args, size=None):
    """
    选择查询
    :param sql: sql语句
    :param args: SQL参数
    :param size:
    :return:
    """
    # log(sql, args)
    global __pool
    with (yield from __pool) as conn:
        cur = yield




if __name__ == '__main__':
    create_pool('s')