import os
import ccxt
from dotenv import load_dotenv

load_dotenv()

# 配置代理 - 请将端口号替换成你代理软件的端口
proxies = {
    'http': os.environ.get('PROXY_HOST') + ':' + os.environ.get('PROXY_PORT'),   # 通常是7890或1080/10809
    'https': os.environ.get('PROXY_HOST') + ':' + os.environ.get('PROXY_PORT')   # https也使用http代理即可
}

# 初始化交易所时传入代理配置
exchange = ccxt.binance({
    'apiKey': os.environ.get('BINANCE_API_KEY'),
    'secret': os.environ.get('BINANCE_SECRET'),
    'enableRateLimit': True,
    'proxies': proxies,  # 关键：加入代理设置
    'options': {
        'adjustForTimeDifference': True,
    }
})

# 可选：增加超时时间（默认10秒可能不够）
exchange.timeout = 30000  # 设置为30秒，单位毫秒

try:
    print("正在通过代理连接币安...")
    balance = exchange.fetch_balance()
    print("连接成功！账户总价值（USDT 估值）:", balance['total'].get('USDT', 0))
except ccxt.RequestTimeout as e:
    print("请求超时，请检查代理设置和网络连接")
    print("错误详情:", e)
except Exception as e:
    print("其他错误:", e)