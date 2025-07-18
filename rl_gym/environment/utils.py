# environment/utils.py (一个建议的新文件，用于存放辅助函数)

import orekit
from orekit.pyhelpers import setup_orekit_cur_jar

def init_orekit():
    """
    初始化 Orekit VM 并下载所需的数据。
    在任何 Orekit 操作之前，必须在主程序入口调用此函数一次。
    """
    # orekit.initVM() # 老版本用法
    # setup_orekit_cur_jar() # 推荐的新版本用法，它会自动处理VM和数据
    # 为了兼容性，我们假设用户已经通过某种方式初始化了orekit
    # 比如在主程序开头执行了下面这句
    # orekit.pyhelpers.download_orekit_data_cur()
    # 为确保代码能运行，我们在这里调用它，但实际项目中应放在main.py
    try:
        # 检查 orekit vm 是否已经启动
        orekit.JArray_double(1)
    except Exception:
        # 如果没有，则初始化
        vm = orekit.initVM()
        # 如果需要下载数据，可以取消下面的注释
        # from orekit.pyhelpers import download_orekit_data_cur
        # download_orekit_data_cur()
    print("Orekit VM is running.")