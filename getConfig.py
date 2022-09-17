# 引入configparser包
import configparser


# 定义读取配置文件函数
# 分别读取section的配置参数
# section包括ints、floats、strings
def get_config(config_file='config.txt'):
    parser = configparser.ConfigParser()
    parser.read(config_file)
    # 获取整型参数，按照key-value的形式保存
    _conf_ints = [(key, int(value)) for key, value in parser.items('ints')]
    # 获取浮点型参数，按照key-value的形式保存
    # _conf_floats = [(key, float(value)) for key, value in parser.items('floats')]
    # 获取字符型参数，按照key-value的形式保存
    _conf_strings = [(key, str(value)) for key, value in parser.items('strings')]
    # 返回字典类型
    # return dict(_conf_ints + _conf_floats + _conf_strings)
    return dict(_conf_ints + _conf_strings)

