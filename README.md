用MiniConda安装
Windows需要安装Visual C++ [VC_redist.x64]





1. numpy报错，重新安装numpy
```
    pip uninstall -y numpy
    pip uninstall -y setuptools
    ---
    pip install setuptools
    pip install numpy
    pip install numpy==1.22.0
```
2. 找不到tensorflow模块
```
    pip install tensorflow-gpu==2.6.0
```
3. A NumPy version >=1.16.5 and <1.23.0 is required for this version of SciPy (detected version 1.23.3
  warnings.warn(f"A NumPy version >={np_minversion} and <{np_maxversion}"
```
    pip uninstall numpy
    pip install numpy==1.22.0
```
4. Could not load dynamic library 'cudart64_110.dll'
```
  https://blog.csdn.net/u013421629/article/details/124627858
  

```




常见命令
```
  1) 查看以创建的虚拟环境： conda info --envs / conda env list
  2) 激活创建的环境：conda activate xxx(虚拟环境名称)
  3) 退出激活的环境：conda deactivate
  4) 删除一个已有虚拟环境：conda remove --name(已创建虚拟环境名称) tensorflow --all
  5) 创建一个新的虚拟环境：conda create --name tensorflow python=3.7.3
  6) 查看安装的包：conda list/pip list
  7) 可以通过克隆的方式更改创建虚拟环境的名称：conda create --name 新名字 --clone 已创               建虚拟环境名称
  8) 安装包：pip install scipy -i https://pypi.douban.com/simple

                        或者 conda install scipy -i https://pypi.douban.com/simple
  9) 卸载包：pip uninstall xxxx；conda uninstall xxxx
  10) 检查conda版本：conda --version
  11) 安装tensorflow的脚本：pip install tensorflow-cpu2.3.0 -i https://pypi.douban.com/simple/
  12) 更新升级工具包：conda upgrade --all
  13) 升级pip：python -m pip install --upgrade pip
  14) 使用镜像源网站安装需要的库：pip install -i https://pypi.tuna.tsinghua.edu.cn/simple/                     tensorflow2.0.0 安装其他库则只需将后面的库替换掉即可
  15)  常用镜像网站：
    阿里云      http://mirrors.aliyun.com/pypi/simple/
    中国科技大学   https://pypi.mirrors.ustc.edu.cn/simple/
    豆瓣(douban)   http://pypi.douban.com/simple/
    清华大学     https://pypi.tuna.tsinghua.edu.cn/simple/
    中国科学技术大学 http://pypi.mirrors.ustc.edu.cn/simple/
————————————————
版权声明：本文为CSDN博主「彩色海绵」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
原文链接：https://blog.csdn.net/m0_63172128/article/details/124217796
```