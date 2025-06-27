作业步骤：

1. 下载HPatches 数据集，解压到指定目录。
目录格式如下：

hpatches
  - i_xx
    - 1.ppm
    - 2.ppm
    - ...
  - v_xx
    - 1.ppm
    - 2.ppm
    - ...

2. 运行datasets/hpatches.py 可视化数据集，查看数据集是否下载成功。

3. 按照代码中的提示，补全models/SuperPoint.py 中的代码，实现SuperPoint模型，参考 https://github.com/magicleap/SuperPointPretrainedNetwork

4. 按照代码中的提示，补全tasks/repeatability.py 中的代码，实现计算重复性的函数。
运行 main.py, 查看重复性的计算结果。
python3 main.py -c config/task_1.yaml


