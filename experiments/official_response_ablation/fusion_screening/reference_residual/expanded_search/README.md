# 扩展边界共同结构搜索

本轮把物理平滑尺度扩展为 `{1, 1.5, 2, 2.5, 3, 4}`，局部窗口半径扩展为 `{0.5, 1, 2, 3, 4, 6}`。默认用`run_expanded_fine.py`复现细阈值`0.01:0.01:0.99`；粗阈值对照`run_expanded.py`使用`0.05:0.05:0.95`。G、L、Mean、PositiveResidual和RefResidual各自进行留一被试内层full F1选参。

扩展 `a0` 时，core 在内存中使用封存实现同样的卷积宽度、峰检测、spread、alignment、局部归一化、median 和 missing=0 规则。没有修改封存源文件。`a0` 改变候选峰，所以同一 `a0` 下方法共享候选，跨 `a0` 的候选池可以不同。G 不依赖 rho，驱动只搜索一次每个 `(a0,tau)`。

细网格总配置数14,850：G594个，其他方法各3,564个。粗网格2,850个。Colab中断用EXP_RESUME续跑同一模式；不要跨模式复用输出目录。结果写到`MyDrive/GLSD_RESIDUAL_EXPANDED`。

`run_original_fine.py`为旧物理尺度集合及旧rho={1,2,3}的阈值加密对照。`supplement_no_global.py --dump 响应目录 --result 已完成结果目录 --output 新补充目录`重放预选参数下去G分数的消融，不重新选参。

输出保存所有配置的 raw/full 张量、选参、事件账本、D特征和接口审计。不要把固定结构诊断中最高的一格作为主结果，主结果只看每个方法预先声明的留一内层选参。

这轮仍是开发实验，不能声称独立验证。若扩展结构后 RefResidual 仍不能超过 PositiveResidual，下一步应改残差的定义或改为事件级去重/可靠性校正，而不是继续无限扩大网格。
