# Web_LidarDetector_test 工程功能总结

这个工程是一个面向激光雷达点云的 3D 目标检测工程，核心用途是围绕自定义数据集完成数据预处理、模型训练、评估、预测，以及模型导出，主要服务于 PointPillars 检测链路，同时保留了一套 PointNet 相关配置与模板。

## 主要功能

1. **点云数据预处理**
   - 支持原始激光雷达数据集的划分、过滤、加载和中间结果生成。
   - 支持生成训练/测试所需的数据配置。
   - 支持将原始点云整理成模型可消费的输入格式。
   - 在旧流程中还包含 cluster 数据集生成等能力，主要用于 PointNet 流程。

2. **3D 目标检测模型训练**
   - 提供统一训练入口 `Z_rectified_srcipts/train_flow.py`。
   - 通过 `LidarDetector` 串联数据模块与模型模块，完成 epoch 级训练、验证与模型保存。
   - 支持学习率调度、断点续训、冻结部分层、损失记录等训练控制能力。
   - 当前重点模型为 PointPillars，多类别检测配置覆盖 Car、Bus、Tricycle、Mbike、Pedestrian 等类别。

3. **模型评估与结果统计**
   - 支持验证集/测试集评估。
   - 支持基于 IoU 的检测关联与指标统计。
   - 支持将 loss 和评估结果保存到结果目录，供后续展示使用。

4. **预测与结果落盘**
   - 提供预测流程 `Z_rectified_srcipts/model_predict.py`。
   - 可对待预测点云执行推理，并将结果转换为指定 JSON 标注格式。
   - 预测结果会和原始点云文件一起拷贝到输出目录，便于下游平台或可视化工具消费。

5. **PointPillars 模型实现**
   - `PointPillars/model/model/networks.py` 实现了完整网络拼装。
   - 网络包含 VFE、PFN、Scatter、2D Backbone、Deblock、SharedConv、多检测头和 Merge 等模块。
   - 支持将体素化点云转换为 BEV 特征，并输出分类与框回归结果。

6. **模型导出能力**
   - 支持把训练好的 PointPillars 模型拆分导出为多个 ONNX 子模型。
   - 当前导出粒度包括：
     - `vfe.onnx`
     - `backbone2D.onnx`
     - `rpn.onnx`
   - 这种拆分方式说明该工程不仅关注训练，也兼顾部署侧的分模块推理需求。

7. **模板化交付能力**
   - `user_download_template` 提供了一套可交付给外部用户或平台的算法模板目录。
   - 模板中包含数据处理、模型计算、后处理、评估、网络定义等骨架代码，便于二次集成。

## 工程结构理解

- `web_lidardetector/`
  - 通用训练/数据/模型框架。
  - 包含数据加载、评估基类、模型组件封装、接口定义等。

- `PointPillars/`
  - 具体的 PointPillars 算法实现。
  - 包括网络结构、后处理、损失、评估与模型导出。

- `Z_rectified_srcipts/`
  - 面向实际运行的主入口脚本。
  - 用于训练、测试、预测等流程调度。

- `user_download_template/`
  - 对外导出的算法模板示例。

## 总结

从功能上看，这个工程不是单纯的“模型定义仓库”，而是一套较完整的激光雷达 3D 检测工作流工程。它覆盖了数据准备、训练、评估、预测、结果格式转换和 ONNX 导出，重点面向 PointPillars 模型在自定义数据集上的训练与部署。