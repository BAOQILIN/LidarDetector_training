## HX 数据集说明与适配方案

### 1. 数据集真实情况

新的 HX 数据集位于：

`/home/bql/ARS/ARS_Data/ars_hx_train_data/origin_type`

我已经实际核对过这个目录，结论如下：

- 这是一个**扁平目录**，不是 `Lidar/`、`Label/` 这种分层结构。
- 目录下直接存放大量同名配对文件：
  - `时间戳.json`
  - `时间戳.pcd`
- 当前数据集中 `.json` 与 `.pcd` 是按同一 stem 一一对应的。

示例：

```text
/home/bql/ARS/ARS_Data/ars_hx_train_data/origin_type/
├── 1733898949738.json
├── 1733898949738.pcd
├── 1733898949938.json
├── 1733898949938.pcd
├── 1733898950139.json
├── 1733898950139.pcd
├── ...
```

说明：你原文中的 `173389895013910.pcd` 是笔误，实际应为 `1733898950139.pcd`。

---

### 2. PCD 文件格式核对结果

当前 HX 数据集中的 `.pcd` 文件可以被归类为 **ASCII PCD v0.7**，这一点与你原文判断基本一致。

不过，原文示例有一处需要修正：真实样本里 intensity 字段并不是你示例中那种 `4-byte float`，而更接近下面这种头部：

```text
# .PCD v0.7 - Point Cloud Data file format
VERSION 0.7
FIELDS x y z intensity
SIZE 4 4 4 1
TYPE F F F U
COUNT 1 1 1 1
...
DATA ascii
```

也就是说：

- `x y z` 是浮点
- `intensity` 更像是无符号整型字段
- 当前工程如果只取前 4 列作为点特征，整体上仍然是可适配的

---

### 3. 坐标系说明

你文档中写的是：

- 已统一到 LiDAR / 车辆坐标系
- `x` 前、`y` 左、`z` 上
- 原始数据中的 pcd 已经转换到了车辆坐标系

这个描述**目前可以先作为适配假设保留**，但还不能只凭文档完全确认。后续真正接入训练前，仍然需要做一次抽样验证，重点看：

- 框中心点位置是否落在合理 ROI 内
- 朝向角 `yaw` 是否与点云朝向一致
- 长宽高的定义是否与当前 PointPillars 训练逻辑一致

---

### 4. 原始 JSON 标签格式核对结果

你给出的 HX 标签 JSON 结构与实际抽样结果是一致的，训练真正需要关注的是：

- 顶层目标列表：`movingObjects`
- 每个目标的类别：`objectType`
- 三维框标注：`annotationTool.cuboid3D`
- 尺寸：`cuboidExtent`
- 中心点：`position`
- 航向角：`orientation[2]`

也就是训练阶段主要从每个目标提取：

- `objectType`
- `cuboidExtent`
- `position`
- `orientation[2]`

说明：你原文里的 `rorientation` 是笔误，应改成 `orientation`。

---

### 5. 当前工程真正需要的数据合同

当前工程的训练主链路并不直接消费 HX 原始 JSON。

真正被训练流程使用的是 `PointPillars/algo/data_preprocessor.py` 预处理后生成的 pickle 样本，其结构是：

```python
{
  'lidar_path': str,
  'gt_boxes': np.ndarray,   # [N, 7] -> [x, y, z, dx, dy, dz, heading]
  'gt_names': np.ndarray,   # [N] -> 类别名
  'num_lidar_pts': np.ndarray
}
```

因此，HX 数据集适配的关键不是去改训练器，而是：

**把 HX 的 `.pcd + .json` 原始格式转换成上述中间结构。**

---

### 6. 当前工程对旧数据格式的假设

当前 `PointPillars/algo/data_preprocessor.py` 默认假设旧数据满足：

```text
data_root/
├── Lidar/
├── Label/
│   ├── label.json
│   ├── xxx.json
│   └── ...
```

并且标签是旧平台格式：

- `label.json` 提供 `label_id -> class_name` 映射
- 每帧标签中通过 `annotation.annotation[]` 读取目标

这和 HX 数据集完全不同，所以不能直接复用旧预处理逻辑。

---

### 7. HX 数据集适配总体策略

推荐采用：

**只改预处理层，不大改训练/损失/评估主链路。**

也就是：

1. 在 `PointPillars/algo/data_preprocessor.py` 中增加 HX 数据分支
2. 让 HX 原始标注转换成当前训练流程已经接受的 pickle 样本
3. 尽量不改下面这些文件：
   - `PointPillars/algo/data_dataset.py`
   - `web_lidardetector/DataUtils/data_loader.py`
   - `web_lidardetector/train_flow.py`
   - `PointPillars/algo/loss_computers.py`
   - `PointPillars/algo/data_evaluater.py`

这样风险最小，也最符合当前代码结构。

---

### 8. 推荐的类别映射方案

当前工程里固定的训练类别是：

- `Pedestrian`
- `Mbike`
- `Car`
- `Bus`
- `Tricycle`

而在 HX 数据集的实际抽样中，我已经看到这些原始类别：

- `Pedestrian`
- `Bicycle`
- `Car`
- `Heavy Truck`
- `Bus`

因此推荐先采用下面这套映射：

| HX 原始类别 | 映射到训练类别 | 说明 |
|---|---|---|
| Pedestrian | Pedestrian | 直接映射 |
| Bicycle | Mbike | 归并到骑行类 |
| Car | Car | 直接映射 |
| Heavy Truck | Bus | 归并到大车类 |
| Bus | Bus | 直接映射 |

说明：

- 当前数据里暂时没看到 `Tricycle`，所以这个类别可以先保留在模型类别集合中，但允许训练集为空类。
- 不在这张映射表中的 HX 类别，建议训练阶段直接过滤。

---

### 9. 你原文类别表里需要修正的地方

你原文中的类别说明表存在不一致，建议修正为明确的“训练阶段归并关系”：

- `Heavy Truck -> Bus`：合理，可以保留
- `Bus -> Bus`：合理
- 但你原表最后一行中文备注写成了“三轮车”，这是错误的，应改成“大巴/公交车”或直接写“Bus”

---

### 10. 具体适配实施方案

#### Phase 1：在预处理配置中增加 HX 数据源标志
建议在：

- `PointPillars/algo/algo_config.yaml`

的 `PREPROCESS` 段增加一个数据风格配置，例如：

```yaml
PREPROCESS:
  DATASET_STYLE: ['hx_flat', 1, '数据集风格：旧平台/扁平HX']
```

用途：

- `old_platform`：旧 `Lidar/Label/label.json` 模式
- `hx_flat`：HX 扁平目录模式

#### Phase 2：在 `data_preprocessor.py` 中增加 HX 分支
核心文件：

- `PointPillars/algo/data_preprocessor.py`

建议改造思路：

1. 当 `DATASET_STYLE == hx_flat` 时：
   - 扫描目录中全部 `.pcd`
   - 查找同 stem 的 `.json`
   - 成功配对后进入样本构建

2. 训练/测试阶段：
   - 读取 HX `.json`
   - 遍历 `movingObjects`
   - 只保留 `cuboid3D.flag == 1` 的对象
   - 抽取：
     - `position`
     - `cuboidExtent`
     - `orientation[2]`
     - `objectType`
   - 执行类别映射
   - 生成：
     - `gt_boxes`
     - `gt_names`
     - `num_lidar_pts`

3. 预测阶段：
   - 只保留 `lidar_path`
   - `gt_boxes`、`gt_names`、`num_lidar_pts` 置空

#### Phase 3：保持下游训练链路不变
尽量保持以下逻辑不动：

- `PointPillars/algo/data_dataset.py` 继续从 pickle 读 `gt_boxes/gt_names`
- `loss_computers.py` 继续使用固定 5 类
- `data_evaluater.py` 继续按固定类别评估

只有在后续验证发现不兼容时，再决定是否改 evaluator 或 dataset loader。

---

### 11. 风险点

#### 11.1 类别分布风险
- HX 数据中可能没有 `Tricycle`
- `Heavy Truck` 被并入 `Bus` 后，会影响这一类 anchor 和评估语义

#### 11.2 PCD 解析鲁棒性风险
当前工程已经改成动态识别 PCD 头部中的 `DATA` 行，再读取点数据，不再依赖固定 `skiprows=11`。

但仍然要注意：

- PCD 必须是 ASCII 格式
- 必须至少有 4 列点特征
- 如果个别文件损坏，预处理会跳过并在日志中记录

#### 11.3 坐标系风险
虽然当前文档写的是已经转到车体坐标系，但执行阶段必须抽样核对，不然会直接影响框位置与朝向训练质量。

#### 11.4 输出格式风险
当前 `PointPillars/algo/data_postprocessor.py` 仍然更偏向旧平台的输出结构。

所以：

- **本次适配优先解决训练数据输入**
- 如果后续还要让预测结果回写成 HX 风格 JSON，需要单独再做一轮输出适配

---

### 12. 验证步骤

完成适配后，建议按下面顺序验证：

#### 12.1 静态样本验证
抽样读取若干对 `.pcd/.json`：

- 打印类别分布
- 打印框数量
- 打印 `position / extent / yaw` 的范围
- 确认映射后的类别都属于：
  - `Pedestrian`
  - `Mbike`
  - `Car`
  - `Bus`
  - `Tricycle`

#### 12.2 预处理验证
运行 HX 预处理，确认生成：

- `training.pkl`
- `validation.pkl`
- `testing.pkl`
- `prediction.pkl`

并检查 pickle 单条样本结构是否正确。

#### 12.3 数据加载验证
直接实例化当前 `data_dataset.py`，跑一个 batch，确认：

- 点云能正常 voxelize
- `gt_boxes` 维度正确
- 类别不会越界

#### 12.4 训练冒烟验证
用极少量样本跑 1 个 epoch，确认：

- 前向可跑
- loss 可算
- 反向传播不报错

#### 12.5 评估验证
跑一个验证 batch，确认 evaluator 不会因为空类或类别映射崩溃。

---

### 13. 最终建议

本次 HX 数据集适配，推荐采用下面这条主线：

> **仅在 `PointPillars/algo/data_preprocessor.py` 中增加 HX 扁平目录解析与标签转换逻辑，把 HX 原始 `.pcd + .json` 转成当前训练链路已经支持的 pickle 中间格式；训练、损失、评估主链路先保持不变。**

这样做的优点是：

- 改动范围最小
- 风险最低
- 最容易快速验证是否能训通
- **不会复制整套 `.pcd/.json` 原始数据，只会生成少量 `pkl` 中间文件，因此不会明显增加硬盘占用**
- 后续如果要继续适配 HX 输出格式，也有清晰的第二阶段边界

---

### 14. 本轮结论摘要

已确认：

1. HX 数据集确实是**扁平目录 + 同名 `.pcd/.json` 配对**
2. `.pcd` 为 **ASCII PCD v0.7**，但头部字段细节和你原示例略有不同
3. `hx_data_training.md` 中有几处笔误和类别表不一致，已在本文中指出
4. 最优适配入口是：
   - `PointPillars/algo/data_preprocessor.py`
5. 当前推荐类别映射为：
   - `Pedestrian -> Pedestrian`
   - `Bicycle -> Mbike`
   - `Car -> Car`
   - `Heavy Truck -> Bus`
   - `Bus -> Bus`

---

### 15. 当前代码已经实现的内容

目前已经完成以下改造：

1. `PointPillars/algo/data_preprocessor.py`
   - 已支持 `hx_flat` 模式
   - 直接扫描扁平目录中的同 stem `.pcd/.json`
   - **不会拷贝原始数据**，只会生成：
     - `training.pkl`
     - `validation.pkl`
     - `testing.pkl`
     - `prediction.pkl`

2. `PointPillars/algo/algo_config.yaml`
   - 已增加：
     - `DATASET_STYLE`
     - `CLASS_MAPPING`

3. `PointPillars/algo/utils.py`
   - 已增加统一的 ASCII PCD 读取函数

4. `PointPillars/algo/data_dataset.py`
   - 已改成复用统一 PCD 读取逻辑
   - 不再依赖固定 `skiprows=11`

---

### 16. 正式运行前需要知道的一个环境阻塞

当前 Linux 环境下，训练链路需要：

- `munkres`

当前已处理结果：

- `sklearn`：已安装
- `matplotlib`：已安装
- `numba`：已安装
- `munkres`：**已安装**

安装命令为：

```bash
pip install munkres
```

---

### 17. 推荐的数据目录组织方式

#### 17.1 原始数据目录

HX 原始数据继续保持不动：

```text
/home/bql/ARS/ARS_Data/ars_hx_train_data/origin_type/
├── 1733898949738.json
├── 1733898949738.pcd
├── 1733898949938.json
├── 1733898949938.pcd
├── ...
```

#### 17.2 预处理输出目录

建议把中间文件输出到：

```text
/home/bql/ARS/ARS_Data/ars_hx_train_data/hx_preprocess/
```

里面将包含：

```text
hx_preprocess/
├── training.pkl
├── validation.pkl
├── testing.pkl
└── prediction.pkl
```

#### 17.3 训练结果目录

建议训练结果输出到：

```text
/home/bql/ARS/ARS_Data/ars_hx_train_data/hx_result/
```

例如：

```text
hx_result/
├── loss_eva/
├── model_epoch/
├── pictures/
└── predictions/
```

---

### 18. 正式运行步骤与命令

下面给出推荐的 Linux 环境运行顺序。

#### Step 1：安装依赖

确认依赖：

```bash
pip install munkres
```

如果还没有确认 PyTorch 环境，也建议检查：

```bash
python - <<'PY'
import torch
print(torch.__version__)
print('cuda_available =', torch.cuda.is_available())
PY
```

---

#### Step 2：准备正式配置

当前仓库里的 `PointPillars/algo/algo_config.yaml` 已经支持 HX 模式，但默认值还是旧平台路径语义。

正式跑 HX 时，关键配置建议为：

```yaml
PREPROCESS:
  DATASET_STYLE: ['hx_flat', 1, '数据集风格']
  ORI_DATA_PATH: ['origin_type', 1, 'HX 扁平原始数据目录']
  SAVE_DATA_PATH: ['hx_preprocess', 1, '预处理输出目录']
```

类别映射保持为：

```yaml
CLASS_MAPPING:
  Pedestrian: ['Pedestrian', 1, '']
  Bicycle: ['Mbike', 1, '']
  Car: ['Car', 1, '']
  Heavy Truck: ['Bus', 1, '']
  Bus: ['Bus', 1, '']
```

如果你不想覆盖原配置，建议使用专门的 HX 配置文件：

```text
PointPillars/algo/algo_config_hx.yaml
```

---

#### Step 3：执行 HX 数据预处理

命令如下：

```bash
python - <<'PY'
import os
import sys
import yaml

repo = '/home/bql/ARS/ARS_Project/Web_LidarDetector_test'
data_root = '/home/bql/ARS/ARS_Data/ars_hx_train_data'

sys.path.insert(0, os.path.join(repo, 'PointPillars/algo'))
from data_preprocessor import DataPreprocessor

cfg_path = os.path.join(repo, 'PointPillars/algo/algo_config_hx.yaml')
with open(cfg_path, encoding='utf-8') as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)

params = cfg['PREPROCESS']
res = {'msg': []}
pre = DataPreprocessor(params, data_root, train_flag=True, test_flag=False, res_dict=res)
pre.data_preprocess()
print('\n'.join(str(x) for x in res['msg']))
PY
```

执行成功后，你应该能看到：

```text
/home/bql/ARS/ARS_Data/ars_hx_train_data/hx_preprocess/
├── training.pkl
└── validation.pkl
```

如果要生成测试集或预测集 pickle，则分别改成：

##### 生成 testing.pkl

```bash
python - <<'PY'
import os
import sys
import yaml

repo = '/home/bql/ARS/ARS_Project/Web_LidarDetector_test'
data_root = '/home/bql/ARS/ARS_Data/ars_hx_train_data'

sys.path.insert(0, os.path.join(repo, 'PointPillars/algo'))
from data_preprocessor import DataPreprocessor

cfg_path = os.path.join(repo, 'PointPillars/algo/algo_config_hx.yaml')
with open(cfg_path, encoding='utf-8') as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)

params = cfg['PREPROCESS']
res = {'msg': []}
pre = DataPreprocessor(params, data_root, train_flag=False, test_flag=True, res_dict=res)
pre.data_preprocess()
print('\n'.join(str(x) for x in res['msg']))
PY
```

##### 生成 prediction.pkl

```bash
python - <<'PY'
import os
import sys
import yaml

repo = '/home/bql/ARS/ARS_Project/Web_LidarDetector_test'
data_root = '/home/bql/ARS/ARS_Data/ars_hx_train_data'

sys.path.insert(0, os.path.join(repo, 'PointPillars/algo'))
from data_preprocessor import DataPreprocessor

cfg_path = os.path.join(repo, 'PointPillars/algo/algo_config_hx.yaml')
with open(cfg_path, encoding='utf-8') as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)

params = cfg['PREPROCESS']
res = {'msg': []}
pre = DataPreprocessor(params, data_root, train_flag=False, test_flag=False, res_dict=res)
pre.data_preprocess()
print('\n'.join(str(x) for x in res['msg']))
PY
```

---

#### Step 4：训练前做数据加载冒烟验证

建议先验证训练集能否正常取出 batch：

```bash
python - <<'PY'
import os
import sys
import yaml

repo = '/home/bql/ARS/ARS_Project/Web_LidarDetector_test'
preprocess_dir = '/home/bql/ARS/ARS_Data/ars_hx_train_data/hx_preprocess'

sys.path.insert(0, os.path.join(repo, 'PointPillars/algo'))
from data_dataset import dataset

cfg_path = os.path.join(repo, 'PointPillars/algo/algo_config_hx.yaml')
with open(cfg_path, encoding='utf-8') as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)

train_params = cfg['TRAIN_MODEL']
ds = dataset(train_params, preprocess_dir, train_params['TRAIN']['OVERALL']['TRAIN_PREFIX'][0], {'msg': []})
inputs, labels, filenames = ds[0]
print(inputs[0].shape, inputs[1].shape, inputs[2].shape)
print(labels[0].shape)
print(len(filenames))
PY
```

如果这一步通过，说明：

- pickle 合同正确
- PCD 读取正常
- voxelize 正常
- 类别映射未越界

---

#### Step 5：训练 1 个 epoch 冒烟

下面是推荐的最小训练冒烟命令：

```bash
python - <<'PY'
import os
import sys
import yaml

repo = '/home/bql/ARS/ARS_Project/Web_LidarDetector_test'
data_root = '/home/bql/ARS/ARS_Data/ars_hx_train_data'
result_root = '/home/bql/ARS/ARS_Data/ars_hx_train_data/hx_result'
model_epoch_root = os.path.join(result_root, 'model_epoch')

sys.path.insert(0, os.path.join(repo, 'PointPillars/algo'))
sys.path.insert(0, os.path.join(repo, 'PointPillars/model/model'))
sys.path.insert(0, os.path.join(repo, 'PointPillars/model/layer'))
sys.path.insert(0, os.path.join(repo, 'web_lidardetector'))

from interface import ITrain

cfg_path = os.path.join(repo, 'PointPillars/algo/algo_config_hx.yaml')
with open(cfg_path, encoding='utf-8') as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)

train_cfg = cfg['TRAIN_MODEL']
train_cfg['TRAIN']['CTRL']['CTRL_']['EPOCH_NUM'][0] = 1
train_cfg['TRAIN']['CTRL']['DATA']['BATCH_SIZE'][0] = 1
train_cfg['TRAIN']['OVERALL']['INITIAL_RESULT'][0] = True

res = {'msg': []}
ITrain(train_cfg, data_root, result_root, model_epoch_root, pretrained_path=None, check_flag=True, res_dict=res)
print('\n'.join(str(x) for x in res['msg'][-20:]))
PY
```

说明：

- `check_flag=True`：当前代码里会在训练和验证阶段尽快提前 break，适合做冒烟
- `EPOCH_NUM=1`：只验证链路是否跑通
- `BATCH_SIZE=1`：减少显存和排障成本

---

### 19. 当前我实际验证到哪一步了

我已经实际验证通过：

1. HX 小样本预处理可运行
2. 可生成 `training.pkl` / `validation.pkl`
3. `data_dataset.py` 可读取这些 pickle
4. 能成功构造一个训练 batch
5. 已成功跑通 1 个最小训练冒烟

---

### 20. 推荐的正式执行顺序

建议你后续正式跑的时候按这个顺序：

1. 确认依赖
   - `pip install munkres`
2. 使用专门的 HX 配置
3. 先跑预处理
4. 再跑数据加载冒烟
5. 再跑 1 epoch 训练冒烟
6. 最后再跑正式训练

---

### 21. 后续如果需要进一步自动化

如果后面你希望流程更顺一些，建议再做两件事：

1. 增加一份专门的 HX 配置文件
   - `PointPillars/algo/algo_config_hx.yaml`
2. 增加一个 Linux 友好的入口脚本
   - `scripts/run_hx_preprocess.py`
   - `scripts/run_hx_train.py`

现在这两件事都已经完成，所以后续可以直接用脚本运行，不需要每次都写 heredoc Python 命令。

#### 21.1 使用脚本执行 HX 预处理

生成训练/验证集：

```bash
python scripts/run_hx_preprocess.py \
  --data-root /home/bql/ARS/ARS_Data/ars_hx_train_data \
  --config PointPillars/algo/algo_config_hx.yaml \
  --mode train
```

生成测试集：

```bash
python scripts/run_hx_preprocess.py \
  --data-root /home/bql/ARS/ARS_Data/ars_hx_train_data \
  --config PointPillars/algo/algo_config_hx.yaml \
  --mode test
```

生成预测集：

```bash
python scripts/run_hx_preprocess.py \
  --data-root /home/bql/ARS/ARS_Data/ars_hx_train_data \
  --config PointPillars/algo/algo_config_hx.yaml \
  --mode predict
```

#### 21.2 使用脚本执行 HX 训练

先确保已经完成训练模式预处理，并且目录下存在：

- `hx_preprocess/training.pkl`
- `hx_preprocess/validation.pkl`

执行正式训练：

```bash
python scripts/run_hx_train.py \
  --data-root /home/bql/ARS/ARS_Data/ars_hx_train_data \
  --result-root /home/bql/ARS/ARS_Data/ars_hx_train_data/hx_result \
  --config PointPillars/algo/algo_config_hx.yaml
```

执行 1 epoch 冒烟训练：

```bash
python scripts/run_hx_train.py \
  --data-root /home/bql/ARS/ARS_Data/ars_hx_train_data \
  --result-root /home/bql/ARS/ARS_Data/ars_hx_train_data/hx_result_smoke \
  --config PointPillars/algo/algo_config_hx.yaml \
  --smoke
```

也可以临时覆盖 epoch 或 batch size：

```bash
python scripts/run_hx_train.py \
  --data-root /home/bql/ARS/ARS_Data/ars_hx_train_data \
  --result-root /home/bql/ARS/ARS_Data/ars_hx_train_data/hx_result \
  --config PointPillars/algo/algo_config_hx.yaml \
  --epochs 10 \
  --batch-size 2
```

---

### 22. 实际冒烟验证结果（已完成）

我已经在当前 Linux 环境里完成了真实冒烟验证，分两步：

#### 22.1 HX 预处理小样本冒烟

使用 8 个真实 HX 样本：

- 成功执行 `hx_flat` 预处理
- 成功生成：
  - `training.pkl`
  - `validation.pkl`
- `data_dataset.py` 成功读取这些 pickle
- 成功构造训练 batch 并完成 voxelize

#### 22.2 HX 训练小样本冒烟

在安装 `munkres` 后，我又继续跑了 1 个最小训练冒烟：

- 训练成功进入第 1 个 batch
- 成功完成一次 loss 计算和学习率更新
- 成功进入 `eva_vali`
- 成功结束验证阶段
- 成功保存模型文件：

```text
/home/bql/ARS/ARS_Data/ars_hx_train_data/hx_result_smoke/model_epoch/PointPillars_0.torch
```

训练日志中的关键信息包括：

- `1/9 lr: 0.000390`
- `########## evaluate in eva_vali ##########`
- `No prediction, Evaluation Done!`
- `############### end ###############`

说明：

- 这次冒烟使用的是极小样本和 `check_flag=True`
- 目标是验证 HX 数据链路、模型前向、loss、验证入口、模型保存能否跑通
- **结论：当前 HX 非拷贝适配方案已经可以跑通最小训练链路**
- 另外，现已增加 `COUNT_LIDAR_POINTS` 开关；在 HX 配置中默认关闭后，小样本预处理从约 `6s/8帧` 提升到约 `1.1s/8帧`，速度提升明显
- 现已增加 `USE_MULTIPROCESS` 与 `NUM_WORKERS` 开关；在 16 帧、4 进程的小样本测试中，预处理耗时约 `0.595s`，多进程路径已验证可用

---

### 23. 为了跑通冒烟，我顺手修复的旧代码问题

在这次训练冒烟过程中，还暴露并修复了几处原仓库的兼容性问题：

1. `PointPillars/algo/model_computers.py`
   - 构造函数参数与调用方不匹配
   - 已改成兼容 `pretrained_path`

2. `PointPillars/model/layer/Deblock.py`
   - 使用了 `np.int`，在新 NumPy 下报错
   - 已替换为 `int`

3. `PointPillars/algo/data_evaluater.py`
   - 使用了 `np.int`
   - 同时还有 `astype(int16/int8)` 这种未定义类型名问题
   - 已修正为兼容的新写法

这些修复并不是 HX 数据专属逻辑，但它们是当前环境下把训练冒烟真正跑通所必需的。