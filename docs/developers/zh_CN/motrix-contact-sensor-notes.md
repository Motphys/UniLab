# Motrix Contact Sensor 适配笔记

语言: 简体中文

## 背景

Sharpa Inhand Rotation 任务的触觉观测依赖 contact sensor 数据（5 个指尖与物体的接触力）。当前代码在 MuJoCo 后端下工作正常，Motrix 后端的 contact sensor 返回格式存在差异，需要适配。

## Contact Sensor 配置

场景 XML（`right_sharpa_wave.xml`）中的定义：

```xml
<contact name="contact_right_thumb_elastomer_force" geom1="right_thumb_elastomer" geom2="object" num="1" data="force" reduce="netforce"/>
<!-- 共 5 个指尖 sensor，配置相同 -->
```

参数说明：
- `num="1"` — 最多报告 1 个接触点
- `data="force"` — 只返回力数据
- `reduce="netforce"` — 合成为单个接触点，全局坐标系

## 两个后端的返回格式差异

### MuJoCo

返回 `(num_envs, 3)` 的力向量，全局坐标系：

```
[[fx, fy, fz],    # env 0
 [fx, fy, fz],    # env 1
 ...]
```

### Motrix

返回 `(num_envs, 1 + num * stride)` 的扁平数组，首个元素为实际接触点数：

```
[[count, fx, fy, fz],    # env 0, shape = (4,) 因为 num=1, stride=3
 [count, fx, fy, fz],    # env 1
 ...]
```

对于 `reduce="netforce"` + `data="force"` + `num=1`：
- shape = `(num_envs, 4)`
- `[0]` = 接触点数（0 或 1）
- `[1:4]` = 全局坐标系的合力向量（netforce 模式下使用全局坐标）

### 多接触点的通用格式（Motrix）

当 `num=4, data="force pos normal tangent"` 无 reduce 时：

```
shape = (num_envs, 1 + 4 * 12) = (num_envs, 49)

[count,
 f1_normal, f1_tangent0, f1_tangent1,   # contact 1 force (contact 坐标系)
 f1_x, f1_y, f1_z,                       # contact 1 position
 f1_nx, f1_ny, f1_nz,                    # contact 1 normal
 f1_tx, f1_ty, f1_tz,                    # contact 1 tangent
 f2_normal, f2_tangent0, f2_tangent1,    # contact 2 ...
 ...
 padding_zeros]                           # 不足 num 个的用 0 填充
```

注意：无 reduce 时力是 **contact 坐标系**（法向 + 两个切向标量），不是全局 xyz 向量。只有 `reduce="netforce"` 返回全局坐标。

## 当前代码的问题

`base.py` 中 `_extract_sensor_scalar()` 的处理逻辑：

```python
def _extract_sensor_scalar(self, sensor_name: str) -> np.ndarray:
    data = np.asarray(self._backend.get_sensor_data(sensor_name), dtype=self._np_dtype)
    if data.ndim == 1:
        return data
    if data.ndim == 2 and data.shape[1] == 1:
        return data[:, 0]
    if data.ndim == 2 and data.shape[1] >= 3:
        return np.asarray(np.linalg.norm(data[:, :3], axis=1), dtype=self._np_dtype)
    flat = data.reshape(data.shape[0], -1)
    return np.asarray(flat[:, 0], dtype=self._np_dtype)
```

- MuJoCo 返回 `(N, 3)` → 走第三个分支 `norm(data[:, :3])` → 正确
- Motrix 返回 `(N, 4)` → 也走第三个分支 `norm(data[:, :3])` → **错误**
  - `data[:, :3]` 取到的是 `[count, fx, fy]`，把 count 当成了力分量
  - 漏掉了 `fz`

## 静态验证数据

用 `make('SharpaInhandRotation', sim_backend=..., num_envs=4)` 测试，PD gain 不随机化：

| 指标 | MuJoCo | Motrix |
|------|--------|--------|
| virtual_torque max | 0.21 ~ 0.38 | 0.31 ~ 0.74 |
| virtual_torque mean | 0.04 ~ 0.05 | 0.05 ~ 0.06 |
| dof_vel max | 0.43 ~ 0.58 | 0.51 ~ 0.97 |
| work_penalty mean | 0.01 | 0.00 ~ 0.02 |

虚拟 torque（PD 公式计算）在两个后端下量级一致。work reward 爆炸（训练日志中 -19169）与 PD gain 随机化（0.5x ~ 2.0x）有关，非 Motrix 特有问题。

## 建议的适配方案

### 方案：Backend 接口层增加专用方法

在 `SimBackend` 接口增加 `get_contact_force_magnitude(sensor_name) -> np.ndarray`，返回 `(num_envs,)` 的标量力大小。每个 backend 自行实现：

- **MuJoCo**：`norm(get_sensor_data(name), axis=1)` — 直接对 3D 力向量取范数
- **Motrix**：根据 sensor 的 reduce 模式和数据布局做不同处理
  - `reduce="netforce"`：取 `[1:4]` 三个分量 → norm
  - 无 reduce 多 contact：对所有接触力求和后取 norm
  - `reduce="maxforce"`：取力最大的那个

env 层的 `_read_tactile_force()` 对 contact sensor 走 `get_contact_force_magnitude()` 路径，普通 sensor 仍走 `_extract_sensor_scalar()`。

### 优点

- env 代码保持 backend 无关
- 未来新增 backend 只需实现接口方法
- 多 contact 场景可扩展

### 需要关注的 Motrix sensor 配置变体

| reduce 模式 | 坐标系 | 返回内容 |
|-------------|--------|----------|
| `netforce` | 全局 | 单个合成力向量 |
| `maxforce` | contact | 力最大的那个接触点 |
| `mindist` | contact | 穿透最浅的那个接触点 |
| 无 | contact | 前 `num` 个接触点 |

其中只有 `netforce` 返回全局坐标系力向量，其他模式返回 contact 坐标系（法向标量 + 两个切向标量）。

## 相关文件

| 文件 | 说明 |
|------|------|
| `src/unilab/envs/manipulation/sharpa_inhand/base.py` | `_extract_sensor_scalar()`, `_read_tactile_force()` |
| `src/unilab/envs/manipulation/sharpa_inhand/rotation.py` | reward 计算，virtual torque |
| `src/unilab/assets/robots/sharpa_wave/right_sharpa_wave.xml` | contact sensor XML 定义 |
| `src/unilab/base/backend/motrix/backend.py` | Motrix `get_sensor_data()` 实现 |
| `src/unilab/base/backend/mujoco/backend.py` | MuJoCo `get_sensor_data()` 实现 |
| `motrixsim-docs/.../sensor/contact.md` | Motrix contact sensor 文档 |

## Navigation

- Index: [Documentation](../../README.md)
