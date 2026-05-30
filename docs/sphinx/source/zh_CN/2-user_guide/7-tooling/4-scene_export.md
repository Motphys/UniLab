# 场景导出

语言: 简体中文

场景导出由 `src/unilab/tools/export_scene.py` 实现，并在 `pyproject.toml` 中注册为 `unilab-export-scene` 控制台入口。它接受一个 MuJoCo XML 或 MJB 模型路径，写出 `scene.xml`，在能够发现 mesh asset 时复制它们，并且可以创建一个 zip 归档。

对于 task 级别的实例化检查，请使用从 registry 和 owner config 构造 env 的脚本：

```bash
uv run scripts/visualize_task_env.py --task Go2JoystickRough --backend mujoco --num_envs 4
```

`tests/test_export_scene.py` 覆盖了导出辅助逻辑，包括 `scene.xml` 的创建、可重新加载性以及 zip 输出。

## Navigation

- Index: [文档](0-index.md)
