#!/usr/bin/env python3
"""
Motrix 后端性能对比测试
测试 Go1JoystickFlatTerrain 环境下 MotrixBackend vs MotrixNumbaBackend 的性能差异
"""

import sys
import time
import numpy as np
from pathlib import Path

# 添加项目路径
ROOT_DIR = Path(__file__).parent
sys.path.insert(0, str(ROOT_DIR))

from unilab.utils.algo_utils import ensure_registries
from unilab.envs import registry


def setup_environment(backend_type: str, num_envs: int = 512):
    """创建环境实例"""
    ensure_registries()
    env = registry.make(
        "Go1JoystickFlatTerrain",
        num_envs=num_envs,
        sim_backend=backend_type
    )
    return env


def warmup(env, num_steps: int = 10):
    """预热环境"""
    env_indices = np.arange(env.num_envs, dtype=np.int32)
    _, obs, _ = env.reset(env_indices)
    for _ in range(num_steps):
        actions = np.random.rand(*env.action_space.shape).astype(np.float32)
        obs, reward, done, truncated, info = env.step(actions)


def benchmark_step(env, num_iterations: int = 100, num_steps_per_iter: int = 10):
    """测试 step 操作性能"""
    # 预热
    warmup(env, num_steps=5)
    env_indices = np.arange(env.num_envs, dtype=np.int32)
    _, obs, _ = env.reset(env_indices)
    actions = np.random.rand(*env.action_space.shape).astype(np.float32)

    # 测试
    times = []
    for _ in range(num_iterations):
        start = time.perf_counter()
        for _ in range(num_steps_per_iter):
            obs, reward, done, truncated, info = env.step(actions)
        end = time.perf_counter()
        times.append(end - start)

    avg_time = np.mean(times)
    std_time = np.std(times)
    total_steps = num_iterations * num_steps_per_iter
    throughput = total_steps / np.sum(times)

    return {
        'avg_time_ms': avg_time * 1000,
        'std_time_ms': std_time * 1000,
        'throughput': throughput,
        'total_steps': total_steps
    }


def benchmark_set_state(env, num_iterations: int = 100):
    """测试 set_state 操作性能"""
    # 预热
    warmup(env, num_steps=5)

    # 准备测试数据
    num_reset = 64  # 每次重置的环境数量
    env_indices = np.arange(num_reset)

    # 从环境获取维度
    env_indices = np.arange(env.num_envs, dtype=np.int32)
    _, obs, _ = env.reset(env_indices)
    qpos = env._backend.get_qpos()[:num_reset].copy()
    qvel = env._backend.get_dof_vel()[:num_reset].copy()

    # 预热 set_state
    for _ in range(10):
        env._backend.set_state(env_indices, qpos, qvel)

    # 测试
    times = []
    for _ in range(num_iterations):
        start = time.perf_counter()
        env._backend.set_state(env_indices, qpos, qvel)
        end = time.perf_counter()
        times.append(end - start)

    avg_time = np.mean(times)
    std_time = np.std(times)
    ops_per_sec = num_iterations / np.sum(times)

    return {
        'avg_time_ms': avg_time * 1000,
        'std_time_ms': std_time * 1000,
        'ops_per_sec': ops_per_sec
    }


def benchmark_get_data(env, num_iterations: int = 1000):
    """测试数据获取性能"""
    # 预热
    warmup(env, num_steps=5)

    # 测试 dof_pos 获取
    start = time.perf_counter()
    for _ in range(num_iterations):
        dof_pos = env._backend.get_dof_pos()
    end = time.perf_counter()
    get_dof_pos_time = (end - start) / num_iterations * 1000

    # 测试 dof_vel 获取
    start = time.perf_counter()
    for _ in range(num_iterations):
        dof_vel = env._backend.get_dof_vel()
    end = time.perf_counter()
    get_dof_vel_time = (end - start) / num_iterations * 1000

    # 测试 qpos 获取
    start = time.perf_counter()
    for _ in range(num_iterations):
        qpos = env._backend.get_qpos()
    end = time.perf_counter()
    get_qpos_time = (end - start) / num_iterations * 1000

    return {
        'get_dof_pos_ms': get_dof_pos_time,
        'get_dof_vel_ms': get_dof_vel_time,
        'get_qpos_ms': get_qpos_time
    }


def benchmark_quaternion_conversion(env, num_iterations: int = 1000):
    """测试四元数转换性能（通过 set_state 触发）"""
    from unilab.envs.backend.motrix_numba_backend import _convert_quaternion_wxyz_to_xyzw

    # 准备测试数据
    num_envs = 512
    qpos = np.random.rand(num_envs, 35).astype(np.float32)

    # 测试 Numba 版本
    start = time.perf_counter()
    for _ in range(num_iterations):
        result = _convert_quaternion_wxyz_to_xyzw(qpos)
    end = time.perf_counter()
    numba_time = (end - start) / num_iterations * 1000

    # 测试 NumPy 版本
    start = time.perf_counter()
    for _ in range(num_iterations):
        qpos_motrix = qpos.copy()
        qpos_motrix[:, 3:7] = qpos[:, [4, 5, 6, 3]]
    end = time.perf_counter()
    numpy_time = (end - start) / num_iterations * 1000

    speedup = numpy_time / numba_time

    return {
        'numba_ms': numba_time,
        'numpy_ms': numpy_time,
        'speedup': speedup
    }


def run_benchmark_suite(backend_type: str, num_envs: int = 512):
    """运行完整的性能测试套件"""
    print(f"\n{'='*80}")
    print(f"测试后端: {backend_type} (环境数: {num_envs})")
    print(f"{'='*80}")

    try:
        # 创建环境
        print("创建环境...")
        env = setup_environment(backend_type, num_envs)

        results = {}

        # 1. 测试 step 性能
        print("\n[1/4] 测试 step 操作...")
        results['step'] = benchmark_step(env, num_iterations=50, num_steps_per_iter=10)

        # 2. 测试 set_state 性能
        print("[2/4] 测试 set_state 操作...")
        results['set_state'] = benchmark_set_state(env, num_iterations=200)

        # 3. 测试数据获取性能
        print("[3/4] 测试数据获取操作...")
        results['get_data'] = benchmark_get_data(env, num_iterations=1000)

        # 4. 测试四元数转换（仅 Numba 后端）
        if backend_type == "motrix_numba":
            print("[4/4] 测试四元数转换...")
            results['quaternion'] = benchmark_quaternion_conversion(env, num_iterations=1000)

        return results

    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        return None


def print_comparison_results(motrix_results, numba_results):
    """打印对比结果"""
    print(f"\n{'='*80}")
    print("性能对比结果")
    print(f"{'='*80}\n")

    # Step 性能对比
    print("1. Step 操作性能:")
    print(f"  {'指标':<25} {'Motrix':<20} {'Motrix+Numba':<20} {'加速比':<10}")
    print(f"  {'-'*75}")

    motrix_step = motrix_results['step']
    numba_step = numba_results['step']

    step_speedup = numba_step['throughput'] / motrix_step['throughput']
    print(f"  {'平均耗时 (ms)':<25} {motrix_step['avg_time_ms']:<20.3f} {numba_step['avg_time_ms']:<20.3f} {step_speedup:<10.2f}x")
    print(f"  {'吞吐量 (steps/s)':<25} {motrix_step['throughput']:<20.1f} {numba_step['throughput']:<20.1f} {step_speedup:<10.2f}x")

    # Set_state 性能对比
    print(f"\n2. Set_state 操作性能:")
    print(f"  {'指标':<25} {'Motrix':<20} {'Motrix+Numba':<20} {'加速比':<10}")
    print(f"  {'-'*75}")

    motrix_set_state = motrix_results['set_state']
    numba_set_state = numba_results['set_state']

    set_state_speedup = numba_set_state['ops_per_sec'] / motrix_set_state['ops_per_sec']
    print(f"  {'平均耗时 (ms)':<25} {motrix_set_state['avg_time_ms']:<20.3f} {numba_set_state['avg_time_ms']:<20.3f} {set_state_speedup:<10.2f}x")
    print(f"  {'操作数/秒':<25} {motrix_set_state['ops_per_sec']:<20.1f} {numba_set_state['ops_per_sec']:<20.1f} {set_state_speedup:<10.2f}x")

    # 数据获取性能对比
    print(f"\n3. 数据获取性能:")
    print(f"  {'操作':<25} {'Motrix (ms)':<20} {'Motrix+Numba (ms)':<20} {'加速比':<10}")
    print(f"  {'-'*75}")

    motrix_get = motrix_results['get_data']
    numba_get = numba_results['get_data']

    for key in ['get_dof_pos_ms', 'get_dof_vel_ms', 'get_qpos_ms']:
        op_name = key.replace('_ms', '').replace('_', ' ').title()
        motrix_time = motrix_get[key]
        numba_time = numba_get[key]
        speedup = motrix_time / numba_time
        print(f"  {op_name:<25} {motrix_time:<20.4f} {numba_time:<20.4f} {speedup:<10.2f}x")

    # 四元数转换性能（仅 Numba 后端）
    if 'quaternion' in numba_results:
        print(f"\n4. 四元数转换性能 (Numba vs NumPy):")
        quat_results = numba_results['quaternion']
        print(f"  NumPy:  {quat_results['numpy_ms']:.4f} ms")
        print(f"  Numba:  {quat_results['numba_ms']:.4f} ms")
        print(f"  加速比: {quat_results['speedup']:.2f}x")

    # 总结
    print(f"\n{'='*80}")
    print("总结:")
    avg_speedup = (step_speedup + set_state_speedup) / 2
    print(f"  Step 操作加速:        {step_speedup:.2f}x")
    print(f"  Set_state 操作加速:   {set_state_speedup:.2f}x")
    print(f"  平均加速:            {avg_speedup:.2f}x")

    if avg_speedup > 1.0:
        print(f"\n  ✅ Numba 后端整体性能提升 {avg_speedup:.2%}")
    else:
        print(f"\n  ⚠️  Numba 后端性能下降 {-avg_speedup:.2%}")
    print(f"{'='*80}\n")


def main():
    """主测试函数"""
    print("="*80)
    print("Motrix 后端性能对比测试")
    print("测试环境: Go1JoystickFlatTerrain")
    print("="*80)

    # 测试配置
    num_envs_list = [128, 512, 2048]

    for num_envs in num_envs_list:
        print(f"\n\n{'#'*80}")
        print(f"# 测试配置: {num_envs} 个并行环境")
        print(f"{'#'*80}")

        # 测试原始 Motrix 后端
        motrix_results = run_benchmark_suite("motrix", num_envs)

        # 测试 Numba 优化后端
        numba_results = run_benchmark_suite("motrix_numba", num_envs)

        # 打印对比结果
        if motrix_results and numba_results:
            print_comparison_results(motrix_results, numba_results)

    print("\n测试完成!")


if __name__ == "__main__":
    main()
