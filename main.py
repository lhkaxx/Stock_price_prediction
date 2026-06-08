"""
main.py — 主流程编排
一键运行：数据下载 → 特征工程 → LSTM训练 → GRU训练 → 评估可视化
"""
import os
import sys
import time


def section(title: str):
    """打印分节标题"""
    print(f"\n{'#' * 60}")
    print(f"#  {title}")
    print(f"{'#' * 60}")


def step_download():
    """Step 1: 数据下载"""
    section("Step 1/5: 数据下载")
    csv_path = os.path.join("data", "600519_daily.csv")
    if os.path.exists(csv_path):
        print("[main] 数据文件已存在，跳过下载")
        return
    from data_download import download_data
    download_data()


def step_features():
    """Step 2: 特征工程"""
    section("Step 2/5: 特征工程")
    test_path = os.path.join("data", "X_train.npy")
    if os.path.exists(test_path):
        print("[main] 预处理数据已存在，跳过特征工程")
        return
    from feature_engineering import process
    process()


def step_train_lstm():
    """Step 3: LSTM 训练"""
    section("Step 3/5: LSTM 模型训练")
    from model_lstm import run
    run()


def step_train_gru():
    """Step 4: GRU 训练"""
    section("Step 4/5: GRU 模型训练")
    from model_gru import run
    run()


def step_evaluate():
    """Step 5: 评估与可视化"""
    section("Step 5/5: 评估与可视化")
    from evaluation import evaluate
    evaluate()


def main():
    """主入口"""
    print("=" * 60)
    print("   股票价格预测 — LSTM vs GRU 对比实验")
    print("   标的: 贵州茅台 (600519)  时间: 2020-2025")
    print("=" * 60)

    t_start = time.time()

    try:
        step_download()
        step_features()
        step_train_lstm()
        step_train_gru()
        step_evaluate()
    except KeyboardInterrupt:
        print("\n[main] 用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n[main] 运行出错: {e}")
        raise

    elapsed = time.time() - t_start
    section(f"全部完成! 总耗时 {elapsed:.1f}s")
    print("\n输出文件:")
    print("  data/          — 原始数据 + 预处理数组")
    print("  models/        — LSTM/GRU 最优权重 + 训练历史")
    print("  figures/       — 损失曲线 / 预测对比 / 残差分布")


if __name__ == "__main__":
    main()
