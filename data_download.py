"""
data_download.py — 数据获取模块
使用 akshare 拉取贵州茅台(600519) 2020-01 至 2025-12 日线行情数据
"""
import os
import akshare as ak
import pandas as pd


def download_data(symbol: str = "600519",
                  start_date: str = "20200101",
                  end_date: str = "20251231",
                  save_dir: str = "data") -> pd.DataFrame:
    """
    从 akshare 下载股票日线数据并保存为 CSV。

    Args:
        symbol: 股票代码，默认 600519（贵州茅台）
        start_date: 起始日期 YYYYMMDD
        end_date: 截止日期 YYYYMMDD
        save_dir: 保存目录

    Returns:
        pd.DataFrame: 包含 OHLCV 的日线数据
    """
    print(f"[data_download] 正在从 akshare 拉取 {symbol} {start_date}-{end_date} 数据...")

    df = ak.stock_zh_a_hist(
        symbol=symbol,
        period="daily",
        start_date=start_date,
        end_date=end_date,
        adjust="qfq"  # 前复权
    )

    # 统一列名
    column_map = {
        "日期": "Date",
        "开盘": "Open",
        "最高": "High",
        "最低": "Low",
        "收盘": "Close",
        "成交量": "Volume",
    }
    df = df.rename(columns=column_map)

    # 只保留需要的列
    df = df[["Date", "Open", "High", "Low", "Close", "Volume"]]

    # 按日期排序
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    # 保存
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{symbol}_daily.csv")
    df.to_csv(save_path, index=False, encoding="utf-8-sig")

    print(f"[data_download] 数据已保存至 {save_path}")
    print(f"[data_download] 共 {len(df)} 条记录，日期范围 {df['Date'].min().date()} ~ {df['Date'].max().date()}")
    return df


if __name__ == "__main__":
    download_data()
