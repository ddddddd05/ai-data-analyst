"""生成一份模拟的跨境电商（渔具）订单数据，用于演示和测试。

数据完全是随机模拟的，不对应任何真实店铺。为了检验数据清洗能力，
故意加入了：重复订单、评分缺失、单价异常值、数量为负数的错误记录。

用法：python make_sample_data.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent / "sample_data" / "模拟电商订单.csv"


def make(n: int = 2000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    categories = {  # 品类: (基础单价美元, 退货率, 平均评分)
        "钓竿": (45, 0.08, 4.3),
        "渔轮": (60, 0.06, 4.4),
        "鱼线": (12, 0.03, 4.5),
        "鱼饵": (8, 0.04, 4.1),
        "配件": (15, 0.05, 4.2),
    }
    shops = ["店铺A", "店铺B", "店铺C"]
    states = ["加州", "得州", "佛州", "纽约州", "华盛顿州", "密歇根州"]

    cat = rng.choice(list(categories), size=n, p=[0.2, 0.15, 0.25, 0.25, 0.15])
    dates = pd.to_datetime("2025-01-01") + pd.to_timedelta(rng.integers(0, 365, n), unit="D")
    # 夏季是钓鱼旺季：把一部分订单挪到 5～8 月
    summer = rng.random(n) < 0.25
    dates = dates.where(~summer, pd.to_datetime("2025-05-01")
                        + pd.to_timedelta(rng.integers(0, 120, n), unit="D"))

    base = np.array([categories[c][0] for c in cat])
    price = np.round(base * rng.uniform(0.7, 1.4, n), 2)
    qty = rng.choice([1, 1, 1, 2, 2, 3, 4], size=n)
    ad = np.round(price * qty * rng.uniform(0.05, 0.35, n), 2)
    ret = rng.random(n) < np.array([categories[c][1] for c in cat])
    rating = np.clip(np.round(rng.normal([categories[c][2] for c in cat], 0.8)), 1, 5)

    df = pd.DataFrame({
        "订单号": [f"OD{100000 + i}" for i in range(n)],
        "下单日期": dates.strftime("%Y-%m-%d"),
        "店铺": rng.choice(shops, size=n, p=[0.45, 0.35, 0.2]),
        "品类": cat,
        "地区": rng.choice(states, size=n),
        "单价": price,
        "数量": qty,
        "广告花费": ad,
        "是否退货": np.where(ret, "是", "否"),
        "评分": rating,
    })
    df["销售额"] = np.round(df["单价"] * df["数量"], 2)

    # ---- 故意加入的数据问题 ----
    df.loc[rng.choice(n, int(n * 0.05), replace=False), "评分"] = np.nan    # 5% 评分缺失
    bad = rng.choice(n, 6, replace=False)
    df.loc[bad, "单价"] = df.loc[bad, "单价"] * 100                          # 单价录入错误
    df.loc[bad, "销售额"] = np.round(df.loc[bad, "单价"] * df.loc[bad, "数量"], 2)
    neg = rng.choice(n, 3, replace=False)
    df.loc[neg, "数量"] = -df.loc[neg, "数量"]                              # 数量为负
    df.loc[neg, "销售额"] = np.round(df.loc[neg, "单价"] * df.loc[neg, "数量"], 2)
    dup = df.sample(30, random_state=seed)                                  # 30 条重复订单
    df = pd.concat([df, dup]).sample(frac=1, random_state=seed).reset_index(drop=True)
    return df


if __name__ == "__main__":
    OUT.parent.mkdir(exist_ok=True)
    data = make()
    data.to_csv(OUT, index=False, encoding="utf-8-sig")
    print(f"已生成 {len(data)} 行 -> {OUT}")
