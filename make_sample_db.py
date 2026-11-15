"""生成一个模拟的跨境电商（渔具）SQLite 数据库，演示 Text-to-SQL（多表、外键、需要 JOIN）。

和 make_sample_data.py 的区别：那份是平台导出的一张大表（故意带脏数据，考数据清洗）；
这份是业务系统里的数据库（有约束、比较干净，考“该用哪几张表、怎么 JOIN、口径对不对”）。

| 表 | 内容 |
|---|---|
| shops | 店铺：名称、类型、开店日期 |
| products | 商品：名称、品类、标价 |
| orders | 订单（一单一行）：下单日期、店铺、收货地区 |
| order_items | 订单明细（一单可以有多件商品）：成交单价、数量、是否退货、评分 |
| ad_spend | 广告花费：按 月份 × 店铺 × 品类 汇总 |

两个常见陷阱（评测里专门考）：
1. 一个订单有多行明细：数订单要 COUNT(DISTINCT order_id)，直接 COUNT(*) 数的是明细行；
2. 广告花费是按月汇总的，直接和明细 JOIN 会把一笔广告费重复算很多次（扇出），要先各自汇总再关联。

数据完全随机模拟，不对应任何真实店铺。用法：python make_sample_db.py
"""
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent / "sample_data" / "电商经营库.db"

SCHEMA = """
CREATE TABLE shops (
    shop_id TEXT PRIMARY KEY,
    shop_name TEXT NOT NULL,
    shop_type TEXT NOT NULL,
    open_date TEXT NOT NULL
);
CREATE TABLE products (
    product_id TEXT PRIMARY KEY,
    product_name TEXT NOT NULL,
    category TEXT NOT NULL,
    list_price REAL NOT NULL
);
CREATE TABLE orders (
    order_id TEXT PRIMARY KEY,
    order_date TEXT NOT NULL,
    shop_id TEXT NOT NULL REFERENCES shops(shop_id),
    region TEXT NOT NULL
);
CREATE TABLE order_items (
    item_id INTEGER PRIMARY KEY,
    order_id TEXT NOT NULL REFERENCES orders(order_id),
    product_id TEXT NOT NULL REFERENCES products(product_id),
    unit_price REAL NOT NULL CHECK (unit_price > 0),
    quantity INTEGER NOT NULL CHECK (quantity > 0),
    is_returned INTEGER NOT NULL CHECK (is_returned IN (0, 1)),
    rating INTEGER CHECK (rating BETWEEN 1 AND 5)
);
CREATE TABLE ad_spend (
    month TEXT NOT NULL,
    shop_id TEXT NOT NULL REFERENCES shops(shop_id),
    category TEXT NOT NULL,
    spend REAL NOT NULL,
    PRIMARY KEY (month, shop_id, category)
);
"""

# 数据字典：表和列的中文说明（schema linking 和写 SQL 都靠它把中文问题对应到英文列名）
DOCS = {"tables": {
    "shops": {"desc": "店铺信息：店铺名称、店铺类型、开店日期",
              "columns": {"shop_id": "店铺编号", "shop_name": "店铺名称",
                          "shop_type": "店铺类型：自营旗舰店、授权专营店、新开店铺",
                          "open_date": "开店日期 YYYY-MM-DD"}},
    "products": {"desc": "商品信息：商品名称、所属品类、标价",
                 "columns": {"product_id": "商品编号", "product_name": "商品名称",
                             "category": "品类：钓竿、渔轮、鱼线、鱼饵、配件",
                             "list_price": "标价（美元），不是成交价，计算销售额要用 order_items.unit_price"}},
    "orders": {"desc": "订单：一个订单一行，记录下单日期、所属店铺、收货地区。一个订单可能包含多件商品，明细在 order_items",
               "columns": {"order_id": "订单号", "order_date": "下单日期 YYYY-MM-DD，按月汇总用 strftime('%Y-%m', order_date)",
                           "shop_id": "订单所属店铺", "region": "收货地区（美国的州）"}},
    "order_items": {"desc": "订单明细：每个订单里的每一件商品，成交单价、数量、是否退货、评分。"
                            "销售额 = unit_price × quantity；数订单数要用 COUNT(DISTINCT order_id)",
                    "columns": {"item_id": "明细编号", "order_id": "所属订单", "product_id": "商品",
                                "unit_price": "成交单价（美元，折扣后）",
                                "quantity": "购买数量（件数、销量）",
                                "is_returned": "这件商品是否退货：1=退货，0=未退货。退货率按订单计算（订单里有任意一件退货就算退货订单），见指标口径",
                                "rating": "客户评分 1～5 星，未评价为 NULL（计算平均分时排除，AVG 会自动跳过 NULL）"}},
    "ad_spend": {"desc": "广告花费：按 月份 × 店铺 × 品类 汇总的站内广告费用。"
                         "和订单关联前要先各自汇总，直接 JOIN 明细会重复计算广告费",
                 "separate": True,          # 汇总表：和明细分别汇总后按维度关联，不参与明细的 JOIN 路径
                 "columns": {"month": "月份 YYYY-MM", "shop_id": "店铺", "category": "品类",
                             "spend": "广告花费（美元）"}},
},
    # 语义层：指标（统一口径 + 需要哪些表）和维度（中文说法对应哪一列）。
    # schema linking 先查这里，比单纯按关键词匹配表说明准：问“广告投入产出比”时，
    # 字面上没有“销售额”，但口径决定了它一定要用到订单明细。
    "metrics": {
        "销售额": {"aliases": ["销售额", "GMV", "营业额", "收入", "卖了多少钱"], "tables": ["order_items"],
                "formula": "SUM(order_items.unit_price * order_items.quantity)，包含已退货的明细"},
        "净销售额": {"aliases": ["净销售额", "扣除退货", "扣掉退货"], "tables": ["order_items"],
                  "formula": "SUM(CASE WHEN is_returned = 0 THEN unit_price * quantity ELSE 0 END)"},
        "订单数": {"aliases": ["订单数", "订单量", "单量", "多少单"], "tables": ["orders"],
                "formula": "COUNT(DISTINCT order_id)；和明细 JOIN 后不能用 COUNT(*)"},
        "销量": {"aliases": ["销量", "件数", "卖了多少", "卖得最多"], "tables": ["order_items"],
               "formula": "SUM(order_items.quantity)"},
        "退货率": {"aliases": ["退货率", "退货", "退款"], "tables": ["order_items"],
                "formula": "按订单计算（公司默认口径）：COUNT(DISTINCT CASE WHEN is_returned = 1 THEN order_id END) * 1.0 "
                           "/ COUNT(DISTINCT order_id)，订单里有任意一件退货就算退货订单；按品类等分组时在组内计算"},
        "退货明细率": {"aliases": ["退货明细率", "按明细", "按件"], "tables": ["order_items"],
                  "formula": "AVG(order_items.is_returned)，退货的明细行数 ÷ 明细行数；只有用户明确要求按明细或按件时才用"},
        "客单价": {"aliases": ["客单价", "平均订单金额", "每单金额"], "tables": ["order_items", "orders"],
                "formula": "销售额 ÷ COUNT(DISTINCT order_id)"},
        "平均评分": {"aliases": ["评分", "好评", "差评", "星级", "满意度"], "tables": ["order_items"],
                 "formula": "AVG(order_items.rating)，NULL（未评价）不计入"},
        "广告花费": {"aliases": ["广告花费", "广告费", "推广费", "广告投入"], "tables": ["ad_spend"],
                 "formula": "SUM(ad_spend.spend)"},
        "ROAS": {"aliases": ["ROAS", "投入产出比", "广告回报", "广告效果"], "tables": ["ad_spend", "order_items"],
                 "formula": "销售额 ÷ 广告花费。两边先按同一维度（店铺 / 品类 / 月份）分别汇总，再按维度关联；"
                            "不要把 ad_spend 直接 JOIN 到明细上求和，否则广告费会被重复计算"},
        "ACOS": {"aliases": ["ACOS", "广告成本占比", "广告占比"], "tables": ["ad_spend", "order_items"],
                 "formula": "广告花费 ÷ 销售额，汇总方式同 ROAS"},
    },
    # 逻辑关联：不是外键、但业务上可以关联的列（广告表按品类名关联商品表）
    "joins": [{"left": "ad_spend.category", "right": "products.category", "note": "按品类名关联，不是外键"}],
    "dimensions": {
        "品类": {"aliases": ["品类", "类目", "类别"], "column": "products.category"},
        "商品": {"aliases": ["商品", "产品", "单品", "SKU"], "column": "products.product_name"},
        "店铺": {"aliases": ["店铺", "门店"], "column": "shops.shop_name"},
        "地区": {"aliases": ["地区", "州", "区域", "收货地"], "column": "orders.region"},
        "日期": {"aliases": ["月", "季度", "趋势", "日期", "旺季", "淡季"], "column": "orders.order_date"},
    },
}

CATEGORIES = {  # 品类: (商品名称列表, 基础价格, 退货率, 平均评分)
    "钓竿": (["碳素路亚竿", "矶钓竿", "海竿", "溪流竿"], 45, 0.08, 4.3),
    "渔轮": (["纺车轮", "水滴轮", "鼓式轮"], 60, 0.06, 4.4),
    "鱼线": (["尼龙线", "碳线", "PE线"], 12, 0.03, 4.5),
    "鱼饵": (["软饵套装", "米诺", "亮片", "VIB"], 8, 0.04, 4.1),
    "配件": (["钓箱", "竿架", "钓鱼钳", "失手绳"], 15, 0.05, 4.2),
}
SHOPS = [("S1", "店铺A", "自营旗舰店", "2021-03-15", 0.45),
         ("S2", "店铺B", "授权专营店", "2022-07-01", 0.35),
         ("S3", "店铺C", "新开店铺", "2024-10-08", 0.20)]
REGIONS = ["加州", "得州", "佛州", "纽约州", "华盛顿州", "密歇根州"]


def make(n_orders: int = 2500, seed: int = 7) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    shops = pd.DataFrame([s[:4] for s in SHOPS], columns=["shop_id", "shop_name", "shop_type", "open_date"])

    prods = []
    for cat, (names, base, _, _) in CATEGORIES.items():
        for name in names:
            prods.append((f"P{len(prods) + 1:02d}", name, cat, round(base * rng.uniform(0.8, 1.3), 2)))
    products = pd.DataFrame(prods, columns=["product_id", "product_name", "category", "list_price"])

    dates = pd.to_datetime("2025-01-01") + pd.to_timedelta(rng.integers(0, 365, n_orders), unit="D")
    summer = rng.random(n_orders) < 0.25          # 夏季旺季：一部分订单挪到 5～8 月
    dates = dates.where(~summer, pd.to_datetime("2025-05-01")
                        + pd.to_timedelta(rng.integers(0, 120, n_orders), unit="D"))
    orders = pd.DataFrame({
        "order_id": [f"OD{200000 + i}" for i in range(n_orders)],
        "order_date": dates.strftime("%Y-%m-%d"),
        "shop_id": rng.choice([s[0] for s in SHOPS], n_orders, p=[s[4] for s in SHOPS]),
        "region": rng.choice(REGIONS, n_orders),
    }).sort_values("order_date").reset_index(drop=True)

    cat_of = dict(zip(products.product_id, products.category))
    price_of = dict(zip(products.product_id, products.list_price))
    items = []
    for oid in orders.order_id:
        for pid in rng.choice(products.product_id, rng.choice([1, 1, 1, 2, 2, 3]), replace=False):
            _, _, ret, rate = CATEGORIES[cat_of[pid]]
            rating = int(np.clip(round(rng.normal(rate, 0.8)), 1, 5)) if rng.random() > 0.08 else None
            items.append((len(items) + 1, oid, pid, round(price_of[pid] * rng.uniform(0.75, 1.0), 2),
                          int(rng.choice([1, 1, 1, 2, 2, 3, 4])), int(rng.random() < ret), rating))
    order_items = pd.DataFrame(items, columns=["item_id", "order_id", "product_id", "unit_price",
                                               "quantity", "is_returned", "rating"])
    order_items["rating"] = order_items["rating"].astype("Int64")

    # 广告花费：按月 × 店铺 × 品类，约为当月销售额的 8%～30%
    m = order_items.merge(orders, on="order_id").merge(products[["product_id", "category"]], on="product_id")
    m["month"] = m["order_date"].str[:7]
    m["sales"] = m["unit_price"] * m["quantity"]
    ad = m.groupby(["month", "shop_id", "category"], as_index=False)["sales"].sum()
    ad["spend"] = (ad["sales"] * rng.uniform(0.08, 0.30, len(ad))).round(2)
    ad_spend = ad[["month", "shop_id", "category", "spend"]]
    return {"shops": shops, "products": products, "orders": orders,
            "order_items": order_items, "ad_spend": ad_spend}


def build(path: Path = OUT, **kwargs) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.unlink(missing_ok=True)
    conn = sqlite3.connect(path)
    try:
        conn.executescript(SCHEMA)
        for name, df in make(**kwargs).items():
            df.to_sql(name, conn, if_exists="append", index=False)
        conn.commit()
    finally:
        conn.close()
    path.with_suffix(".schema.json").write_text(json.dumps(DOCS, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


if __name__ == "__main__":
    p = build()
    conn = sqlite3.connect(p)
    for t in DOCS["tables"]:
        print(f"{t}: {conn.execute(f'SELECT COUNT(*) FROM {t}').fetchone()[0]} 行")
    conn.close()
    print(f"已生成 -> {p}")
