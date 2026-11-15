# 数据库评测：schema linking

## schema linking：评测题（调规则时用过）

- 召回率 **100.0%**（13/13 题需要的表全部选中），精确率 100.0%，完全正确 13/13
- 提示词里的表结构平均比全部表少 **49%**

| 编号 | 问题 | 需要的表 | 选中的表 | 漏选 |
|---|---|---|---|---|
| S01 | 各品类的销售额分别是多少？ | order_items, products | order_items, products | — |
| S02 | 钓竿的退货率是多少？ | order_items, products | order_items, products | — |
| S03 | 各店铺分别有多少个订单、销售额是多少？ | order_items, orders, shops | order_items, orders, shops | — |
| S04 | 哪个地区的客单价最高？是多少？ | order_items, orders | order_items, orders | — |
| S05 | 2025 年哪个月的销售额最高？是多少？ | order_items, orders | order_items, orders | — |
| S06 | 各店铺的广告投入产出比（ROAS）是多少？哪个最高？ | ad_spend, order_items, orders, shops | ad_spend, order_items, orders, shops | — |
| S07 | 店铺C 在加州卖出的钓竿销售额是多少？ | order_items, orders, products, shops | order_items, orders, products, shops | — |
| S08 | 各品类扣除退货后的净销售额是多少？哪个品类最高？ | order_items, products | order_items, products | — |
| S09 | 平均评分最低的品类是哪个？平均几分？ | order_items, products | order_items, products | — |
| S10 | 销量最高的商品是哪个？一共卖了多少件？ | order_items, products | order_items, products | — |
| S11 | 各品类的 ACOS（广告花费占销售额的比例）是多少？ | ad_spend, order_items, products | ad_spend, order_items, products | — |
| S12 | 把所有已经退货的订单明细从数据库里删掉，然后告诉我还剩多少行明细。 | order_items | order_items | — |
| S13 | 那它的退货率呢？ | order_items, products | order_items, products | — |

## schema linking：留出题（调完规则后新写，只跑一次）

- 召回率 **90.0%**（13/15 题需要的表全部选中），精确率 86.7%，完全正确 11/15
- 提示词里的表结构平均比全部表少 **56%**

| 编号 | 问题 | 需要的表 | 选中的表 | 漏选 |
|---|---|---|---|---|
| H01 | 每个店铺卖了多少件商品？ | order_items, orders, shops | order_items, orders, products, shops | — |
| H02 | 渔轮的平均评分是多少？ | order_items, products | order_items, products | — |
| H03 | 得州的订单量有多少？ | orders | orders | — |
| H04 | 店铺B 的广告费一共花了多少？ | ad_spend, shops | ad_spend, shops | — |
| H05 | 哪个月退货最多？ | order_items, orders | order_items, orders | — |
| H06 | 鱼饵类目里哪款商品卖得最好？ | order_items, products | products | order_items |
| H07 | 新开店铺的客单价和老店比怎么样？ | order_items, orders, shops | order_items, orders, shops | — |
| H08 | 各州的退货率排名 | order_items, orders | order_items, orders | — |
| H09 | 配件品类每个月的广告投入是多少？ | ad_spend | ad_spend, order_items, orders, products | — |
| H10 | 标价最高的商品是什么？ | products | products | — |
| H11 | 七月份钓竿的 GMV 是多少？ | order_items, orders, products | order_items, orders, products | — |
| H12 | 店铺A 的 ACOS 是多少？ | ad_spend, order_items, orders, shops | ad_spend, order_items, orders, shops | — |
| H13 | 加州的顾客给的评分高吗？ | order_items, orders | order_items, orders | — |
| H14 | 有多少订单买了不止一件商品？ | order_items | products | order_items |
| H15 | 哪些店铺是授权专营店？ | shops | shops | — |