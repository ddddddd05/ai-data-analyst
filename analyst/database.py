"""数据库数据源（Text-to-SQL）：连接 SQLite 数据库，读取表结构，按问题挑出相关的表，安全地执行只读 SQL。

为什么需要它：真实的电商数据在数据库里，分成订单、明细、商品、店铺、广告等多张表，
不是一张导出好的大表。Agent 要自己判断用哪几张表、怎么 JOIN，再写 SQL 取数。

三个难点和做法：
| 难点 | 做法 |
|---|---|
| 表多、列名是英文，问题是中文：该用哪几张表？ | schema linking：语义层（指标、维度）→ 取值匹配 → 关键词兜底，再沿外键补上 JOIN 需要的中间表 |
| 模型写的 SQL 可能改数据、删表、跑很久 | 三层护栏：只读方式打开数据库；授权回调只放行“读”操作；超时中断 + 返回行数上限 |
| SQL 写错（列名错、JOIN 错、结果为空） | 报错信息交回模型修改；“列不存在”时把全部表结构附上，弥补 schema linking 漏掉的表 |

表和列的中文说明放在数据库旁边的 <库名>.schema.json 里（相当于数据字典），没有时只用列名。
同一个文件里还有“语义层”：
- 指标（metrics）：别名、统一口径（SQL 写法）、需要哪些表，如 ROAS = 销售额 ÷ 广告花费，要用 ad_spend 和 order_items；
- 维度（dimensions）：中文说法对应哪一列，如“品类”→ products.category；
- 逻辑关联（joins）：不是外键、但业务上能关联的列，如 ad_spend.category = products.category。
匹配到的指标口径会交给写 SQL 的模型，保证不同问题、不同次运行算出来的口径一致。
"""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .knowledge import BM25, tokenize

# 授权回调里放行的操作：读表、SELECT、调用函数（SUM、strftime……）、WITH RECURSIVE。
# 其他一律拒绝：INSERT/UPDATE/DELETE、建表删表、ATTACH 别的数据库文件、PRAGMA 等。
_ALLOWED_ACTIONS = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION,
                    getattr(sqlite3, "SQLITE_RECURSIVE", 33)}
# 追问的指代词：出现时才从上一轮对话里找取值（“那它的退货率呢”里的“它”指上一轮的“钓竿”）
FOLLOW_UP_CUES = ("它", "那", "这个", "这些", "其中", "上面", "刚才", "该品类", "该店铺", "同样", "呢")
_START = re.compile(r"^\s*(?:--[^\n]*\n\s*|/\*.*?\*/\s*)*\(?\s*(select|with)\b", re.I | re.S)


@dataclass
class Column:
    name: str
    type: str
    desc: str = ""
    samples: list[str] = field(default_factory=list)   # 文字列的常见取值（用于“值匹配”，如问题里的“钓竿”）


@dataclass
class Table:
    name: str
    desc: str
    rows: int
    columns: list[Column]
    foreign_keys: list[tuple[str, str, str]] = field(default_factory=list)   # (本表列, 引用表, 引用列)
    separate: bool = False     # 粒度不同的汇总表（如按月的广告花费）：先各自汇总再按维度关联，不参与 JOIN 路径


@dataclass
class SQLResult:
    ok: bool
    sql: str
    df: pd.DataFrame | None = None
    error: str = ""
    truncated: bool = False
    seconds: float = 0.0


def check_sql(sql: str) -> list[str]:
    """执行前的静态检查：只允许一条 SELECT / WITH 查询。真正的拦截靠只读连接和授权回调。"""
    text = sql.strip().rstrip(";").strip()
    if not text:
        return ["SQL 是空的"]
    if not _START.match(text):
        return ["只允许查询（SELECT / WITH 开头），不允许修改数据或表结构"]
    if ";" in _strip_strings(text):
        return ["一次只能执行一条 SQL"]
    return []


def _strip_strings(sql: str) -> str:
    return re.sub(r"'(?:[^']|'')*'|\"(?:[^\"]|\"\")*\"", "''", sql)


class Database:
    def __init__(self, path: Path | str, timeout: float = 10.0, max_rows: int = 200):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"数据库文件不存在：{self.path}")
        self.timeout, self.max_rows = timeout, max_rows
        self.name = self.path.name
        docs = _load_docs(self.path)
        self.metrics: dict[str, dict] = docs.get("metrics", {})
        self.dimensions: dict[str, dict] = docs.get("dimensions", {})
        self.joins: list[dict] = docs.get("joins", [])
        self.tables = self._introspect(docs.get("tables", {}))
        self.by_name = {t.name: t for t in self.tables}
        texts = [self._table_text(t) for t in self.tables]
        self.bm25 = BM25([tokenize(x) for x in texts])

    # ---------------- 连接：只读 + 授权回调 + 超时 ----------------
    def _connect(self) -> sqlite3.Connection:
        uri = f"file:{self.path.resolve().as_posix()}?mode=ro"
        conn = sqlite3.connect(uri, uri=True, check_same_thread=False)
        conn.set_authorizer(lambda action, *_: sqlite3.SQLITE_OK if action in _ALLOWED_ACTIONS
                            else sqlite3.SQLITE_DENY)
        return conn

    def run_sql(self, sql: str) -> SQLResult:
        sql = sql.strip().rstrip(";").strip()
        problems = check_sql(sql)
        if problems:
            return SQLResult(False, sql, error="安全检查未通过，SQL 没有执行：" + "；".join(problems))
        t0 = time.perf_counter()
        conn = self._connect()
        deadline = t0 + self.timeout
        conn.set_progress_handler(lambda: 1 if time.perf_counter() > deadline else 0, 10000)
        try:
            cur = conn.execute(sql)
            rows = cur.fetchmany(self.max_rows + 1)
            cols = [d[0] for d in cur.description or []]
        except sqlite3.DatabaseError as e:
            msg = str(e)
            if "interrupted" in msg:
                msg = f"查询超过 {self.timeout:g} 秒，已中断（可能 JOIN 条件缺失导致数据量爆炸）"
            elif "not authorized" in msg:
                msg = "数据库是只读的，不允许修改数据、表结构或访问其他数据库文件（not authorized）"
            return SQLResult(False, sql, error=msg, seconds=time.perf_counter() - t0)
        finally:
            conn.close()
        truncated = len(rows) > self.max_rows
        df = pd.DataFrame(rows[: self.max_rows], columns=cols)
        return SQLResult(True, sql, df, truncated=truncated, seconds=time.perf_counter() - t0)

    def fingerprint(self) -> str:
        """数据库文件内容的哈希：任务存档用它区分“是不是同一份数据”。"""
        h = hashlib.sha1(self.path.read_bytes()).hexdigest()[:16]
        return f"db-{self.name}-{h}"

    # ---------------- 读取表结构 ----------------
    def _introspect(self, docs: dict) -> list[Table]:
        conn = sqlite3.connect(f"file:{self.path.resolve().as_posix()}?mode=ro", uri=True)
        try:
            names = [r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name")]
            tables = []
            for name in names:
                doc = docs.get(name, {})
                col_docs = doc.get("columns", {})
                q = _quote(name)
                rows = conn.execute(f"SELECT COUNT(*) FROM {q}").fetchone()[0]
                cols = []
                for _, cname, ctype, *_rest in conn.execute(f"PRAGMA table_info({q})"):
                    samples = []
                    if "CHAR" in ctype.upper() or "TEXT" in ctype.upper() or not ctype:
                        top = conn.execute(f"SELECT {_quote(cname)}, COUNT(*) c FROM {q} WHERE {_quote(cname)} "
                                           f"IS NOT NULL GROUP BY 1 ORDER BY c DESC LIMIT 13").fetchall()
                        if len(top) <= 12:                     # 取值少的才算“类别列”，订单号这类不列样例
                            samples = [str(v) for v, _ in top]
                    cols.append(Column(cname, ctype, col_docs.get(cname, ""), samples))
                fks = [(r[3], r[2], r[4]) for r in conn.execute(f"PRAGMA foreign_key_list({q})")]
                tables.append(Table(name, doc.get("desc", ""), int(rows), cols, fks, bool(doc.get("separate"))))
            return tables
        finally:
            conn.close()

    @staticmethod
    def _table_text(t: Table) -> str:
        parts = [t.name, t.desc]
        for c in t.columns:
            parts += [c.name, c.desc, " ".join(c.samples)]
        return " ".join(parts)

    # ---------------- schema linking：挑出和问题相关的表 ----------------
    def link(self, question: str, context: str = "") -> list[tuple[str, str]]:
        """返回 [(表名, 选中原因)]。context 是上一轮对话（问题 + 结论），只在问题像追问时用来找取值。

        按以下顺序：

        1. 指标：问题里出现指标的别名（“投入产出比”→ ROAS），加入口径需要的表；
        2. 维度：出现维度的说法（“品类”“每个月”），加入该列所在的表；
        3. 取值：出现某列的取值（“钓竿”“店铺C”“加州”），加入这张表；同一个取值在几张表里都有时，
           优先已经选中的表，其次维度表（被别的表外键引用的表，如 products），再次关键词得分高的表
           （“钓竿”既是商品表的品类，也是广告表的品类，应该取商品表）；
        4. 关键词兜底：前三步什么都没匹配到时，按问题和“表说明 + 列说明”的 BM25 得分，取最高分一半以上的表；
        5. 追问：问题里有“它、那、这个……”时，上一轮对话里出现的取值也算（“那它的退货率呢”）；
        6. 补中间表：选中的表之间没有外键直接相连时，沿外键找最短路径，把路上的表补进来（JOIN 要用）。
           只走外键、不走逻辑关联，汇总表（separate）也不参与：图上连得通不等于 JOIN 语义对。
           例如“各店铺的 ROAS”，如果允许 店铺 → 广告表 → 商品表 这条“捷径”，就会漏掉算店铺销售额必需的 orders。
        仍然一张都没有时返回全部表。
        """
        chosen: dict[str, str] = {}
        q = question.upper()
        for name in self.matched_metrics(question):
            for t in self.metrics[name].get("tables", []):
                chosen.setdefault(t, f"指标“{name}”的口径需要")
        for name, d in self.dimensions.items():
            alias = next((a for a in d.get("aliases", []) if a.upper() in q), None)
            table = d.get("column", "").split(".")[0]
            if alias and table in self.by_name:
                chosen.setdefault(table, f"维度“{name}”（{alias}）在 {d['column']}")
        scores = dict(zip((t.name for t in self.tables), self.bm25.scores(tokenize(question))))
        follow_up = bool(context) and any(w in question for w in FOLLOW_UP_CUES)
        texts = [(question, "问题里")] + ([(context, "上一轮对话提到")] if follow_up else [])
        referenced = {ref for t in self.tables for _, ref, _ in t.foreign_keys}
        for text, where in texts:
            values: dict[str, list[tuple[str, str]]] = {}
            for t in self.tables:
                for c in t.columns:
                    for v in c.samples:
                        if len(v) >= 2 and v in text:
                            values.setdefault(v, []).append((t.name, c.name))
            for v, places in values.items():
                places.sort(key=lambda p: (p[0] not in chosen, p[0] not in referenced, -scores.get(p[0], 0)))
                t, c = places[0]
                chosen.setdefault(t, f"{where}的“{v}”是 {t}.{c} 的取值")
        if not chosen:
            top = max(scores.values(), default=0)
            for name, sc in scores.items():
                if top > 0 and sc >= top * 0.5:
                    chosen[name] = f"问题和表/列说明相关（关键词得分 {sc:.1f}）"
        if not chosen:
            return [(t.name, "没有明确相关的表，提供全部表") for t in self.tables]
        detail = [n for n in chosen if not self.by_name[n].separate]
        for name in self._bridges(detail):
            chosen.setdefault(name, "连接其他表需要经过它（外键）")
        return [(t.name, chosen[t.name]) for t in self.tables if t.name in chosen]

    def matched_metrics(self, question: str) -> list[str]:
        q = question.upper()
        return [n for n, m in self.metrics.items() if any(a.upper() in q for a in m.get("aliases", []))]

    def metrics_text(self, question: str) -> str:
        """问题涉及的指标口径（语义层），交给写 SQL 的模型。"""
        names = self.matched_metrics(question)
        if not names:
            return "（问题没有直接对应的预定义指标，按表和列的说明计算）"
        return "\n".join(f"- {n}：{self.metrics[n]['formula']}" for n in names)

    def _graph(self) -> dict[str, set[str]]:
        """外键图（不含汇总表）。"""
        g: dict[str, set[str]] = {t.name: set() for t in self.tables if not t.separate}
        for t in self.tables:
            for _, ref, _ in t.foreign_keys:
                if t.name in g and ref in g:
                    g[t.name].add(ref)
                    g[ref].add(t.name)
        return g

    def _bridges(self, names: list[str]) -> list[str]:
        if not names:
            return []
        """把选中的表连成一个整体所需的中间表（逐个用 BFS 找到已连通部分的最短路径）。
        邻居按表名排序后再遍历，同样长的路径有多条时结果固定，不随运行变化。"""
        g = {k: sorted(v) for k, v in self._graph().items()}
        connected, extra = {names[0]}, []
        for target in names[1:]:
            if target in connected:
                continue
            prev, queue, found = {target: None}, deque([target]), None
            while queue:
                cur = queue.popleft()
                if cur in connected:
                    found = cur
                    break
                for nb in g[cur]:
                    if nb not in prev:
                        prev[nb] = cur
                        queue.append(nb)
            node = found
            while node is not None:            # 从连通部分沿路径走回 target，路上的表都加入
                if node not in connected:
                    connected.add(node)
                    if node not in names:
                        extra.append(node)
                node = prev.get(node)
            connected.add(target)
        return extra

    # ---------------- 交给模型看的文字 ----------------
    def overview(self) -> str:
        """所有表的一行简介（工具模式开场时给模型看，详细结构用 describe_table 查）。"""
        return "\n".join(f"- {t.name}（{t.rows} 行）：{t.desc or '无说明'}" for t in self.tables)

    def schema_text(self, names: list[str] | None = None) -> str:
        """指定表的完整结构：列、类型、说明、取值样例、外键。"""
        parts = []
        for t in self.tables:
            if names is not None and t.name not in names:
                continue
            lines = [f"表 {t.name}（{t.rows} 行）：{t.desc}" + ("【汇总表：先单独汇总，再按维度和其他结果关联】" if t.separate else "")]
            for c in t.columns:
                line = f"  - {c.name} {c.type}" + (f"：{c.desc}" if c.desc else "")
                if c.samples:
                    line += f"；取值：{'、'.join(c.samples)}"
                lines.append(line)
            for col, ref, rcol in t.foreign_keys:
                lines.append(f"  外键：{t.name}.{col} → {ref}.{rcol}")
            for j in self.joins:
                if j["left"].split(".")[0] == t.name:
                    lines.append(f"  关联：{j['left']} = {j['right']}（{j.get('note', '逻辑关联')}）")
            parts.append("\n".join(lines))
        return "\n\n".join(parts)

    def profile(self) -> dict:
        """和表格数据的 profile() 同样的格式，供报告、核对数字使用。"""
        return {"rows": sum(t.rows for t in self.tables), "cols": sum(len(t.columns) for t in self.tables),
                "duplicate_rows": 0, "columns": [],
                "tables": [{"name": t.name, "rows": t.rows, "desc": t.desc,
                            "columns": [c.name for c in t.columns]} for t in self.tables]}

    def numbers(self) -> list[float]:
        """表的行数也是真实数字（报告写“订单明细共 6012 行”时算有出处）。"""
        return [float(t.rows) for t in self.tables]


def _quote(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _load_docs(path: Path) -> dict:
    side = path.with_suffix(".schema.json")
    if not side.exists():
        return {}
    try:
        data = json.loads(side.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        return {}
