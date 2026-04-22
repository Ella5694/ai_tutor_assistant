"""
run_eval.py — 项目二评测脚本 v1.1
放置路径：ai_tutor_assistant/test/run_eval.py
评测集路径：ai_tutor_assistant/test/项目二_评测集_v1.0.csv
结果输出：ai_tutor_assistant/test/项目二_评测结果_v1.0.csv（新文件，不覆盖原始评测集）

使用方法：
  cd ai_tutor_assistant
  python test/run_eval.py
"""

import csv
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests

# ── 路径配置 ────────────────────────────────────────────────
TEST_DIR = Path(__file__).resolve().parent          # test/
ROOT_DIR = TEST_DIR.parent                          # 项目根目录（app.py所在）
CSV_INPUT  = TEST_DIR / "项目二_评测集_v1.0.csv"
CSV_OUTPUT = TEST_DIR / "项目二_评测结果_v1.0.csv"

# ── 环境变量加载（从根目录的 .env 读取）────────────────────
def _load_env() -> None:
    env_file = ROOT_DIR / ".env"
    if not env_file.exists():
        return
    with env_file.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = val

_load_env()

DIFY_API_KEY = os.getenv("DIFY_API_KEY", "").strip()
DIFY_API_URL = os.getenv("DIFY_API_URL", "https://api.dify.ai/v1").rstrip("/")

# ── Dify 通信 ────────────────────────────────────────────────
def stream_chat(query: str, conversation_id: str = "", user_id: str = "eval-v1") -> Tuple[str, str]:
    """
    向 Dify 发送一条消息，返回 (完整回复文本, conversation_id)。
    conversation_id 为空字符串时开启新会话。
    """
    url = f"{DIFY_API_URL}/chat-messages"
    headers = {
        "Authorization": f"Bearer {DIFY_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "inputs": {},
        "query": query,
        "response_mode": "streaming",
        "conversation_id": conversation_id,
        "user": user_id,
    }

    answer = ""
    conv_id = conversation_id

    try:
        with requests.post(url, headers=headers, json=payload, stream=True, timeout=120) as r:
            r.raise_for_status()
            for raw in r.iter_lines(decode_unicode=True):
                if not raw:
                    continue
                line = raw.strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                try:
                    evt = json.loads(data)
                except Exception:
                    continue
                if evt.get("conversation_id"):
                    conv_id = evt["conversation_id"]
                if evt.get("event") in ("message", "agent_message"):
                    answer += evt.get("answer", "")
                if evt.get("event") == "message_end":
                    break
    except requests.HTTPError as e:
        code = e.response.status_code if e.response is not None else "?"
        body = ""
        try:
            body = e.response.text[:200]
        except Exception:
            pass
        print(f"  [HTTP错误 {code}] {body}", file=sys.stderr)
    except Exception as e:
        print(f"  [请求异常] {e}", file=sys.stderr)

    return answer.strip(), conv_id


# ── 自动评分 ────────────────────────────────────────────────
def _split_keywords(text: str) -> List[str]:
    """用中文标点和分号分割关键词列表，过滤空项和'无'"""
    parts = re.split(r"[；;]", text or "")
    return [p.strip() for p in parts if p.strip() and p.strip() not in ("无", "")]


def auto_score(expected: str, forbidden: str, actual: str) -> Tuple[int, str]:
    """
    简单规则评分（0-5分）：
    - 初始 5 分
    - 命中禁止内容每项 -2 分
    - 期望方向关键词覆盖率低于 30% 再 -1 分
    安全下限 0，上限 5。
    """
    if not actual.strip():
        return 0, "无有效输出"

    score = 5
    reasons: List[str] = []

    # 禁止内容检查
    forbidden_hits = [f for f in _split_keywords(forbidden) if f in actual]
    if forbidden_hits:
        deduct = min(4, len(forbidden_hits) * 2)
        score -= deduct
        reasons.append(f"命中禁止内容：{'|'.join(forbidden_hits[:3])}")

    # 期望方向关键词覆盖（取长度≥4的词，最多取前6个）
    exp_tokens = [t for t in re.split(r"[，,。；;、\s]+", expected or "") if len(t) >= 4][:6]
    if exp_tokens:
        hit_count = sum(1 for t in exp_tokens if t in actual)
        ratio = hit_count / len(exp_tokens)
        if ratio < 0.3:
            score -= 1
            reasons.append(f"期望关键词覆盖率低（{hit_count}/{len(exp_tokens)}）")
        else:
            reasons.append(f"期望关键词覆盖率{hit_count}/{len(exp_tokens)}")

    score = max(0, min(5, score))
    return score, "；".join(reasons) if reasons else "基本符合"


# ── 多轮会话预热 ────────────────────────────────────────────
def warmup_conversation(case_id: str, user_id: str) -> str:
    """
    对需要上下文的 case，先跑若干轮预热，返回 conversation_id。
    case 19：连续3轮不知道（测试提示升级）
    case 20：模拟5轮完整引导对话后触发报告
    case 21：仅1轮浅对话后触发报告（测试不足时边界）
    case 27：先问一道椭圆题被拒，再切换全域模式
    """
    conv_id = ""

    if case_id == "19":
        _, conv_id = stream_chat("求 f(x)=x³-3x 的极值", "", user_id)
        time.sleep(0.5)
        _, conv_id = stream_chat("不知道", conv_id, user_id)
        time.sleep(0.5)
        _, conv_id = stream_chat("不知道", conv_id, user_id)
        time.sleep(0.5)
        # 第3次不知道由主循环发送

    elif case_id == "20":
        _, conv_id = stream_chat("求 f(x)=x³-3x 的极值", "", user_id)
        time.sleep(0.5)
        _, conv_id = stream_chat("我会求导 f'(x)=3x²-3 然后不知道怎么办", conv_id, user_id)
        time.sleep(0.5)
        _, conv_id = stream_chat("解出 x=1 和 x=-1", conv_id, user_id)
        time.sleep(0.5)
        _, conv_id = stream_chat("x=1时从负变正是极小值 x=-1时从正变负是极大值", conv_id, user_id)
        time.sleep(0.5)
        _, conv_id = stream_chat("复合函数的链式法则是什么", conv_id, user_id)
        time.sleep(0.5)
        # 主循环发送「生成薄弱点报告」

    elif case_id == "21":
        _, conv_id = stream_chat("导数是什么意思？", "", user_id)
        time.sleep(0.5)
        # 主循环发送「生成薄弱点报告」

    elif case_id == "27":
        _, conv_id = stream_chat("椭圆 x²/4+y²/3=1 的焦点坐标是多少？", "", user_id)
        time.sleep(0.5)
        # 主循环发送「开启全域模式」

    return conv_id


# ── 主流程 ───────────────────────────────────────────────────
def main() -> None:
    # 前置检查
    if not DIFY_API_KEY:
        sys.exit("❌ 缺少 DIFY_API_KEY，请在根目录 .env 中配置后重试。")
    if not CSV_INPUT.exists():
        sys.exit(f"❌ 评测集文件不存在：{CSV_INPUT}")

    # 读取原始评测集
    with CSV_INPUT.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        original_fieldnames = list(reader.fieldnames or [])
        rows: List[Dict[str, str]] = list(reader)

    if not rows:
        sys.exit("❌ 评测集为空")

    print(f"✅ 读取评测集：{len(rows)} 条用例")
    print(f"   输入文件：{CSV_INPUT}")
    print(f"   结果输出：{CSV_OUTPUT}\n")

    # 输出列名 = 原始列 + 3个新列
    result_fieldnames = original_fieldnames + ["实际输出", "评分(0-5)", "评分说明"]
    results: List[Dict[str, str]] = []

    # 记录多轮对话的 conversation_id（key = case_id，value = conv_id）
    conv_map: Dict[str, str] = {}

    for row in rows:
        case_id = row["序号"].strip()
        raw_input = row["输入（学生发送的内容）"].strip()
        expected  = row["期望输出方向"].strip()
        forbidden = row["期望不出现的内容"].strip()

        # ── 确定发给 Dify 的实际 query ──────────────────────
        # 去掉「【接续N】」前缀标记（这是评测集的说明，不是实际输入）
        query = re.sub(r"^【接续\d+】", "", raw_input).strip()

        # ── 确定 conversation_id ───────────────────────────
        conv_id = ""

        # 接续上一轮的情形：前缀为「【接续N】」
        m = re.match(r"^【接续(\d+)】", raw_input)
        if m:
            ref_id = m.group(1)
            conv_id = conv_map.get(ref_id, "")

        # 需要预热的特殊用例
        if case_id in ("19", "20", "21", "27"):
            conv_id = warmup_conversation(case_id, f"eval-{case_id}")

        # ── 发送请求 ──────────────────────────────────────
        print(f"[{case_id:>2}/30] 发送：{query[:50]}{'...' if len(query)>50 else ''}")
        actual, new_conv = stream_chat(query, conv_id, f"eval-{case_id}")
        conv_map[case_id] = new_conv

        # ── 自动评分 ──────────────────────────────────────
        score, reason = auto_score(expected, forbidden, actual)
        print(f"         评分：{score}/5  |  {reason}")

        result_row = dict(row)
        result_row["实际输出"]  = actual
        result_row["评分(0-5)"] = str(score)
        result_row["评分说明"]  = reason
        results.append(result_row)

        # 避免频繁请求触发限流
        time.sleep(0.8)

    # ── 写入结果文件 ───────────────────────────────────────
    with CSV_OUTPUT.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=result_fieldnames, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(results)

    # ── 汇总统计 ──────────────────────────────────────────
    scores = [int(r["评分(0-5)"]) for r in results]
    avg    = sum(scores) / len(scores)
    dist   = {s: scores.count(s) for s in range(6)}

    print("\n" + "="*50)
    print(f"  评测完成  |  共 {len(results)} 条  |  平均分 {avg:.2f}/5")
    print(f"  分布：" + "  ".join(f"{s}分×{dist[s]}" for s in range(5, -1, -1) if dist[s]))
    print(f"  结果已写入：{CSV_OUTPUT}")
    print("="*50)


if __name__ == "__main__":
    main()