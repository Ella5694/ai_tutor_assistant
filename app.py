import base64
import io
import json
from dataclasses import dataclass
from typing import Any, Dict, Generator, List, Optional, Tuple

import requests
import streamlit as st  # type: ignore[reportMissingImports]

# 依赖准备（请先在终端运行一次）：
#   pip install zhipuai PyMuPDF python-docx
# Eye-Brain 双模型流水线：
# - 智谱 GLM-4V：仅做“视觉/OCR 提取”
# - DeepSeek：仅做“逻辑推理 + 苏格拉底式教学”
try:
    import fitz  # type: ignore  # PyMuPDF
except Exception:
    fitz = None  # type: ignore

try:
    import docx  # type: ignore
except Exception:
    docx = None  # type: ignore

try:
    from zhipuai import ZhipuAI  # type: ignore
except Exception:
    ZhipuAI = None  # type: ignore


BASE_URL = "https://api.deepseek.com"


# 全局 LaTeX 格式铁律，所有 Prompt 必须引用此规则
LATEX_FORMAT_RULE = (
    "【格式铁律】所有数学公式必须使用标准的 LaTeX 格式。"
    "独立成行公式使用 $$公式$$，行内公式使用 $公式$。"
    "严禁使用 \\[ \\] 或 \\( \\)。"
    "所有英文下标必须包裹在 \\text{} 中，例如 $W_{\\text{in}}$。"
)


@dataclass
class QuestionContext:
    source_name: str
    source_type: str  # pdf | docx | image | text
    extracted_text: str = ""
    image_bytes: Optional[bytes] = None
    image_mime: Optional[str] = None


SOCRATIC_SYSTEM_PROMPT = f"""你是一位擅长苏格拉底式教学的老师，正在辅导学生做题。

【角色】
你是一个苏格拉底式的、严格且聪明的 AI 老师。

【排版规则】
- 语言必须极其精简，避免长段废话。
- 所有关键变量名、关键中间结论和最终结论必须使用 **加粗** 形式展示。

【公式铁律】
- 所有数学公式、数字计算必须严格包裹在 $公式$（行内）或 $$公式$$（独立行）中。
- 绝对禁止输出“裸露”的 LaTeX 片段，例如不能单独输出 \\frac、\\times 等，而是必须放在 $...$ 或 $$...$$ 中。
- {LATEX_FORMAT_RULE}

【解题逻辑（防幻觉）】
- 在你阅读题目后，必须先在心里明确列出所有已知变量（如：**输入尺寸**、**卷积核大小**、**步长**、**填充** 等），
  然后严格按照标准公式一步一步推导，不要被用户给出的错误中间结果带偏。
- 如果题目信息不足，必须先向学生确认或索取缺失条件，不能主观臆造。

【互动原则】
- 你的第一句话绝对不能给出答案或直接给出完整公式，必须先用 1-2 句简要描述考点，然后反问学生目前的思路或卡点。
- 在学生没有明确表示“完全不会”之前，每次只推进一个小步骤的思考（例如：先确认已知条件，再确认目标，再选择公式）。
- 用户上传的图片、PDF 或 Word 截图已经被 OCR 提取为文本交给了你。你收到题目后，**绝对不要在回复中重复或复述题目原文**，请直接根据题目内容开始你的第一轮苏格拉底式提问。
- 在排版时，严格使用 Markdown 标准语法进行换行，**绝对禁止输出任何 HTML 标签（如 <br> 等）**。

【终止条件（核心）】
- 如果你判断学生已经说出了**正确答案**，或者学生明确表示“我会了/不用你引导了”，
  那么你必须立刻停止继续追问，而是：
  1. 给出本题的【完整正确答案】（用 LaTeX 规范排版公式）；
  2. 给出【详细解析】，包含每一步推导的依据；
  3. 在输出的最后一行，独占一行严格输出标记：`[教学结束]`。

【对话启动规则】
- 当你识别到题目内容后，你的第一句话必须：
  1. 用 1-2 句概括本题涉及的考点或知识模块（不要解题）；
  2. 随后用提问的方式询问学生目前的思路或卡点，例如：“**你现在卡在了哪一步？**”。

【引导策略】
- 优先用提问方式定位学生的具体卡点（条件、目标、未知量、关键约束、方法选择等）。
- 必要时可以给出一个“更简单、类比本题结构”的小例子帮助学生迁移。
- Hint 只能给关键公式或思路的“线索”，不要把完整推导一次性写完，除非触发【终止条件】。
- 输出尽量分段、短句、易读。
"""


WEAKNESS_REPORT_SYSTEM_PROMPT = f"""你是一位经验丰富的学习诊断老师。
基于学生与老师的完整对话记录，诊断学生的：
1) 逻辑漏洞（推理跳步、条件遗漏、因果混淆等）
2) 知识点盲区（概念不清、公式误用、方法选择不当等）
3) 解题心态（急于求成、畏难、注意力分散等）

{LATEX_FORMAT_RULE}

请输出一份 200 字以内的中文诊断报告，内容具体、可执行、不要空泛说教。"""


ZHIPU_OCR_PROMPT = (
    "你是一个高精度的 OCR 引擎。"
    "请提取图片中的所有文字、数学公式和变量。"
    "不要尝试解答问题，只输出原始的题目文本和 LaTeX 公式。"
)


def _pdf_to_text_fitz(pdf_bytes: bytes) -> str:
    if fitz is None:
        return "（未能解析 PDF：缺少 PyMuPDF 依赖）"
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        texts: List[str] = []
        for page in doc:
            texts.append(page.get_text().strip())
        return "\n".join(t for t in texts if t).strip()
    except Exception as e:
        st.error(f"解析 PDF 文本时出错：{e}")
        return ""


def _pdf_first_page_image(pdf_bytes: bytes) -> Tuple[Optional[bytes], Optional[str]]:
    if fitz is None:
        st.error("当前环境未安装 PyMuPDF，无法对 PDF 截图。请运行 pip install PyMuPDF。")
        return None, None
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        if doc.page_count == 0:
            st.error("PDF 文件没有任何页面。")
            return None, None
        page = doc.load_page(0)
        pix = page.get_pixmap(dpi=200)
        img_bytes = pix.tobytes("png")
        return img_bytes, "image/png"
    except Exception as e:
        st.error(f"将 PDF 转为图片时出错：{e}")
        return None, None


def _docx_text_and_first_image(docx_bytes: bytes) -> Tuple[str, Optional[bytes], Optional[str]]:
    if docx is None:
        st.error("当前环境未安装 python-docx，无法读取 Word。请运行 pip install python-docx。")
        return "", None, None
    try:
        document = docx.Document(io.BytesIO(docx_bytes))
    except Exception as e:
        st.error(f"读取 Word 文档时出错：{e}")
        return "", None, None

    # 文本
    paras = [p.text.strip() for p in document.paragraphs if p.text and p.text.strip()]
    text = "\n".join(paras).strip()

    # 第一张图片（如果有）
    image_bytes: Optional[bytes] = None
    image_mime: Optional[str] = None
    try:
        rels = document.part.rels
        for rel in rels.values():
            if "image" in rel.target_ref:
                image_bytes = rel.target_part.blob
                image_mime = "image/png"  # Word 内部通常为 png/jpg，这里统一视作 png 处理
                break
    except Exception:
        image_bytes = None
        image_mime = None

    return text, image_bytes, image_mime


def _encode_image_b64_url(image_bytes: bytes, mime: str) -> str:
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def init_state() -> None:
    if "deepseek_api_key" not in st.session_state:
        st.session_state.deepseek_api_key = ""
    if "zhipu_api_key" not in st.session_state:
        st.session_state.zhipu_api_key = ""
    if "ocr_cache_fp" not in st.session_state:
        st.session_state.ocr_cache_fp = None
    if "ocr_text" not in st.session_state:
        st.session_state.ocr_text = ""
    if "ocr_displayed_fp" not in st.session_state:
        st.session_state.ocr_displayed_fp = None
    if "current_extracted_text" not in st.session_state:
        st.session_state.current_extracted_text = None
    if "messages" not in st.session_state:
        st.session_state.messages = []  # List[Dict[str, Any]]
    if "question_ctx" not in st.session_state:
        st.session_state.question_ctx = None
    if "question_fingerprint" not in st.session_state:
        st.session_state.question_fingerprint = None
    if "auto_analysis_done_fp" not in st.session_state:
        st.session_state.auto_analysis_done_fp = None
    if "report_text" not in st.session_state:
        st.session_state.report_text = ""


def question_fingerprint(file_name: str, file_bytes: bytes) -> str:
    head = base64.b64encode(file_bytes[:64]).decode("utf-8")
    return f"{file_name}:{len(file_bytes)}:{head}"


def deepseek_stream_chat(
    *,
    api_key: str,
    model: str,
    messages: List[Dict[str, Any]],
    base_url: str = BASE_URL,
    temperature: float = 0.4,
) -> Generator[str, None, None]:
    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "stream": True,
    }

    with requests.post(url, headers=headers, json=payload, stream=True, timeout=120) as r:
        r.raise_for_status()
        for raw in r.iter_lines(decode_unicode=True):
            if not raw:
                continue
            line = raw.strip()
            if not line.startswith("data:"):
                continue
            data = line[len("data:") :].strip()
            if data == "[DONE]":
                break
            try:
                evt = json.loads(data)
                delta = evt["choices"][0].get("delta", {})
                chunk = delta.get("content")
                if chunk:
                    yield chunk
            except Exception:
                continue


def deepseek_chat_once(
    *,
    api_key: str,
    model: str,
    messages: List[Dict[str, Any]],
    base_url: str = BASE_URL,
    temperature: float = 0.4,
) -> str:
    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "stream": False,
    }
    r = requests.post(url, headers=headers, json=payload, timeout=120)
    r.raise_for_status()
    data = r.json()
    return data["choices"][0]["message"]["content"]


def zhipu_ocr_image_once(*, api_key: str, image_bytes: bytes, mime: str) -> str:
    """
    使用智谱 GLM-4V 仅做 OCR/视觉文字提取，不做解题。
    """
    if ZhipuAI is None:
        raise RuntimeError("当前环境未安装 zhipuai，请先运行 pip install zhipuai。")
    img_url = _encode_image_b64_url(image_bytes, mime)
    client = ZhipuAI(api_key=api_key)
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": ZHIPU_OCR_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "请对这张题目截图进行 OCR 提取："},
                {"type": "image_url", "image_url": {"url": img_url}},
            ],
        },
    ]
    resp = client.chat.completions.create(model="glm-4v", messages=messages, stream=False)
    try:
        msg = resp.choices[0].message
        content = getattr(msg, "content", None)
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts: List[str] = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    parts.append(part.get("text", ""))
            return "".join(parts).strip()
    except Exception:
        pass
    return str(resp).strip()


def build_context_block(ctx: Optional[QuestionContext]) -> str:
    if not ctx:
        return ""
    text = (ctx.extracted_text or "").strip()
    if ctx.source_type == "image":
        return f"题目来自图片文件：{ctx.source_name}。如果你无法看清，请提示学生重新上传更清晰的截图。"
    if ctx.source_type == "pdf":
        return f"题目来自 PDF 文件：{ctx.source_name}\n提取到的文本如下：\n{text}"
    if ctx.source_type == "docx":
        return f"题目来自 Word 文件：{ctx.source_name}\n提取到的文本如下：\n{text}"
    return text


def build_deepseek_messages_for_turn(
    *,
    history: List[Dict[str, Any]],
    ctx: Optional[QuestionContext],
    user_input: Optional[str],
) -> Tuple[str, List[Dict[str, Any]]]:
    context_block = build_context_block(ctx)
    model = "deepseek-chat"
    msgs: List[Dict[str, Any]] = [{"role": "system", "content": SOCRATIC_SYSTEM_PROMPT}]
    if context_block:
        msgs.append({"role": "system", "content": f"【题目上下文】\n{context_block}"})

    for m in history:
        # 允许注入“系统提示”作为隐藏上下文（例如 OCR 结果）
        if m["role"] in ("user", "assistant", "system"):
            msgs.append({"role": m["role"], "content": m["content"]})

    if user_input is not None:
        msgs.append({"role": "user", "content": user_input})
    return model, msgs


def _split_safe_math_prefix(text: str) -> Tuple[str, str]:
    in_inline = False
    in_block = False
    i = 0
    last_safe = 0

    while i < len(text):
        if text.startswith("$$", i):
            in_block = not in_block
            i += 2
        elif text[i] == "$":
            if not in_block:
                in_inline = not in_inline
            i += 1
        else:
            i += 1

        if not in_inline and not in_block:
            last_safe = i

    return text[:last_safe], text[last_safe:]


def sanitize_markdown(text: str) -> str:
    if not text:
        return text

    # HTML 换行标签 -> Markdown 换行
    text = (
        text.replace("<br/>", "\n")
        .replace("<br />", "\n")
        .replace("<br>", "\n")
        .replace("</br>", "\n")
    )

    # 如果完全没有 $ 却出现明显 LaTeX 片段，尽量整体包裹一次，避免“裸露公式”
    if "$" not in text and any(tok in text for tok in ["\\frac", "\\times", "\\cdot"]):
        text = f"${text}$"

    # 仅在非数学环境中做替换，避免破坏正确的 LaTeX
    parts = text.split("$")
    for i in range(0, len(parts), 2):
        # 偶数索引：不在数学环境内
        segment = parts[i]
        segment = segment.replace("\\[", "$$").replace("\\]", "$$")
        # 如果智谱偶尔输出 \times / \cdot 等裸露符号，在非数学环境下转为普通字符
        segment = segment.replace("\\times", "×").replace("\\cdot", "·")
        parts[i] = segment

    return "$".join(parts)


def stream_assistant_reply() -> Optional[str]:
    deepseek_key = (st.session_state.deepseek_api_key or "").strip()
    zhipu_key = (st.session_state.zhipu_api_key or "").strip()
    if not deepseek_key or not zhipu_key:
        st.warning("请先在侧边栏同时填写 **DeepSeek API Key** 和 **智谱 API Key**。")
        return None

    ctx: Optional[QuestionContext] = st.session_state.question_ctx

    with st.chat_message("assistant"):
        container = st.empty()
        acc = ""
        try:
            with st.spinner("🔍 老师正在查看题目并思考下一步引导..."):
                model_name, api_messages = build_deepseek_messages_for_turn(
                    history=st.session_state.messages,
                    ctx=ctx,
                    user_input=None,
                )
                for chunk in deepseek_stream_chat(api_key=deepseek_key, model=model_name, messages=api_messages):
                    if not chunk:
                        continue
                    acc += chunk
                    safe_prefix, _ = _split_safe_math_prefix(acc)
                    if safe_prefix:
                        container.markdown(sanitize_markdown(safe_prefix), help=None)
        except Exception as e:
            container.empty()
            st.error(f"发生具体错误: {e}")
            return None

    final_safe, _ = _split_safe_math_prefix(acc)
    if not final_safe:
        return None

    st.session_state.messages.append({"role": "assistant", "content": final_safe})

    # 如果本轮回复中包含教学结束标记，立刻强制刷新页面，
    # 使下方历史渲染逻辑接管替换与按钮展示，避免原始暗号在界面停留。
    if "[教学结束]" in final_safe:
        st.rerun()

    return final_safe


def _generate_weakness_report(trigger: str = "sidebar") -> None:
    """
    统一的薄弱点报告生成逻辑。
    trigger: "sidebar" 或 "inline"，用于区分加载文案。
    """
    deepseek_key = (st.session_state.deepseek_api_key or "").strip()
    zhipu_key = (st.session_state.zhipu_api_key or "").strip()
    if not deepseek_key or not zhipu_key:
        st.warning("请先在侧边栏同时填写 **DeepSeek API Key** 和 **智谱 API Key**。")
        return
    if not st.session_state.messages:
        st.warning("当前还没有对话记录，先和老师聊几轮再生成报告。")
        return

    convo_lines: List[str] = []
    for m in st.session_state.messages:
        role_cn = "学生" if m["role"] == "user" else "老师"
        convo_lines.append(f"{role_cn}：{m['content']}")
    convo_text = "\n".join(convo_lines).strip()

    report_messages = [
        {"role": "system", "content": WEAKNESS_REPORT_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "以下是我与学生在做题过程中的完整对话记录，请基于这些信息生成一份精炼的能力诊断报告：\n"
                f"{convo_text}\n\n"
                "请严格遵守上面的【格式铁律】，在需要使用公式的地方使用 LaTeX 表达。"
            ),
        },
    ]

    spinner_text = "正在生成报告..." if trigger == "inline" else "📊 AI 老师正在深度复盘您的答题表现，请稍候..."
    with st.container():
        with st.spinner(spinner_text):
            try:
                report = deepseek_chat_once(
                    api_key=deepseek_key, model="deepseek-chat", messages=report_messages
                )
                st.session_state.report_text = report.strip()
            except Exception as e:
                st.error(f"发生具体错误: {e}")
                st.session_state.report_text = "老师刚才走神了，报告生成失败，请稍后再试一次。"

    if st.session_state.report_text:
        st.success("📊 个人能力诊断报告")
        st.markdown(sanitize_markdown(st.session_state.report_text), help=None)


def main() -> None:
    st.set_page_config(page_title="AI 启发式错题助手", layout="wide")
    init_state()

    st.title("AI 启发式错题助手")

    with st.sidebar:
        st.subheader("API 设置（眼脑分离）")
        st.session_state.deepseek_api_key = st.text_input(
            "DeepSeek API Key（逻辑大脑）",
            value=st.session_state.deepseek_api_key,
            type="password",
            placeholder="请输入你的 DeepSeek API Key",
        )
        st.text_input("DeepSeek Base URL", value=BASE_URL, disabled=True)
        st.session_state.zhipu_api_key = st.text_input(
            "智谱 API Key（视觉 OCR）",
            value=st.session_state.zhipu_api_key,
            type="password",
            placeholder="请输入你的 ZhipuAI API Key",
        )

        st.divider()
        st.subheader("功能操作区")
        gen_report = st.button("生成薄弱点报告", use_container_width=True)

    # 上传组件（全场景：图片/PDF/Word）
    upload_types = ["pdf", "docx", "png", "jpg", "jpeg"]
    upload_label = "上传题目的 PDF、Word 或图片（JPG/PNG）"
    upload_help = "上传含截图的题目后，将先进行 OCR 扫描，再由 DeepSeek 老师进行启发式引导。"

    uploaded = st.file_uploader(
        label=upload_label,
        type=upload_types,
        accept_multiple_files=False,
        help=upload_help,
    )

    if uploaded is not None:
        file_bytes = uploaded.read()
        fp = question_fingerprint(uploaded.name, file_bytes)
        if st.session_state.question_fingerprint != fp:
            st.session_state.question_fingerprint = fp
            st.session_state.report_text = ""
            st.session_state.messages = []
            st.session_state.auto_analysis_done_fp = None
            st.session_state.ocr_cache_fp = None
            st.session_state.ocr_text = ""
            st.session_state.current_extracted_text = None

            name_lower = uploaded.name.lower()
            ctx: Optional[QuestionContext]

            try:
                with st.container():
                    with st.spinner("🔍 老师正在仔细查看您的题目排版和图片内容，请稍候..."):
                        deepseek_key = (st.session_state.deepseek_api_key or "").strip()
                        zhipu_key = (st.session_state.zhipu_api_key or "").strip()
                        if not deepseek_key or not zhipu_key:
                            st.warning("请先在侧边栏同时填写 **DeepSeek API Key** 和 **智谱 API Key**。")
                            ctx = None
                        else:
                            ocr_image_bytes: Optional[bytes] = None
                            ocr_mime: Optional[str] = None
                            local_text = ""

                            if name_lower.endswith((".png", ".jpg", ".jpeg")):
                                ocr_image_bytes = file_bytes
                                ocr_mime = "image/png" if name_lower.endswith(".png") else "image/jpeg"
                                local_text = ""
                            elif name_lower.endswith(".pdf"):
                                local_text = _pdf_to_text_fitz(file_bytes)
                                ocr_image_bytes, ocr_mime = _pdf_first_page_image(file_bytes)
                            elif name_lower.endswith(".docx"):
                                local_text, img_bytes, mime = _docx_text_and_first_image(file_bytes)
                                ocr_image_bytes, ocr_mime = img_bytes, mime
                            else:
                                st.error("暂不支持该文件格式，请上传 PDF、Word 或常见图片格式（PNG/JPG）。")
                                ctx = None

                            if ocr_image_bytes and ocr_mime:
                                # 调用智谱 GLM-4V 做 OCR（眼睛）
                                ocr_text = zhipu_ocr_image_once(
                                    api_key=zhipu_key, image_bytes=ocr_image_bytes, mime=ocr_mime
                                )
                                st.session_state.ocr_cache_fp = fp
                                st.session_state.ocr_text = ocr_text

                                merged = ocr_text.strip()
                                if local_text.strip():
                                    merged = f"{merged}\n\n（文档可复制文本补充）\n{local_text.strip()}"

                                ctx = QuestionContext(
                                    source_name=uploaded.name,
                                    source_type="text",
                                    extracted_text=merged.strip(),
                                )
                            else:
                                # 无图片可 OCR（例如纯文字 Word），直接走本地文本提取
                                ctx = QuestionContext(
                                    source_name=uploaded.name,
                                    source_type="text",
                                    extracted_text=local_text.strip(),
                                )
            except Exception as e:
                st.error(f"处理上传文件时发生错误：{e}")
                ctx = None

            st.session_state.question_ctx = ctx

            if ctx:
                extracted = (ctx.extracted_text or "").strip()
                if len(extracted) < 5:
                    st.warning(
                        "⚠️ 无法读取题目内容。您上传的文档可能是一张图片截图。目前的 AI 老师更擅长阅读“纯文本”文档，"
                        "请尝试将题目文字直接复制到文档中再上传，或直接在下方输入框告诉老师题目。"
                    )
                    extracted = ""

                if extracted:
                    cleaned_extracted = sanitize_markdown(extracted).strip()
                    # OCR/解析结果写入“当前题目内容”，用于置顶展示
                    st.session_state.ocr_text = cleaned_extracted
                    st.session_state.current_extracted_text = cleaned_extracted
                    # 注入一条“隐藏系统提示”保存 OCR/解析结果（不会在聊天流展示给用户）
                    st.session_state.messages.append(
                        {
                            "role": "system",
                            "content": f"[系统提示：用户上传了文件，内容识别为：{cleaned_extracted}]",
                        }
                    )
                    # 不在聊天里重复展示整段题干，给一个简短的启动消息即可
                    st.session_state.messages.append(
                        {
                            "role": "user",
                            "content": "我已上传题目，请直接开始你的第一轮苏格拉底式提问。",
                        }
                    )
                else:
                    st.session_state.messages.append(
                        {
                            "role": "user",
                            "content": "这是我上传的题目内容：\n\n（未能从文件中提取到可用文本，请你提示我如何补充题干信息。）",
                        }
                    )

    # 置顶展示区：当前题目内容（可折叠）
    if st.session_state.current_extracted_text:
        with st.expander("🔍 当前题目内容 (点击展开/收起)", expanded=True):
            st.info(st.session_state.current_extracted_text)

    # 聊天历史渲染（支持“[教学结束]”内联报告按钮）
    for idx, m in enumerate(st.session_state.messages):
        is_assistant = m["role"] == "assistant"
        raw_content = m.get("content", "") or ""

        # 隐藏仅供 DeepSeek 使用的系统提示（OCR 结果等），不在前端展示
        if m["role"] == "system" and "[系统提示：" in raw_content:
            continue

        has_end_marker = "[教学结束]" in raw_content
        if is_assistant and has_end_marker:
            display_content = raw_content.replace("[教学结束]", "").strip()
        else:
            display_content = raw_content.strip()

        with st.chat_message("assistant" if is_assistant else "user"):
            st.markdown(sanitize_markdown(display_content), help=None)

            # 如果本轮教学已经结束，提供行内报告按钮
            if is_assistant and has_end_marker:
                st.markdown("✅ **教学结束，恭喜你掌握了本题！**", help=None)
                if st.button(
                    "📝 恭喜完成！点击生成本次诊断报告",
                    key=f"report_{idx}",
                ):
                    _generate_weakness_report(trigger="inline")

    # 上传后自动触发首轮分析（只触发一次）
    if (
        st.session_state.question_fingerprint
        and st.session_state.messages
        and st.session_state.messages[-1]["role"] == "user"
        and st.session_state.auto_analysis_done_fp != st.session_state.question_fingerprint
    ):
        reply = stream_assistant_reply()
        if reply is not None:
            st.session_state.auto_analysis_done_fp = st.session_state.question_fingerprint

    # 生成薄弱点报告
    if gen_report:
        _generate_weakness_report(trigger="sidebar")

    # 聊天输入
    user_text = st.chat_input("把你目前的思路/卡点发给老师…")
    if user_text:
        st.session_state.report_text = ""
        st.session_state.messages.append({"role": "user", "content": user_text})

        with st.chat_message("user"):
            st.markdown(user_text)
        stream_assistant_reply()


if __name__ == "__main__":
    main()
