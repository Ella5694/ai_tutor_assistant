import base64
import io
import json
import os
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import requests
import streamlit as st  # type: ignore[reportMissingImports]
from dotenv import load_dotenv  # type: ignore[reportMissingImports]

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
except Exception as _zhipu_import_err:
    ZHIPUAI_IMPORT_ERROR = repr(_zhipu_import_err)
    ZhipuAI = None  # type: ignore
else:
    ZHIPUAI_IMPORT_ERROR = ""

load_dotenv()

ZHIPU_API_KEY = os.getenv("ZHIPU_API_KEY", "").strip()
DIFY_API_KEY = os.getenv("DIFY_API_KEY", "").strip()
DIFY_API_URL = os.getenv("DIFY_API_URL", "https://api.dify.ai/v1").rstrip("/")

ZHIPU_OCR_PROMPT = (
    "你是一个高精度的 OCR 引擎。"
    "请提取图片中的所有文字、数学公式和变量。"
    "不要尝试解答问题，只输出原始的题目文本和 LaTeX 公式。"
)


@dataclass
class QuestionContext:
    source_name: str
    source_type: str  # pdf | docx | image | text
    extracted_text: str = ""
    image_bytes: Optional[bytes] = None
    image_mime: Optional[str] = None


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
        st.error("当前环境未安装 PyMuPDF，无法对 PDF 截图。")
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
        st.error("当前环境未安装 python-docx，无法读取 Word。")
        return "", None, None
    try:
        document = docx.Document(io.BytesIO(docx_bytes))
    except Exception as e:
        st.error(f"读取 Word 文档时出错：{e}")
        return "", None, None

    paras = [p.text.strip() for p in document.paragraphs if p.text and p.text.strip()]
    text = "\n".join(paras).strip()
    image_bytes: Optional[bytes] = None
    image_mime: Optional[str] = None
    try:
        rels = document.part.rels
        for rel in rels.values():
            if "image" in rel.target_ref:
                image_bytes = rel.target_part.blob
                image_mime = "image/png"
                break
    except Exception:
        image_bytes, image_mime = None, None
    return text, image_bytes, image_mime


def _encode_image_b64_url(image_bytes: bytes, mime: str) -> str:
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def sanitize_markdown(text: str) -> str:
    """
    仅做 HTML 换行标签替换，不对 $ 符号做任何处理。
    Dify 已确保输出 $$ 格式，Streamlit 的 st.markdown() 原生支持 $$ 渲染。
    """
    if not text:
        return text
    text = (
        text.replace("<br/>", "\n")
        .replace("<br />", "\n")
        .replace("<br>", "\n")
        .replace("</br>", "\n")
    )
    return text


def init_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "question_ctx" not in st.session_state:
        st.session_state.question_ctx = None
    if "question_fingerprint" not in st.session_state:
        st.session_state.question_fingerprint = None
    if "auto_analysis_done_fp" not in st.session_state:
        st.session_state.auto_analysis_done_fp = None
    if "current_extracted_text" not in st.session_state:
        st.session_state.current_extracted_text = None
    if "report_text" not in st.session_state:
        st.session_state.report_text = ""
    if "dify_conversation_id" not in st.session_state:
        st.session_state.dify_conversation_id = ""
    if "user_id" not in st.session_state:
        st.session_state.user_id = f"student-{uuid.uuid4().hex[:8]}"
    if "turn_count" not in st.session_state:
        st.session_state.turn_count = 0


def question_fingerprint(file_name: str, file_bytes: bytes) -> str:
    head = base64.b64encode(file_bytes[:64]).decode("utf-8")
    return f"{file_name}:{len(file_bytes)}:{head}"


def _ensure_env_keys() -> bool:
    missing: List[str] = []
    if not ZHIPU_API_KEY:
        missing.append("ZHIPU_API_KEY")
    if not DIFY_API_KEY:
        missing.append("DIFY_API_KEY")
    if not DIFY_API_URL:
        missing.append("DIFY_API_URL")
    if missing:
        st.error(f"环境变量缺失：{', '.join(missing)}。请配置 .env 后重启应用。")
        return False
    return True


def zhipu_ocr_image_once(*, image_bytes: bytes, mime: str) -> str:
    if ZhipuAI is None:
        raise RuntimeError(f"zhipuai 无法导入：{ZHIPUAI_IMPORT_ERROR}")
    img_url = _encode_image_b64_url(image_bytes, mime)
    client = ZhipuAI(api_key=ZHIPU_API_KEY)
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


def _format_dify_error(exc: Exception) -> str:
    if isinstance(exc, requests.HTTPError) and exc.response is not None:
        code = exc.response.status_code
        body = ""
        try:
            body = exc.response.text[:500]
        except Exception:
            pass
        if code == 401:
            return f"Dify 认证失败（401）。请检查 DIFY_API_KEY。{body}"
        return f"Dify 请求失败（HTTP {code}）。{body or str(exc)}"
    return f"发生具体错误: {exc}"


def dify_stream_chat(query: str):
    url = f"{DIFY_API_URL}/chat-messages"
    headers = {
        "Authorization": f"Bearer {DIFY_API_KEY}",
        "Content-Type": "application/json",
    }
    payload: Dict[str, Any] = {
        "inputs": {},
        "query": query,
        "response_mode": "streaming",
        "conversation_id": st.session_state.dify_conversation_id or "",
        "user": st.session_state.user_id,
    }
    conv_id = st.session_state.dify_conversation_id or ""

    with requests.post(url, headers=headers, json=payload, stream=True, timeout=180) as r:
        r.raise_for_status()
        for raw in r.iter_lines(decode_unicode=True):
            if not raw:
                continue
            line = raw.strip()
            if not line.startswith("data:"):
                continue
            data = line[len("data:") :].strip()
            try:
                evt = json.loads(data)
            except Exception:
                continue
            if evt.get("conversation_id"):
                conv_id = evt.get("conversation_id", conv_id)
                yield {"event": "conversation", "conversation_id": conv_id}
            if evt.get("event") in ("message", "agent_message"):
                chunk = evt.get("answer", "")
                if chunk:
                    yield {"event": "chunk", "text": chunk, "conversation_id": conv_id}
            if evt.get("event") == "message_end":
                yield {"event": "end", "conversation_id": conv_id}
                break


def stream_assistant_reply(query: Optional[str] = None) -> Optional[str]:
    if not _ensure_env_keys():
        return None
    if query is None:
        for m in reversed(st.session_state.messages):
            if m.get("role") == "user":
                query = m.get("content", "")
                break
    if not query:
        return None

    with st.chat_message("assistant"):
        container = st.empty()
        acc = ""
        try:
            with st.spinner("🔍 老师正在思考下一步引导..."):
                for evt in dify_stream_chat(query):
                    if evt.get("event") == "conversation" and evt.get("conversation_id"):
                        st.session_state.dify_conversation_id = evt["conversation_id"]
                        continue
                    if evt.get("event") == "chunk":
                        acc += evt.get("text", "")
                        container.markdown(sanitize_markdown(acc), help=None)
        except Exception as e:
            container.empty()
            st.error(_format_dify_error(e))
            return None

    final_safe = sanitize_markdown(acc).strip()
    if not final_safe:
        return None
    st.session_state.messages.append({"role": "assistant", "content": final_safe})
    st.session_state.turn_count += 1
    if "[教学结束]" in final_safe:
        st.rerun()
    return final_safe


def _generate_weakness_report(trigger: str = "sidebar") -> None:
    """
    只发送触发词「生成薄弱点报告」，让 Dify 的 System Prompt 接管报告生成逻辑。
    """
    if not _ensure_env_keys():
        return
    if not st.session_state.messages:
        st.warning("当前还没有对话记录，先和老师聊几轮再生成报告。")
        return
    trigger_query = "生成薄弱点报告"

    spinner_text = "正在生成报告..." if trigger == "inline" else "📊 AI 老师正在深度复盘您的答题表现，请稍候..."
    with st.container():
        with st.spinner(spinner_text):
            try:
                report = ""
                for evt in dify_stream_chat(trigger_query):
                    if evt.get("event") == "conversation" and evt.get("conversation_id"):
                        st.session_state.dify_conversation_id = evt["conversation_id"]
                    if evt.get("event") == "chunk":
                        report += evt.get("text", "")
                st.session_state.report_text = report.strip()
                st.session_state.messages.append({"role": "user", "content": trigger_query})
                st.session_state.messages.append({"role": "assistant", "content": report.strip()})
            except Exception as e:
                st.error(_format_dify_error(e))
                st.session_state.report_text = "报告生成失败，请稍后再试。"

    if st.session_state.report_text:
        st.success("📊 个人能力诊断报告")
        st.markdown(sanitize_markdown(st.session_state.report_text), help=None)


def main() -> None:
    st.set_page_config(page_title="AI 启发式错题助手", layout="wide")
    init_state()
    st.title("AI 启发式错题助手")

    # 侧边栏：模式切换与报告（点击即向 Dify 发送对应 query，由云端应用识别）
    sidebar_query: Optional[str] = None
    with st.sidebar:
        st.caption(f"当前对话轮次：{st.session_state.get('turn_count', 0)}")
        if st.button("开启全域模式", use_container_width=True, key="sb_global_mode"):
            sidebar_query = "开启全域模式"
        if st.button("回到教材模式", use_container_width=True, key="sb_textbook_mode"):
            sidebar_query = "回到教材模式"
        if st.button("生成薄弱点报告", use_container_width=True, key="sb_weak_report"):
            sidebar_query = "生成薄弱点报告"

    upload_types = ["pdf", "docx", "png", "jpg", "jpeg"]
    uploaded = st.file_uploader(
        "上传题目的 PDF、Word 或图片（JPG/PNG）",
        type=upload_types,
        accept_multiple_files=False,
        help="上传后先由智谱 GLM-4V 提取题干，再由 Dify 对话引擎进行启发式讲解。",
    )

    if uploaded is not None:
        file_bytes = uploaded.read()
        fp = question_fingerprint(uploaded.name, file_bytes)
        if st.session_state.question_fingerprint != fp:
            st.session_state.question_fingerprint = fp
            st.session_state.messages = []
            st.session_state.report_text = ""
            st.session_state.current_extracted_text = None
            st.session_state.auto_analysis_done_fp = None
            st.session_state.dify_conversation_id = ""

            name_lower = uploaded.name.lower()
            ctx: Optional[QuestionContext] = None
            try:
                with st.spinner("🔍 老师正在仔细查看您的题目排版和图片内容，请稍候..."):
                    ocr_image_bytes: Optional[bytes] = None
                    ocr_mime: Optional[str] = None
                    local_text = ""
                    if name_lower.endswith((".png", ".jpg", ".jpeg")):
                        ocr_image_bytes = file_bytes
                        ocr_mime = "image/png" if name_lower.endswith(".png") else "image/jpeg"
                    elif name_lower.endswith(".pdf"):
                        local_text = _pdf_to_text_fitz(file_bytes)
                        ocr_image_bytes, ocr_mime = _pdf_first_page_image(file_bytes)
                    elif name_lower.endswith(".docx"):
                        local_text, img_bytes, mime = _docx_text_and_first_image(file_bytes)
                        ocr_image_bytes, ocr_mime = img_bytes, mime
                    else:
                        st.error("暂不支持该文件格式，请上传 PDF、Word 或常见图片格式。")

                    merged = local_text.strip()
                    if ocr_image_bytes and ocr_mime:
                        ocr_text = zhipu_ocr_image_once(image_bytes=ocr_image_bytes, mime=ocr_mime).strip()
                        merged = ocr_text if not merged else f"{ocr_text}\n\n（文档可复制文本补充）\n{merged}"
                    ctx = QuestionContext(source_name=uploaded.name, source_type="text", extracted_text=merged.strip())
            except Exception as e:
                st.error(f"处理上传文件时发生错误：{e}")

            st.session_state.question_ctx = ctx
            extracted = (ctx.extracted_text or "").strip() if ctx else ""
            if len(extracted) < 5:
                st.warning("⚠️ 未能提取到足够题干文本，请上传更清晰的截图或可复制文字文档。")
            else:
                cleaned = sanitize_markdown(extracted)
                st.session_state.current_extracted_text = cleaned
                # 仍保留隐藏上下文消息，便于前端逻辑一致
                st.session_state.messages.append({"role": "system", "content": f"[系统提示：OCR 结果：{cleaned}]"})
                st.session_state.messages.append({"role": "user", "content": "我已上传题目，请开始第一轮提问。"})

    if st.session_state.current_extracted_text:
        with st.expander("🔍 当前题目内容 (点击展开/收起)", expanded=True):
            st.info(st.session_state.current_extracted_text)

    for idx, m in enumerate(st.session_state.messages):
        raw = m.get("content", "") or ""
        role = m.get("role", "assistant")
        if role == "system" and "[系统提示：" in raw:
            continue
        is_assistant = role == "assistant"
        has_end = is_assistant and "[教学结束]" in raw
        display = raw.replace("[教学结束]", "").strip() if has_end else raw.strip()
        with st.chat_message("assistant" if is_assistant else "user"):
            st.markdown(sanitize_markdown(display), help=None)
            if has_end:
                st.markdown("✅ **教学结束，恭喜你掌握了本题！**", help=None)
                if st.button("📝 恭喜完成！点击生成本次诊断报告", key=f"inline_report_{idx}"):
                    _generate_weakness_report(trigger="inline")

    if (
        st.session_state.question_fingerprint
        and st.session_state.messages
        and st.session_state.messages[-1]["role"] == "user"
        and st.session_state.auto_analysis_done_fp != st.session_state.question_fingerprint
    ):
        ocr_preview = (st.session_state.current_extracted_text or "")[:200]
        query = f"我上传了一道题，OCR识别内容如下，请先确认识别是否正确，再开始引导：\n\n{ocr_preview}"
        reply = stream_assistant_reply(query=query)
        if reply is not None:
            st.session_state.auto_analysis_done_fp = st.session_state.question_fingerprint

    if sidebar_query:
        st.session_state.report_text = ""
        if _ensure_env_keys():
            st.session_state.messages.append({"role": "user", "content": sidebar_query})
            with st.chat_message("user"):
                st.markdown(sidebar_query, help=None)
            stream_assistant_reply(query=sidebar_query)

    user_text = st.chat_input("把你目前的思路/卡点发给老师…")
    if user_text:
        st.session_state.report_text = ""
        st.session_state.messages.append({"role": "user", "content": user_text})
        with st.chat_message("user"):
            st.markdown(user_text, help=None)
        stream_assistant_reply(query=user_text)


if __name__ == "__main__":
    main()
