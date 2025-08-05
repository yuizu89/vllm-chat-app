from __future__ import annotations
import argparse
import asyncio, json, multiprocessing as mp, uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import re
import requests
from bs4 import BeautifulSoup

import gradio as gr
from duckduckgo_search import DDGS
from transformers import AutoTokenizer
from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams


# ───────── 基本設定 ─────────
HROOT          = Path("chat_histories"); HROOT.mkdir(exist_ok=True)

# ───────── Utility ──────────
def fetch_url_content(url: str, max_chars: int = 50000) -> str:
    """
    指定 URL のページを取得し、主要コンテンツ領域からテキストをまとめて返す。
    長すぎる場合は先頭 max_chars 文字に切り詰め、末尾に "…" を付加します。
    """
    try:
        resp = requests.get(
            url,
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=5  # 5秒タイムアウト
        )
        resp.raise_for_status()  # HTTPエラーは例外化
    except requests.RequestException:
        # 例外時は空文字を返す（必要なら logging.error でログ出力）
        return ""

    soup = BeautifulSoup(resp.text, "html.parser")
    # main タグまたは markdown クラスを優先、それ以外は全体を対象
    container = (
        soup.select_one("main") or
        soup.select_one("div.markdown") or
        soup
    )
    # 全テキストを改行区切りで取得
    text = container.get_text(separator="\n", strip=True)
    # 文字数制限
    return text[:max_chars] + ("…" if len(text) > max_chars else "")

def web_search(q: str, n: int = 3) -> str:
    if not q.strip():
        return ""
    try:
        with DDGS() as ddgs:
            return "\n".join(
                "• {title}: {body} ({href})".format(**r)
                for r in ddgs.text(q, max_results=n)
            )
    except Exception:
        return ""

def save_conv(cid: str, msgs: List[Dict[str, str]], title: str) -> None:
    """
    先頭行に ``role=meta`` を置き、以降に通常メッセージを jsonl で保存する。
    """
    path = HROOT / f"{cid}.jsonl"
    with path.open("w", encoding="utf-8") as f:
        meta = {"role": "meta", "title": title,
                "ts": datetime.utcnow().isoformat()}
        json.dump(meta, f, ensure_ascii=False)
        f.write("\n")
        for m in msgs:
            json.dump({"ts": datetime.utcnow().isoformat(), **m},
                      f, ensure_ascii=False)
            f.write("\n")

def choice(conv: dict) -> tuple[str, str]:
    """Radio 用 (label, value) タプル"""
    return (conv["title"], conv["id"])


# ──────── VllmChatSession ────────
class VllmChatSession:
    _engine: AsyncLLMEngine = None
    _tok:    AutoTokenizer  = None
    _max_tokens: int        = 32_768

    @classmethod
    def init_backend(cls, eng: AsyncLLMEngine, tok: AutoTokenizer, max_tokens: int):
        cls._engine     = eng
        cls._tok        = tok
        cls._max_tokens = max_tokens

    def __init__(self):
        self.messages: List[Dict[str, str]] = []
        self._cancel = asyncio.Event()

    def cancel(self):                          # Stop ボタンから呼ぶ
        self._cancel.set()

    def sampling(self, thinking: bool) -> SamplingParams:
        return SamplingParams(
            max_tokens=self._max_tokens,
            temperature=0.6 if thinking else 0.7,
            top_p=0.95 if thinking else 0.8,
            top_k=20,
        )

    async def ask(self, user_txt, *, thinking, search):
        # (0) ユーザ入力中の URL を検出し、ページ内容を取得して先頭に挿入 ---
        urls = re.findall(r"https?://[^\s]+", user_txt)
        for url in urls:
            content = fetch_url_content(url)
            if content:
                # 入力の先頭にページ内容を追加
                user_txt = f"[Web page content: {url}]\n{content}\n\n{user_txt}"

        # (A) Web 検索結果を前置
        if search and (ctx := web_search(user_txt)):
            user_txt = f"[Web search results]\n{ctx}\n\n{user_txt}"

        # (B) ユーザ発話を追加・即表示
        self.messages.append({"role": "user", "content": user_txt})
        yield self.messages

        # (C) 空の assistant メッセージを追加
        assistant = {"role": "assistant", "content": ""}
        self.messages.append(assistant)
        yield self.messages

        prompt = self._tok.apply_chat_template(
            self.messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=thinking
        )
        params = self.sampling(thinking)
        req_id = str(uuid.uuid4())
        partial = ""

        try:
            async for out in self._engine.generate(prompt, params, req_id):
                if self._cancel.is_set():                  # Stop 要求
                    await self._engine.abort(req_id)
                    if self.messages and self.messages[-1]["role"] == "user":
                        self.messages.pop()               # 質問ごと取消
                    raise asyncio.CancelledError

            #ans = out.outputs[0].text.strip()
            #self.messages.append({"role": "assistant", "content": ans})
                text = out.outputs[0].text
                if text != partial:                # 差分があれば更新
                    assistant["content"] = text
                    partial = text
                    yield self.messages

        finally:
            self._cancel.clear()

    async def summarize_title(self) -> str:
        if len(self.messages) < 2:
            return ""
        prompt = self._tok.apply_chat_template(
            self.messages + [{"role":"system",
                               "content":"上の会話を5~10文字で要約してください。"}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
            )
        req_id = str(uuid.uuid4())
        params = SamplingParams(max_tokens=16, temperature=0.3)
        async for ch in self._engine.generate(prompt, params, req_id):
            if ch.finished:
              return ch.outputs[0].text.strip()
        return ""

# ─────────── Gradio UI ───────────
def build_ui(model_name: str, max_tokens: int, history_root: str):
    global HROOT
    HROOT = Path(history_root)
    HROOT.mkdir(exist_ok=True)

    # 既存 *.jsonl を読み込み
    convs: List[Dict[str, Any]] = []
    for path in HROOT.glob("*.jsonl"):
        cid = path.stem
        sess = VllmChatSession()
        title = "(untitled)"
        try:
            with path.open(encoding="utf-8") as f:
                # 1 行目 = メタ
                meta = json.loads(f.readline())
                title = meta.get("title", title)
                # 残り行 = 発話
                for line in f:
                    rec = json.loads(line)
                    sess.messages.append(
                        {"role": rec["role"], "content": rec["content"]})
        except Exception:
            pass
        convs.append({"id": cid, "title": title, "session": sess})

    # ─ 初期選択肢とデフォルト値を決定 ─
    if convs:
        init_id = convs[0]["id"]          # 既存履歴がある
    else:
        init_id = datetime.utcnow().strftime("%Y%m%d%H%M%S")   # 新規
        convs.append({"id": init_id, "title": "新規チャット",
                      "session": VllmChatSession()})
        save_conv(init_id, [], "新規チャット")

    radio_choices = [choice(c) for c in convs]

    with gr.Blocks(css="""
        /* 全要素に一律で18pxを適用 */
        * {
            font-size: 24px !important;
        }

        /* ボタン色指定 */
        #new-btn { background-color: #d0e8ff !important; color: #000; }
        #del-btn { background-color: #ffe6c7 !important; color: #000; }
        #stop-btn { background-color: #ffdcdc !important; color: #000; }

    """) as demo:
        gr.Markdown(f"### {model_name} Chat (vLLM)")

        with gr.Row():
            # ─ 左ペイン ─
            with gr.Column(scale=1, min_width=220):
                new_btn = gr.Button("+ 新規チャット", elem_id="new-btn")
                del_btn = gr.Button("✖️ 削除",       elem_id="del-btn")

                radio = gr.Radio(
                    label="会話履歴",
                    choices=radio_choices,
                    value=init_id,
                    interactive=True,
                )

            # ─ 右ペイン ─
            with gr.Column(scale=4):
                bot  = gr.Chatbot(type="messages", height=960,
                                  allow_tags=["think"])
                with gr.Row():
                    txt  = gr.Textbox(placeholder="メッセージ…",
                                      show_label=False, scale=4)
                    send = gr.Button("Send", variant="primary")
                    stop = gr.Button("⏹ Stop", elem_id="stop-btn")
                with gr.Row():
                    think = gr.Checkbox(label="Thinking On", value=False)
                    srch  = gr.Checkbox(label="Web Search ON")

        st = gr.State(convs)

        # ─ helpers ─
        def get_conv(cid): return next(c for c in st.value if c["id"] == cid)
        def load(cid):     return get_conv(cid)["session"].messages

        def new_chat():
            cid = datetime.utcnow().strftime("%Y%m%d%H%M%S")
            st.value.append({"id": cid, "title":"新規チャット",
                             "session": VllmChatSession()})
            save_conv(cid, [], "新規チャット")
            ch = [choice(c) for c in st.value]
            return gr.update(choices=ch, value=cid), "", []

        def delete_chat(cid):
            if not st.value:
                return gr.update(), [], ""
            st.value = [c for c in st.value if c["id"] != cid]
            try:
                (HROOT/f"{cid}.jsonl").unlink(missing_ok=True)
            except:
                pass
            if not st.value:
                new_chat()
            ch = [choice(c) for c in st.value]
            first = st.value[0]
            return gr.update(choices=ch, value=first["id"]), \
                   first["session"].messages, ""

        def stop_req(cid): get_conv(cid)["session"].cancel()

        # ─ main async handler ─
        async def respond(msg, cid, thinking, do_search):
            if not msg.strip():
                yield gr.update(), "", gr.update(); return
            conv, sess = get_conv(cid), get_conv(cid)["session"]

            try:
                async for messages in sess.ask(msg, thinking=thinking, search=do_search):
                    yield messages, "", gr.update()
            except asyncio.CancelledError:
                yield sess.messages, msg, gr.update(); return

            yield sess.messages, "", gr.update()

            # タイトル確定
            if conv["title"] == "新規チャット":
                title = await sess.summarize_title()
                if title:
                    conv["title"] = title
                    ch = [choice(c) for c in st.value]
                    yield sess.messages, "", \
                          gr.update(choices=ch, value=cid)
            # 毎ターン保存
            save_conv(cid, sess.messages, conv["title"])

        # ─ wire events ─
        radio.change(load, inputs=radio, outputs=bot)
        new_btn.click(new_chat, outputs=[radio, txt, bot])
        del_btn.click(delete_chat, inputs=radio,
                      outputs=[radio, bot, txt])

        send_evt = send.click(
            respond,
            inputs=[txt, radio, think, srch],
            outputs=[bot, txt, radio],
        ).then(lambda: None)      # 警告抑止

        stop.click(stop_req, inputs=radio, cancels=[send_evt])

    return demo

# ─────────── main ───────────
def main():
    parser = argparse.ArgumentParser(description="vLLM + Gradio Chat App")
    parser.add_argument("-m", "--model", default="Qwen/Qwen3-4B",
                        help="モデル名またはローカルディレクトリのパス")
    parser.add_argument("--dtype", default="bfloat16",
                        help="エンジンに渡す dtype (float16, bfloat16, fp32)")
    parser.add_argument("-p", "--port", type=int, default=7860,
                        help="Gradio サーバーの起動ポート")
    parser.add_argument("-r", "--history-root", default="chat_histories",
                        help="会話履歴を保存するディレクトリ")
    parser.add_argument("-t", "--max-tokens", type=int, default=32768,
                        help="LLM に渡す max_tokens の値")
    args = parser.parse_args()

    mp.set_start_method("spawn", force=True)

    eng = AsyncLLMEngine.from_engine_args(
        AsyncEngineArgs(model=args.model, dtype=args.dtype))
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    VllmChatSession.init_backend(eng, tok, args.max_tokens)

    demo = build_ui(args.model, args.max_tokens, args.history_root)
    demo.queue().launch(share=True, server_port=args.port)

if __name__ == "__main__":
    asyncio.run(main())