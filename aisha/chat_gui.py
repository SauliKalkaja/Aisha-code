"""
chat_gui.py — minimal Tk desktop interface for Aisha.

Run:  python chat_gui.py

Layout:
  • Top: scrollable transcript (auto-scrolls)
  • Bottom: input entry + Send button
  • Status bar: routed style + target register per turn

The pipeline is loaded lazily on first send so the window opens fast.
Generation runs in a background thread so the UI doesn't freeze.
"""
from __future__ import annotations

import queue
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import scrolledtext, ttk

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


class ChatApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Aisha")
        self.root.geometry("680x540")
        self.root.minsize(420, 320)

        self._pipeline = None
        self._gen_queue: queue.Queue = queue.Queue()
        self._busy = False

        self._build_ui()
        self._poll_queue()

        # Friendly opener so the window isn't blank
        self._append("aisha", "Loading… ask me anything.", style="—")

    # ---------- UI ---------------------------------------------------

    def _build_ui(self):
        # Transcript
        self.transcript = scrolledtext.ScrolledText(
            self.root, wrap=tk.WORD, state="disabled",
            font=("Helvetica", 11), padx=8, pady=8,
            background="#fafafa")
        self.transcript.pack(fill="both", expand=True, padx=8, pady=(8, 4))

        # Tags for speaker styling
        self.transcript.tag_configure("you",
            foreground="#1f4ea0", font=("Helvetica", 11, "bold"))
        self.transcript.tag_configure("aisha",
            foreground="#5a3a99", font=("Helvetica", 11, "bold"))
        self.transcript.tag_configure("meta",
            foreground="#888888", font=("Helvetica", 9, "italic"))

        # Input row
        row = tk.Frame(self.root)
        row.pack(fill="x", padx=8, pady=(0, 4))
        self.input = tk.Entry(row, font=("Helvetica", 11))
        self.input.pack(side="left", fill="x", expand=True, padx=(0, 4))
        self.input.bind("<Return>", lambda _e: self._send())
        self.send_btn = tk.Button(row, text="Send", width=8,
                                    command=self._send)
        self.send_btn.pack(side="right")
        self.input.focus()

        # Status bar
        self.status = tk.Label(self.root, text="ready", anchor="w",
                                background="#eeeeee",
                                font=("Helvetica", 9), padx=6)
        self.status.pack(fill="x", side="bottom")

    def _append(self, who: str, text: str, style: str | None = None):
        self.transcript.configure(state="normal")
        prefix = {"you": "You: ", "aisha": "Aisha: "}.get(who, "")
        self.transcript.insert("end", prefix, who)
        self.transcript.insert("end", text + "\n")
        if style is not None:
            self.transcript.insert("end", f"   ({style})\n", "meta")
        self.transcript.configure(state="disabled")
        self.transcript.see("end")

    def _set_status(self, text: str):
        self.status.configure(text=text)

    # ---------- send / generation -----------------------------------

    def _send(self):
        if self._busy:
            return
        msg = self.input.get().strip()
        if not msg:
            return
        self.input.delete(0, "end")
        self._append("you", msg)
        self._busy = True
        self._set_status("thinking…")
        self.send_btn.configure(state="disabled")
        threading.Thread(target=self._generate, args=(msg,),
                          daemon=True).start()

    def _ensure_pipeline(self):
        if self._pipeline is None:
            self._set_status("loading manifold…")
            from pipeline import Pipeline
            self._pipeline = Pipeline()
            # Pick an LLM polish backend.  Prefer Ollama (real LM with style
            # tone prompting) over MockLLM (raw word-stream concatenation).
            # If neither succeeds the pipeline still falls back to MockLLM.
            self._llm_fn = self._pick_llm()

    def _pick_llm(self):
        """Try to construct an OllamaLLM with the first available installed
        model.  Falls back to MockLLM if Ollama isn't running."""
        try:
            import requests
            r = requests.get("http://localhost:11434/api/tags", timeout=2)
            r.raise_for_status()
            models = [m["name"] for m in r.json().get("models", [])]
            if not models:
                return None
            # prefer small chat-tuned models
            preference = ["qwen2.5:0.5b", "llama3.2:1b", "qwen2.5:1.5b"]
            chosen = next((m for m in preference if m in models), models[0])
            from pipeline import OllamaLLM
            self._set_status(f"using LLM polish: {chosen}")
            return OllamaLLM(model=chosen).__call__
        except Exception:
            return None

    def _generate(self, msg: str):
        try:
            self._ensure_pipeline()
            out = self._pipeline.respond(msg, llm_fn=self._llm_fn,
                                          verbose=False)
            self._gen_queue.put(("ok", out))
        except Exception as e:
            self._gen_queue.put(("err", str(e)))

    def _poll_queue(self):
        try:
            while True:
                kind, payload = self._gen_queue.get_nowait()
                if kind == "ok":
                    s = payload["structure"]
                    style = s.get("style", "?")
                    tv = s.get("target_register", {}).get("v", 0)
                    ta = s.get("target_register", {}).get("a", 0)
                    gate = payload.get("gate", "?")
                    gate_short = ("polish" if gate == "ok"
                                    else "raw (gate rejected)")
                    meta = (f"style={style}  v={tv:+.2f} a={ta:+.2f}  "
                              f"·  {gate_short}")
                    self._append("aisha", payload["response"], style=meta)
                    self._set_status(f"ready · {style} · {gate_short}")
                else:
                    self._append("aisha", f"[error] {payload}",
                                  style="error")
                    self._set_status("error")
                self._busy = False
                self.send_btn.configure(state="normal")
                self.input.focus()
        except queue.Empty:
            pass
        self.root.after(80, self._poll_queue)


def main():
    root = tk.Tk()
    ChatApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
