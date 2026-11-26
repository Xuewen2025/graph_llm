"""Export the Claude Code session transcript to docs/conversation_log.md (messages only, secrets redacted).

    python scripts/export_conversation.py <transcript.jsonl>
"""
import json
import re
import sys

MID = re.compile(r"The user sent a new message while you were working:\n(.*?)\n\nThis is how", re.S)


def clean(t):
    for tag in ("system-reminder", "ide_selection", "ide_opened_file", "task-notification"):
        t = re.sub(rf"<{tag}>.*?</{tag}>", "", t, flags=re.S)
    t = re.sub(r"sk-or-v1-[A-Za-z0-9]+", "sk-or-v1-<REDACTED>", t)
    t = re.sub(r"sk-[A-Za-z0-9_-]{20,}", "sk-<REDACTED>", t)
    t = re.sub(r"[\w.+-]+@[\w-]+\.(com|org|net|io)", "<email>", t)
    return t.strip()


def main(src, dst="docs/conversation_log.md"):
    out = ["# SAGE-Drive reproduction: conversation log", "",
           "Conversation between the user and Claude Code (Claude Opus 5.5) while reproducing the SAGE-Drive paper.",
           "Only messages are kept (tool calls and command outputs are omitted). API keys are redacted.", ""]
    seen = set()
    for line in open(src):
        d = json.loads(line)
        att = d.get("attachment") or {}
        if d.get("type") == "attachment" and att.get("type") == "queued_command":  # message sent mid-turn
            pr = att.get("prompt")
            d = dict(d, type="user", message={"role": "user", "content": pr if isinstance(pr, list) else str(pr)})
        if d.get("type") not in ("user", "assistant") or d.get("isMeta") or d.get("isCompactSummary"):
            continue
        m = d.get("message") or {}
        c, role = m.get("content"), m.get("role")
        blocks = [{"type": "text", "text": c}] if isinstance(c, str) else c
        items = []
        for b in blocks:
            if b.get("type") == "text":
                raw = b["text"]
            elif b.get("type") == "tool_result":
                r = b.get("content")
                raw = r if isinstance(r, str) else " ".join(x.get("text", "") for x in r if isinstance(x, dict))
                if raw.startswith("The user answered:"):
                    items.append(("answer", raw))
            else:
                continue
            items += [("user", mm) for mm in MID.findall(raw)]          # messages sent mid-turn
            if b.get("type") == "text":
                items.append(("assistant" if role == "assistant" else "user", raw))
        for kind, t in items:
            t = clean(t)
            if not t or t.startswith("<command-") or t.startswith("[Request interrupted") or (kind, t) in seen:
                continue
            seen.add((kind, t))
            who = {"assistant": "**Claude**", "user": "**User**", "answer": "**User (answers to Claude's questions)**"}[kind]
            ts = d.get("timestamp", "")[:16].replace("T", " ")
            out += [f"### {who}  <sub>{ts} UTC</sub>", "", t, "", "---", ""]
    open(dst, "w").write("\n".join(out))
    print(len(seen), "messages ->", dst)


if __name__ == "__main__":
    main(sys.argv[1])
