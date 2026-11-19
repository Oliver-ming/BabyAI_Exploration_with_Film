"""Generate 讲稿与QA准备.md from the speaker notes inside the built .pptx (single source of truth)."""
import re
from pathlib import Path
from pptx import Presentation

ROOT = Path(__file__).resolve().parents[1]
prs = Presentation(ROOT / "AIAA4220_Project3_Presentation.pptx")
tpl = (ROOT / "scripts/讲稿与QA准备_template.md").read_text()

rows, script, total = [], [], 0
for i, s in enumerate(prs.slides, 1):
    title = s.shapes.title.text_frame.text if s.shapes.title is not None else "(title)"
    if i == 1:
        title = "Title: " + title
    n = s.notes_slide.notes_text_frame.text.strip()
    m = re.match(r"【(?:讲者 )?([^｜]+)｜([^｜]+)｜([^】]+)】", n)
    body = n[m.end():].strip() if m else n
    chars = len(re.findall(r"[\u4e00-\u9fff]", body)) + len(re.findall(r"[A-Za-z0-9.+\-±→]+", body))
    if i <= 9:
        sec = int(re.search(r"(\d+)\s*秒", m.group(3)).group(1))
        total += sec
        rows.append(f"| {i} | {title} | {m.group(1)} | {m.group(2)} | {sec} s | {chars} |")
        script.append(f"### 第 {i} 页 · {title}\n**{m.group(1)}｜{m.group(2)}｜{m.group(3)}**\n\n{body}\n")
    else:
        script.append(f"### 备份 · 第 {i} 页 · {title}\n{n}\n")
rows.append(f"| | **合计** | | | **{total} s = {total // 60} min {total % 60:02d} s** | |")
out = tpl.replace("__TIMELINE__", "\n".join(rows)).replace("__SCRIPT__", "\n".join(script))
(ROOT / "讲稿与QA准备.md").write_text(out)
print("total talk seconds:", total)
