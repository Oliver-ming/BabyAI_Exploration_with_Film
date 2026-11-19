"""Generate 讲稿与QA准备.md from the speaker notes inside the built .pptx (single source of truth)."""
import re
from pathlib import Path
from pptx import Presentation

ROOT = Path(__file__).resolve().parents[1]
prs = Presentation(ROOT / "AIAA4220_Project3_Presentation.pptx")
tpl = (ROOT / "scripts/讲稿与QA准备_template.md").read_text()

# Dynamically locate the Backup section divider so core slide count is flexible
backup_idx = len(prs.slides) + 1
for idx, sl in enumerate(prs.slides, 1):
    ttl = sl.shapes.title.text_frame.text if sl.shapes.title is not None else ""
    if sl.slide_layout.name == "SECTION_HEADER" or ttl.startswith("Backup slides"):
        backup_idx = idx
        break

rows, script, total = [], [], 0
for i, s in enumerate(prs.slides, 1):
    title = s.shapes.title.text_frame.text if s.shapes.title is not None else "(title)"
    if i == 1:
        title = "Title: " + title
    n = s.notes_slide.notes_text_frame.text.strip()
    m = re.match(r"【(?:讲者 )?([^｜]+)｜([^｜]+)｜([^】]+)】", n)
    body = n[m.end():].strip() if m else n
    chars = len(re.findall(r"[\u4e00-\u9fff]", body)) + len(re.findall(r"[A-Za-z0-9.+\-±→]+", body))
    if i < backup_idx:
        sec = int(re.search(r"(\d+)\s*秒", m.group(3)).group(1))
        total += sec
        rows.append(f"| {i} | {title} | {m.group(1)} | {m.group(2)} | {sec} s | {chars} |")
        script.append(f"### 第 {i} 页 · {title}\n**{m.group(1)}｜{m.group(2)}｜{m.group(3)}**\n\n{body}\n")
    else:
        script.append(f"### 备份 · 第 {i} 页 · {title}\n{n}\n")
rows.append(f"| | **合计** | | | **{total} s = {total // 60} min {total % 60:02d} s** | |")
out = tpl.replace("__TIMELINE__", "\n".join(rows)).replace("__SCRIPT__", "\n".join(script))
(ROOT / "讲稿与QA准备.md").write_text(out)
print(f"core slides: {backup_idx - 1} | total slides: {len(prs.slides)} | total talk seconds: {total}")
