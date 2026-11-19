"""Automated checks on the built deck: notes/speaker tags, font sizes, word counts, bounds, GIF, number traceability."""
import io, json, re, glob
from pathlib import Path
from pptx import Presentation
from pptx.util import Emu
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
prs = Presentation(ROOT / "AIAA4220_Project3_Presentation_Attention.pptx")
SW, SH = prs.slide_width, prs.slide_height

# ---- allowed numbers: every decimal in approved reports / case analysis / raw JSON (rounded 1-3 dp)
allowed = set()
for f in ["src/remote/FINAL_REPORT_EN.md", "src/remote/FINAL_REPORT_CN.md", "src/remote/CASE_ANALYSIS.md", "assets/numbers.json", "assets/cases.json",
          "src/remote/json/b5_data_eff_summary.md", "src/remote/json/data_efficiency.csv"]:
    allowed |= set(re.findall(r"\d+\.\d+", (ROOT / f).read_text()))
def walk(o):
    if isinstance(o, dict):
        for v in o.values(): yield from walk(v)
    elif isinstance(o, list):
        for v in o: yield from walk(v)
    elif isinstance(o, float):
        yield o
for f in glob.glob(str(ROOT / "src/remote/json/*.json")) + glob.glob(str(ROOT / "src/hist/*.json")):
    for x in walk(json.load(open(f))):
        for nd in (1, 2, 3):
            allowed.add(f"{x:.{nd}f}"); allowed.add(f"{abs(x):.{nd}f}")
        allowed.add(f"{x * 100:.1f}")
for s in [str(round(x, 3)) for x in []]: pass

issues = []
for i, s in enumerate(prs.slides, 1):
    core = i <= 9
    n = s.notes_slide.notes_text_frame.text.strip() if s.has_notes_slide else ""
    if not n: issues.append(f"slide {i}: missing notes")
    if core and not re.match(r"【讲者 [ABC]｜", n): issues.append(f"slide {i}: notes lack speaker tag")
    words, sizes, texts = 0, [], []
    for sh in s.shapes:
        if sh.left is not None and (sh.left < 0 or sh.top < 0 or sh.left + sh.width > SW + 10 or sh.top + sh.height > SH + 10):
            issues.append(f"slide {i}: shape '{sh.name}' out of bounds")
        frames = []
        if sh.has_text_frame: frames.append(sh.text_frame)
        if sh.has_table:
            frames += [c.text_frame for r in sh.table.rows for c in r.cells]
        for tf in frames:
            for p in tf.paragraphs:
                for r in p.runs:
                    if not r.text.strip(): continue
                    sz = r.font.size.pt if r.font.size else None
                    texts.append(r.text)
                    if sz: sizes.append(sz)
                    if core and sz and sz >= 14 and sh.shape_type == 17 and not sh.is_placeholder:  # text boxes only (not diagram labels/tables)
                        words += len(r.text.split())
    body_min = min([x for x in sizes if x >= 14], default=None)
    small = sorted(set(x for x in sizes if x < 14))
    print(f"slide {i:2d} | text-box words (excl. title/diagram/footer) {words:3d} | min body pt {body_min} | small pt (captions/diagram/footer) {small}")
    for t in texts + [n]:
        for num in re.findall(r"(?<![\w.])\d+\.\d+(?![\w.])", t):
            if num not in allowed and num not in ("1.1", "3.12.14", "2.15", "3.2.0", "1.4.0", "2.5.4", "2.15.0", "0.95", "1709.07871", "1810.08272", "2501.15270"):
                issues.append(f"slide {i}: number {num} not found in reports/JSON")

# ---- GIF embedded intact
for i, s in enumerate(prs.slides, 1):
    for sh in s.shapes:
        if sh.shape_type == 13 and sh.image.ext == "gif":
            im = Image.open(io.BytesIO(sh.image.blob)); n = 0; dur = 0
            try:
                while True:
                    dur += im.info.get("duration", 0); n += 1; im.seek(im.tell() + 1)
            except EOFError:
                pass
            print(f"slide {i}: embedded GIF {n} frames, {dur / 1000:.1f} s")
print("\nISSUES:" if issues else "\nNo issues.")
for x in sorted(set(issues)): print(" -", x)
