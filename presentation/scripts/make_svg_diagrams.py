"""Generate publication-grade SVG architecture diagrams, flowcharts, and mechanism schematics
for the AIAA 4220 Project 3 BabyAI Attention presentation.

Saves:
  - Raw editable W3C vector SVGs in assets/svg/*.svg
  - 2800-px Retina PNG renders in assets/svg_*.png (rendered via macOS WebKit / qlmanage)
"""
import subprocess
import tempfile
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
A = ROOT / "assets"
SVG_DIR = A / "svg"
SVG_DIR.mkdir(parents=True, exist_ok=True)

COMMON_DEFS = """
  <defs>
    <filter id="sh" x="-4%" y="-6%" width="108%" height="118%">
      <feDropShadow dx="0" dy="4" stdDeviation="6" flood-color="#0F172A" flood-opacity="0.08"/>
    </filter>
    <filter id="shSm" x="-4%" y="-6%" width="108%" height="118%">
      <feDropShadow dx="0" dy="2" stdDeviation="3.5" flood-color="#0F172A" flood-opacity="0.07"/>
    </filter>
    <linearGradient id="gBlue" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="#EFF6FF"/>
      <stop offset="100%" stop-color="#DBEAFE"/>
    </linearGradient>
    <linearGradient id="gBlueHdr" x1="0" y1="0" x2="1" y2="0">
      <stop offset="0%" stop-color="#1D4ED8"/>
      <stop offset="100%" stop-color="#2563EB"/>
    </linearGradient>
    <linearGradient id="gTeal" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="#F0FDFA"/>
      <stop offset="100%" stop-color="#CCFBF1"/>
    </linearGradient>
    <linearGradient id="gTealHdr" x1="0" y1="0" x2="1" y2="0">
      <stop offset="0%" stop-color="#0F766E"/>
      <stop offset="100%" stop-color="#0D9488"/>
    </linearGradient>
    <linearGradient id="gSlate" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="#F8FAFC"/>
      <stop offset="100%" stop-color="#E2E8F0"/>
    </linearGradient>
    <linearGradient id="gAmber" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="#FFFBEB"/>
      <stop offset="100%" stop-color="#FDE68A"/>
    </linearGradient>
    <linearGradient id="gOrange" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="#FFF7ED"/>
      <stop offset="100%" stop-color="#FFEDD5"/>
    </linearGradient>
    <linearGradient id="gRose" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="#FEF2F2"/>
      <stop offset="100%" stop-color="#FECACA"/>
    </linearGradient>
    <linearGradient id="gPurple" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="#FAF5FF"/>
      <stop offset="100%" stop-color="#E9D5FF"/>
    </linearGradient>
    <marker id="arrSlate" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="6.5" markerHeight="6.5" orient="auto">
      <path d="M 0 1.2 L 9.5 5 L 0 8.8 z" fill="#475569"/>
    </marker>
    <marker id="arrBlue" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="6.5" markerHeight="6.5" orient="auto">
      <path d="M 0 1.2 L 9.5 5 L 0 8.8 z" fill="#2563EB"/>
    </marker>
    <marker id="arrTeal" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="6.5" markerHeight="6.5" orient="auto">
      <path d="M 0 1.2 L 9.5 5 L 0 8.8 z" fill="#0F766E"/>
    </marker>
    <marker id="arrOrange" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="6.5" markerHeight="6.5" orient="auto">
      <path d="M 0 1.2 L 9.5 5 L 0 8.8 z" fill="#D97706"/>
    </marker>
    <marker id="arrPurple" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="6.5" markerHeight="6.5" orient="auto">
      <path d="M 0 1.2 L 9.5 5 L 0 8.8 z" fill="#6D28D9"/>
    </marker>
    <marker id="arrRed" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="6.5" markerHeight="6.5" orient="auto">
      <path d="M 0 1.2 L 9.5 5 L 0 8.8 z" fill="#DC2626"/>
    </marker>
    <style>
      text { font-family: Arial, Helvetica, sans-serif; }
      .mono { font-family: Menlo, Consolas, monospace; }
    </style>
  </defs>
"""


def render_svg(name: str, w: int, h: int, body: str, px_size: int = 2600) -> Path:
    # 1. Save clean standalone W3C SVG in assets/svg/{name}.svg
    clean_svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">\n{COMMON_DEFS}\n  <rect width="{w}" height="{h}" fill="#FFFFFF"/>\n{body}\n</svg>\n'
    svg_path = SVG_DIR / f"{name}.svg"
    svg_path.write_text(clean_svg)

    # 2. Render via macOS WebKit (qlmanage) using 760-pt viewport width so WebKit never clips wide viewBoxes
    wk_w = 760.0
    wk_h = wk_w * h / w
    wk_svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="{wk_w:.3f}" height="{wk_h:.3f}" viewBox="0 0 {w} {h}">\n{COMMON_DEFS}\n  <rect width="{w}" height="{h}" fill="#FFFFFF"/>\n{body}\n</svg>\n'
    with tempfile.TemporaryDirectory() as td:
        tmp_svg = Path(td) / f"{name}.svg"
        tmp_svg.write_text(wk_svg)
        subprocess.run(
            ["qlmanage", "-t", "-s", str(px_size), "-o", td, str(tmp_svg)],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        im = Image.open(Path(td) / f"{name}.svg.png").convert("RGB")
        crop_h = round(im.width * h / w)
        im = im.crop((0, 0, im.width, crop_h))
        out_png = A / f"{name}.png"
        im.save(out_png, optimize=True)
    print(f"Rendered {svg_path.name} -> {out_png.name} ({im.width}x{im.height})")
    return out_png


def draw_tensor_stack(x: int, y: int, w: int, h: int, fill1: str, stroke: str, n: int = 3, dx: int = 10, dy: int = -8) -> str:
    parts = []
    for k in range(n - 1, -1, -1):
        ox, oy = x + k * dx, y + k * dy
        op = 0.45 if k == 2 else (0.72 if k == 1 else 1.0)
        parts.append(f'<rect x="{ox}" y="{oy}" width="{w}" height="{h}" rx="10" fill="{fill1}" stroke="{stroke}" stroke-width="2.2" opacity="{op}"/>')
    return "\n".join(parts)


# ==============================================================================
# 1. SLIDE 3 · Prescribed Baseline BCNet & Late-Concatenation Bottleneck (1860 x 520)
# ==============================================================================
def make_slide03_bcnet():
    W, H = 1860, 520
    # Mini 7x7 symbolic grid icon on top-left
    gx, gy, cs = 42, 68, 17
    grid_cells = []
    for r in range(7):
        for c in range(7):
            is_wall = r < 2 or c == 0 or c == 6
            bg = "#CBD5E1" if is_wall else "#F8FAFC"
            grid_cells.append(f'<rect x="{gx + c*cs}" y="{gy + r*cs}" width="{cs}" height="{cs}" fill="{bg}" stroke="#94A3B8" stroke-width="1"/>')
    # Add purple key, yellow key, red agent triangle
    grid_cells.append(f'<circle cx="{gx + 5.5*cs}" cy="{gy + 3.5*cs}" r="5.5" fill="#9333EA"/>')
    grid_cells.append(f'<circle cx="{gx + 4.5*cs}" cy="{gy + 4.5*cs}" r="5.5" fill="#EAB308"/>')
    grid_cells.append(f'<polygon points="{gx+3.5*cs},{gy+5.2*cs} {gx+3.15*cs},{gy+5.85*cs} {gx+3.85*cs},{gy+5.85*cs}" fill="#DC2626"/>')
    grid_svg = "\n".join(grid_cells)

    # Token pills for middle stream
    toks = [("go", "#F1F5F9", "#475569"), ("to", "#F1F5F9", "#475569"), ("the", "#F1F5F9", "#475569"),
            ("purple", "#F3E8FF", "#7E22CE"), ("key", "#EFF6FF", "#1D4ED8")]
    tok_svg = []
    tx = 42
    for t_str, t_bg, t_fg in toks:
        tw = 68 if len(t_str) <= 3 else 92
        tok_svg.append(f'<rect x="{tx}" y="278" width="{tw}" height="42" rx="10" fill="{t_bg}" stroke="{t_fg}" stroke-width="1.8"/>')
        tok_svg.append(f'<text x="{tx + tw/2}" y="305" font-size="22" font-weight="700" fill="{t_fg}" text-anchor="middle">{t_str}</text>')
        tx += tw + 8

    body = f"""
    <!-- Mission-blind dashed container around Vision Trunk -->
    <rect x="24" y="22" width="1215" height="212" rx="18" fill="#FFFBEB" fill-opacity="0.55" stroke="#D97706" stroke-width="2.4" stroke-dasharray="10,7"/>
    <rect x="48" y="6" width="724" height="32" rx="8" fill="#FEF3C7" stroke="#D97706" stroke-width="1.6"/>
    <text x="410" y="28" font-size="20" font-weight="700" fill="#B45309" text-anchor="middle">100% Mission-Blind Vision Trunk  (Δh₂ ≡ 0 under color / type swap)</text>

    <!-- Obs 7x7x3 -->
    {grid_svg}
    <text x="101" y="212" font-size="22" font-weight="700" fill="#1E293B" text-anchor="middle">Obs 7×7×3</text>
    <line x1="172" y1="128" x2="220" y2="128" stroke="#475569" stroke-width="3.2" marker-end="url(#arrSlate)"/>

    <!-- Embed(11,8)x3 -->
    {draw_tensor_stack(232, 78, 195, 102, "#F8FAFC", "#475569", 3, 9, -7)}
    <text x="329" y="122" font-size="23" font-weight="700" fill="#0F172A" text-anchor="middle">Embed(11,8)×3</text>
    <text x="329" y="154" font-size="21" fill="#475569" text-anchor="middle">24 × 7 × 7</text>
    <line x1="452" y1="128" x2="500" y2="128" stroke="#475569" stroke-width="3.2" marker-end="url(#arrSlate)"/>

    <!-- Conv1 -->
    {draw_tensor_stack(512, 78, 205, 102, "#F1F5F9", "#475569", 3, 9, -7)}
    <text x="614" y="122" font-size="23" font-weight="700" fill="#0F172A" text-anchor="middle">Conv1 3×3</text>
    <text x="614" y="154" font-size="21" fill="#475569" text-anchor="middle">ReLU · 32×7×7</text>
    <line x1="742" y1="128" x2="790" y2="128" stroke="#475569" stroke-width="3.2" marker-end="url(#arrSlate)"/>

    <!-- Conv2 -->
    {draw_tensor_stack(802, 78, 205, 102, "#F1F5F9", "#475569", 3, 9, -7)}
    <text x="904" y="122" font-size="23" font-weight="700" fill="#0F172A" text-anchor="middle">Conv2 3×3</text>
    <text x="904" y="154" font-size="21" fill="#475569" text-anchor="middle">ReLU · 32×7×7</text>
    <line x1="1032" y1="128" x2="1076" y2="128" stroke="#475569" stroke-width="3.2" marker-end="url(#arrSlate)"/>

    <!-- Flatten 1568 -->
    <rect x="1086" y="68" width="132" height="122" rx="14" fill="url(#gSlate)" stroke="#475569" stroke-width="2.4" filter="url(#shSm)"/>
    <text x="1152" y="122" font-size="23" font-weight="700" fill="#0F172A" text-anchor="middle">Flatten</text>
    <text x="1152" y="154" font-size="22" font-weight="700" fill="#334155" text-anchor="middle">h₂ (1,568)</text>

    <!-- Stream 2: Mission Language -->
    <text x="42" y="266" font-size="21" font-weight="700" fill="#1E3A8A">Mission Tokens (L ≤ 5)</text>
    {"".join(tok_svg)}
    <line x1="454" y1="299" x2="508" y2="299" stroke="#2563EB" stroke-width="3.2" marker-end="url(#arrBlue)"/>
    <rect x="518" y="258" width="440" height="82" rx="14" fill="url(#gBlue)" stroke="#2563EB" stroke-width="2.2" filter="url(#shSm)"/>
    <text x="738" y="294" font-size="23" font-weight="700" fill="#1E3A8A" text-anchor="middle">Token Embed(14, 32) + Masked Mean</text>
    <text x="738" y="324" font-size="21" fill="#1D4ED8" text-anchor="middle">Sentence Vector  ℓ ∈ ℝ³²  (bag-of-words)</text>

    <!-- Stream 3: Egocentric Heading -->
    <rect x="42" y="384" width="196" height="74" rx="14" fill="url(#gSlate)" stroke="#64748B" stroke-width="2" filter="url(#shSm)"/>
    <text x="140" y="416" font-size="22" font-weight="700" fill="#0F172A" text-anchor="middle">Heading Dir</text>
    <text x="140" y="444" font-size="19" fill="#475569" text-anchor="middle">N / E / S / W (0..3)</text>
    <line x1="238" y1="421" x2="300" y2="421" stroke="#475569" stroke-width="3.2" marker-end="url(#arrSlate)"/>
    <rect x="310" y="384" width="230" height="74" rx="14" fill="url(#gSlate)" stroke="#64748B" stroke-width="2" filter="url(#shSm)"/>
    <text x="425" y="416" font-size="22" font-weight="700" fill="#0F172A" text-anchor="middle">One-Hot Vector</text>
    <text x="425" y="444" font-size="20" fill="#475569" text-anchor="middle">d ∈ ℝ⁴</text>

    <!-- Merge curves into Late Concat Bottleneck -->
    <path d="M 1218 128 C 1265 128, 1265 215, 1306 215" stroke="#D97706" stroke-width="3.4" fill="none" marker-end="url(#arrOrange)"/>
    <path d="M 958 299 C 1140 299, 1220 260, 1306 260" stroke="#2563EB" stroke-width="3.2" stroke-dasharray="9,6" fill="none" marker-end="url(#arrBlue)"/>
    <path d="M 540 421 C 920 421, 1210 305, 1306 305" stroke="#64748B" stroke-width="3.0" fill="none" marker-end="url(#arrSlate)"/>

    <!-- Late Concat Bottleneck Box -->
    <rect x="1316" y="110" width="196" height="295" rx="18" fill="url(#gOrange)" stroke="#D97706" stroke-width="3.2" filter="url(#sh)"/>
    <rect x="1338" y="128" width="152" height="34" rx="8" fill="#D97706"/>
    <text x="1414" y="152" font-size="20" font-weight="700" fill="#FFFFFF" text-anchor="middle">BOTTLENECK</text>
    <text x="1414" y="205" font-size="26" font-weight="700" fill="#9A3412" text-anchor="middle">Late Concat</text>
    <text x="1414" y="242" font-size="24" font-weight="700" fill="#B45309" text-anchor="middle">1,604-d</text>
    <text x="1414" y="284" font-size="19" fill="#7C2D12" text-anchor="middle">W_v h₂ + W_l ℓ</text>
    <text x="1414" y="314" font-size="18" fill="#9A3412" text-anchor="middle">(additive only —</text>
    <text x="1414" y="338" font-size="18" fill="#9A3412" text-anchor="middle">no × binding!)</text>
    <text x="1414" y="382" font-size="19" font-weight="700" fill="#DC2626" text-anchor="middle">Swap flip: 0.051</text>

    <!-- Policy MLP Head -->
    <line x1="1512" y1="205" x2="1564" y2="205" stroke="#475569" stroke-width="3.2" marker-end="url(#arrSlate)"/>
    <rect x="1574" y="142" width="256" height="118" rx="16" fill="url(#gSlate)" stroke="#475569" stroke-width="2.2" filter="url(#shSm)"/>
    <text x="1702" y="194" font-size="24" font-weight="700" fill="#0F172A" text-anchor="middle">Linear(1604→128)</text>
    <text x="1702" y="228" font-size="21" fill="#475569" text-anchor="middle">+ ReLU  (205k params!)</text>

    <line x1="1702" y1="260" x2="1702" y2="308" stroke="#475569" stroke-width="3.2" marker-end="url(#arrSlate)"/>
    <rect x="1574" y="318" width="256" height="118" rx="16" fill="url(#gBlue)" stroke="#1D4ED8" stroke-width="2.4" filter="url(#shSm)"/>
    <text x="1702" y="368" font-size="24" font-weight="700" fill="#1E3A8A" text-anchor="middle">Linear(128 → 7)</text>
    <text x="1702" y="402" font-size="21" font-weight="700" fill="#1D4ED8" text-anchor="middle">Greedy argmax a_t</text>

    <!-- Bottom param badge -->
    <rect x="650" y="458" width="1180" height="46" rx="12" fill="#F8FAFC" stroke="#CBD5E1" stroke-width="1.6"/>
    <text x="1240" y="489" font-size="21" fill="#334155" text-anchor="middle">
      <tspan font-weight="700" fill="#0F172A">Total: 223,071 params</tspan>  (92% concentrated in first FC layer)  ·  38.3 / 300 representation grounding failures
    </text>
    """
    return render_svg("svg_slide03_bcnet", W, H, body)


# ==============================================================================
# 2. SLIDE 5 · Controlled Experimental Protocol & 3-Layer Causal Probes (1860 x 690)
# ==============================================================================
def make_slide05_protocol():
    W, H = 1860, 690
    body = """
    <!-- TIER 1: Leak-Free Seed Split & OOD Scale Ribbon -->
    <rect x="24" y="18" width="1812" height="196" rx="18" fill="#F8FAFC" stroke="#CBD5E1" stroke-width="2" filter="url(#shSm)"/>
    <text x="52" y="56" font-size="24" font-weight="700" fill="#0F172A">1 · Leak-Free Seed Partition &amp; Multi-Scale Evaluation Suite</text>
    <rect x="1488" y="30" width="320" height="36" rx="18" fill="#DCFCE7" stroke="#16A34A" stroke-width="1.8"/>
    <text x="1648" y="55" font-size="20" font-weight="700" fill="#15803D" text-anchor="middle">assert_no_leak() PASSED</text>

    <!-- Segmented Seed Bar -->
    <rect x="52" y="76" width="560" height="72" rx="10" fill="url(#gBlueHdr)"/>
    <text x="332" y="109" font-size="22" font-weight="700" fill="#FFFFFF" text-anchor="middle">Train [0 : 4000]</text>
    <text x="332" y="136" font-size="18" fill="#DBEAFE" text-anchor="middle">4,000 unique expert episodes</text>

    <rect x="620" y="76" width="215" height="72" rx="10" fill="#B45309"/>
    <text x="727" y="109" font-size="22" font-weight="700" fill="#FFFFFF" text-anchor="middle">Val [4000:4500]</text>
    <text x="727" y="136" font-size="18" fill="#FEF3C7" text-anchor="middle">500 eps (model sel.)</text>

    <rect x="843" y="76" width="235" height="72" rx="10" fill="#0E9664"/>
    <text x="960" y="109" font-size="22" font-weight="700" fill="#FFFFFF" text-anchor="middle">Test [4500:4800]</text>
    <text x="960" y="136" font-size="18" fill="#DCFCE7" text-anchor="middle">300 unseen S6N3</text>

    <rect x="1086" y="76" width="225" height="72" rx="10" fill="url(#gBlueHdr)"/>
    <text x="1198" y="109" font-size="22" font-weight="700" fill="#FFFFFF" text-anchor="middle">Train [4800:5300]</text>
    <text x="1198" y="136" font-size="18" fill="#DBEAFE" text-anchor="middle">+500 disjoint eps</text>

    <line x1="1322" y1="112" x2="1376" y2="112" stroke="#0F766E" stroke-width="3.5" marker-end="url(#arrTeal)"/>

    <rect x="1386" y="72" width="422" height="80" rx="12" fill="url(#gTealHdr)" filter="url(#shSm)"/>
    <text x="1597" y="106" font-size="22" font-weight="700" fill="#FFFFFF" text-anchor="middle">Zero-Shot OOD [10000 : 10300]</text>
    <text x="1597" y="136" font-size="19" fill="#CCFBF1" text-anchor="middle">S7N4 (7×7, N=4)  &amp;  S8N6 (8×8, N=6)</text>

    <text x="52" y="188" font-size="21" fill="#334155">
      <tspan font-weight="700" fill="#1E3A8A">Frozen Budget:</tspan> 4,500 train + 500 val = 5,000 demos  ·  Adam lr=1e-3, wd=0, 30 epochs  ·  Step-matched 10% / 20% / 50% / 100% scaling
    </text>

    <!-- Connector Banner: 14 Controlled Models -->
    <rect x="24" y="236" width="1812" height="58" rx="14" fill="#1E293B" filter="url(#shSm)"/>
    <text x="930" y="274" font-size="23" font-weight="700" fill="#FFFFFF" text-anchor="middle">
      14 Controlled Architectures (223k–231k params)  →  Evaluated via Three Orthogonal Causal Probes (n = 3 init seeds, ddof = 0)
    </text>

    <!-- TIER 2: Three Causal Probe Cards -->
    <!-- Card 1: Behavioral -->
    <rect x="24" y="318" width="580" height="348" rx="18" fill="url(#gBlue)" stroke="#2563EB" stroke-width="2.4" filter="url(#sh)"/>
    <rect x="48" y="342" width="532" height="48" rx="10" fill="url(#gBlueHdr)"/>
    <text x="314" y="374" font-size="23" font-weight="700" fill="#FFFFFF" text-anchor="middle">Probe 1 · Behavioral Counterfactual Swap</text>
    <!-- Mini swap diagram -->
    <rect x="64" y="410" width="210" height="54" rx="10" fill="#FFFFFF" stroke="#2563EB" stroke-width="1.8"/>
    <text x="169" y="445" font-size="20" font-weight="700" fill="#1E3A8A" text-anchor="middle">"go to blue key"</text>
    <text x="314" y="446" font-size="28" font-weight="700" fill="#2563EB" text-anchor="middle">⇄</text>
    <rect x="354" y="410" width="210" height="54" rx="10" fill="#FFFFFF" stroke="#7C3AED" stroke-width="1.8"/>
    <text x="459" y="445" font-size="20" font-weight="700" fill="#6D28D9" text-anchor="middle">"go to red key"</text>
    <text x="52" y="508" font-size="21" fill="#1E293B">• Fix exact (obs, dir) across <tspan font-weight="700">183 multi-object states</tspan></text>
    <text x="52" y="546" font-size="21" fill="#1E293B">• Swap color (138 pairs) or type (59 pairs) to an</text>
    <text x="68" y="576" font-size="21" fill="#1E293B">actual distractor present inside the room</text>
    <text x="52" y="618" font-size="21" font-weight="700" fill="#1D4ED8">→ Quantifies causal language-to-action sensitivity</text>

    <!-- Card 2: Feature-Space -->
    <rect x="640" y="318" width="580" height="348" rx="18" fill="url(#gTeal)" stroke="#0F766E" stroke-width="2.4" filter="url(#sh)"/>
    <rect x="664" y="342" width="532" height="48" rx="10" fill="url(#gTealHdr)"/>
    <text x="930" y="374" font-size="23" font-weight="700" fill="#FFFFFF" text-anchor="middle">Probe 2 · Hidden Feature-Space Geometry</text>
    <rect x="666" y="410" width="528" height="54" rx="10" fill="#FFFFFF" stroke="#0F766E" stroke-width="1.8"/>
    <text x="930" y="444" font-size="19" font-weight="700" fill="#0F766E" text-anchor="middle">h_mod (post-fusion)  &amp;  h₂ (post-Conv2) · 150 scenes</text>
    <text x="668" y="508" font-size="21" fill="#1E293B">• <tspan font-weight="700">Pre-Conv2 SNR:</tspan>  ‖h_mod(T)‖₂ / max ‖h_mod(D)‖₂</text>
    <text x="668" y="546" font-size="21" fill="#1E293B">• <tspan font-weight="700">Spatial Selectivity:</tspan>  Δh₂(Target) / Δh₂(Background)</text>
    <text x="668" y="584" font-size="21" fill="#1E293B">• <tspan font-weight="700">Background Drift:</tspan>  non-target perturbation ratio</text>
    <text x="668" y="622" font-size="21" font-weight="700" fill="#0F766E">→ Directly opens black box on S6N3, S7N4 &amp; S8N6</text>

    <!-- Card 3: Closed-Loop Taxonomy -->
    <rect x="1256" y="318" width="580" height="348" rx="18" fill="url(#gPurple)" stroke="#6D28D9" stroke-width="2.4" filter="url(#sh)"/>
    <rect x="1280" y="342" width="532" height="48" rx="10" fill="#6D28D9"/>
    <text x="1546" y="374" font-size="23" font-weight="700" fill="#FFFFFF" text-anchor="middle">Probe 3 · 300-Episode Failure Taxonomy</text>
    <rect x="1300" y="410" width="232" height="54" rx="10" fill="#FEE2E2" stroke="#DC2626" stroke-width="1.8"/>
    <text x="1416" y="445" font-size="20" font-weight="700" fill="#B91C1C" text-anchor="middle">Repr. Grounding</text>
    <rect x="1556" y="410" width="236" height="54" rx="10" fill="#DBEAFE" stroke="#2563EB" stroke-width="1.8"/>
    <text x="1674" y="445" font-size="20" font-weight="700" fill="#1D4ED8" text-anchor="middle">Coverage Loops</text>
    <text x="1284" y="508" font-size="21" fill="#1E293B">• Classifies every failed rollout over <tspan font-weight="700">300 unseen seeds</tspan></text>
    <text x="1284" y="546" font-size="21" fill="#1E293B">• Separates wrong-object / attribute confusion from</text>
    <text x="1300" y="576" font-size="21" fill="#1E293B">out-of-view POMDP observation-aliasing loops</text>
    <text x="1284" y="618" font-size="21" font-weight="700" fill="#6D28D9">→ Pinpoints exact physical ceiling of 1-frame BC</text>
    """
    return render_svg("svg_slide05_protocol", W, H, body)


# ==============================================================================
# 3. SLIDE 6 · Method I: Spatial Competition vs Channel Broadcast (1860 x 650)
# ==============================================================================
def make_slide06_competition():
    W, H = 1860, 650
    # Vector 7x7 heatmap for Seed 4502 ("go to the purple key", t=2) on the right
    gx, gy, cs = 1335, 198, 62
    # Exact pattern of seed 4502 ego-view (rows 0..2 walls, cols 0..1 & col 6 walls, lit 4x4 interior rows 3..6 cols 2..5)
    weights_4502 = {
        (3, 5): ("6.26×", "#881337", "#FFFFFF", "purple_key"),
        (4, 5): ("1.79×", "#FDE68A", "#78350F", "yellow_key"),
        (5, 4): ("1.70×", "#FEF08A", "#78350F", "yellow_key"),
    }
    cells_svg = []
    for r in range(7):
        for c in range(7):
            x0, y0 = gx + c * cs, gy + r * cs
            unlit = (r < 3) or (c < 2) or (c == 6)
            if (r, c) in weights_4502:
                bg = weights_4502[(r, c)][1]
            elif (r, c) == (2, 5):
                bg = "#D6BFA3"
            elif unlit:
                bg = "#CBD5E1"
            elif (r, c) in ((4, 4), (3, 4)):
                bg = "#FEF3C7"
            else:
                bg = "#FFFBEB"
            cells_svg.append(f'<rect x="{x0}" y="{y0}" width="{cs}" height="{cs}" fill="{bg}" stroke="#94A3B8" stroke-width="1.4"/>')
            if (r, c) in weights_4502:
                lbl, _, fg, kind = weights_4502[(r, c)]
                kcol = "#A855F7" if kind == "purple_key" else "#EAB308"
                kstr = "#581C87" if kind == "purple_key" else "#713F12"
                # Crisp vector key icon (plus/key glyph)
                cx, cy = x0 + cs / 2, y0 + 21
                cells_svg.append(f'<circle cx="{cx}" cy="{cy-4}" r="8" fill="{kcol}" stroke="{kstr}" stroke-width="2"/>')
                cells_svg.append(f'<rect x="{cx-3}" y="{cy+2}" width="6" height="13" fill="{kcol}" stroke="{kstr}" stroke-width="1.6"/>')
                cells_svg.append(f'<rect x="{x0+5}" y="{y0+38}" width="{cs-10}" height="20" rx="5" fill="#FFFFFF" fill-opacity="0.92"/>')
                cells_svg.append(f'<text x="{cx}" y="{y0+53}" font-size="16" font-weight="700" fill="#0F172A" text-anchor="middle">{lbl}</text>')
            if (r, c) == (6, 3):
                cx, cy = x0 + cs / 2, y0 + cs / 2
                cells_svg.append(f'<polygon points="{cx},{cy-16} {cx-14},{cy+14} {cx+14},{cy+14}" fill="#DC2626" stroke="#FFFFFF" stroke-width="2"/>')

    body = f"""
    <!-- Top Controlled Insertion Pipeline -->
    <rect x="24" y="14" width="1812" height="96" rx="16" fill="#F8FAFC" stroke="#CBD5E1" stroke-width="1.8"/>
    <rect x="46" y="28" width="210" height="68" rx="12" fill="#FFFFFF" stroke="#64748B" stroke-width="2"/>
    <text x="151" y="70" font-size="23" font-weight="700" fill="#0F172A" text-anchor="middle">Obs 7×7×3</text>
    <line x1="256" y1="62" x2="306" y2="62" stroke="#475569" stroke-width="3" marker-end="url(#arrSlate)"/>

    <rect x="316" y="28" width="305" height="68" rx="12" fill="#FFFFFF" stroke="#64748B" stroke-width="2"/>
    <text x="468" y="58" font-size="22" font-weight="700" fill="#0F172A" text-anchor="middle">Embed + Conv1 (3×3)</text>
    <text x="468" y="84" font-size="19" fill="#475569" text-anchor="middle">Visual Grid H ∈ ℝ³²ˣ⁷ˣ⁷</text>
    <line x1="621" y1="62" x2="675" y2="62" stroke="#2563EB" stroke-width="3.4" marker-end="url(#arrBlue)"/>

    <rect x="685" y="22" width="490" height="80" rx="14" fill="url(#gBlue)" stroke="#2563EB" stroke-width="3" filter="url(#shSm)"/>
    <text x="930" y="55" font-size="23" font-weight="700" fill="#1D4ED8" text-anchor="middle">Early Cross-Modal Fusion Site</text>
    <text x="930" y="84" font-size="18.5" font-weight="700" fill="#1E40AF" text-anchor="middle">Mission ℓ ∈ ℝ³² × Grid H (frozen elsewhere)</text>
    <line x1="1175" y1="62" x2="1229" y2="62" stroke="#2563EB" stroke-width="3.4" marker-end="url(#arrBlue)"/>

    <rect x="1239" y="28" width="255" height="68" rx="12" fill="#FFFFFF" stroke="#64748B" stroke-width="2"/>
    <text x="1366" y="58" font-size="22" font-weight="700" fill="#0F172A" text-anchor="middle">Conv2 (3×3) + ReLU</text>
    <text x="1366" y="84" font-size="19" fill="#475569" text-anchor="middle">32 × 7 × 7</text>
    <line x1="1494" y1="62" x2="1544" y2="62" stroke="#475569" stroke-width="3" marker-end="url(#arrSlate)"/>

    <rect x="1554" y="28" width="260" height="68" rx="12" fill="#FFFFFF" stroke="#64748B" stroke-width="2"/>
    <text x="1684" y="58" font-size="22" font-weight="700" fill="#0F172A" text-anchor="middle">Flatten + MLP</text>
    <text x="1684" y="84" font-size="19" fill="#475569" text-anchor="middle">→ 7 Actions</text>

    <!-- LEFT CARD 1: Control · Channel FiLM -->
    <rect x="24" y="132" width="1245" height="232" rx="18" fill="url(#gSlate)" stroke="#64748B" stroke-width="2.2" filter="url(#shSm)"/>
    <rect x="48" y="152" width="545" height="40" rx="9" fill="#475569"/>
    <text x="320" y="179" font-size="21" font-weight="700" fill="#FFFFFF" text-anchor="middle">Control · Channel FiLM  (225,183 params · broadcast)</text>
    <rect x="615" y="152" width="628" height="46" rx="9" fill="#FFFFFF" stroke="#94A3B8" stroke-width="1.6"/>
    <text x="929" y="183" font-size="21" font-weight="700" fill="#334155" text-anchor="middle">H' = (1 + Δγ(ℓ)) ⊙ H + Δβ(ℓ)    [identical ∀ 49 cells]</text>
    <text x="52" y="236" font-size="20.5" fill="#1E293B">• <tspan font-weight="700">Zero inter-cell competition:</tspan> identical channel weights ∀ 49 cells → pre-Conv2 T/maxD = <tspan font-weight="700" fill="#475569">1.005×</tspan> (S8N6: <tspan font-weight="700" fill="#DC2626">0.939×</tspan>)</text>
    <text x="52" y="278" font-size="20.5" fill="#1E293B">• <tspan font-weight="700">Global background drift:</tspan> additive Δβ perturbs empty/wall cells (drift <tspan font-weight="700">0.787</tspan>); distractor noise scales as <tspan font-weight="700">O(N−1)</tspan></text>
    <text x="52" y="322" font-size="22" font-weight="700" fill="#B91C1C">→ Fails crowded seed 4502 (0 / 3 seeds solved) &amp; drops 22.1 pts from S6N3 (0.959) to S8N6 (0.738)</text>

    <!-- LEFT CARD 2: Protagonist · Spatial-Softmax Attention -->
    <rect x="24" y="384" width="1245" height="248" rx="18" fill="url(#gBlue)" stroke="#2563EB" stroke-width="3.0" filter="url(#sh)"/>
    <rect x="48" y="404" width="565" height="42" rx="9" fill="url(#gBlueHdr)"/>
    <text x="330" y="432" font-size="21" font-weight="700" fill="#FFFFFF" text-anchor="middle">Ours · Spatial-Softmax  (225,183 params · exact tie!)</text>
    <rect x="630" y="402" width="615" height="48" rx="9" fill="#FFFFFF" stroke="#2563EB" stroke-width="2.0"/>
    <text x="937" y="434" font-size="22" font-weight="700" fill="#1E3A8A" text-anchor="middle">α = softmax₄₉(⟨W_q ℓ, W_k H⟩ / √C),   H' = (49·α) ⊙ H</text>
    <text x="52" y="490" font-size="22" fill="#0F172A">• <tspan font-weight="700" fill="#1D4ED8">Global 49-cell partition function:</tspan> conjunctive (color ∧ type) target suppresses partial-match distractors by <tspan font-weight="700" fill="#1D4ED8">exp(Δs)</tspan></text>
    <text x="52" y="534" font-size="22" fill="#0F172A">• <tspan font-weight="700" fill="#1D4ED8">7.6× SNR jump:</tspan> lifts pre-Conv2 h_mod T/maxD to <tspan font-weight="700" fill="#1D4ED8">7.669×</tspan> (S6N3) / <tspan font-weight="700" fill="#1D4ED8">5.193×</tspan> (S8N6) with zero additive empty-cell bias</text>
    <text x="52" y="582" font-size="22" font-weight="700" fill="#0F766E">→ Solves seed 4502 (3 / 3 in 7 steps!), beats FiLM by +14.9% on S8N6 (0.887) &amp; +8.4% at 20% data (0.898)</text>

    <!-- RIGHT PANEL: Real Seed 4502 Vector Heatmap -->
    <rect x="1294" y="132" width="542" height="500" rx="18" fill="#FFFFFF" stroke="#2563EB" stroke-width="2.4" filter="url(#sh)"/>
    <text x="1565" y="168" font-size="22" font-weight="700" fill="#1E3A8A" text-anchor="middle">Measured 49·α Map · Seed 4502 (t=2)</text>
    <text x="1565" y="191" font-size="18" font-weight="700" fill="#6D28D9" text-anchor="middle">Mission: "go to the purple key"</text>
    {"".join(cells_svg)}
    """
    return render_svg("svg_slide06_competition", W, H, body)


# ==============================================================================
# 4. SLIDE 7 · Method II: Fixing Naive Attention Pitfalls — Laws 1 & 2 (1860 x 650)
# ==============================================================================
def make_slide07_laws12():
    W, H = 1860, 650
    body = """
    <!-- LEFT COLUMN: LAW 1 (NoPE vs Absolute pos_emb) -->
    <rect x="24" y="16" width="886" height="258" rx="18" fill="url(#gRose)" stroke="#DC2626" stroke-width="2.4" filter="url(#shSm)"/>
    <rect x="48" y="36" width="520" height="42" rx="9" fill="#DC2626"/>
    <text x="308" y="64" font-size="22" font-weight="700" fill="#FFFFFF" text-anchor="middle">Pitfall 1 · Learned 2D Absolute pos_emb</text>
    <!-- Mini 6x6 vs 8x8 mismatch icon -->
    <rect x="705" y="38" width="76" height="76" rx="6" fill="#FFFFFF" stroke="#DC2626" stroke-width="2"/>
    <text x="743" y="83" font-size="20" font-weight="700" fill="#DC2626" text-anchor="middle">6×6 P</text>
    <text x="797" y="84" font-size="24" font-weight="700" fill="#DC2626" text-anchor="middle">≠</text>
    <rect x="814" y="28" width="76" height="88" rx="6" fill="#FEE2E2" stroke="#991B1B" stroke-width="2" stroke-dasharray="5,4"/>
    <text x="852" y="80" font-size="20" font-weight="700" fill="#991B1B" text-anchor="middle">8×8</text>
    <text x="50" y="118" font-size="21" font-weight="700" fill="#7F1D1D">Standard Cross-MHA adds P ∈ ℝ¹ˣ⁴⁹ˣ³² trained only on 6×6 rooms:</text>
    <text x="50" y="160" font-size="21" fill="#1E293B">• Memorizes 6×6 wall / corner priors inside the egocentric grid</text>
    <text x="50" y="198" font-size="21" fill="#1E293B">• Causes <tspan font-weight="700" fill="#DC2626">worst OOD background drift (0.823)</tspan> when room scales to 8×8</text>
    <text x="50" y="240" font-size="21" font-weight="700" fill="#B91C1C">→ Zero-shot S8N6 drops to 0.747 ± 0.054 (high variance across seeds)</text>

    <!-- Transformation Arrow Left -->
    <line x1="467" y1="276" x2="467" y2="348" stroke="#6D28D9" stroke-width="4" marker-end="url(#arrPurple)"/>
    <rect x="217" y="290" width="500" height="42" rx="21" fill="#6D28D9" filter="url(#shSm)"/>
    <text x="467" y="318" font-size="20" font-weight="700" fill="#FFFFFF" text-anchor="middle">Remove Absolute P  (Keep Conv1 Equivariance)</text>

    <!-- Law 1 Solution Box -->
    <rect x="24" y="358" width="886" height="274" rx="18" fill="url(#gPurple)" stroke="#6D28D9" stroke-width="3.0" filter="url(#sh)"/>
    <rect x="48" y="380" width="600" height="42" rx="9" fill="#6D28D9"/>
    <text x="348" y="408" font-size="22" font-weight="700" fill="#FFFFFF" text-anchor="middle">Design Law 1 · Translation-Equivariant NoPE</text>
    <rect x="666" y="380" width="218" height="42" rx="9" fill="#FFFFFF" stroke="#6D28D9" stroke-width="1.8"/>
    <text x="775" y="408" font-size="20" font-weight="700" fill="#6D28D9" text-anchor="middle">227,360 params</text>
    <text x="50" y="462" font-size="22" font-weight="700" fill="#4C1D95">cross_mha_nopos:  Grid Q = W_q H  attends to  K, V = W_{k,v} E_tokens</text>
    <text x="50" y="506" font-size="21" fill="#1E293B">• 3×3 Conv1 already encodes local spatial neighborhoods equivariantly</text>
    <text x="50" y="546" font-size="21" fill="#1E293B">• Cuts S8N6 background drift <tspan font-weight="700" fill="#6D28D9">0.823 → 0.749</tspan> &amp; solves seed 4502 (<tspan font-weight="700">3/3</tspan>)</text>
    <text x="50" y="594" font-size="22" font-weight="700" fill="#5B21B6">→ Single-factor ablation lifts zero-shot S8N6 by +6.2% (0.747 → 0.809)!</text>


    <!-- RIGHT COLUMN: LAW 2 (Attn-Map-Inject vs Obstacle Erasure) -->
    <rect x="950" y="16" width="886" height="258" rx="18" fill="url(#gOrange)" stroke="#D97706" stroke-width="2.4" filter="url(#shSm)"/>
    <rect x="974" y="36" width="530" height="42" rx="9" fill="#D97706"/>
    <text x="1239" y="64" font-size="22" font-weight="700" fill="#FFFFFF" text-anchor="middle">Pitfall 2 · Multiplicative Obstacle Erasure</text>
    <!-- Mini 3-cell obstacle erasure icon -->
    <rect x="1530" y="34" width="282" height="58" rx="10" fill="#FFFFFF" stroke="#D97706" stroke-width="1.8"/>
    <text x="1671" y="70" font-size="19" font-weight="700" fill="#9A3412" text-anchor="middle">▲ → [Box 0.60×!] → Key 6.6×</text>
    <text x="976" y="118" font-size="21" font-weight="700" fill="#7C2D12">Pure multiplicative gating  H' = (49·α) ⊙ H  scales non-targets → 0:</text>
    <text x="976" y="160" font-size="21" fill="#1E293B">• On seed 4501, blocking <tspan font-weight="700">yellow_box</tspan> is dimmed to <tspan font-weight="700" fill="#DC2626">0.60×</tspan> (seed 2: <tspan font-weight="700" fill="#DC2626">0.16×</tspan>!)</text>
    <text x="976" y="198" font-size="21" fill="#1E293B">• Conv2 loses physical obstacle boundaries right in front of the agent</text>
    <text x="976" y="240" font-size="21" font-weight="700" fill="#B45309">→ Agent walks into the blocking box &amp; solves seed 4501 on only 1 / 3 seeds!</text>

    <!-- Transformation Arrow Right -->
    <line x1="1393" y1="276" x2="1393" y2="348" stroke="#0F766E" stroke-width="4" marker-end="url(#arrTeal)"/>
    <rect x="1128" y="290" width="530" height="42" rx="21" fill="#0F766E" filter="url(#shSm)"/>
    <text x="1393" y="318" font-size="20" font-weight="700" fill="#FFFFFF" text-anchor="middle">Decouple Raw Features H  from  Pointer Maps 49α</text>

    <!-- Law 2 Solution Box with Channel Concat Mini-Diagram -->
    <rect x="950" y="358" width="886" height="274" rx="18" fill="url(#gTeal)" stroke="#0F766E" stroke-width="3.0" filter="url(#sh)"/>
    <rect x="974" y="380" width="590" height="42" rx="9" fill="url(#gTealHdr)"/>
    <text x="1269" y="408" font-size="22" font-weight="700" fill="#FFFFFF" text-anchor="middle">Design Law 2 · Uncorrupted Map Injection</text>
    <rect x="1584" y="380" width="226" height="42" rx="9" fill="#FFFFFF" stroke="#0F766E" stroke-width="1.8"/>
    <text x="1697" y="408" font-size="20" font-weight="700" fill="#0F766E" text-anchor="middle">226,335 params</text>

    <!-- Channel Concat Schematic Strip -->
    <rect x="976" y="438" width="250" height="54" rx="10" fill="#FFFFFF" stroke="#475569" stroke-width="2"/>
    <text x="1101" y="472" font-size="20" font-weight="700" fill="#0F172A" text-anchor="middle">Raw H (32×7×7 · 1.0×)</text>
    <text x="1244" y="473" font-size="26" font-weight="700" fill="#0F766E" text-anchor="middle">⊕</text>
    <rect x="1264" y="438" width="240" height="54" rx="10" fill="#DBEAFE" stroke="#2563EB" stroke-width="2"/>
    <text x="1384" y="472" font-size="20" font-weight="700" fill="#1D4ED8" text-anchor="middle">4 Maps 49α (4×7×7)</text>
    <line x1="1508" y1="465" x2="1552" y2="465" stroke="#0F766E" stroke-width="3" marker-end="url(#arrTeal)"/>
    <rect x="1560" y="438" width="250" height="54" rx="10" fill="#CCFBF1" stroke="#0F766E" stroke-width="2.2"/>
    <text x="1685" y="472" font-size="20" font-weight="700" fill="#0F766E" text-anchor="middle">Conv2 (36 → 32)</text>

    <text x="976" y="536" font-size="21" fill="#1E293B">• Obstacles &amp; walls stay <tspan font-weight="700" fill="#0F766E">100% unmodified (1.0× floor)</tspan> while maps point to target</text>
    <text x="976" y="590" font-size="22" font-weight="700" fill="#0F766E">→ Lowest drift of all models (0.569) &amp; solves seed 4501 3/3 in 8/8/8 steps!</text>
    """
    return render_svg("svg_slide07_laws12", W, H, body)


# ==============================================================================
# 5. SLIDE 8 · Method III: Egocentric Relational & Token Co-Attention (1860 x 650)
# ==============================================================================
def make_slide08_relational():
    W, H = 1860, 650
    body = """
    <!-- TOP ARCHITECTURE FLOWCHART: MH-Spatial-Relational (229,572 params) -->
    <rect x="24" y="12" width="1812" height="232" rx="18" fill="#F8FAFC" stroke="#CBD5E1" stroke-width="2" filter="url(#shSm)"/>
    <text x="48" y="42" font-size="20" font-weight="700" fill="#0F766E">MH-Spatial-Relational Architecture  (229,572 params · Law 3 End-to-End Pipeline)</text>

    <!-- Node 1: Conv1 H -->
    <rect x="46" y="92" width="206" height="102" rx="14" fill="#FFFFFF" stroke="#64748B" stroke-width="2.2" filter="url(#shSm)"/>
    <text x="149" y="135" font-size="22" font-weight="700" fill="#0F172A" text-anchor="middle">Conv1 Features</text>
    <text x="149" y="168" font-size="20" fill="#475569" text-anchor="middle">H ∈ ℝ³²ˣ⁷ˣ⁷</text>
    <line x1="252" y1="143" x2="306" y2="143" stroke="#2563EB" stroke-width="3.2" marker-end="url(#arrBlue)"/>

    <!-- Node 2: 4-Head Residual Spatial-Softmax -->
    <rect x="316" y="82" width="376" height="122" rx="14" fill="url(#gBlue)" stroke="#2563EB" stroke-width="2.6" filter="url(#shSm)"/>
    <text x="504" y="120" font-size="21.5" font-weight="700" fill="#1E3A8A" text-anchor="middle">4-Head Residual Softmax</text>
    <text x="504" y="152" font-size="19.5" font-weight="700" fill="#1D4ED8" text-anchor="middle">H_res = H · (1 + w ⊙ (49α − 1))</text>
    <text x="504" y="182" font-size="17.5" fill="#334155" text-anchor="middle">Mission ℓ grounds target salience</text>
    <!-- Residual skip arc 1 (routed cleanly below header text!) -->
    <path d="M 195 92 C 235 54, 625 54, 668 82" stroke="#2563EB" stroke-width="2.2" stroke-dasharray="7,5" fill="none" marker-end="url(#arrBlue)"/>

    <!-- Coord injection node (+) -->
    <line x1="692" y1="143" x2="768" y2="143" stroke="#0F766E" stroke-width="3.2" marker-end="url(#arrTeal)"/>
    <circle cx="800" cy="143" r="23" fill="#CCFBF1" stroke="#0F766E" stroke-width="2.6"/>
    <text x="800" y="152" font-size="27" font-weight="700" fill="#0F766E" text-anchor="middle">+</text>
    <!-- Smooth 2D Ego Coords badge feeding (+) -->
    <rect x="665" y="190" width="270" height="44" rx="10" fill="#FFFFFF" stroke="#0F766E" stroke-width="2"/>
    <text x="800" y="218" font-size="17.2" font-weight="700" fill="#0F766E" text-anchor="middle">Coords [-1,1]² → Lin(2→32)</text>
    <line x1="800" y1="190" x2="800" y2="169" stroke="#0F766E" stroke-width="2.6" marker-end="url(#arrTeal)"/>

    <line x1="823" y1="143" x2="884" y2="143" stroke="#0F766E" stroke-width="3.2" marker-end="url(#arrTeal)"/>

    <!-- Node 3: 49x49 Multi-Head Relational Self-Attention -->
    <rect x="894" y="82" width="420" height="122" rx="14" fill="url(#gTeal)" stroke="#0F766E" stroke-width="2.8" filter="url(#shSm)"/>
    <text x="1104" y="120" font-size="21.5" font-weight="700" fill="#0F766E" text-anchor="middle">49×49 Egocentric Self-Attention</text>
    <text x="1104" y="152" font-size="19.5" font-weight="700" fill="#115E59" text-anchor="middle">MHA( Q=K=H_res+C,  V=H_res )</text>
    <text x="1104" y="182" font-size="17.5" fill="#334155" text-anchor="middle">Target ↔ obstacle bearing exchange</text>
    <!-- Residual skip arc 2 -->
    <path d="M 742 143 C 760 48, 1340 48, 1420 86" stroke="#0F766E" stroke-width="2.2" stroke-dasharray="7,5" fill="none" marker-end="url(#arrTeal)"/>

    <line x1="1314" y1="143" x2="1368" y2="143" stroke="#0F766E" stroke-width="3.2" marker-end="url(#arrTeal)"/>

    <!-- Node 4: LayerNorm + Conv2 + Policy -->
    <rect x="1378" y="84" width="434" height="118" rx="14" fill="#FFFFFF" stroke="#0F766E" stroke-width="2.4" filter="url(#shSm)"/>
    <text x="1595" y="122" font-size="21" font-weight="700" fill="#0F172A" text-anchor="middle">LN( H_res + γ_rel · Attn )</text>
    <text x="1595" y="155" font-size="19.5" font-weight="700" fill="#0F766E" text-anchor="middle">→ Conv2(32) + MLP → 7 Actions</text>
    <text x="1595" y="184" font-size="17" fill="#475569" text-anchor="middle">(learned γ_rel = 1.07–1.20 across seeds)</text>

    <!-- BOTTOM LEFT: Law 3 Relational Message Passing Schematic -->
    <rect x="24" y="262" width="912" height="370" rx="18" fill="url(#gTeal)" stroke="#0F766E" stroke-width="2.6" filter="url(#sh)"/>
    <rect x="48" y="284" width="620" height="42" rx="9" fill="url(#gTealHdr)"/>
    <text x="358" y="312" font-size="21" font-weight="700" fill="#FFFFFF" text-anchor="middle">Design Law 3 · Non-Local Target ↔ Obstacle Reasoning</text>

    <!-- Mini 5x5 relational message passing grid on left of card -->
    <rect x="52" y="346" width="225" height="225" rx="12" fill="#FFFFFF" stroke="#0F766E" stroke-width="2"/>
    <!-- Grid lines -->
    <line x1="97" y1="346" x2="97" y2="571" stroke="#E2E8F0"/><line x1="142" y1="346" x2="142" y2="571" stroke="#E2E8F0"/>
    <line x1="187" y1="346" x2="187" y2="571" stroke="#E2E8F0"/><line x1="232" y1="346" x2="232" y2="571" stroke="#E2E8F0"/>
    <line x1="52" y1="391" x2="277" y2="391" stroke="#E2E8F0"/><line x1="52" y1="436" x2="277" y2="436" stroke="#E2E8F0"/>
    <line x1="52" y1="481" x2="277" y2="481" stroke="#E2E8F0"/><line x1="52" y1="526" x2="277" y2="526" stroke="#E2E8F0"/>
    <!-- Target blue key, blocking yellow box, corridor cell, agent -->
    <rect x="144" y="393" width="41" height="41" rx="6" fill="#DBEAFE" stroke="#2563EB" stroke-width="2.2"/>
    <text x="164" y="420" font-size="16" font-weight="700" fill="#1D4ED8" text-anchor="middle">Key</text>
    <rect x="144" y="458" width="41" height="41" rx="6" fill="#FEF3C7" stroke="#D97706" stroke-width="2.2"/>
    <text x="164" y="485" font-size="16" font-weight="700" fill="#B45309" text-anchor="middle">Box</text>
    <rect x="191" y="458" width="40" height="41" rx="6" fill="#CCFBF1" stroke="#0F766E" stroke-width="2" stroke-dasharray="4,3"/>
    <polygon points="164,534 150,562 178,562" fill="#DC2626"/>
    <!-- Relational attention curved arrows from Key to Box & Corridor -->
    <path d="M 152 434 C 125 446, 125 462, 144 474" stroke="#0F766E" stroke-width="3" fill="none" marker-end="url(#arrTeal)"/>
    <path d="M 184 422 C 215 426, 218 442, 212 456" stroke="#2563EB" stroke-width="3" fill="none" marker-end="url(#arrBlue)"/>

    <text x="300" y="378" font-size="21" fill="#0F172A">• Local 3×3 Conv2 plans myopically when obstacle</text>
    <text x="316" y="408" font-size="21" fill="#0F172A">sits between agent &amp; distant target</text>
    <text x="300" y="452" font-size="21" fill="#0F172A">• <tspan font-weight="700" fill="#0F766E">49×49 self-attention</tspan> broadcasts target bearing</text>
    <text x="316" y="482" font-size="21" fill="#0F172A">directly to blocking box &amp; flanking corridor</text>
    <rect x="300" y="510" width="612" height="96" rx="12" fill="#FFFFFF" stroke="#0F766E" stroke-width="2"/>
    <text x="606" y="546" font-size="20" font-weight="700" fill="#0F766E" text-anchor="middle">Strict Ablation vs Parent MH-Res-Softmax:</text>
    <text x="606" y="582" font-size="20" font-weight="700" fill="#0F172A" text-anchor="middle">S6N3: 0.959 → 0.972 (#1)  ·  Repr: 5.7 → 2.3  ·  4501: 0/3 → 3/3!</text>

    <!-- BOTTOM RIGHT: Token-CoAttn-Spatial Bilinear Schematic -->
    <rect x="962" y="262" width="874" height="370" rx="18" fill="url(#gBlue)" stroke="#2563EB" stroke-width="2.4" filter="url(#sh)"/>
    <rect x="986" y="284" width="635" height="42" rx="9" fill="url(#gBlueHdr)"/>
    <text x="1303" y="312" font-size="21" font-weight="700" fill="#FFFFFF" text-anchor="middle">Complementary · Token-CoAttn-Spatial (227,435 p)</text>

    <!-- Word-to-Head Bilinear Split Diagram -->
    <rect x="988" y="350" width="215" height="50" rx="10" fill="#FFFFFF" stroke="#64748B" stroke-width="1.8"/>
    <text x="1095" y="382" font-size="19" font-weight="700" fill="#475569" text-anchor="middle">"go" · "to" · "the"</text>
    <rect x="988" y="420" width="215" height="54" rx="10" fill="#F3E8FF" stroke="#7C3AED" stroke-width="2.2"/>
    <text x="1095" y="454" font-size="20" font-weight="700" fill="#6D28D9" text-anchor="middle">&lt;color&gt; + &lt;type&gt;</text>

    <path d="M 1203 447 L 1288 386" stroke="#2563EB" stroke-width="3.2" marker-end="url(#arrBlue)"/>
    <path d="M 1203 375 L 1288 450" stroke="#0F766E" stroke-width="3.0" stroke-dasharray="7,5" marker-end="url(#arrTeal)"/>

    <rect x="1298" y="346" width="512" height="62" rx="10" fill="#FFFFFF" stroke="#2563EB" stroke-width="2"/>
    <text x="1554" y="385" font-size="20" font-weight="700" fill="#1E3A8A" text-anchor="middle">Heads 1–2 (0.70 / 0.73 mass) → Lock Target</text>
    <rect x="1298" y="422" width="512" height="62" rx="10" fill="#FFFFFF" stroke="#0F766E" stroke-width="2"/>
    <text x="1554" y="461" font-size="20" font-weight="700" fill="#0F766E" text-anchor="middle">Head 3 (0.32 'the' / 0.17 'go') → Free Corridor</text>

    <text x="988" y="534" font-size="21" fill="#1E293B">• Replaces pooled ℓ with full <tspan font-weight="700" fill="#1D4ED8">5-token × 49-cell</tspan> bilinear affinity</text>
    <text x="988" y="576" font-size="21" fill="#1E293B">• <tspan font-weight="700" fill="#0F766E">Zero supervision:</tspan> syntactic-to-spatial role division emerges</text>
    <text x="1004" y="608" font-size="21" fill="#1E293B">automatically during standard BC imitation (Lu/Kim [6])</text>
    """
    return render_svg("svg_slide08_relational", W, H, body)


# ==============================================================================
# 6. SLIDE 13 RIGHT · 4 Diagnostic Rollout Scorecard SVG (860 x 740)
# ==============================================================================
def make_slide13_demo_cards():
    W, H = 860, 740
    cards = [
        ("Clip 1 · seed 4502  ('go to the purple key')",
         "3/3 · 7 STEPS", "#0F766E", "#CCFBF1", "url(#gBlue)", "#2563EB",
         "Crowded yellow-key distractors: BC30 (0/3) &amp; Channel FiLM",
         "(0/3) both timeout; Spatial-SS &amp; MH-Rel lock target cleanly."),
        ("Clip 2 · seed 4501  ('go to the blue key')",
         "3/3 · 8 STEPS", "#0F766E", "#CCFBF1", "url(#gTeal)", "#0F766E",
         "Yellow box blocks direct path: v1 Spatial-SS dims box (1/3);",
         "Map-Inject (1.0× h₁) &amp; MH-Rel detour optimally in 8 steps."),
        ("Clip 3 · seed 4516  ('go to the purple box')",
         "3/3 · 8 STEPS", "#1D4ED8", "#DBEAFE", "url(#gBlue)", "#2563EB",
         "Same-color purple key vs purple box: BC30 fails (0/3)",
         "attribute confusion; conjunctive softmax binds color ∧ type."),
        ("Clip 4 · seed 4568  ('go to the blue box')",
         "0/3 · POMDP LIMIT", "#B91C1C", "#FEE2E2", "url(#gRose)", "#DC2626",
         "Target spawns behind agent (outside 7×7 ego-cone): BC30,",
         "FiLM &amp; Attention all oscillate L↔R without temporal memory."),
    ]
    parts = []
    yy = 12
    for ttl, badge, b_fg, b_bg, fl, st, l1, l2 in cards:
        parts.append(f"""
        <rect x="14" y="{yy}" width="832" height="166" rx="16" fill="{fl}" stroke="{st}" stroke-width="2.4" filter="url(#shSm)"/>
        <text x="38" y="{yy+44}" font-size="22" font-weight="700" fill="#0F172A">{ttl}</text>
        <rect x="612" y="{yy+18}" width="212" height="36" rx="18" fill="{b_bg}" stroke="{b_fg}" stroke-width="1.8"/>
        <text x="718" y="{yy+43}" font-size="17" font-weight="700" fill="{b_fg}" text-anchor="middle">{badge}</text>
        <text x="38" y="{yy+94}" font-size="20" fill="#1E293B">{l1}</text>
        <text x="38" y="{yy+130}" font-size="20" fill="#1E293B">{l2}</text>
        """)
        yy += 184
    return render_svg("svg_slide13_demo_cards", W, H, "\n".join(parts))


# ==============================================================================
# 7. SLIDE 14 RIGHT · POMDP Blind-Spot Ego-Cone & Failure Funnel (970 x 584)
# ==============================================================================
def make_slide14_pomdp():
    W, H = 970, 584
    # Vector 6x6 room schematic of seed 4568 showing lit ego-cone in front vs unlit blind spot behind!
    rx, ry, cs = 42, 46, 42
    grid = []
    for r in range(6):
        for c in range(6):
            x0, y0 = rx + c * cs, ry + r * cs
            is_wall = r == 0 or r == 5 or c == 0 or c == 5
            in_cone = (not is_wall) and (c >= 3)  # Agent at (2,3) facing East -> behind (cols 1,2) is blind spot!
            bg = "#94A3B8" if is_wall else ("#FEF9C3" if in_cone else "#334155")
            grid.append(f'<rect x="{x0}" y="{y0}" width="{cs}" height="{cs}" fill="{bg}" stroke="#64748B" stroke-width="1.2"/>')
    # Blue box target behind agent at (1,1), purple key at (3,1), green ball in front at (1,4), agent at (2,3) facing left/east
    grid.append(f'<rect x="{rx+1*cs+7}" y="{ry+1*cs+7}" width="28" height="28" rx="4" fill="none" stroke="#3B82F6" stroke-width="3.5"/>')
    grid.append(f'<circle cx="{rx+4.5*cs}" cy="{ry+1.5*cs}" r="12" fill="#22C55E"/>')
    cx_a, cy_a = rx + 3.5 * cs, ry + 2.5 * cs
    grid.append(f'<polygon points="{cx_a+14},{cy_a} {cx_a-12},{cy_a-12} {cx_a-12},{cy_a+12}" fill="#DC2626" stroke="#FFFFFF" stroke-width="2"/>')

    body = f"""
    <!-- Top Card: Blind-Spot Observation Aliasing Schematic -->
    <rect x="14" y="14" width="942" height="292" rx="18" fill="url(#gRose)" stroke="#DC2626" stroke-width="2.4" filter="url(#shSm)"/>
    {"".join(grid)}
    <rect x="30" y="18" width="276" height="28" rx="7" fill="#DC2626"/>
    <text x="168" y="38" font-size="14.6" font-weight="700" fill="#FFFFFF" text-anchor="middle">seed 4568 · Ego Cone vs Blind Spot</text>

    <text x="322" y="58" font-size="23" font-weight="700" fill="#991B1B">Hard POMDP Blind-Spot Limit</text>
    <text x="322" y="102" font-size="20" fill="#1E293B">• Target <tspan font-weight="700" fill="#1D4ED8">blue_box</tspan> spawns in dark blind spot</text>
    <text x="338" y="132" font-size="20" fill="#1E293B">behind agent (outside 7×7 forward view)</text>
    <text x="322" y="176" font-size="20" fill="#1E293B">• Memoryless <tspan class="mono" font-weight="700">argmax π(o_t)</tspan> maps aliased</text>
    <text x="338" y="206" font-size="20" fill="#1E293B">observations into deterministic <tspan font-weight="700" fill="#DC2626">L ↔ R loops</tspan></text>
    <rect x="316" y="232" width="618" height="50" rx="10" fill="#FFFFFF" stroke="#DC2626" stroke-width="1.8"/>
    <text x="625" y="263" font-size="17.2" font-weight="700" fill="#B91C1C" text-anchor="middle">0/3 solved across ALL 14 single-frame models (BC, FiLM &amp; Attention)</text>

    <!-- Bottom Card: Quantitative 300-Episode Reduction Funnel -->
    <rect x="14" y="324" width="942" height="246" rx="18" fill="#F8FAFC" stroke="#1E3A8A" stroke-width="2.4" filter="url(#shSm)"/>
    <rect x="38" y="344" width="610" height="40" rx="9" fill="#112B5C"/>
    <text x="343" y="371" font-size="20" font-weight="700" fill="#FFFFFF" text-anchor="middle">300-Episode Failure Funnel  (Mean of 3 Init Seeds)</text>
    <text x="40" y="424" font-size="20" fill="#0F172A">• <tspan font-weight="700" fill="#DC2626">Representation Errors (-94%):</tspan>  BC30 <tspan font-weight="700">38.3</tspan> → FiLM <tspan font-weight="700">6.3</tspan> → Spatial-SS <tspan font-weight="700" fill="#1D4ED8">3.7</tspan> → MH-Rel <tspan font-weight="700" fill="#0F766E">2.3</tspan></text>
    <text x="58" y="458" font-size="19" font-weight="700" fill="#0F766E">(attribute_confusion cut to just 1.0 / 300 episodes!)</text>
    <text x="40" y="504" font-size="20" fill="#0F172A">• <tspan font-weight="700" fill="#2563EB">Coverage Loops (~5.0–6.0 floor):</tspan>  orthogonal to single-frame fusion —</text>
    <text x="58" y="538" font-size="20" fill="#334155">requires <tspan font-weight="700" fill="#1E3A8A">Recurrent Memory (ConvGRU)</tspan> or loop-unsticking <tspan font-weight="700" fill="#1E3A8A">DAgger (B12)</tspan></text>
    """
    return render_svg("svg_slide14_pomdp", W, H, body)


# ==============================================================================
# 8. SLIDE 15 · Executive Synthesis Infographic: 3 Laws & Tradeoffs (1860 x 560)
# ==============================================================================
def make_slide15_conclusions():
    W, H = 560, 1860  # Note: W=1860, H=560
    W, H = 1860, 560
    body = """
    <!-- Banner 1: Equal-Param Victory -->
    <rect x="24" y="14" width="1812" height="142" rx="18" fill="url(#gBlue)" stroke="#2563EB" stroke-width="2.6" filter="url(#shSm)"/>
    <circle cx="92" cy="85" r="42" fill="url(#gBlueHdr)"/>
    <text x="92" y="98" font-size="34" font-weight="700" fill="#FFFFFF" text-anchor="middle">01</text>
    <text x="158" y="60" font-size="24" font-weight="700" fill="#1E3A8A">Spatial Competition Beats Channel Broadcast at Identical Parameters (225,183 p)</text>
    <text x="158" y="102" font-size="21" fill="#1E293B">49-cell softmax lifts pre-Conv2 T/maxD SNR <tspan font-weight="700" fill="#1D4ED8">1.005× → 7.669×</tspan> &amp; spatial selectivity <tspan font-weight="700" fill="#1D4ED8">1.90× → 3.59×</tspan>,</text>
    <text x="158" y="134" font-size="21" fill="#1E293B">beating Channel FiLM by <tspan font-weight="700" fill="#1D4ED8">+14.9% on S8N6</tspan> (0.887 vs 0.738) and matching 100% FiLM with <tspan font-weight="700" fill="#1D4ED8">50% data (0.958)</tspan>.</text>

    <!-- Banner 2: Three v2 Geometric Laws -->
    <rect x="24" y="176" width="1812" height="156" rx="18" fill="url(#gTeal)" stroke="#0F766E" stroke-width="2.6" filter="url(#shSm)"/>
    <circle cx="92" cy="254" r="42" fill="url(#gTealHdr)"/>
    <text x="92" y="267" font-size="34" font-weight="700" fill="#FFFFFF" text-anchor="middle">02</text>
    <text x="158" y="220" font-size="24" font-weight="700" fill="#0F766E">Three Verified v2 Geometric Design Laws Fix Naive Attention Pitfalls</text>
    <!-- 3 mini pills inside Banner 2 -->
    <rect x="158" y="240" width="520" height="72" rx="12" fill="#FFFFFF" stroke="#6D28D9" stroke-width="2"/>
    <text x="418" y="272" font-size="19" font-weight="700" fill="#6D28D9" text-anchor="middle">(a) Translation-Equivariant NoPE</text>
    <text x="418" y="300" font-size="18" fill="#334155" text-anchor="middle">Cuts drift 0.823→0.749 · +6.2% S8N6</text>

    <rect x="698" y="240" width="540" height="72" rx="12" fill="#FFFFFF" stroke="#0F766E" stroke-width="2"/>
    <text x="968" y="272" font-size="19" font-weight="700" fill="#0F766E" text-anchor="middle">(b) Uncorrupted Map-Inject [H; 49α]</text>
    <text x="968" y="300" font-size="18" fill="#334155" text-anchor="middle">1.0× obstacle floor · min drift 0.569 · 4501 3/3</text>

    <rect x="1258" y="240" width="554" height="72" rx="12" fill="#FFFFFF" stroke="#1D4ED8" stroke-width="2"/>
    <text x="1535" y="272" font-size="19" font-weight="700" fill="#1D4ED8" text-anchor="middle">(c) Egocentric MH-Relational</text>
    <text x="1535" y="300" font-size="18" fill="#334155" text-anchor="middle">#1 S6N3 (0.972) · 2.3/300 repr · 4501 &amp; 4502 3/3</text>

    <!-- Banner 3: Honest Bias-Variance Tradeoff & Next Step -->
    <rect x="24" y="352" width="1812" height="126" rx="18" fill="url(#gSlate)" stroke="#475569" stroke-width="2.2" filter="url(#shSm)"/>
    <circle cx="92" cy="415" r="42" fill="#334155"/>
    <text x="92" y="428" font-size="34" font-weight="700" fill="#FFFFFF" text-anchor="middle">03</text>
    <text x="158" y="396" font-size="24" font-weight="700" fill="#0F172A">Honest Bias–Variance Tradeoff &amp; Principled Next Step</text>
    <text x="158" y="434" font-size="20" fill="#1E293B">1-head <tspan font-weight="700" fill="#1D4ED8">Spatial-SS</tspan> excels on OOD scale (0.888) &amp; low data (0.802 @10%, 0.898 @20%); <tspan font-weight="700" fill="#0F766E">MH-Rel</tspan> leads at full data (0.972) &amp; detours (3/3).</text>
    <text x="158" y="463" font-size="20" fill="#334155"><tspan font-weight="700" fill="#991B1B">Next diagnostic:</tspan> pair MH-Spatial-Relational with recurrent <tspan font-weight="700">ConvGRU memory</tspan> to resolve the 6.0 / 300 blind-spot coverage ceiling.</text>

    <!-- Bottom Team Ownership Ribbon -->
    <rect x="24" y="496" width="1812" height="50" rx="12" fill="#112B5C"/>
    <text x="930" y="529" font-size="20" font-weight="700" fill="#FFFFFF" text-anchor="middle">
      Speaker A: Task, Baseline &amp; Protocol (1–5)    ·    Speaker B: Attention Laws &amp; Feature Proof (6–11)    ·    Speaker C: Diagnostics &amp; Limits (12–15)
    </text>
    """
    return render_svg("svg_slide15_conclusions", W, H, body)


if __name__ == "__main__":
    make_slide03_bcnet()
    make_slide05_protocol()
    make_slide06_competition()
    make_slide07_laws12()
    make_slide08_relational()
    make_slide13_demo_cards()
    make_slide14_pomdp()
    make_slide15_conclusions()
    print("ALL 8 PUBLICATION-GRADE SVG DIAGRAMS GENERATED SUCCESSFULLY.")
