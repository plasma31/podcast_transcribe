#!/usr/bin/env python3
"""Build a single PDF from the docs/thesis markdown files.

Run with the full venv:
    .venv/bin/python docs/thesis/_build_pdf.py

Output:
    dist/thesis_documentation.pdf
"""
import re
from pathlib import Path

import markdown
from fpdf import FPDF

THESIS = Path(__file__).resolve().parent
ROOT = THESIS.parents[1]
FILES = [
    "README.md",
    "01_pipeline_and_data.md",
    "02_topic_modeling.md",
    "03_results_and_figures.md",
]
DEJAVU = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
DEJAVU_B = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
DEJAVU_O = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Oblique.ttf"
DEJAVU_BO = "/usr/share/fonts/truetype/dejavu/DejaVuSans-BoldOblique.ttf"
DEJAVU_M = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"

# ---------- assemble markdown ----------
parts = []
for fname in FILES:
    md = (THESIS / fname).read_text(encoding="utf-8")
    # Strip mermaid fences — fpdf2 can't render them
    md = re.sub(
        r"```mermaid.*?```",
        "*[Pipeline flow diagram — Mermaid source in docs/thesis/01_pipeline_and_data.md]*",
        md,
        flags=re.DOTALL,
    )
    # Rewrite relative image paths to absolute so fpdf2 finds them
    def _abs_img(m):
        alt, path = m.group(1), m.group(2)
        if path.startswith(("http://", "https://", "/")):
            return m.group(0)
        return f"![{alt}]({(THESIS / path).resolve()})"
    md = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", _abs_img, md)
    parts.append(md)

combined_md = "\n\n<hr/>\n\n".join(parts)

# ---------- markdown -> HTML ----------
html_body = markdown.markdown(
    combined_md,
    extensions=["tables", "fenced_code", "sane_lists"],
)

# fpdf2's write_html doesn't accept inline tags inside <td>/<th>. Strip every
# inline tag (b/strong/i/em/code/a) only within table cells, preserving them
# in body text where they render fine.
html_body = re.sub(r"</?blockquote>", "", html_body)
# Normalise <strong>/<em> to <b>/<i> globally
html_body = html_body.replace("<strong>", "<b>").replace("</strong>", "</b>")
html_body = html_body.replace("<em>", "<i>").replace("</em>", "</i>")

def _strip_inline_in_cell(m):
    inner = m.group(2)
    inner = re.sub(r"</?(?:b|i|code|strong|em|a(?:\s[^>]*)?)>", "", inner)
    return f"<{m.group(1)}>{inner}</{m.group(1).split()[0]}>"

html_body = re.sub(
    r"<(td|th)>(.*?)</\1>",
    _strip_inline_in_cell,
    html_body,
    flags=re.DOTALL,
)

# Wrap with the minimal styling fpdf2 understands.
html = f"""
<h1>Podcast Thesis — Documentation</h1>
<p><i>Methodological documentation of the German-language podcast processing
pipeline and the first topic models. Generated from docs/thesis/.</i></p>
<hr/>
{html_body}
"""

# ---------- PDF ----------
pdf = FPDF(orientation="P", unit="mm", format="A4")
pdf.set_margins(left=18, top=18, right=18)
pdf.set_auto_page_break(auto=True, margin=18)
pdf.add_font("DejaVu", "", DEJAVU)
pdf.add_font("DejaVu", "B", DEJAVU_B)
# DejaVuSans-Oblique isn't packaged on this distro; alias regular file to the
# italic styles so write_html doesn't fail. Italic emphasis is lost visually.
pdf.add_font("DejaVu", "I", DEJAVU)
pdf.add_font("DejaVu", "BI", DEJAVU_B)
pdf.add_font("DejaVuMono", "", DEJAVU_M)
pdf.add_font("DejaVuMono", "B", DEJAVU_M)
pdf.add_font("DejaVuMono", "I", DEJAVU_M)
pdf.add_font("DejaVuMono", "BI", DEJAVU_M)
pdf.set_font("DejaVu", size=10)
pdf.add_page()
pdf.write_html(
    html,
    font_family="DejaVu",
    pre_code_font="DejaVuMono",
    table_line_separators=True,
)

out = ROOT / "dist" / "thesis_documentation.pdf"
out.parent.mkdir(parents=True, exist_ok=True)
pdf.output(str(out))
print("wrote:", out, "(%d pages)" % pdf.pages_count)
