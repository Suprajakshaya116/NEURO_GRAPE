"""
Convert Neuro-GRAPE Research Paper Markdown to a publication-quality PDF
using markdown-it-py, MathJax, Mermaid.js, and headless Google Chrome.
"""

import os
import re
import base64
import subprocess
from pathlib import Path
from markdown_it import MarkdownIt

def build_pdf():
    workspace = Path(r"d:\projects\NEURO_GRAPE")
    paper_dir = workspace / "paper"
    md_file = paper_dir / "neuro_grape_research_paper.md"
    html_file = paper_dir / "neuro_grape_research_paper.html"
    pdf_file = paper_dir / "neuro_grape_research_paper.pdf"
    
    print(f"Reading {md_file}...")
    content = md_file.read_text(encoding="utf-8")
    
    # 1. Protect display math ($$...$$) and inline math ($...$) from markdown mangling
    display_math = []
    inline_math = []
    
    def repl_display_math(match):
        idx = len(display_math)
        display_math.append(match.group(1))
        return f"%%DISPLAYMATH{idx}%%"
    
    def repl_inline_math(match):
        idx = len(inline_math)
        inline_math.append(match.group(1))
        return f"%%INLINEMATH{idx}%%"
    
    # Protect $$ ... $$
    content = re.sub(r"\$\$(.*?)\$\$", repl_display_math, content, flags=re.DOTALL)
    # Protect $ ... $
    content = re.sub(r"(?<!\$)\$(?!\$)([^\n]+?)(?<!\$)\$(?!\$)", repl_inline_math, content)
    
    # 2. Render Markdown to HTML
    md = MarkdownIt('commonmark').enable(['table', 'strikethrough'])
    html_body = md.render(content)
    
    # 3. Restore Math
    for idx, math in enumerate(display_math):
        html_body = html_body.replace(f"%%DISPLAYMATH{idx}%%", f"$${math}$$")
        # Also handle potential HTML escaping
        html_body = html_body.replace(f"&percnt;&percnt;DISPLAYMATH{idx}&percnt;&percnt;", f"$${math}$$")
        
    for idx, math in enumerate(inline_math):
        html_body = html_body.replace(f"%%INLINEMATH{idx}%%", f"${math}$")
        html_body = html_body.replace(f"&percnt;&percnt;INLINEMATH{idx}&percnt;&percnt;", f"${math}$")
    
    # 4. Transform mermaid code blocks to <div class="mermaid">
    html_body = re.sub(
        r'<pre><code class="language-mermaid">(.*?)</code></pre>',
        r'<div class="mermaid">\1</div>',
        html_body,
        flags=re.DOTALL
    )
    # Unescape HTML entities inside mermaid divs
    def clean_mermaid(match):
        code = match.group(1)
        code = code.replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&").replace("&quot;", '"')
        return f'<div class="mermaid">\n{code}\n</div>'
    html_body = re.sub(r'<div class="mermaid">(.*?)</div>', clean_mermaid, html_body, flags=re.DOTALL)
    
    # 5. Embed local images as base64 data URIs so Chrome can load them offline
    def repl_img(match):
        img_src = match.group(1)
        # Check if local file
        local_path = paper_dir / img_src
        if local_path.exists() and local_path.suffix.lower() == '.png':
            b64_data = base64.b64encode(local_path.read_bytes()).decode('utf-8')
            return f'src="data:image/png;base64,{b64_data}"'
        return match.group(0)
    
    html_body = re.sub(r'src="(figures/[^"]+)"', repl_img, html_body)
    
    # 6. Build Complete Standalone HTML with Academic Typography & MathJax & Mermaid
    full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Neuro-GRAPE: Group-Robust Adaptive Data Mixture Optimization for Multimodal Neural Representation Learning</title>
<!-- MathJax 3 with support for both $ and $$ -->
<script>
window.MathJax = {{
  tex: {{
    inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
    displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']],
    processEscapes: true
  }},
  options: {{
    skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code']
  }}
}};
</script>
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>

<!-- Mermaid.js -->
<script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
<script>
document.addEventListener('DOMContentLoaded', function() {{
  mermaid.initialize({{
    startOnLoad: true,
    theme: 'neutral',
    flowchart: {{ useMaxWidth: true, htmlLabels: true, curve: 'basis' }}
  }});
}});
</script>

<style>
@page {{
  size: letter;
  margin: 20mm 18mm 20mm 18mm;
  @bottom-right {{
    content: counter(page);
  }}
}}

body {{
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, "Liberation Sans", sans-serif;
  font-size: 10pt;
  line-height: 1.5;
  color: #1a1a1a;
  background-color: #ffffff;
  margin: 0;
  padding: 0;
}}

h1 {{
  font-size: 19pt;
  font-weight: 700;
  text-align: center;
  margin-top: 10pt;
  margin-bottom: 12pt;
  color: #0f172a;
  line-height: 1.25;
}}

.authors-block {{
  text-align: center;
  font-size: 10pt;
  color: #334155;
  margin-bottom: 24pt;
}}

h2 {{
  font-size: 13pt;
  font-weight: 700;
  color: #0f172a;
  border-bottom: 1.5px solid #0284c7;
  padding-bottom: 3pt;
  margin-top: 18pt;
  margin-bottom: 8pt;
}}

h3 {{
  font-size: 11pt;
  font-weight: 600;
  color: #1e293b;
  margin-top: 14pt;
  margin-bottom: 6pt;
}}

p {{
  margin-top: 0;
  margin-bottom: 8pt;
  text-align: justify;
}}

hr {{
  border: none;
  border-top: 1px solid #e2e8f0;
  margin: 16pt 0;
}}

table {{
  width: 100%;
  border-collapse: collapse;
  margin: 14pt 0;
  font-size: 8.5pt;
  page-break-inside: avoid;
}}

th, td {{
  padding: 6pt 8pt;
  text-align: left;
  border-top: 1px solid #cbd5e1;
  border-bottom: 1px solid #cbd5e1;
}}

th {{
  background-color: #f1f5f9;
  font-weight: 700;
  color: #0f172a;
  border-top: 2px solid #334155;
  border-bottom: 1.5px solid #334155;
}}

tr:hover {{
  background-color: #f8fafc;
}}

img {{
  max-width: 95%;
  height: auto;
  display: block;
  margin: 12pt auto;
  border-radius: 4px;
  box-shadow: 0 1px 3px rgba(0,0,0,0.12);
  page-break-inside: avoid;
}}

em {{
  font-style: italic;
}}

p > em {{
  display: block;
  text-align: center;
  font-size: 8.5pt;
  color: #475569;
  margin-top: -6pt;
  margin-bottom: 12pt;
}}

pre {{
  background-color: #f8fafc;
  border: 1px solid #e2e8f0;
  border-left: 3px solid #0284c7;
  padding: 8pt 10pt;
  font-family: "Cascadia Code", "Consolas", "Courier New", monospace;
  font-size: 8pt;
  overflow-x: auto;
  page-break-inside: avoid;
  margin: 10pt 0;
}}

code {{
  font-family: "Cascadia Code", "Consolas", "Courier New", monospace;
  font-size: 8.5pt;
  background-color: #f1f5f9;
  padding: 1pt 3pt;
  border-radius: 3px;
}}

pre code {{
  background-color: transparent;
  padding: 0;
}}

.mermaid {{
  margin: 14pt auto;
  text-align: center;
  page-break-inside: avoid;
  background-color: #ffffff;
  padding: 10pt;
  border: 1px solid #e2e8f0;
  border-radius: 6px;
}}

ul, ol {{
  margin-top: 0;
  margin-bottom: 8pt;
  padding-left: 20pt;
}}

li {{
  margin-bottom: 3pt;
}}

/* Clean page break management */
h2, h3 {{
  page-break-after: avoid;
}}

table, .mermaid, img, figure {{
  page-break-inside: avoid;
}}
</style>
</head>
<body>
{html_body}
</body>
</html>
"""
    
    print(f"Writing self-contained HTML to {html_file}...")
    html_file.write_text(full_html, encoding="utf-8")
    
    # 7. Convert HTML to PDF using Chrome Headless
    chrome_path = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
    if not os.path.exists(chrome_path):
        chrome_path = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
        
    print(f"Converting HTML to PDF with headless browser: {chrome_path}...")
    cmd = [
        chrome_path,
        "--headless=new",
        "--disable-gpu",
        "--allow-file-access-from-files",
        "--enable-local-file-accesses",
        "--run-all-compositor-stages-before-draw",
        "--virtual-time-budget=8000",  # Wait 8s for MathJax & Mermaid rendering
        f"--print-to-pdf={str(pdf_file)}",
        str(html_file)
    ]
    
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode == 0 and pdf_file.exists():
        size_kb = pdf_file.stat().st_size / 1024
        print(f"SUCCESS: Generated PDF at {pdf_file} ({size_kb:.1f} KB)")
        return True
    else:
        print(f"Error executing Chrome PDF generation: {res.stderr}")
        return False

if __name__ == "__main__":
    build_pdf()
