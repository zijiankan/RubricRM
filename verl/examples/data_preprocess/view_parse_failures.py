#!/usr/bin/env python3
"""Generate a standalone HTML viewer for V2 parse failure logs.

Usage:
    python view_parse_failures.py <failure_dir> [-o output.html]

Example:
    python view_parse_failures.py logs/reward/v2_20260424_183738/v2_parse_failures
"""

import argparse
import glob
import html
import json
import os
import sys


def load_records(failure_dir: str) -> list[dict]:
    """Load all JSONL records from the failure directory."""
    records = []
    for path in sorted(glob.glob(os.path.join(failure_dir, "*.jsonl"))):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


def build_html(records: list[dict], failure_dir: str) -> str:
    """Build a standalone HTML page."""
    # Compute stats
    reason_counts: dict[str, int] = {}
    for r in records:
        reason = r.get("reason", "unknown")
        reason_counts[reason] = reason_counts.get(reason, 0) + 1

    records_json = json.dumps(records, ensure_ascii=False)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>V2 Parse Failures Viewer</title>
<style>
  :root {{
    --bg: #0d1117; --surface: #161b22; --border: #30363d;
    --text: #e6edf3; --text2: #8b949e; --accent: #58a6ff;
    --red: #f85149; --green: #3fb950; --yellow: #d29922;
    --orange: #db6d28;
  }}
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
    background: var(--bg); color: var(--text); line-height: 1.6;
  }}
  .container {{ max-width: 1200px; margin: 0 auto; padding: 20px; }}
  h1 {{ color: var(--accent); margin-bottom: 8px; font-size: 24px; }}
  .meta {{ color: var(--text2); margin-bottom: 20px; font-size: 14px; }}

  /* Stats bar */
  .stats {{
    display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 20px;
  }}
  .stat-card {{
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 8px; padding: 12px 18px; min-width: 120px;
  }}
  .stat-card .num {{ font-size: 28px; font-weight: 700; color: var(--accent); }}
  .stat-card .label {{ font-size: 12px; color: var(--text2); }}

  /* Filters */
  .filters {{
    display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 16px;
    align-items: center;
  }}
  .filters label {{ color: var(--text2); font-size: 13px; margin-right: 4px; }}
  .filter-btn {{
    padding: 4px 12px; border-radius: 16px; border: 1px solid var(--border);
    background: var(--surface); color: var(--text2); cursor: pointer;
    font-size: 13px; transition: all .15s;
  }}
  .filter-btn:hover {{ border-color: var(--accent); color: var(--text); }}
  .filter-btn.active {{ background: var(--accent); color: #000; border-color: var(--accent); }}
  .search-box {{
    padding: 6px 12px; border-radius: 6px; border: 1px solid var(--border);
    background: var(--surface); color: var(--text); font-size: 13px;
    width: 260px; margin-left: auto;
  }}
  .search-box::placeholder {{ color: var(--text2); }}

  /* Cards */
  .card {{
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 8px; margin-bottom: 12px; overflow: hidden;
    transition: border-color .15s;
  }}
  .card:hover {{ border-color: var(--accent); }}
  .card-header {{
    padding: 12px 16px; cursor: pointer; display: flex;
    align-items: center; gap: 12px; user-select: none;
  }}
  .card-header:hover {{ background: rgba(88,166,255,.05); }}
  .badge {{
    padding: 2px 10px; border-radius: 12px; font-size: 11px;
    font-weight: 600; white-space: nowrap;
  }}
  .badge-red {{ background: rgba(248,81,73,.15); color: var(--red); }}
  .badge-orange {{ background: rgba(219,109,40,.15); color: var(--orange); }}
  .badge-yellow {{ background: rgba(210,153,34,.15); color: var(--yellow); }}
  .card-index {{ color: var(--text2); font-size: 13px; font-family: monospace; }}
  .card-reason {{ color: var(--text); font-size: 14px; flex: 1; }}
  .card-gt {{ font-size: 13px; font-weight: 600; }}
  .card-gt.A {{ color: var(--green); }}
  .card-gt.B {{ color: var(--yellow); }}
  .arrow {{ color: var(--text2); transition: transform .2s; font-size: 12px; }}
  .card.open .arrow {{ transform: rotate(90deg); }}

  /* Card body */
  .card-body {{ display: none; padding: 0 16px 16px; }}
  .card.open .card-body {{ display: block; }}
  .section-title {{
    font-size: 12px; font-weight: 600; color: var(--accent);
    text-transform: uppercase; letter-spacing: .5px;
    margin: 12px 0 6px; padding-bottom: 4px;
    border-bottom: 1px solid var(--border);
  }}
  .text-block {{
    background: var(--bg); border: 1px solid var(--border);
    border-radius: 6px; padding: 12px; font-size: 13px;
    line-height: 1.7; white-space: pre-wrap; word-break: break-word;
    max-height: 400px; overflow-y: auto; font-family: 'SF Mono', Monaco,
    'Cascadia Code', 'Roboto Mono', Consolas, monospace;
  }}
  .text-block.short {{ max-height: 120px; }}
  .score-grid {{
    display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
    gap: 8px; margin-top: 6px;
  }}
  .score-item {{
    background: var(--bg); border: 1px solid var(--border);
    border-radius: 6px; padding: 8px 12px; font-size: 13px;
  }}
  .score-item .dim-name {{ color: var(--accent); font-weight: 600; margin-bottom: 2px; }}
  .score-item .dim-detail {{ color: var(--text2); }}

  /* Highlight markers in output */
  .hl-score {{ color: var(--green); font-weight: 700; }}
  .hl-weight {{ color: var(--yellow); font-weight: 700; }}
  .hl-section {{ color: var(--accent); font-weight: 700; }}

  /* Pagination */
  .pagination {{
    display: flex; justify-content: center; gap: 8px;
    margin-top: 20px; align-items: center;
  }}
  .page-btn {{
    padding: 6px 14px; border-radius: 6px; border: 1px solid var(--border);
    background: var(--surface); color: var(--text); cursor: pointer; font-size: 13px;
  }}
  .page-btn:hover {{ border-color: var(--accent); }}
  .page-btn:disabled {{ opacity: .4; cursor: default; }}
  .page-info {{ color: var(--text2); font-size: 13px; }}
</style>
</head>
<body>
<div class="container">
  <h1>V2 Parse Failures Viewer</h1>
  <div class="meta">Source: {html.escape(failure_dir)} &middot; {len(records)} records</div>

  <div class="stats" id="stats"></div>
  <div class="filters">
    <label>Filter:</label>
    <div id="filter-btns"></div>
    <input class="search-box" id="search" placeholder="Search output / prompt text..." />
  </div>
  <div id="cards"></div>
  <div class="pagination" id="pagination"></div>
</div>

<script>
const DATA = {records_json};
const PAGE_SIZE = 20;
let currentFilter = "all";
let currentSearch = "";
let currentPage = 1;

// ── Helpers ──
function esc(s) {{
  const d = document.createElement("div");
  d.textContent = s || "";
  return d.innerHTML;
}}
function highlightOutput(text) {{
  let s = esc(text);
  s = s.replace(/(\\*{{0,2}}(?:Score|score)\\*{{0,2}}\\s*[:：]\\s*\\d+(?:\\.\\d+)?\\s*\\/\\s*4)/g,
    '<span class="hl-score">$1</span>');
  s = s.replace(/(\\(\\d+(?:\\.\\d+)?%\\))/g, '<span class="hl-weight">$1</span>');
  s = s.replace(/(Detailed\\s+Evaluation|Final\\s+Conclusion)/gi,
    '<span class="hl-section">$1</span>');
  return s;
}}
function badgeClass(reason) {{
  if (reason.includes("parsed=0")) return "badge-red";
  if (reason.includes("fallback")) return "badge-red";
  return "badge-orange";
}}
function parseScoreRaw(raw) {{
  if (!raw) return null;
  try {{ return typeof raw === "string" ? JSON.parse(raw) : raw; }}
  catch {{ return null; }}
}}

// ── Render ──
function getFiltered() {{
  return DATA.filter(r => {{
    if (currentFilter !== "all" && r.reason !== currentFilter) return false;
    if (currentSearch) {{
      const q = currentSearch.toLowerCase();
      const hay = ((r.output||"") + (r.prompt||"") + (r.reason||"")).toLowerCase();
      if (!hay.includes(q)) return false;
    }}
    return true;
  }});
}}

function renderStats() {{
  const counts = {{}};
  DATA.forEach(r => {{ counts[r.reason] = (counts[r.reason]||0) + 1; }});
  const el = document.getElementById("stats");
  el.innerHTML = `<div class="stat-card"><div class="num">${{DATA.length}}</div><div class="label">Total Failures</div></div>`
    + Object.entries(counts).sort((a,b)=>b[1]-a[1]).map(([k,v]) =>
      `<div class="stat-card"><div class="num">${{v}}</div><div class="label">${{esc(k)}}</div></div>`
    ).join("");

  const fb = document.getElementById("filter-btns");
  fb.innerHTML = `<button class="filter-btn ${{currentFilter==="all"?"active":""}}" data-f="all">All (${{DATA.length}})</button>`
    + Object.entries(counts).sort((a,b)=>b[1]-a[1]).map(([k,v]) =>
      `<button class="filter-btn ${{currentFilter===k?"active":""}}" data-f="${{esc(k)}}">${{esc(k.replace(/dim_mismatch\\(/, "").replace(")", ""))}} (${{v}})</button>`
    ).join("");
  fb.querySelectorAll(".filter-btn").forEach(btn => {{
    btn.onclick = () => {{ currentFilter = btn.dataset.f; currentPage = 1; render(); }};
  }});
}}

function renderCards() {{
  const filtered = getFiltered();
  const totalPages = Math.max(1, Math.ceil(filtered.length / PAGE_SIZE));
  if (currentPage > totalPages) currentPage = totalPages;
  const start = (currentPage - 1) * PAGE_SIZE;
  const page = filtered.slice(start, start + PAGE_SIZE);

  const el = document.getElementById("cards");
  el.innerHTML = page.map((r, idx) => {{
    const scoreInfo = parseScoreRaw(r.score_raw);
    const dims = scoreInfo?.dimensions || [];
    return `<div class="card" id="card-${{start+idx}}">
      <div class="card-header" onclick="this.parentElement.classList.toggle('open')">
        <span class="arrow">▶</span>
        <span class="card-index">#${{r.index ?? "?"}}</span>
        <span class="badge ${{badgeClass(r.reason)}}">${{esc(r.reason)}}</span>
        <span class="card-reason"></span>
        <span class="card-gt ${{esc(r.ground_truth||"")}}">${{esc(r.ground_truth||"")}}</span>
      </div>
      <div class="card-body">
        ${{dims.length ? `<div class="section-title">GT Dimensions</div>
        <div class="score-grid">${{dims.map(d =>
          `<div class="score-item">
            <div class="dim-name">${{esc(d.name)}} (${{(d.weight*100).toFixed(0)}}%)</div>
            <div class="dim-detail">A: ${{d.score_a}} &nbsp; B: ${{d.score_b}}</div>
          </div>`
        ).join("")}}</div>` : ""}}
        <div class="section-title">Model Output</div>
        <div class="text-block">${{highlightOutput(r.output||"")}}</div>
        <div class="section-title">Prompt (compact)</div>
        <div class="text-block short">${{esc(r.prompt||"")}}</div>
      </div>
    </div>`;
  }}).join("");

  // Pagination
  const pg = document.getElementById("pagination");
  pg.innerHTML = `
    <button class="page-btn" ${{currentPage<=1?"disabled":""}} onclick="currentPage--;render()">← Prev</button>
    <span class="page-info">${{start+1}}-${{Math.min(start+PAGE_SIZE, filtered.length)}} of ${{filtered.length}}</span>
    <button class="page-btn" ${{currentPage>=totalPages?"disabled":""}} onclick="currentPage++;render()">Next →</button>
  `;
}}

function render() {{ renderStats(); renderCards(); }}

document.getElementById("search").addEventListener("input", e => {{
  currentSearch = e.target.value; currentPage = 1; render();
}});

render();
</script>
</body>
</html>"""


def main():
    parser = argparse.ArgumentParser(description="Generate V2 parse failure viewer HTML")
    parser.add_argument("failure_dir", help="Path to v2_parse_failures directory")
    parser.add_argument("-o", "--output", default=None, help="Output HTML path (default: <dir>/viewer.html)")
    args = parser.parse_args()

    if not os.path.isdir(args.failure_dir):
        print(f"Error: {args.failure_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    records = load_records(args.failure_dir)
    if not records:
        print("No records found.", file=sys.stderr)
        sys.exit(1)

    output_path = args.output or os.path.join(args.failure_dir, "viewer.html")
    html_content = build_html(records, os.path.abspath(args.failure_dir))

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"Generated {output_path} ({len(records)} records)")


if __name__ == "__main__":
    main()
