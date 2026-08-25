#!/usr/bin/env python3
"""Build the FBTriton GitHub Pages site from website/content and curated guides."""

from __future__ import annotations

import html
import re
from dataclasses import dataclass
from pathlib import Path

from guide_content import GUIDE_CONTENT

SITE_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = SITE_ROOT.parent
REPOSITORY_URL = "https://github.com/facebookexperimental/triton"
STYLESHEET_VERSION = "20260824c"


@dataclass(frozen=True)
class Page:
    slug: str
    title: str
    summary: str
    section: str = "tlx"


CONTENT_ROOT = SITE_ROOT / "content"

TLX_PAGES = (
    Page("tlx", "TLX", "What TLX is, and the hardware tags used throughout."),
    Page(
        "buffers",
        "Memory",
        "Allocate, view, slice, and reuse local and remote buffers.",
    ),
    Page(
        "global-memory",
        "Global memory access",
        "Address global memory directly with scalar-base buffer operations.",
    ),
    Page(
        "async-memory",
        "Async memory access",
        "Move data asynchronously with descriptors, TMA, TDM, and copy groups.",
    ),
    Page(
        "async-compute",
        "Tensor core operations",
        "Issue and schedule synchronous and asynchronous tensor-core work.",
    ),
    Page(
        "synchronization",
        "Synchronization",
        "Barriers, scheduling barriers, and memory fences.",
    ),
    Page(
        "warp-specialization",
        "Warp specialization",
        "Assign warps to concurrent tasks and build explicit pipelines.",
    ),
    Page(
        "clusters",
        "Clusters and Cluster Launch Control",
        "Cooperate across CTAs and distribute persistent work in hardware.",
    ),
    Page(
        "layouts",
        "Layout control and diagnostics",
        "Pin, release, and verify register and shared-memory layouts.",
    ),
    Page(
        "utilities",
        "Other operations",
        "Thread, type, timing, and stochastic-rounding utilities.",
    ),
    Page(
        "kernels",
        "Kernels implemented with TLX",
        "GEMM and attention kernels implemented with TLX.",
    ),
    Page(
        "testing",
        "Testing",
        "Correctness and performance scripts for the TLX tutorial kernels.",
    ),
    Page(
        "resources",
        "Further reading",
        "Additional TLX documentation and conference material.",
    ),
)

HOME_PAGE = Page(
    "home",
    "Overview",
    "Explore Triton, TLX, TorchTLX, and the tooling used to build and optimize GPU kernels.",
    "home",
)
TRITON_PAGE = Page(
    "triton",
    "Triton",
    "Compiler-managed performance portability and automatic warp specialization.",
    "triton",
)
COMPILER_PAGE = Page(
    "compiler",
    "Compiler features",
    "AutoWS enablement and knobs, and deterministic reduction ordering.",
    "triton",
)
TORCHTLX_PAGE = Page(
    "torchtlx",
    "TorchTLX",
    "Bring TLX kernels into PyTorch 2 through Inductor templates and epilogue fusion.",
    "torchtlx",
)
CI_PAGE = Page(
    "ci",
    "CI",
    "Workflows, runners, nightly failure handling, and per-project test coverage.",
    "ci",
)
TOOLING_PAGE = Page(
    "tooling",
    "Tooling",
    "Tools for tracing, profiling, validating, and benchmarking Triton kernels.",
    "tooling",
)

TRITON_PAGES = (TRITON_PAGE, COMPILER_PAGE)

SECTIONS = {
    "triton": TRITON_PAGES,
    "tlx": TLX_PAGES,
    "torchtlx": (TORCHTLX_PAGE,),
    "ci": (CI_PAGE,),
    "tooling": (TOOLING_PAGE,),
}
SECTION_PAGES = tuple(pages[0] for pages in SECTIONS.values())
SECTION_LABELS = {
    "triton": "Triton",
    "tlx": "TLX",
    "torchtlx": "TorchTLX",
    "ci": "CI",
    "tooling": "Tooling",
}
PAGES = (HOME_PAGE, *TRITON_PAGES, *TLX_PAGES, TORCHTLX_PAGE, CI_PAGE, TOOLING_PAGE)

def page_source(page: Page) -> str:
    """Read a page body from website/content/, falling back to guide_content."""
    path = CONTENT_ROOT / f"{page.slug}.md"
    if path.exists():
        return path.read_text(encoding="utf-8")
    return GUIDE_CONTENT[page.slug].strip() + "\n"


def collect_page_sources() -> list[tuple[Page, str]]:
    return [(page, page_source(page)) for page in PAGES]


def slugify(value: str) -> str:
    value = re.sub(r"<[^>]+>", "", value).lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    return value.strip("-")


def repository_link(target: str) -> str:
    target = target.strip()
    if target.startswith(("http://", "https://", "#", "mailto:", "./", "../")):
        return target
    if target.endswith(".html"):
        return target
    if target.endswith("/"):
        return f"{REPOSITORY_URL}/tree/main/{target.lstrip('./')}"
    return f"{REPOSITORY_URL}/blob/main/{target.lstrip('./')}"


def render_inline(value: str) -> str:
    code_spans: list[str] = []

    def stash_code(match: re.Match[str]) -> str:
        code_spans.append(f"<code>{html.escape(match.group(1))}</code>")
        return f"\x00CODE{len(code_spans) - 1}\x00"

    value = re.sub(r"`([^`]+)`", stash_code, value)
    value = html.escape(value, quote=False)

    def render_link(match: re.Match[str]) -> str:
        label = match.group(1)
        target = repository_link(html.unescape(match.group(2)))
        external = target.startswith(("http://", "https://"))
        attrs = ' target="_blank" rel="noreferrer"' if external else ""
        return f'<a href="{html.escape(target, quote=True)}"{attrs}>{label}</a>'

    value = re.sub(r"\[([^]]+)]\(([^)]+)\)", render_link, value)
    value = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", value)
    for index, code in enumerate(code_spans):
        value = value.replace(f"\x00CODE{index}\x00", code)
    return value


def is_table_separator(line: str) -> bool:
    cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
    return bool(cells) and all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells)


def table_row(line: str, tag: str) -> str:
    cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
    return ("<tr>" + "".join(f"<{tag}>{render_inline(cell)}</{tag}>" for cell in cells) + "</tr>")


def render_markdown(source: str) -> str:
    lines = source.splitlines()
    output: list[str] = []
    paragraph: list[str] = []
    in_code = False
    code_language = ""
    code_lines: list[str] = []
    index = 0

    def flush_paragraph() -> None:
        if paragraph:
            output.append(f"<p>{render_inline(' '.join(part.strip() for part in paragraph))}</p>")
            paragraph.clear()

    while index < len(lines):
        line = lines[index]
        fence = re.match(r"^\s*```\s*([^ ]*)\s*$", line)
        if fence:
            flush_paragraph()
            if in_code:
                language = (f' class="language-{html.escape(code_language)}"' if code_language else "")
                output.append(f"<pre><code{language}>{html.escape(chr(10).join(code_lines))}</code></pre>")
                code_lines.clear()
                in_code = False
            else:
                in_code = True
                code_language = fence.group(1)
            index += 1
            continue

        if in_code:
            code_lines.append(line)
            index += 1
            continue

        if line.startswith(">"):
            flush_paragraph()
            quote_lines = []
            while index < len(lines) and lines[index].startswith(">"):
                quote_lines.append(lines[index][1:].strip())
                index += 1
            output.append(f"<blockquote><p>{render_inline(' '.join(quote_lines))}</p></blockquote>")
            continue

        heading = re.match(r"^(#{1,6})\s+(.+)$", line)
        if heading:
            flush_paragraph()
            level = len(heading.group(1))
            title = heading.group(2).strip()
            output.append(f'<h{level} id="{slugify(title)}">{render_inline(title)}</h{level}>')
            index += 1
            continue

        if (index + 1 < len(lines) and "|" in line and is_table_separator(lines[index + 1])):
            flush_paragraph()
            rows = ["<table><thead>", table_row(line, "th"), "</thead><tbody>"]
            index += 2
            while index < len(lines) and "|" in lines[index] and lines[index].strip():
                rows.append(table_row(lines[index], "td"))
                index += 1
            rows.append("</tbody></table>")
            output.append("".join(rows))
            continue

        item = re.match(r"^(\s*)[-*]\s+(.+)$", line)
        if item:
            flush_paragraph()
            depth = min(len(item.group(1)) // 2, 3)
            output.append(f'<div class="list-item depth-{depth}"><span aria-hidden="true">•</span>'
                          f"<div>{render_inline(item.group(2))}</div></div>")
            index += 1
            continue

        if not line.strip():
            flush_paragraph()
        else:
            paragraph.append(line)
        index += 1

    flush_paragraph()
    if in_code:
        raise RuntimeError("Unclosed Markdown code fence")
    return "\n".join(output)


def content_without_page_heading(source: str) -> str:
    lines = source.splitlines()
    if lines and lines[0].startswith("# "):
        lines = lines[1:]
    while lines and not lines[0].strip():
        lines.pop(0)
    return "\n".join(lines)


def page_href(page: Page, from_root: bool) -> str:
    if page.slug == "home":
        return "./" if from_root else "../"
    prefix = "website/" if from_root else ""
    return f"{prefix}{page.slug}.html"


def top_navigation(page: Page, from_root: bool) -> str:
    items = []
    for section_page in SECTION_PAGES:
        href = page_href(section_page, from_root)
        current = ' data-current="true"' if section_page.section == page.section else ""
        label = SECTION_LABELS[section_page.slug]
        items.append(f'<a href="{href}"{current}>{html.escape(label)}</a>')
    return "\n".join(items)


def sub_headings(source: str) -> list[str]:
    """Level-2 headings of a page body, ignoring fenced code."""
    out, in_code = [], False
    for line in source.splitlines():
        if re.match(r"^\s*```", line):
            in_code = not in_code
            continue
        if in_code:
            continue
        heading = re.match(r"^## (.+)$", line)
        if heading:
            out.append(heading.group(1).strip())
    return out


def section_navigation(page: Page, source: str, from_root: bool) -> str:
    pages = SECTIONS[page.section]
    items = []
    for entry in pages:
        href = page_href(entry, from_root)
        current = ' aria-current="page"' if entry.slug == page.slug else ""
        label = "Overview" if len(pages) > 1 and entry is pages[0] else entry.title
        items.append(f'<a href="{href}"{current}>{html.escape(label)}</a>')
        if entry.slug == page.slug:
            for heading in sub_headings(source):
                items.append(
                    f'<a class="sub" href="#{slugify(heading)}">{html.escape(heading)}</a>')
    return "\n".join(items)


def page_links(page: Page, from_root: bool) -> str:
    if page.section == "home":
        return ""
    # Page within its own section when the section has several; otherwise step
    # across the top-level sections.
    sequence = SECTIONS[page.section]
    if len(sequence) == 1:
        sequence = SECTION_PAGES
    position = sequence.index(page)
    previous = sequence[position - 1] if position else None
    following = sequence[position + 1] if position + 1 < len(sequence) else None
    links = []
    if previous:
        href = page_href(previous, from_root)
        links.append(f'<a href="{href}">← {html.escape(previous.title)}</a>')
    if following:
        href = page_href(following, from_root)
        links.append(f'<a class="next" href="{href}">{html.escape(following.title)} →</a>')
    return "".join(links)


def render_page(page: Page, source: str) -> str:
    body = render_markdown(content_without_page_heading(source))
    from_root = page.slug == "home"
    stylesheet_path = "website/assets/tlx.css" if from_root else "assets/tlx.css"
    stylesheet = f"{stylesheet_path}?v={STYLESHEET_VERSION}"
    home = "./" if from_root else "../"
    eyebrow = {
        "home": "Triton at Meta",
        "triton": "Triton compiler",
        "tlx": "TLX documentation",
        "torchtlx": "TorchTLX",
        "ci": "Continuous integration",
        "tooling": "Developer tooling",
    }[page.section]
    pager = page_links(page, from_root)
    footer = f'<footer class="pager">{pager}</footer>' if pager else ""
    sidebar = ""
    shell_class = "shell no-sidebar"
    if page.section in SECTIONS:
        shell_class = "shell"
        sidebar = f"""<aside class="sidebar">
      <p class="eyebrow">{html.escape(eyebrow)}</p>
      <nav>{section_navigation(page, source, from_root)}</nav>
    </aside>"""
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="description" content="{html.escape(page.summary, quote=True)}">
  <title>{html.escape(page.title)} · FBTriton</title>
  <link rel="stylesheet" href="{stylesheet}">
</head>
<body>
  <header class="topbar">
    <a class="brand" href="{home}"><span>FBTriton</span></a>
    <nav class="topnav" aria-label="Top-level sections">{top_navigation(page, from_root)}</nav>
    <a class="repo-link" href="{REPOSITORY_URL}" target="_blank" rel="noreferrer">View on GitHub ↗</a>
  </header>
  <div class="{shell_class}">
{sidebar}
    <main>
      <p class="eyebrow">{eyebrow}</p>
      <h1>{html.escape(page.title)}</h1>
      <p class="lede">{html.escape(page.summary)}</p>
      <article>{body}</article>
{footer}
    </main>
  </div>
</body>
</html>
"""


def main() -> None:
    chunks = collect_page_sources()

    for page, source in chunks:
        output = (REPOSITORY_ROOT / "index.html" if page.slug == "home" else SITE_ROOT / f"{page.slug}.html")
        output.write_text(render_page(page, source), encoding="utf-8")

    print(f"Generated {len(chunks)} pages from website/content and guide content")


if __name__ == "__main__":
    main()
