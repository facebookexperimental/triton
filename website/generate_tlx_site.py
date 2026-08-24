#!/usr/bin/env python3
"""Build the FBTriton GitHub Pages site from README.md and curated guides."""

from __future__ import annotations

import html
import re
from dataclasses import dataclass
from pathlib import Path

from guide_content import GUIDE_CONTENT

SITE_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = SITE_ROOT.parent
REPOSITORY_URL = "https://github.com/facebookexperimental/triton"
STYLESHEET_VERSION = "20260821b"


@dataclass(frozen=True)
class Page:
    slug: str
    title: str
    start: str | None
    end: str | None
    summary: str
    section: str = "tlx"


README_PAGES = (
    Page(
        "tlx",
        "TLX",
        None,
        "## The DSL Extension",
        "What TLX is and when to use it.",
    ),
    Page(
        "buffers",
        "Local and remote buffers",
        "## The DSL Extension",
        "### Async memory access",
        "Allocate, view, load, store, and share hardware-near buffers.",
    ),
    Page(
        "async-memory",
        "Async memory access",
        "### Async memory access",
        "### Async tensor core operations",
        "Move data asynchronously with descriptors, TMA, and copy groups.",
    ),
    Page(
        "async-compute",
        "Async tensor core operations",
        "### Async tensor core operations",
        "### Barrier operations",
        "Issue and coordinate asynchronous tensor-core work.",
    ),
    Page(
        "synchronization",
        "Barriers and Cluster Launch Control",
        "### Barrier operations",
        "### Warp Specialization operations",
        "Synchronize tasks and distribute persistent work across CTAs.",
    ),
    Page(
        "warp-specialization",
        "Warp specialization and clustering",
        "### Warp Specialization operations",
        "### Other operations",
        "Assign warps to concurrent tasks and configure CTA clusters.",
    ),
    Page(
        "utilities",
        "Other operations",
        "### Other operations",
        "## Kernels Implemented with TLX",
        "Thread, type, timing, and stochastic-rounding utilities.",
    ),
    Page(
        "kernels",
        "Example kernels",
        "## Kernels Implemented with TLX",
        "## Build and install TLX from source",
        "GEMM and attention kernels implemented with TLX.",
    ),
    Page(
        "install-and-test",
        "Build, install, and test",
        "## Build and install TLX from source",
        "## More reading materials",
        "Build TLX and run its correctness and performance scripts.",
    ),
    Page(
        "resources",
        "More reading",
        "## More reading materials",
        None,
        "Additional TLX documentation and conference material.",
    ),
)

HOME_PAGE = Page(
    "home",
    "Overview",
    None,
    None,
    "Explore Triton, TLX, and the tooling used to build and optimize GPU kernels.",
    "home",
)
GETTING_STARTED_PAGE = Page(
    "getting-started",
    "Getting started",
    None,
    None,
    "Start with TLX imports, tutorials, and a minimal warp-specialized kernel.",
)
HARDWARE_SUPPORT_PAGE = Page(
    "hardware-support",
    "Hardware support",
    None,
    None,
    "Understand TLX capabilities across Hopper, Blackwell, and AMD CDNA GPUs.",
)
PERFORMANCE_PAGE = Page(
    "performance-optimization",
    "Performance optimization",
    None,
    None,
    "Structure pipelines, buffers, fusion, and scheduling for high utilization.",
)
DEBUGGING_PAGE = Page(
    "debugging",
    "Debugging performance and numerics",
    None,
    None,
    "Diagnose compiler, runtime, performance, and numerical issues systematically.",
)
CASE_STUDIES_PAGE = Page(
    "production-case-studies",
    "Production case studies",
    None,
    None,
    "See how TLX has been applied to large-scale training and inference workloads.",
)
TRITON_PAGE = Page(
    "triton",
    "Triton",
    None,
    None,
    "Compiler-managed performance portability and automatic warp specialization.",
    "triton",
)
TOOLING_PAGE = Page(
    "tooling",
    "Tooling",
    None,
    None,
    "Tools for tracing, profiling, validating, and benchmarking Triton kernels.",
    "tooling",
)

TLX_PAGES = (
    README_PAGES[0],
    GETTING_STARTED_PAGE,
    HARDWARE_SUPPORT_PAGE,
    *README_PAGES[1:7],
    PERFORMANCE_PAGE,
    DEBUGGING_PAGE,
    README_PAGES[7],
    CASE_STUDIES_PAGE,
    *README_PAGES[8:],
)
SECTION_PAGES = (TRITON_PAGE, README_PAGES[0], TOOLING_PAGE)
SECTION_LABELS = {
    "triton": "Triton",
    "tlx": "TLX",
    "tooling": "Tooling",
}
TRITON_NAV_ITEMS = (
    ("Overview", "triton.html"),
    ("Automatic warp specialization", "#automatic-warp-specialization"),
    ("Compiler pipeline", "#the-compiler-pipeline-today"),
    ("TLX and AutoWS", "#tlx-and-autows"),
    ("Roadmap", "#roadmap"),
    ("Design article", "#read-the-design-article"),
)
PAGES = (HOME_PAGE, TRITON_PAGE, *TLX_PAGES, TOOLING_PAGE)


def read_readme() -> str:
    return (REPOSITORY_ROOT / "README.md").read_text(encoding="utf-8")


def split_readme(source: str) -> list[tuple[Page, str]]:
    chunks: list[tuple[Page, str]] = []
    for page in README_PAGES:
        start = source.index(page.start) if page.start else 0
        end = source.index(page.end) if page.end else len(source)
        chunks.append((page, source[start:end]))

    if "".join(chunk for _, chunk in chunks) != source:
        raise RuntimeError("Page boundaries do not reproduce README.md exactly")
    return chunks


def collect_page_sources(source: str) -> list[tuple[Page, str]]:
    sources = {page.slug: chunk for page, chunk in split_readme(source)}
    sources["tlx"] = sources["tlx"].replace(
        "Primarily targeting NVIDIA GPUs (for now), TLX extends Triton to support:",
        "TLX targets NVIDIA and AMD GPUs and supports:",
        1,
    )
    overview_marker = "## Nightly builds (fbtriton)"
    if overview_marker not in sources["tlx"]:
        raise RuntimeError(f"Missing overview insertion point: {overview_marker}")
    overview_addition = GUIDE_CONTENT["tlx-overview"].strip()
    sources["tlx"] = sources["tlx"].replace(
        overview_marker,
        f"{overview_addition}\n\n{overview_marker}",
        1,
    )

    for slug, content in GUIDE_CONTENT.items():
        if slug != "tlx-overview":
            sources[slug] = content.strip() + "\n"

    return [(page, sources[page.slug]) for page in PAGES]


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
    if lines and lines[0].startswith("#"):
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


def documentation_navigation(active_slug: str, from_root: bool) -> str:
    items = []
    for page in TLX_PAGES:
        href = page_href(page, from_root)
        current = ' aria-current="page"' if page.slug == active_slug else ""
        label = "Overview" if page.slug == "tlx" else page.title
        items.append(f'<a href="{href}"{current}>{html.escape(label)}</a>')
    return "\n".join(items)


def triton_navigation() -> str:
    items = []
    for index, (label, href) in enumerate(TRITON_NAV_ITEMS):
        current = ' aria-current="page"' if index == 0 else ""
        items.append(f'<a href="{href}"{current}>{html.escape(label)}</a>')
    return "\n".join(items)


def page_links(page: Page, from_root: bool) -> str:
    if page.section == "home":
        return ""
    sequence = TLX_PAGES if page.section == "tlx" else SECTION_PAGES
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
        "tooling": "Developer tooling",
    }[page.section]
    pager = page_links(page, from_root)
    footer = f'<footer class="pager">{pager}</footer>' if pager else ""
    sidebar = ""
    shell_class = "shell no-sidebar"
    if page.section == "tlx":
        shell_class = "shell"
        sidebar = f"""<aside class="sidebar">
      <p class="eyebrow">TLX documentation</p>
      <nav>{documentation_navigation(page.slug, from_root)}</nav>
    </aside>"""
    elif page.section == "triton":
        shell_class = "shell"
        sidebar = f"""<aside class="sidebar">
      <p class="eyebrow">Triton documentation</p>
      <nav>{triton_navigation()}</nav>
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
    <a class="brand" href="{home}"><span>FB</span>Triton</a>
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
    chunks = collect_page_sources(read_readme())

    for page, source in chunks:
        output = (REPOSITORY_ROOT / "index.html" if page.slug == "home" else SITE_ROOT / f"{page.slug}.html")
        output.write_text(render_page(page, source), encoding="utf-8")

    print(f"Generated {len(chunks)} pages from README.md and guide content")


if __name__ == "__main__":
    main()
