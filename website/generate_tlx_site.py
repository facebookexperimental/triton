#!/usr/bin/env python3
"""Build the TLX GitHub Pages site from README.md on the main branch."""

from __future__ import annotations

import html
import re
from dataclasses import dataclass
from pathlib import Path

SITE_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = SITE_ROOT.parent
REPOSITORY_URL = "https://github.com/facebookexperimental/triton"


@dataclass(frozen=True)
class Page:
    slug: str
    title: str
    start: str | None
    end: str | None
    summary: str


PAGES = (
    Page(
        "overview",
        "Overview",
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


def read_readme() -> str:
    return (REPOSITORY_ROOT / "README.md").read_text(encoding="utf-8")


def split_readme(source: str) -> list[tuple[Page, str]]:
    chunks: list[tuple[Page, str]] = []
    for page in PAGES:
        start = source.index(page.start) if page.start else 0
        end = source.index(page.end) if page.end else len(source)
        chunks.append((page, source[start:end]))

    if "".join(chunk for _, chunk in chunks) != source:
        raise RuntimeError("Page boundaries do not reproduce README.md exactly")
    return chunks


def slugify(value: str) -> str:
    value = re.sub(r"<[^>]+>", "", value).lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    return value.strip("-")


def repository_link(target: str) -> str:
    target = target.strip()
    if target.startswith(("http://", "https://", "#", "mailto:")):
        return target
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
    if page.slug == "overview":
        return "./" if from_root else "../"
    prefix = "website/" if from_root else ""
    return f"{prefix}{page.slug}.html"


def navigation(active_slug: str, from_root: bool) -> str:
    items = []
    for page in PAGES:
        href = page_href(page, from_root)
        current = ' aria-current="page"' if page.slug == active_slug else ""
        items.append(f'<a href="{href}"{current}>{html.escape(page.title)}</a>')
    return "\n".join(items)


def page_links(page: Page, from_root: bool) -> str:
    position = PAGES.index(page)
    previous = PAGES[position - 1] if position else None
    following = PAGES[position + 1] if position + 1 < len(PAGES) else None
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
    from_root = page.slug == "overview"
    stylesheet = "website/assets/tlx.css" if from_root else "assets/tlx.css"
    home = "./" if from_root else "../"
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="description" content="{html.escape(page.summary, quote=True)}">
  <title>{html.escape(page.title)} · TLX</title>
  <link rel="stylesheet" href="{stylesheet}">
</head>
<body>
  <header class="topbar">
    <a class="brand" href="{home}"><span>TLX</span> Triton Low-level Language Extensions</a>
    <a class="repo-link" href="{REPOSITORY_URL}" target="_blank" rel="noreferrer">View on GitHub ↗</a>
  </header>
  <div class="shell">
    <aside class="sidebar">
      <p class="eyebrow">Documentation</p>
      <nav>{navigation(page.slug, from_root)}</nav>
    </aside>
    <main>
      <p class="eyebrow">TLX documentation</p>
      <h1>{html.escape(page.title)}</h1>
      <p class="lede">{html.escape(page.summary)}</p>
      <article>{body}</article>
      <footer class="pager">{page_links(page, from_root)}</footer>
    </main>
  </div>
</body>
</html>
"""


def main() -> None:
    chunks = split_readme(read_readme())

    for page, source in chunks:
        output = (REPOSITORY_ROOT / "index.html" if page.slug == "overview" else SITE_ROOT / f"{page.slug}.html")
        output.write_text(render_page(page, source), encoding="utf-8")

    print(f"Generated {len(chunks)} pages from main:README.md")


if __name__ == "__main__":
    main()
