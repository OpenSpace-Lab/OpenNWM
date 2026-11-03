#!/usr/bin/env python3
"""Render a local model/dataset card from the canonical repository README.

Usage and link mappings: tools/RELEASE_README.md. This script never publishes.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
from urllib.parse import urlsplit, urlunsplit


REPO_BLOB = "https://github.com/OpenSpace-Lab/OpenNWM/blob/main/"
REPO_RAW = "https://raw.githubusercontent.com/OpenSpace-Lab/OpenNWM/main/"
SITE_RAW = "https://raw.githubusercontent.com/OpenSpace-Lab/OpenNWM-site/main/"


def _frontmatter(text: str) -> tuple[str, str]:
    match = re.match(r"\A---[ \t]*\r?\n.*?^---[ \t]*(?:\r?\n|$)", text, re.M | re.S)
    return (match[0], text[match.end():]) if match else ("", text)


def _english_metadata(metadata: str) -> str:
    """Keep existing YAML formatting; remove Chinese from known language values."""
    lines = metadata.splitlines(keepends=True)
    for index, line in enumerate(lines):
        match = re.match(r"^language:[ \t]*(.*?)(\r?\n)?$", line)
        if not match:
            continue
        value = match[1].split("#", 1)[0].strip()
        newline = match[2] or ""
        if value:
            values = [item.strip().strip("\"'") for item in value.strip("[]").split(",")]
            if "zh" in values and set(values) <= {"en", "zh"}:
                comment = " #" + match[1].split("#", 1)[1] if "#" in match[1] else ""
                lines[index] = f"language: en{comment}{newline}"
            break
        entries = []
        cursor = index + 1
        while cursor < len(lines):
            item = re.match(r"^[ \t]*-[ \t]*([^#\r\n]+)", lines[cursor])
            if item:
                entries.append((cursor, item[1].strip().strip("\"'")))
            elif lines[cursor].strip() and not lines[cursor].lstrip().startswith("#"):
                break
            cursor += 1
        values = {value for _, value in entries}
        if "zh" in values and values <= {"en", "zh"}:
            if "en" not in values:
                first = entries[0][0]
                lines[first] = re.sub(r"(?<=-)([ \t]*)[^#\r\n]+", r"\1en", lines[first])
                entries = entries[1:]
            for position, value in entries:
                if value == "zh":
                    lines[position] = ""
        break
    return "".join(lines)


def _absolute_url(url: str, *, image: bool = False) -> str:
    if re.match(r"^(?:[A-Za-z][A-Za-z0-9+.-]*:|//|#|\?|$)", url):
        return url
    parts = urlsplit(url)
    path = parts.path.removeprefix("./").lstrip("/")
    if path.startswith("website/"):
        target = SITE_RAW + path.removeprefix("website/")
    else:
        target = (REPO_RAW if image else REPO_BLOB) + path
    return urlunsplit((*urlsplit(target)[:3], parts.query, parts.fragment))


def _srcset(value: str) -> str:
    # Data URLs contain commas; leave such srcsets intact.
    if "data:" in value:
        return value
    return re.sub(r"(^|,)(\s*)([^\s,]+)", lambda m: m[1] + m[2] + _absolute_url(m[3], image=True), value)


def _rewrite_links(text: str) -> str:
    def html_tag(match: re.Match[str]) -> str:
        def attribute(attr: re.Match[str]) -> str:
            kind = attr[1].lower()
            value = _srcset(attr[4]) if kind == "srcset" else _absolute_url(attr[4], image=kind == "src")
            return attr[1] + attr[2] + attr[3] + value + attr[3]

        return re.sub(r"(?<![\w:-])(href|src|srcset)(\s*=\s*)([\"'])(.*?)\3", attribute, match[0], flags=re.I | re.S)

    text = re.sub(r"<[A-Za-z][^>]*>", html_tag, text)

    def markdown_link(match: re.Match[str]) -> str:
        target = match[2]
        angle = target.startswith("<")
        value = _absolute_url(target[1:-1] if angle else target, image=match[1].startswith("!"))
        return match[1] + (f"<{value}>" if angle else value) + match[3]

    text = re.sub(r"(!?\[[^\]\n]*\]\(\s*)(<[^>\n]*>|[^\s)]+)([^)\n]*\))", markdown_link, text)
    return re.sub(
        r"(?m)^([ \t]{0,3}\[[^\]\n]+\]:[ \t]*)(<[^>\n]*>|[^\s]+)",
        lambda m: m[1] + ("<" + _absolute_url(m[2][1:-1]) + ">" if m[2].startswith("<") else _absolute_url(m[2])),
        text,
    )


def render(readme: str, metadata_from: str = "", appendix: str | None = None) -> str:
    """Return the canonical English body with card metadata and optional appendix.

    Inputs are Markdown strings. Existing card body text is intentionally replaced.
    The appendix is appended unchanged, so platform-relative resource links work.
    """
    metadata, _ = _frontmatter(metadata_from)
    _, body = _frontmatter(readme)
    # Keep command examples, BibTeX and inline code byte-for-byte unchanged.
    chunks = re.split(r"(```[\s\S]*?```|~~~[\s\S]*?~~~|`+[^`]*?`+)", body)
    body = "".join(chunk if index % 2 else _rewrite_links(chunk) for index, chunk in enumerate(chunks))
    # Resource appendices may retain Chinese license or attribution text.
    if not re.search(r"[\u3400-\u4dbf\u4e00-\u9fff]", body + (appendix or "")):
        metadata = _english_metadata(metadata)
    output = (metadata + "\n" if metadata else "") + body.strip() + "\n"
    if appendix and appendix.strip():
        output += "\n" + appendix.strip() + "\n"
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--readme", type=Path, required=True)
    parser.add_argument("--metadata-from", type=Path, required=True)
    parser.add_argument("--appendix", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = render(
        args.readme.read_text(encoding="utf-8"),
        args.metadata_from.read_text(encoding="utf-8"),
        args.appendix.read_text(encoding="utf-8") if args.appendix else None,
    )
    args.output.write_text(result, encoding="utf-8")


if __name__ == "__main__":
    main()
