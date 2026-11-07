#!/usr/bin/env python3
"""
Obsidian Vault -> Quartz content 선택 동기화 스크립트
- Frontmatter에 `publish: true` 또는 `dg-publish: true`가 있는 마크다운 파일만 추출합니다.
- 공개된 노트에서 참조(`![[이미지]]` 등)하는 첨부파일만 자동으로 찾아 복사합니다.
- 비공개 노트(예: 토큰, 개인 일기 등)는 Git 저장소에 절대 포함되지 않도록 보호합니다.
"""

import os
import re
import shutil
from pathlib import Path

VAULT_DIR = Path("/Users/seunghyeonjeong/Documents/Obsidian")
QUARTZ_CONTENT_DIR = Path("/Users/seunghyeonjeong/Documents/quartz/content")

EXCLUDE_DIRS = {".obsidian", ".git", ".trash", "node_modules"}

FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*(?:\n|$)", re.DOTALL)
PUBLISH_TRUE_RE = re.compile(
    r"^(?:publish|dg-publish)\s*:\s*[\"']?true[\"']?\s*$",
    re.MULTILINE | re.IGNORECASE,
)
HOME_TRUE_RE = re.compile(
    r"^(?:dg-home|home)\s*:\s*[\"']?true[\"']?\s*$",
    re.MULTILINE | re.IGNORECASE,
)
WIKILINK_EMBED_RE = re.compile(r"!\[\[([^\]|#]+)(?:[|#][^\]]*)?\]\]")
MD_IMAGE_RE = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")


def get_frontmatter(text: str) -> str:
    match = FRONTMATTER_RE.match(text)
    return match.group(1) if match else ""


def is_published(text: str) -> bool:
    fm = get_frontmatter(text)
    if not fm:
        return False
    return bool(PUBLISH_TRUE_RE.search(fm))


def is_home(text: str) -> bool:
    fm = get_frontmatter(text)
    if not fm:
        return False
    return bool(HOME_TRUE_RE.search(fm))


def build_asset_index(vault_dir: Path) -> dict[str, Path]:
    asset_index: dict[str, Path] = {}
    for root, dirs, files in os.walk(vault_dir):
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
        for f in files:
            if not f.endswith(".md"):
                full_path = Path(root) / f
                asset_index[f] = full_path
                rel_path = full_path.relative_to(vault_dir).as_posix()
                asset_index[rel_path] = full_path
    return asset_index


def extract_referenced_assets(text: str) -> set[str]:
    refs = set()
    for match in WIKILINK_EMBED_RE.finditer(text):
        target = match.group(1).strip()
        if not target.endswith(".md") and "." in Path(target).name:
            refs.add(target)
    for match in MD_IMAGE_RE.finditer(text):
        target = match.group(1).strip()
        if not target.startswith(("http://", "https://", "data:")):
            refs.add(target)
    return refs


def main():
    if not VAULT_DIR.exists():
        print(f"[오류] 옵시디언 폴더를 찾을 수 없습니다: {VAULT_DIR}")
        return

    # 기존 content 폴더 초기화
    if QUARTZ_CONTENT_DIR.exists():
        if QUARTZ_CONTENT_DIR.is_symlink():
            QUARTZ_CONTENT_DIR.unlink()
        else:
            shutil.rmtree(QUARTZ_CONTENT_DIR)
    QUARTZ_CONTENT_DIR.mkdir(parents=True, exist_ok=True)

    asset_index = build_asset_index(VAULT_DIR)
    published_files: list[tuple[Path, str]] = []
    referenced_assets: set[str] = set()
    home_source: Path | None = None

    for root, dirs, files in os.walk(VAULT_DIR):
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
        for f in files:
            if not f.endswith(".md"):
                continue
            md_path = Path(root) / f
            try:
                content = md_path.read_text(encoding="utf-8")
            except Exception:
                continue

            if is_published(content):
                published_files.append((md_path, content))
                referenced_assets.update(extract_referenced_assets(content))
                if md_path.name.lower() == "index.md" and md_path.parent == VAULT_DIR:
                    home_source = md_path
                elif is_home(content) and home_source is None:
                    home_source = md_path

    # 공개 노트 복사
    for md_path, content in published_files:
        rel_path = md_path.relative_to(VAULT_DIR)
        dest_path = QUARTZ_CONTENT_DIR / rel_path
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(md_path, dest_path)
        print(f"[노트 복사] {rel_path}")

    # 참조된 이미지/첨부파일 복사
    for ref in referenced_assets:
        src_asset = asset_index.get(ref) or asset_index.get(Path(ref).name)
        if src_asset and src_asset.exists():
            rel_path = src_asset.relative_to(VAULT_DIR)
            dest_path = QUARTZ_CONTENT_DIR / rel_path
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_asset, dest_path)
            print(f"[첨부 복사] {rel_path}")

    # 루트 index.md 보장 (Quartz 홈페이지 필수 파일)
    root_index = QUARTZ_CONTENT_DIR / "index.md"
    if not root_index.exists():
        links = []
        for md_path, _ in sorted(published_files, key=lambda x: x[0].stem):
            if home_source and md_path == home_source:
                continue
            stem = md_path.stem
            links.append(f"- [[{stem}]]")
        links_md = "\n".join(links) if links else "- 아직 공개된 노트가 없습니다."

        if home_source and home_source.exists():
            home_text = home_source.read_text(encoding="utf-8")
            fm_match = FRONTMATTER_RE.match(home_text)
            body = home_text[fm_match.end():].strip() if fm_match else home_text.strip()
            if body:
                # title 속성이 없으면 원래 노트 파일명(예: Home)을 title로 자동 삽입하여 'index'로 표시되지 않게 함
                if fm_match:
                    fm_content = fm_match.group(1)
                    if not re.search(r"^title\s*:", fm_content, re.MULTILINE | re.IGNORECASE):
                        new_fm = f"---\ntitle: {home_source.stem}\n{fm_content}\n---\n"
                        home_text = new_fm + home_text[fm_match.end():]
                else:
                    home_text = f"---\ntitle: {home_source.stem}\npublish: true\n---\n\n" + home_text
                root_index.write_text(home_text, encoding="utf-8")
            else:
                root_index.write_text(
                    f"""---
title: {home_source.stem}
publish: true
---

## 공개된 노트 목록
{links_md}
""",
                    encoding="utf-8",
                )
            print(f"[홈페이지 설정] '{home_source.relative_to(VAULT_DIR)}' -> index.md")
        else:
            root_index.write_text(
                f"""---
title: Home
publish: true
---

## 공개된 노트 목록
{links_md}
""",
                encoding="utf-8",
            )
            print("[홈페이지 자동 생성] content/index.md")

    print(f"\n완료! 총 {len(published_files)}개의 공개 노트가 Quartz로 동기화되었습니다.")


if __name__ == "__main__":
    main()
