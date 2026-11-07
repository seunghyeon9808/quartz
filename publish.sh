#!/bin/zsh
set -e

export PATH="$HOME/.local/bin:$HOME/.local/node/bin:/opt/homebrew/bin:/usr/local/bin:$PATH"

cd "/Users/seunghyeonjeong/Documents/quartz"

echo "1. 옵시디언 공개 노트 동기화 중 (publish: true / dg-publish: true)..."
python3 sync_obsidian.py

if [[ "$1" == "--serve" || "$1" == "-s" ]]; then
  echo "2. 로컬 미리보기 서버 실행 중 (http://localhost:8080)..."
  npx quartz build --serve
else
  echo "2. GitHub로 변경사항 업로드(Sync) 중..."
  npx quartz sync
fi
