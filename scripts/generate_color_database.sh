#!/bin/bash
set -eu
cd "$(dirname "$0")"

echo -n > result.jsonl
find repos/ -mindepth 2 -maxdepth 2 | while read -r repo; do
  for colorscheme in "$repo"/colors/*; do
    [[ -e $colorscheme ]] || continue
    colorscheme_name=$(basename "$colorscheme")
    echo "$repo" "$colorscheme_name"
    vim -u NONE -N -n -i NONE -e -s --cmd "source ./generate_color_database.vim" -- \
      "$repo" \
      "${repo#repos/}" \
      "${colorscheme_name%.vim}" >> result.jsonl || continue
  done
done
