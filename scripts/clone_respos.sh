#!/bin/bash
set -eu
cd "$(dirname "$0")"
project_root=$(cd .. && pwd)
data_json_path=$project_root/data/vim_colorscheme_repo.json

jq -r ".data.search.edges[].node.url" "$data_json_path" \
  | sort -u \
  | cut -d / -f 4- \
  | xargs -P 10 -I {} git clone --depth 1 https://github.com/{} repos/{}
