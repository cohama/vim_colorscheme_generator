#!/bin/bash
set -eu
cd "$(dirname "$0")"
project_root=$(cd .. && pwd)
data_json_path=$project_root/data/vim_colorscheme_repo.json

tmp_json_path=$(mktemp -t vim_colorscheme_generator_XXX.json)
echo -n > "$data_json_path"
for querystring in 'vim colorscheme' 'neovim colorscheme' 'vim color scheme' 'neovim color scheme' 'vim theme' 'neovim theme' 'vim topic:colorscheme' 'neovim topic:colorscheme'; do
  for star in '0' '1..9' '10..99' '100..*'; do
  query=$(cat <<EOS
query(\$endCursor: String) {
  search(type: REPOSITORY, query: "${querystring} stars:${star}", first: 100, after: \$endCursor) {
    repositoryCount
    edges {
      node {
        ... on Repository{
          url
        }
      }
    }
    pageInfo {
      hasNextPage
      endCursor
    }
  }
}
EOS
)
    gh api graphql --paginate -f "query=${query}" >> "$dtmp_json_path"
  done
done

jq -r ".data.search.edges[].node.url" result.json \
#   | sort -u \
#   | cut -d / -f 4- \
#   | xargs -P 10 -I {} git clone --depth 1 https://github.com/{} repos/{}
