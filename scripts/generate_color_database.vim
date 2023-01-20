function LoadColorscheme(repo_path, colorscheme_name)
  let &rtp = &rtp .. ',' .. a:repo_path
  execute 'colorscheme ' .. a:colorscheme_name
endfunction

function! GetColorData(hi_group_name) abort
  let result = ''
  try
    redir => result
      silent execute 'hi ' .. a:hi_group_name
    redir END
    let guifg = matchstr(result, '\Vguifg=\zs#\(\x\+\)')
    let guibg = matchstr(result, '\Vguibg=\zs#\(\x\+\)')
  catch /.*E411.*/
    let guifg = ''
    let guibg = ''
  endtry
  return [guifg, guibg]
endfunction

function! GetColorschemeData() abort
  let result = {}
  let hi_groups = [
  \ 'Normal',
  \ 'NonText',
  \ 'StatusLine',
  \ 'LineNr',
  \ 'CursorLineNr',
  \ 'Comment',
  \ 'Constant',
  \ 'String',
  \ 'Character',
  \ 'Number',
  \ 'Boolean',
  \ 'Float',
  \ 'Identifier',
  \ 'Function',
  \ 'Statement',
  \ 'Conditional',
  \ 'Repeat',
  \ 'Label',
  \ 'Operator',
  \ 'Keyword',
  \ 'Exception',
  \ 'PreProc',
  \ 'Include',
  \ 'Define',
  \ 'Macro',
  \ 'PreCondit',
  \ 'Type',
  \ 'StorageClass',
  \ 'Structure',
  \ 'Typedef',
  \ 'Special',
  \ 'SpecialChar',
  \ 'Tag',
  \ 'Delimiter',
  \ 'SpecialComment'
  \ ]
  for hi_group in hi_groups
    let [guifg, guibg] = GetColorData(hi_group)
    let result[hi_group] = #{guifg: guifg, guibg: guibg}
  endfor
  return result
endfunction

function! Main() abort
  set termguicolors
  try
    let args = argv()
    if len(args) != 3
      verbose echo 'Error! arg1: repo_path, arg2: repo_name, arg3: colorscheme_name'
      cquit 1
    endif
    let repo_path = args[0]
    let repo_name = args[1]
    let colorscheme_name = args[2]
    call LoadColorscheme(repo_path, colorscheme_name)
    let colorscheme_data = #{repo_name: repo_name, colorscheme_name: colorscheme_name, colorscheme: GetColorschemeData()}
    new | only!
    set nonu nornu nolist
    0put=json_encode(colorscheme_data)
    $ delete _
    %print
    qall!
  catch /.*/
    verbose echo v:exception
    cquit
  endtry
endif
endfunction

augroup x
  autocmd!
  autocmd VimEnter * call Main()
augroup END
