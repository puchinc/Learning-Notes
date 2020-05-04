" multi cursors
" https://medium.com/@schtoeffel/you-don-t-need-more-than-one-cursor-in-vim-2c44117d51db
gn
cgn
.

" increase number: <C-a>

" Visual select and search
1. Select the text in visual mode (v and move cursor to highlight)
2. Yank the text (y)
3. Go into search mode (/)
4. Select registers (CTRL+r)
5. Select the unnamed register, i.e. the last yank or delete (")

"check file exist
if !empty(glob('$(pwd)/.exrc'))
endif

" folding
zd " remove fold
zD " recursive remove fold

" nerdtree cheat shee
" https://www.cheatography.com/stepk/cheat-sheets/vim-nerdtree/
" C: change tree root to selected dir
" u: move tree root up a dir
" U: move tree root up a dir but leave old root open
" r: refresh cursor dir
" R: refresh current root
" cd: change the CWD to the selected dir
" t: open in new tab
" T: open in new tab silently


