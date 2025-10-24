from __future__ import annotations

import argparse
import curses
from dataclasses import dataclass
from enum import Enum, auto
import os
import re
import subprocess
from tempfile import NamedTemporaryFile
from types import TracebackType
from typing import Generator, Literal, NoReturn, Optional, Self

# display 3 windows:
#     top left = new A hunk with diff from original
#     top right = new B hunk with diff from original
#     bottom = merge result
# All 3 windows have titles containing both their filenames and their labels
#
# Key bindings
# (n) next conflict
# (p) previous conflict
# (a) accept version A
# (b) accept version B
# (A) accept combination, A first
# (B) accept combination, B first
# (i) ignore all changes, use original version
# (u) un-resolve
# (t) toggle change diffs
# (e) open the current conflict in an editor
# (E) open the entire merged file in an editor
# (d) diff latest merge results with original
# (w) or (s) save
# (q) quit, prompt to save
# (Q) quit, discard changes
# (?) show keybindings
# (Tab) or (click) move focus between A, B, and merge result
# (arrows) or (h/j/k/l) or (BUTTON4/BUTTON5) scroll focused window
# (-/+) or (drag border) resize top/bottom split
# ([/]) or (drag border) resize left/right split
# (something) to toggle A/B split horiz/vert
# (something) Save, quit, and open editor
# Additional navigation with Home/End/PgUp/PgDn(space)

# Wishlist: Menus

def clamp[T: float](minimum: T, value: T, maximum: T) -> T:
    return max(minimum, min(value, maximum))


class Pane:
    def __init__(
        self,
        filename: str,
        height: int,
        width: int,
        rowmin: int, colmin: int,
        rowmax: int, colmax: int,
        label: Optional[str] = None
    ):
        self.filenmae = filename
        self.rowmin = rowmin; self.rowmax = rowmax
        self.colmin = colmin; self.colmax = colmax
        self.hscroll = 0
        self.vscroll = 0
        self.height = height
        self.width = width
        self.pad = curses.newpad(self.height, self.width)

    #FIXME: scrolling past the right or bottom edge causes repeating lines/chars
    def scroll_vert(self, n: int) -> None:
        self.vscroll = clamp(0, self.vscroll + n, self.height - 1)
        self.noutrefresh()

    def scroll_horiz(self, n: int) -> None:
        self.hscroll = clamp(0, self.hscroll + n, self.width - 1)
        self.noutrefresh()

    def noutrefresh(self) -> None:
        self.pad.noutrefresh(
            self.vscroll, self.hscroll,
            self.rowmin, self.colmin,
            self.rowmax, self.colmax
        )

    def addstr(self, row: int, col: int, s: str) -> None:
        try:
            self.pad.addstr(row, col, s)
        except curses.error:
            pass  # can happen spuriously when writing to bottom right corner


class ChangePane(Pane):
    def __init__(
        self,
        filename: str,
        orig: list[str],
        new: list[str],
        rowmin: int,
        colmin: int,
        rowmax: int,
        colmax: int,
        label: str | None = None
    ):
        with (
            NamedTemporaryFile('w+', delete_on_close=False) as tmporig,
            NamedTemporaryFile('w+', delete_on_close=False) as tmpnew,
        ):
            tmporig.writelines(f'{line}\n' for line in orig)
            tmporig.close()

            tmpnew.writelines(f'{line}\n' for line in new)
            tmpnew.close()

            diff_result = subprocess.run(
                f'diff --text --unified={len(orig) + len(new)}'.split(' ')
                + [tmporig.name, tmpnew.name],
                stdout=subprocess.PIPE,
                encoding='utf-8'
            )
        contents = diff_result.stdout.splitlines()[3:]  # skip headers
        height = len(contents)
        width = max(map(len, contents))
        super().__init__(
            filename, height, width, rowmin, colmin, rowmax, colmax, label)
        for i, line in enumerate(contents):
            self.addstr(i, 0, line)


class OutputPane(Pane):
    def __init__(
        self,
        filename: str,
        contents: list[Decision | list[str]],
        rowmin: int,
        colmin: int,
        rowmax: int,
        colmax: int,
        label: str | None = None
    ):
        height = sum(
            len(e.conflict.base) if isinstance(e, Decision) else len(e)
            for e in contents
        )
        # FIXME: Should be different depending on decision resolution
        lines_lists = (
            e.conflict.base if isinstance(e, Decision) else e
            for e in contents
        )
        width = 2 + max(
            (len(line) for lines in lines_lists for line in lines),
            default=0
        )
        super().__init__(
            filename, height, width, rowmin, colmin, rowmax, colmax, label)
        i = 0
        for e in contents:
            if isinstance(e, Decision):
                #TODO: Show something different depending on resolution
                for line in e.conflict.base:
                    self.pad.addch(i, 0, '!')
                    self.addstr(i, 2, line)
                    i += 1
            else:
                for line in e:
                    self.pad.addch(i, 0, ' ')
                    self.addstr(i, 2, line)
                    i += 1


class Resolution(Enum):
    UNRESOLVED = auto()
    USE_A = auto()
    USE_B = auto()
    USE_A_FIRST = auto()
    USE_B_FIRST = auto()
    USE_BASE = auto()


@dataclass
class Decision:
    conflict: Conflict
    resolution: Resolution = Resolution.UNRESOLVED


def terminal_supports_xterm_mouse():
    xm = curses.tigetstr('XM')
    if xm and b'1006' in xm:
        return True
    terminal_type = os.environ.get('TERM', 'unknown')
    if 'xterm' in terminal_type:
        return True
    return False


def term_enable_mouse_drag(enable: bool = True):
    if not terminal_supports_xterm_mouse():
        return
    c = 'h' if enable else 'l'
    print(f'\033[?1002{c}', flush=True)


class DLMerge:
    def __init__(
        self,
        file_a: str,
        file_b: str,
        file_base: str,
        merge: list[list[str] | Conflict]
    ):
        self._filenames = [file_a, file_b, file_base]
        self._merge = merge
        self._result = [
            Decision(e) if isinstance(e, Conflict) else e
            for e in merge
        ]
        self._vsplit = .5
        self._hsplit = .5
        self._dragging: bool | Literal['hsplit'] | Literal['vsplit'] = False
        self._focused = 2
        self._hscroll = [0, 0, 0]
        self._vscroll = [0, 0, 0]

    def __enter__(self) -> Self:
        self._stdscr = curses.initscr()
        curses.cbreak()
        curses.noecho()
        curses.curs_set(0)
        self._stdscr.keypad(True)
        curses.mousemask(curses.ALL_MOUSE_EVENTS | curses.REPORT_MOUSE_POSITION)
        curses.mouseinterval(0)
        term_enable_mouse_drag()

        return self

    def __exit__(
        self,
        type: type[BaseException] | None,
        value: BaseException | None,
        traceback: TracebackType | None
    ) -> None:
        term_enable_mouse_drag(False)
        curses.endwin()

    @property
    def _vsplit_col(self) -> int:
        MIN_SIZE = 1
        return clamp(
            MIN_SIZE,
            int(curses.COLS * self._vsplit),
            curses.COLS - 1 - MIN_SIZE
        )

    @_vsplit_col.setter
    def _vsplit_col(self, value: int) -> None:
        self._vsplit = value / curses.COLS

    @property
    def _hsplit_row(self) -> int:
        MIN_SIZE = 1
        return clamp(
            MIN_SIZE,
            int(curses.LINES * self._hsplit),
            curses.LINES - 1 - MIN_SIZE
        )

    @_hsplit_row.setter
    def _hsplit_row(self, value: int) -> None:
        self._hsplit = value / curses.LINES

    def _draw_hsplit(self, ch: Optional[int | str] = None) -> None:
        ch = normalize_ch(ch, curses.ACS_HLINE)
        self._stdscr.hline(self._hsplit_row, 0, ch, curses.COLS)

    def _draw_vsplit(self, ch: Optional[int | str] = None) -> None:
        ch = normalize_ch(ch, curses.ACS_VLINE)
        self._stdscr.vline(0, self._vsplit_col, ch, self._hsplit_row)

    def _draw_tee(self, ch: Optional[int | str] = None) -> None:
        ch = normalize_ch(ch, curses.ACS_BTEE)
        self._stdscr.addch(self._hsplit_row, self._vsplit_col, ch)

    def _draw_borders(self) -> None:
        self._draw_hsplit()
        self._draw_vsplit()
        self._draw_tee()
        self._stdscr.noutrefresh()

    def _move_vsplit(self, col: int) -> None:
        if col == self._vsplit_col:
            return
        self._draw_vsplit(' ')
        self._vsplit_col = col
        self._panes[0].colmax = self._vsplit_col - 1
        self._panes[1].colmin = self._vsplit_col + 1
        self._stdscr.touchline(0, self._hsplit_row)
        self._draw_borders()
        self._panes[0].noutrefresh()
        self._panes[1].noutrefresh()

    def _move_hsplit(self, row: int) -> None:
        if row == self._hsplit_row:
            return
        self._draw_hsplit(' ')
        self._draw_vsplit(' ')
        self._hsplit_row = row
        self._panes[0].rowmax = self._hsplit_row - 1
        self._panes[1].rowmax = self._hsplit_row - 1
        self._panes[2].rowmin = self._hsplit_row + 1
        self._stdscr.touchwin()
        self._draw_borders()
        for pane in self._panes:
            pane.noutrefresh()

    def _draw_contents(self) -> None:
        pane_a, pane_b, pane_m = self._panes
        pane_a.noutrefresh()
        pane_b.noutrefresh()
        pane_m.noutrefresh()

    def run(self) -> int:
        conflict_index = 0
        current_decision: Optional[Decision] = None
        i = 0
        for e in self._result:
            if isinstance(e, Decision):
                if i == conflict_index:
                    current_decision = e
                    break
                else:
                    i += 1
        if not current_decision:
            raise ValueError('No decisions to make')
        self._panes: list[Pane] = [
            ChangePane(
                self._filenames[0],
                current_decision.conflict.base, current_decision.conflict.a,
                0, 0, self._hsplit_row - 1, self._vsplit_col - 1
            ),
            ChangePane(
                self._filenames[1],
                current_decision.conflict.base, current_decision.conflict.b,
                0, self._vsplit_col + 1, self._hsplit_row - 1, curses.COLS - 1
            ),
            OutputPane(
                self._filenames[2],
                self._result,
                self._hsplit_row + 1, 0, curses.LINES - 1, curses.COLS - 1
            ),
        ]

        self._draw_borders()
        self._draw_contents()
        while True:
            curses.doupdate()
            c = self._stdscr.getch()

            if c == ord('q'):
                #TODO: This is a temporary debugging interface
                return c
            elif c == ord('\t'):
                self._focused = (self._focused + 1) % 3
            elif c == curses.KEY_BTAB:
                self._focused = (self._focused - 1) % 3
            elif c == curses.KEY_UP:
                self._panes[self._focused].scroll_vert(-1)
            elif c == curses.KEY_DOWN:
                self._panes[self._focused].scroll_vert(1)
            elif c == curses.KEY_LEFT:
                self._panes[self._focused].scroll_horiz(-2)
            elif c == curses.KEY_RIGHT:
                self._panes[self._focused].scroll_horiz(2)
            elif c == curses.KEY_RESIZE:
                curses.update_lines_cols()
                self._stdscr.resize(curses.LINES, curses.COLS)
                self._stdscr.erase()
                self._stdscr.touchwin()
                self._draw_borders()
                self._panes[0].colmax = self._vsplit_col - 1
                self._panes[1].colmin = self._vsplit_col + 1
                self._panes[1].colmax = curses.COLS - 1
                self._panes[2].colmax = curses.COLS - 1
                self._panes[0].rowmax = self._hsplit_row - 1
                self._panes[1].rowmax = self._hsplit_row - 1
                self._panes[2].rowmin = self._hsplit_row + 1
                self._panes[2].rowmax = curses.LINES - 1
                self._draw_contents()
            elif c == curses.KEY_MOUSE:
                _, mcol, mrow, _, bstate = curses.getmouse()
                if bstate & curses.BUTTON1_PRESSED:
                    if mrow == self._hsplit_row:
                        self._dragging = 'hsplit'
                    elif mrow < self._hsplit_row and mcol == self._vsplit_col:
                        self._dragging = 'vsplit'

                if self._dragging == 'hsplit':
                    self._move_hsplit(mrow)
                if self._dragging == 'vsplit':
                    self._move_vsplit(mcol)
                self._stdscr.noutrefresh()

                if bstate & curses.BUTTON1_RELEASED:
                    self._dragging = False


def normalize_ch(ch: int | str | None, default: int) -> int:
    if ch is None:
        return default
    if isinstance(ch, str):
        return ord(ch)
    return ch


@dataclass
class Conflict:
    base_label: str
    base: list[str]
    a_label: str
    a: list[str]
    b_label: str
    b: list[str]


type TokenType = Literal['a', 'base', 'b', 'end', 'text', 'eof']


@dataclass
class Token:
    lineno: int
    type: TokenType
    data: str | None = None


def tokenize_merge(lines: list[str]) -> Generator[Token]:
    A_RE    = re.compile(         r'<<<<<<< (.*)')
    BASE_RE = re.compile(re.escape('|||||||') + r' (.*)')
    B_RE    = re.compile(         r'=======')
    END_RE  = re.compile(         r'>>>>>>> (.*)')

    for i, line in enumerate(lines):
        lineno = i + 1
        if m := A_RE.fullmatch(line):
            yield Token(lineno, 'a', m.group(1))
        elif m := BASE_RE.fullmatch(line):
            yield Token(lineno, 'base', m.group(1))
        elif m := B_RE.fullmatch(line):
            yield Token(lineno, 'b')
        elif m := END_RE.fullmatch(line):
            yield Token(lineno, 'end', m.group(1))
        else:
            yield Token(lineno, 'text', line)
    yield Token(len(lines), 'eof')


class MergeParser:
    def __init__(self, lines: list[str]):
        self._tokens = tokenize_merge(lines)
        self._pushed_back: Optional[Token] = None

    def _next_token(self) -> Token:
        if self._pushed_back:
            try:
                return self._pushed_back
            finally:
                self._pushed_back = None
        return next(self._tokens)

    def _pushback_token(self, tok: Token) -> None:
        if self._pushed_back:
            raise OverflowError('attempted second pushback')
        self._pushed_back = tok

    def parse(self) -> list[list[str] | Conflict]:
        return list(self._parse())

    def _parse(self) -> Generator[list[str] | Conflict]:
        try:
            while (tok := self._next_token()).type != 'eof':
                match tok.type:
                    case 'text':
                        self._pushback_token(tok)
                        yield self._parse_text()
                    case 'a':
                        self._pushback_token(tok)
                        yield self._parse_conflict()
                    case _:
                        self._syntax_error(tok)
        except StopIteration:
            # Shouldn't happen
            raise IndexError('Unexpected end of token stream')

    def _syntax_error(self, tok: Token) -> NoReturn:
        TOKEN_TYPE_NAMES: dict[TokenType, str] = {
            'a': 'file A header',
            'base': 'base file header',
            'b': 'file B header',
            'end': 'conflict footer',
            'text': 'line of text',
            'eof': 'end-of-file',
        }

        lineno = tok.lineno
        type_name = TOKEN_TYPE_NAMES[tok.type]
        data = tok.data

        raise SyntaxError(f'line {lineno}: Unexpected {type_name}: data={data!r}')

    def _parse_text(self) -> list[str]:
        body: list[str] = []
        while (tok := self._next_token()).type == 'text':
            if tok.data is None:
                self._syntax_error(tok)
            body.append(tok.data)
        self._pushback_token(tok)
        return body

    def _parse_conflict(self) -> Conflict:
        a_label, a = self._parse_a()
        base_label, base = self._parse_base()
        b_label, b = self._parse_b()
        return Conflict(
            a_label=a_label,       a=a,
            b_label=b_label,       b=b,
            base_label=base_label, base=base,
        )

    def _parse_a(self) -> tuple[str, list[str]]:
        return self._parse_header_and_body('a')

    def _parse_base(self) -> tuple[str, list[str]]:
        return self._parse_header_and_body('base')

    def _parse_header_and_body(self, type: TokenType) -> tuple[str, list[str]]:
        header = self._next_token()
        if header.type != type or header.data is None:
            self._syntax_error(header)
        label = header.data
        body = self._parse_text()
        return label, body

    def _parse_b(self) -> tuple[str, list[str]]:
        header = self._next_token()
        if header.type != 'b':
            self._syntax_error(header)
        body = self._parse_text()
        end = self._next_token()
        if end.type != 'end' or end.data is None:
            self._syntax_error(end)
        label = end.data
        return label, body


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('file1')
    parser.add_argument('file2')
    parser.add_argument('file3')
    args = parser.parse_args()

    diff3_result = subprocess.run(
        'git merge-file -p --zdiff3 --'.split(' ')
        + [args.file1, args.file2, args.file3],
        stdout=subprocess.PIPE,
        encoding='utf-8',
    )

    merge = MergeParser(diff3_result.stdout.splitlines()).parse()
    with DLMerge(args.file1, args.file3, args.file2, merge) as dlmerge:
        c = dlmerge.run()

    print(f'{curses.keyname(c)!r} (0x{c:02x})')
    exit()


if __name__ == '__main__':
    main()
