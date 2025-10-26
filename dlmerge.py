from __future__ import annotations

import argparse
import curses
import curses.panel as panel
from dataclasses import dataclass
from enum import Enum, IntEnum, auto
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
    MIN_HEIGHT = 2
    MIN_WIDTH = 2

    def __init__(
        self,
        filename: str,
        color: ColorPair,
        nlines: int,
        ncols: int,
        begin_line: int,
        begin_col: int,
        label: Optional[str] = None
    ):
        self.filename = filename
        self._color = color
        self._set_size_attrs(nlines, ncols, begin_line, begin_col)
        self._hscroll = 0
        self._vscroll = 0
        self._focused = False

        self._win, title, gutter, content = self._create_wins()
        self._title_panel = panel.new_panel(title)
        self._gutter_panel = panel.new_panel(gutter)
        self._content_panel = panel.new_panel(content)
        self._gutter_pad = curses.newpad(1, 1)
        self._content_pad = curses.newpad(1, 1)

    def _set_size_attrs(self, nlines: int, ncols: int, begin_line: int, begin_col: int) -> None:
        self._nlines = nlines
        self._ncols = ncols
        self._begin_line = begin_line
        self._begin_col = begin_col

    def _create_wins(self) -> tuple[curses.window, curses.window, curses.window, curses.window]:
        win = curses.newwin(self._nlines, self._ncols, self._begin_line, self._begin_col)
        title = win.derwin(1, self._ncols, 0, 0)
        gutter = win.derwin(self._nlines - 1, 1, 1, 0)
        content = win.derwin(self._nlines - 1, self._ncols - 1, 1, 1)
        return win, title, gutter, content

    @property
    def preferred_height(self) -> int:
        return self._content_height + 1  # for the title bar

    #FIXME: scrolling past the right or bottom edge causes repeating lines/chars
    def scroll_vert(self, n: int) -> None:
        self.scroll_vert_to(self._vscroll + n)

    def scroll_vert_to(self, n: int) -> None:
        self._vscroll = clamp(0, n, self._content_height - 1)
        self._draw()

    def scroll_horiz(self, n: int) -> None:
        self.scroll_horiz_to(self._hscroll + n)

    def scroll_horiz_to(self, n: int) -> None:
        self._hscroll = clamp(0, n, self.width - 1)
        self._draw()

    def _draw_title(self) -> None:
        titlewin = self._title_panel.window()
        titlewin.bkgdset(' ', self._color.attr | curses.A_REVERSE)
        try:
            # TODO: More information in the title
            titlewin.addch(0, 0, '[' if self._focused else ' ')
            titlewin.addstr(0, 1, self.filename)
            titlewin.addch(']' if self._focused else ' ')
            titlewin.clrtobot()
        except curses.error:
            pass

    def focus(self, on: bool) -> None:
        self._focused = on
        self._draw_title()

    @property
    def width(self) -> int:
        _, cols = self._content_pad.getmaxyx()
        return cols

    @property
    def _content_height(self) -> int:
        lines, _ = self._content_pad.getmaxyx()
        return lines

    def _resize_content(self, nlines: int, ncols: int) -> None:
        self._content_pad.resize(nlines, ncols)

    def resize(self, nlines: int, ncols: int, begin_line: int, begin_col: int) -> None:
        self._set_size_attrs(nlines, ncols, begin_line, begin_col)

        self._win, title, gutter, content = self._create_wins()
        self._title_panel.replace(title)
        self._content_panel.replace(content)
        self._gutter_panel.replace(gutter)

        self._draw()

    def _draw(self) -> None:
        self._draw_title()
        pad_to_win(self._gutter_pad, self._gutter_panel.window(), self._vscroll, 0)
        pad_to_win(self._content_pad, self._content_panel.window(), self._vscroll, self._hscroll)


def pad_to_win(pad: curses.window, win: curses.window, sminline: int, smincol: int) -> None:
    winlines, wincols = win.getmaxyx()
    padlines, padcols = pad.getmaxyx()
    copylines = clamp(0, winlines, padlines - sminline)
    copycols = clamp(0, wincols, padcols - smincol)
    win.erase()
    if copylines and copycols:
        pad.overwrite(win, sminline, smincol, 0, 0, copylines - 1, copycols - 1)


def addstr(window: curses.window, row: int, col: int, s: str, attr: Optional[int] = None) -> None:
        try:
            if attr is None:
                window.addstr(row, col, s)
            else:
                window.addstr(row, col, s, attr)
        except curses.error:
            pass  # can happen spuriously when writing to bottom right corner


class ChangePane(Pane):
    def __init__(
        self,
        filename: str,
        color: ColorPair,
        nlines: int,
        ncols: int,
        begin_line: int,
        begin_col: int,
        label: str | None = None
    ):
        super().__init__(
            filename, color, nlines, ncols, begin_line, begin_col, label)

    def set_change(self, orig: list[str], new: list[str]) -> None:
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
        self._content_pad.erase()
        self._resize_content(height, width)
        for i, line in enumerate(contents):
            if line[0] == '+':
                addstr(self._content_pad, i, 0, line, ColorPair.DIFF_ADDED.attr)
            elif line[0] == '-':
                addstr(self._content_pad, i, 0, line, ColorPair.DIFF_REMOVED.attr)
            else:
                addstr(self._content_pad, i, 0, line)
        self._draw()


class MergeOutput:
    def __init__(self, merge: list[list[str] | Conflict]) -> None:
        self.chunks = [Decision(c) if isinstance(c, Conflict) else c for c in merge]
        self.decisions = [c for c in self.chunks if isinstance(c, Decision)]

    @property
    def height(self) -> int:
        return sum(
            e.linecount if isinstance(e, Decision) else len(e)
            for e in self.chunks
        )

    @property
    def width(self) -> int:
        chunk_widths = (
            e.width
            if isinstance(e, Decision)
            else max((len(line) for line in e), default=0)
            for e in self.chunks
        )
        return 1 + max(chunk_widths, default=0)

    def draw(self, w: curses.window) -> None:
        w.erase()
        lineno = 0
        #FIXME: Deal properly with ^M and other control characters
        for e in self.chunks:
            if isinstance(e, Decision):
                lineno = e.draw(w, lineno)
            else:
                for line in e:
                    w.addch(lineno, 0, ' ')
                    addstr(w, lineno, 1, line)
                    lineno += 1

    def lines(self) -> Generator[str]:
        for e in self.chunks:
            if isinstance(e, Decision):
                yield from e.lines()
            else:
                for line in e:
                    yield line

class OutputPane(Pane):
    def __init__(
        self,
        filename: str,
        resolved_color: ColorPair,
        unresolved_color: ColorPair,
        merge_output: MergeOutput,
        nlines: int,
        ncols: int,
        begin_line: int,
        begin_col: int,
        label: str | None = None
    ):
        self._merge_output = merge_output
        self._resolved_color = resolved_color
        self._unresolved_color = unresolved_color

        super().__init__(filename, unresolved_color, nlines, ncols, begin_line, begin_col, label)
        self._resize_content(merge_output.height, merge_output.width)
        merge_output.draw(self._content_pad)
        self._draw()

    def scroll_to_conflict(self, conflict: int) -> None:
        lineno = 0
        cur_conflict = 0
        for e in self._merge_output.chunks:
            if isinstance(e, Decision):
                if conflict == cur_conflict:
                    pane_height, _ = self._content_panel.window().getmaxyx()
                    conflict_height = e.linecount
                    if conflict_height < pane_height:
                        lineno -= (pane_height - conflict_height) // 2
                    self.scroll_vert_to(lineno)
                    break
                else:
                    lineno += e.linecount
                    cur_conflict += 1
            else:
                lineno += len(e)

    def resolve(self, conflict: int, resolution: Resolution) -> None:
        self._merge_output.decisions[conflict].resolution = resolution
        self._resize_content(self._merge_output.height, self._merge_output.width)
        if any(d.resolution == Resolution.UNRESOLVED for d in self._merge_output.decisions):
            self._color = self._unresolved_color
        else:
            self._color = self._resolved_color
        self._merge_output.draw(self._content_pad)
        self._draw()

class Resolution(Enum):
    UNRESOLVED = auto()
    USE_A = auto()
    USE_B = auto()
    USE_A_FIRST = auto()
    USE_B_FIRST = auto()
    USE_BASE = auto()


class ColorPair(IntEnum):
    DIFF_REMOVED = 1
    DIFF_ADDED = auto()
    A = auto()
    B = auto()
    BASE = auto()
    UNRESOLVED = auto()

    @classmethod
    def init(cls) -> None:
        curses.init_pair(cls.DIFF_REMOVED, curses.COLOR_RED, -1)
        curses.init_pair(cls.DIFF_ADDED, curses.COLOR_GREEN, -1)
        curses.init_pair(cls.A, curses.COLOR_CYAN, -1)
        curses.init_pair(cls.B, curses.COLOR_BLUE, -1)
        curses.init_pair(cls.BASE, -1, -1)
        curses.init_pair(cls.UNRESOLVED, curses.COLOR_MAGENTA, -1)

    @property
    def attr(self) -> int:
        return curses.color_pair(self)


@dataclass
class Decision:
    conflict: Conflict
    resolution: Resolution = Resolution.UNRESOLVED

    @property
    def linecount(self) -> int:
        match self.resolution:
            case Resolution.UNRESOLVED | Resolution.USE_BASE:
                return self._text_height(self.conflict.base)
            case Resolution.USE_A:
                return self._text_height(self.conflict.a)
            case Resolution.USE_B:
                return self._text_height(self.conflict.b)
            case Resolution.USE_A_FIRST | Resolution.USE_B_FIRST:
                return (self._text_height(self.conflict.a)
                        + self._text_height(self.conflict.b))

    def _text_height(self, text: list[str]) -> int:
        return max(1, len(text))

    def _text_width(self, text: list[str]) -> int:
        return max(map(len, text), default=len('-'))

    @property
    def width(self) -> int:
        match self.resolution:
            case Resolution.UNRESOLVED | Resolution.USE_BASE:
                return self._text_width(self.conflict.base)
            case Resolution.USE_A:
                return self._text_width(self.conflict.a)
            case Resolution.USE_B:
                return self._text_width(self.conflict.b)
            case Resolution.USE_A_FIRST | Resolution.USE_B_FIRST:
                return max(
                    self._text_width(self.conflict.a),
                    self._text_width(self.conflict.b)
                )

    def _draw_with_prefix(
        self,
        window: curses.window,
        text: list[str],
        color: ColorPair,
        prefix: str,
        lineno: int
    ) -> int:
        if text:
            for line in text:
                window.addch(lineno, 0, prefix, color.attr | curses.A_STANDOUT)
                addstr(window, lineno, 1, line, color.attr | curses.A_BOLD)
                lineno += 1
        else:
            window.addch(lineno, 0, prefix, color.attr | curses.A_STANDOUT)
            window.addch(lineno, 1, curses.ACS_HLINE, color.attr | curses.A_STANDOUT)
            lineno += 1
        return lineno

    def _draw_a(self, window: curses.window, lineno: int) -> int:
        return self._draw_with_prefix(window, self.conflict.a, ColorPair.A, 'A', lineno)

    def _draw_b(self, window: curses.window, lineno: int) -> int:
        return self._draw_with_prefix(window, self.conflict.b, ColorPair.B, 'B', lineno)

    def _draw_base(self, window: curses.window, color: ColorPair, p: str, lineno: int) -> int:
        return self._draw_with_prefix(window, self.conflict.base, color, p, lineno)

    def draw(self, window: curses.window, lineno: int) -> int:
        match self.resolution:
            case Resolution.UNRESOLVED:
                lineno = self._draw_base(window, ColorPair.UNRESOLVED, '!', lineno)
            case Resolution.USE_A:
                lineno = self._draw_a(window, lineno)
            case Resolution.USE_B:
                lineno = self._draw_b(window, lineno)
            case Resolution.USE_A_FIRST:
                lineno = self._draw_a(window, lineno)
                lineno = self._draw_b(window, lineno)
            case Resolution.USE_B_FIRST:
                lineno = self._draw_b(window, lineno)
                lineno = self._draw_a(window, lineno)
            case Resolution.USE_BASE:
                lineno = self._draw_base(window, ColorPair.BASE, ' ', lineno)
        return lineno

    def lines(self) -> Generator[str]:
        match self.resolution:
            case Resolution.UNRESOLVED:
                #TODO: Write out the diff3 block
                yield from self.conflict.base
            case Resolution.USE_A:
                yield from self.conflict.a
            case Resolution.USE_B:
                yield from self.conflict.b
            case Resolution.USE_A_FIRST:
                yield from self.conflict.a
                yield from self.conflict.b
            case Resolution.USE_B_FIRST:
                yield from self.conflict.b
                yield from self.conflict.a
            case Resolution.USE_BASE:
                yield from self.conflict.base

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
        self._merge_output = MergeOutput(merge)
        self._vsplit = .5
        self._hsplit = .5
        self._dragging: bool | Literal['hsplit'] | Literal['vsplit'] = False

    def __enter__(self) -> Self:
        self._stdscr = curses.initscr()
        curses.start_color()
        curses.use_default_colors()
        ColorPair.init()
        curses.cbreak()
        curses.noecho()
        curses.curs_set(0)
        self._stdscr.leaveok(True)
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
        _, cols = self._stdscr.getmaxyx()
        return clamp(
            ChangePane.MIN_WIDTH,
            round(cols * self._vsplit),
            cols - 1 - ChangePane.MIN_WIDTH
        )

    @_vsplit_col.setter
    def _vsplit_col(self, value: int) -> None:
        _, cols = self._stdscr.getmaxyx()
        self._vsplit = clamp(0, value / cols, 1)

    @property
    def _hsplit_row(self) -> int:
        lines, _ = self._stdscr.getmaxyx()
        return clamp(
            ChangePane.MIN_HEIGHT,
            round(lines * self._hsplit),
            lines - 1 - OutputPane.MIN_HEIGHT
        )

    @_hsplit_row.setter
    def _hsplit_row(self, value: int) -> None:
        lines, _ = self._stdscr.getmaxyx()
        self._hsplit = clamp(0, value / lines, 1)

    def _draw_borders(self) -> None:
        _, cols = self._stdscr.getmaxyx()
        line_char = 0  # use default horzontal or vertical line
        tee_char = curses.ACS_BTEE
        self._stdscr.erase()
        self._stdscr.vline(0, self._vsplit_col, line_char, self._hsplit_row)
        self._stdscr.hline(self._hsplit_row, 0, line_char, self._vsplit_col)
        self._stdscr.addch(self._hsplit_row, self._vsplit_col, tee_char)
        self._stdscr.hline(self._hsplit_row, self._vsplit_col + 1, line_char, cols - self._vsplit_col - 1)

    def _move_vsplit(self, col: int) -> None:
        if col == self._vsplit_col:
            return
        self._vsplit_col = col
        self._top_half_resized()

    def _top_half_resized(self) -> None:
        self._draw_borders()
        self._change_panes[0].resize(*self._change_a_dim())
        self._change_panes[1].resize(*self._change_b_dim())

    def _move_hsplit(self, row: int) -> None:
        if row == self._hsplit_row:
            return
        self._hsplit_row = row
        self._resized()

    def _resized(self) -> None:
        self._top_half_resized()
        self._output_pane.resize(*self._output_dim())

    def _select_conflict(self, n: int) -> None:
        if n not in range(len(self._merge_output.decisions)):
            return

        self._selected_conflict = n
        current_decision = self._merge_output.decisions[n]

        self._change_panes[0].set_change(current_decision.conflict.base, current_decision.conflict.a)
        self._change_panes[1].set_change(current_decision.conflict.base, current_decision.conflict.b)

        lines, _ = self._stdscr.getmaxyx()
        max_change_height = max(pane.preferred_height for pane in self._change_panes)
        new_hsplit = min(max_change_height, lines // 2)
        self._move_hsplit(new_hsplit)
        self._output_pane.scroll_to_conflict(self._selected_conflict)

    def _change_a_dim(self) -> tuple[int, int, int, int]:
        return self._hsplit_row, self._vsplit_col, 0, 0

    def _change_b_dim(self) -> tuple[int, int, int, int]:
        _, cols = self._stdscr.getmaxyx()
        return self._hsplit_row, cols - self._vsplit_col - 1, 0, self._vsplit_col + 1

    def _output_dim(self) -> tuple[int, int, int, int]:
        lines, cols = self._stdscr.getmaxyx()
        return lines - self._hsplit_row - 1, cols, self._hsplit_row + 1, 0

    def _set_focus(self, n: int) -> None:
        self._focused = n
        for i, pane in enumerate(self._panes):
            pane.focus(i == n)

    def run(self) -> None:
        self._change_panes = [
            ChangePane(self._filenames[0], ColorPair.A, *self._change_a_dim()),
            ChangePane(self._filenames[1], ColorPair.B, *self._change_b_dim()),
        ]
        self._output_pane = OutputPane(
                self._filenames[2], ColorPair.BASE, ColorPair.UNRESOLVED, self._merge_output, *self._output_dim())

        self._panes: list[Pane] = [
            *self._change_panes,
            self._output_pane,
        ]

        self._set_focus(2)
        self._select_conflict(0)
        self._draw_borders()

        while True:
            panel.update_panels()
            curses.doupdate()
            c = self._stdscr.getch()

            if c == ord('q'):
                return
            elif c == ord('\t'):
                self._set_focus((self._focused + 1) % 3)
            elif c == curses.KEY_BTAB:
                self._set_focus((self._focused - 1) % 3)
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
                self._resized()
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
            elif c == ord('p'):
                self._select_conflict(self._selected_conflict - 1)
            elif c == ord('n'):
                self._select_conflict(self._selected_conflict + 1)
            elif c == ord('a'):
                self._output_pane.resolve(self._selected_conflict, Resolution.USE_A)
            elif c == ord('b'):
                self._output_pane.resolve(self._selected_conflict, Resolution.USE_B)
            elif c == ord('A'):
                self._output_pane.resolve(self._selected_conflict, Resolution.USE_A_FIRST)
            elif c == ord('B'):
                self._output_pane.resolve(self._selected_conflict, Resolution.USE_B_FIRST)
            elif c == ord('i'):
                self._output_pane.resolve(self._selected_conflict, Resolution.USE_BASE)
            elif c == ord('u'):
                self._output_pane.resolve(self._selected_conflict, Resolution.UNRESOLVED)
            elif c == ord('w'):
                with open(self._filenames[2], 'w') as f:
                    #FIXME: Deal properly with files that don't have newlines
                    f.writelines(f'{line}\n' for line in self._merge_output.lines())
                    return


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
    parser.add_argument('MYFILE')
    parser.add_argument('OLDFILE')
    parser.add_argument('YOURFILE')
    args = parser.parse_args()

    diff3_result = subprocess.run(
        'git merge-file -p --zdiff3 --'.split(' ')
        + [args.MYFILE, args.OLDFILE, args.YOURFILE],
        stdout=subprocess.PIPE,
        encoding='utf-8',
    )

    merge = MergeParser(diff3_result.stdout.splitlines()).parse()
    with DLMerge(args.MYFILE, args.YOURFILE, args.OLDFILE, merge) as dlmerge:
        dlmerge.run()

    exit()


if __name__ == '__main__':
    main()
