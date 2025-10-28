from __future__ import annotations

from abc import abstractmethod
import argparse
import curses
import curses.panel as panel
from dataclasses import dataclass, field
from enum import Enum, IntEnum, auto
import os
import re
import subprocess
from tempfile import NamedTemporaryFile
from types import MappingProxyType, TracebackType
from typing import Callable, Generator, Iterable, Literal, NoReturn, Optional, Self, Sequence

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
# Additional navigation with Home/End/PgUp/PgDn
# (space) select next visible or page down

# TODO:
# Reconcile navigation and conflict selection
# Make work with xterm-mono

# Wishlist: Menus

def clamp[T: float](minimum: T, value: T, maximum: T) -> T:
    return max(minimum, min(value, maximum))


class Pane:
    MIN_HEIGHT = 2
    MIN_WIDTH = 2

    def __init__(
        self,
        color: ColorPair,
        nlines: int,
        ncols: int,
        begin_line: int,
        begin_col: int,
    ):
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
        return self.content_height + 1  # for the title bar

    #FIXME: scrolling past the right or bottom edge causes repeating lines/chars
    def scroll_vert(self, n: int) -> None:
        self.scroll_vert_to(self._vscroll + n)

    def scroll_vert_to(self, n: int) -> None:
        self._vscroll = clamp(0, n, self.content_height - 1)
        self._draw()

    def scroll_horiz(self, n: int) -> None:
        self.scroll_horiz_to(self._hscroll + n)

    def scroll_horiz_to(self, n: int) -> None:
        self._hscroll = clamp(0, n, self.width - 1)
        self._draw()

    def _draw_titlebar(self) -> None:
        titlewin = self._title_panel.window()
        bg_attr = self._color.attr | curses.A_REVERSE
        if self._focused:
            bg_attr |= curses.A_BOLD
        titlewin.bkgdset(' ', bg_attr)
        noerror(titlewin.addch, 0, 0, '[' if self._focused else ' ')
        self._draw_title()
        noerror(titlewin.addch, ']' if self._focused else ' ')
        titlewin.clrtobot()

    @abstractmethod
    def _draw_title(self) -> None: ...

    def focus(self, on: bool) -> None:
        self._focused = on
        self._draw_titlebar()

    def enclose(self, row: int, col: int) -> bool:
        return self._win.enclose(row, col)

    @property
    def gutter(self) -> curses.window:
        return self._gutter_pad

    @property
    def content(self) -> curses.window:
        return self._content_pad

    @property
    def width(self) -> int:
        _, cols = self._content_pad.getmaxyx()
        return cols

    @property
    def content_height(self) -> int:
        lines, _ = self._content_pad.getmaxyx()
        return lines

    def _resize_content(self, nlines: int, ncols: int) -> None:
        self._content_pad.resize(nlines, ncols)
        self._gutter_pad.resize(nlines, 1)

    def resize(self, nlines: int, ncols: int, begin_line: int, begin_col: int) -> None:
        self._set_size_attrs(nlines, ncols, begin_line, begin_col)

        self._win, title, gutter, content = self._create_wins()
        self._title_panel.replace(title)
        self._content_panel.replace(content)
        self._gutter_panel.replace(gutter)

        self._draw()

    def _draw(self) -> None:
        self._draw_titlebar()
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


class ChangePane(Pane):
    def __init__(
        self,
        file: Revision,
        key: str,
        desc: str,
        color: ColorPair,
        nlines: int,
        ncols: int,
        begin_line: int,
        begin_col: int,
    ):
        self._file = file
        self._key = key
        self._desc = desc
        super().__init__(
            color, nlines, ncols, begin_line, begin_col)

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
        self._hscroll = 0
        self._vscroll = 0
        self._content_pad.erase()
        self._resize_content(height, width)
        for i, line in enumerate(contents):
            prefix = line[0]
            data = line[1:]
            if prefix == '+':
                attr = ColorPair.DIFF_ADDED.attr
            elif prefix == '-':
                attr = ColorPair.DIFF_REMOVED.attr
            else:
                attr = curses.A_NORMAL
            noerror(self._gutter_pad.addch, i, 0, prefix, attr)
            noerror(self._content_pad.addstr, i, 0, data, attr | curses.A_BOLD)
        self._draw()

    def _draw_title(self) -> None:
        titlewin = self._title_panel.window()
        noerror(titlewin.addstr, f'{self._key}:')
        noerror(titlewin.addch, ' ')
        name = self._file.label or self._file.filename
        if name:
            noerror(titlewin.addstr, name)
            if self._desc:
                noerror(titlewin.addch, ' ')
        if self._desc:
            noerror(titlewin.addstr, f'({self._desc})', curses.A_ITALIC)


class MergeOutput:
    def __init__(self, merge: list[list[str] | Conflict]) -> None:
        self.chunks = [Decision(c) if isinstance(c, Conflict) else c for c in merge]
        self.decision_indices: Sequence[int] = [i for (i, c) in enumerate(self.chunks) if isinstance(c, Decision)]
        self.decision_chunk_indices = MappingProxyType({i: v for i, v in enumerate(self.decision_indices)})
        self.edited_text_chunks: list[None | list[str]] = [None] * len(self.chunks)

    def decisions(self) -> Generator[Decision]:
        return (c for c in self.chunks if isinstance(c, Decision))

    def get_decision(self, n: int) -> Decision:
        if n < 0 or n >= len(self.decision_indices):
            raise IndexError(f'Decision index out of range: {n}')
        chunk = self.chunks[self.decision_indices[n]]
        assert(isinstance(chunk, Decision))
        return chunk

    def edited_chunks(self) -> Generator[list[str] | Decision]:
        for chunk, edit in zip(self.chunks, self.edited_text_chunks):
            if edit is not None:
                yield edit
            else:
                yield chunk

    def edited_chunk(self, n: int) -> list[str] | Decision:
        edit = self.edited_text_chunks[n]
        if edit is not None:
            return edit
        return self.chunks[n]

    @property
    def height(self) -> int:
        return sum(
            e.linecount if isinstance(e, Decision) else len(e)
            for e in self.edited_chunks()
        )

    @property
    def width(self) -> int:
        chunk_widths = (
            e.width
            if isinstance(e, Decision)
            else max((len(line) for line in e), default=0)
            for e in self.edited_chunks()
        )
        return 1 + max(chunk_widths, default=0)

    def draw(self, pane: Pane, selected_conflict: int = 0) -> None:
        pane.gutter.erase()
        pane.content.erase()
        lineno = 0
        selected_chunk = self.decision_chunk_indices[selected_conflict]
        #FIXME: Deal properly with ^M and other control characters
        for i, e in enumerate(self.edited_chunks()):
            if isinstance(e, Decision):
                lineno = e.draw(pane, lineno, i == selected_chunk)
            else:
                for line in e:
                    noerror(pane.gutter.addch, lineno, 0, ' ')
                    noerror(pane.content.addstr, lineno, 0, line)
                    lineno += 1

    def lines(self) -> Generator[str]:
        for e in self.edited_chunks():
            if isinstance(e, Decision):
                yield from e.lines()
            else:
                for line in e:
                    yield line


def noerror[**P, T](f: Callable[P, T], *args: P.args, **kwargs: P.kwargs) -> None:
    try:
        f(*args, **kwargs)
    except curses.error:
        pass


class OutputPane(Pane):
    def __init__(
        self,
        file: Revision,
        outfile: str | None,
        resolved_color: ColorPair,
        unresolved_color: ColorPair,
        merge_output: MergeOutput,
        nlines: int,
        ncols: int,
        begin_line: int,
        begin_col: int,
    ):
        self._file = file
        self._outfile = outfile
        self._merge_output = merge_output
        self._resolved_color = resolved_color
        self._unresolved_color = unresolved_color

        super().__init__(unresolved_color, nlines, ncols, begin_line, begin_col)
        self._resize_content(merge_output.height, merge_output.width)
        merge_output.draw(self)
        self._draw()

    def _select_conflict(self, conflict: int) -> None:
        self._selected_conflict = conflict
        self._draw_merge_output()

    def scroll_to_conflict(self, conflict: int, select: bool = True) -> None:
        lineno = 0
        cur_conflict = 0
        if select:
            self._select_conflict(conflict)
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

    def _fully_resolved(self) -> bool:
        return not any(
            d.resolution == Resolution.UNRESOLVED
            for d in self._merge_output.decisions()
        )

    def resolve(self, conflict: int, resolution: Resolution, edit: Optional[Edit] = None) -> None:
        #TODO: If already resolved as edited, confirm before discarding edit
        decision_chunk_index = self._merge_output.decision_chunk_indices[conflict]
        decision = self._merge_output.get_decision(conflict)
        assert(isinstance(decision, Decision))
        decision.resolution = resolution
        if resolution == Resolution.EDITED:
            assert(edit is not None)
            if decision_chunk_index > 0:
                self._merge_output.edited_text_chunks[decision_chunk_index - 1] = edit.prelude
            else:
                assert(not edit.prelude)
            if decision_chunk_index < len(self._merge_output.chunks) - 1:
                self._merge_output.edited_text_chunks[decision_chunk_index + 1] = edit.epilogue
            else:
                assert(not edit.epilogue)
            decision.edit = edit.text
        else:
            assert(edit is None)
            try:
                self._merge_output.edited_text_chunks[decision_chunk_index - 1] = None
            except IndexError:
                pass
            try:
                self._merge_output.edited_text_chunks[decision_chunk_index + 1] = None
            except IndexError:
                pass
        self._resize_content(self._merge_output.height, self._merge_output.width)
        if self._fully_resolved():
            self._color = self._resolved_color
        else:
            self._color = self._unresolved_color
        self._draw_merge_output()
        self._draw()

    def _draw_merge_output(self):
        self._merge_output.draw(self, self._selected_conflict)

    def _draw_title(self) -> None:
        titlewin = self._title_panel.window()

        name = self._file.filename or self._file.label
        if name and self._outfile:
            noerror(titlewin.addstr, 'Base:')
            noerror(titlewin.addch, ' ')
            noerror(titlewin.addstr, name)
            noerror(titlewin.addstr, '; ')
            noerror(titlewin.addstr, 'Output:')
            noerror(titlewin.addch, ' ')
            noerror(titlewin.addstr, self._outfile)
            noerror(titlewin.addch, ' ')
        else:
            merge = name or self._outfile
            if merge:
                noerror(titlewin.addstr, merge)
                noerror(titlewin.addch, ' ')
        noerror(titlewin.addstr, '(Merge)', curses.A_ITALIC)

    def _draw_titlebar(self) -> None:
        super()._draw_titlebar()
        titlewin = self._title_panel.window()
        _, cols = titlewin.getmaxyx()
        status = ' RESOLVED' if self._fully_resolved() else ' UNRESOLVED'
        noerror(titlewin.addstr, 0, cols - 1 - len(status), status)


class Resolution(Enum):
    UNRESOLVED = auto()
    USE_A = auto()
    USE_B = auto()
    USE_A_FIRST = auto()
    USE_B_FIRST = auto()
    USE_BASE = auto()
    EDITED = auto()


class ColorPair(IntEnum):
    DIFF_REMOVED = 1
    DIFF_ADDED = auto()
    A = auto()
    B = auto()
    BASE = auto()
    UNRESOLVED = auto()
    EDITED = auto()

    @classmethod
    def init(cls) -> None:
        if not curses.has_colors():
            return
        bright = 8 if curses.COLORS > 8 else 0
        curses.init_pair(cls.DIFF_REMOVED, curses.COLOR_RED, -1)
        curses.init_pair(cls.DIFF_ADDED, curses.COLOR_GREEN, -1)
        curses.init_pair(cls.A, curses.COLOR_CYAN, -1)
        curses.init_pair(cls.B, curses.COLOR_BLUE + bright, -1)
        curses.init_pair(cls.BASE, -1, -1)
        curses.init_pair(cls.UNRESOLVED, curses.COLOR_MAGENTA, -1)
        curses.init_pair(cls.EDITED, curses.COLOR_YELLOW, -1)

    @property
    def attr(self) -> int:
        if not curses.has_colors():
            return 0
        return curses.color_pair(self)


def common_prefix[T](l1: Iterable[T], l2: Iterable[T]) -> Generator[T]:
    for i1, i2 in zip(l1, l2):
        if i1 != i2:
            break
        yield i1


@dataclass
class Edit:
    prelude: list[str]
    text: list[str]
    epilogue: list[str]


@dataclass
class Decision:
    conflict: Conflict
    resolution: Resolution = Resolution.UNRESOLVED
    edit: list[str] = field(default_factory=list[str])

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
            case Resolution.EDITED:
                return self._text_height(self.edit)

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
            case Resolution.EDITED:
                return self._text_width(self.edit)

    def _draw_with_gutter(
        self,
        pane: Pane,
        text: list[str],
        color: ColorPair,
        prefix: str,
        selected: bool,
        lineno: int,
        end_chunk: bool = True,
    ) -> int:
        gutter_attr = curses.A_STANDOUT if selected else 0
        if text:
            for i, line in enumerate(text):
                if i == 0:
                    this_prefix = ord(prefix)
                elif i == len(text) - 1 and end_chunk:
                    this_prefix = curses.ACS_LLCORNER
                else:
                    this_prefix = curses.ACS_VLINE
                noerror(pane.gutter.addch, lineno, 0, this_prefix, color.attr | gutter_attr)
                noerror(pane.content.addstr, lineno, 0, line, color.attr | curses.A_BOLD)
                lineno += 1
        else:
            noerror(pane.gutter.addch, lineno, 0, prefix, color.attr | gutter_attr)
            pane.content.attron(color.attr | curses.A_BOLD)
            pane.content.hline(lineno, 0, 0, pane.width)
            pane.content.attroff(color.attr | curses.A_BOLD)
            lineno += 1
        return lineno

    def _draw_a(self, pane: Pane, selected: bool, lineno: int, start_chunk: bool = True, end_chunk: bool = True) -> int:
        prefix = 'A'
        return self._draw_with_gutter(pane, self.conflict.a, ColorPair.A, prefix, selected, lineno, end_chunk=end_chunk)

    def _draw_b(self, window: Pane, selected: bool, lineno: int, start_chunk: bool = True, end_chunk: bool = True) -> int:
        prefix = 'B'
        return self._draw_with_gutter(window, self.conflict.b, ColorPair.B, prefix, selected, lineno, end_chunk=end_chunk)

    def _draw_base(self, window: Pane, color: ColorPair, p: str, selected: bool, lineno: int, always_prefix: bool = False) -> int:
        return self._draw_with_gutter(window, self.conflict.base, color, p, selected, lineno)

    def _draw_edit(self, window: Pane, selected: bool, lineno: int) -> int:
        prefix = 'E'
        return self._draw_with_gutter(window, self.edit, ColorPair.EDITED, prefix, selected, lineno)

    def draw(self, window: Pane, lineno: int, selected: bool) -> int:
        match self.resolution:
            case Resolution.UNRESOLVED:
                lineno = self._draw_base(window, ColorPair.UNRESOLVED, '?', selected, lineno, always_prefix=True)
            case Resolution.USE_A:
                lineno = self._draw_a(window, selected, lineno)
            case Resolution.USE_B:
                lineno = self._draw_b(window, selected, lineno)
            case Resolution.USE_A_FIRST:
                lineno = self._draw_a(window, selected, lineno, end_chunk=False)
                lineno = self._draw_b(window, selected, lineno, start_chunk=False)
            case Resolution.USE_B_FIRST:
                lineno = self._draw_b(window, selected, lineno, end_chunk=False)
                lineno = self._draw_a(window, selected, lineno, start_chunk=False)
            case Resolution.USE_BASE:
                lineno = self._draw_base(window, ColorPair.BASE, ' ', selected, lineno)
            case Resolution.EDITED:
                lineno = self._draw_edit(window, selected, lineno)
        return lineno

    def lines(self) -> Generator[str]:
        match self.resolution:
            case Resolution.UNRESOLVED:
                yield from self.conflict_lines()
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
            case Resolution.EDITED:
                yield from self.edit

    def conflict_lines(self) -> Generator[str]:
        yield f'<<<<<<< {self.conflict.a_label}'
        yield from self.conflict.a
        yield f'||||||| {self.conflict.base_label}'
        yield from self.conflict.base
        yield f'======='
        yield from self.conflict.b
        yield f'>>>>>>> {self.conflict.b_label}'

def terminal_supports_xterm_mouse() -> bool:
    xm = curses.tigetstr('XM')
    if xm and b'1006' in xm:
        return True
    terminal_type = os.environ.get('TERM', 'unknown')
    if 'xterm' in terminal_type:
        return True
    return False


def term_enable_mouse_drag(enable: bool = True) -> None:
    if not terminal_supports_xterm_mouse():
        return
    c = 'h' if enable else 'l'
    print(f'\033[?1002{c}', flush=True)


@dataclass
class Revision:
    filename: Optional[str]
    label: Optional[str]

class DLMerge:
    def __init__(
        self,
        file_a: Revision,
        file_b: Revision,
        file_base: Revision,
        merge: list[list[str] | Conflict],
        outfile: Optional[str] = None
    ):
        self._outfile = outfile
        self._files = [file_a, file_b, file_base]
        self._merge = merge
        self._merge_output = MergeOutput(merge)
        self._vsplit = .5
        self._hsplit = .5
        self._dragging: bool | Literal['hsplit'] | Literal['vsplit'] = False

    def __enter__(self) -> Self:
        self._stdscr = curses.initscr()
        curses.start_color()
        noerror(curses.use_default_colors)
        ColorPair.init()
        curses.cbreak()
        curses.noecho()
        noerror(curses.curs_set, 0)
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
        line_char = 0  # use default horzontal or vertical line
        self._stdscr.erase()
        self._stdscr.vline(0, self._vsplit_col, line_char, self._hsplit_row)

    def _move_vsplit(self, n: int) -> None:
        self._move_vsplit_to(self._vsplit_col + n)

    def _move_vsplit_to(self, col: int) -> None:
        if col == self._vsplit_col:
            return
        self._vsplit_col = col
        self._top_half_resized()

    def _top_half_resized(self) -> None:
        self._draw_borders()
        self._change_panes[0].resize(*self._change_a_dim())
        self._change_panes[1].resize(*self._change_b_dim())

    def _move_hsplit(self, n: int) -> None:
        self._move_hsplit_to(self._hsplit_row + n)

    def _move_hsplit_to(self, row: int) -> None:
        if row == self._hsplit_row:
            return
        self._hsplit_row = row
        self._resized()

    def _resized(self) -> None:
        self._top_half_resized()
        self._output_pane.resize(*self._output_dim())

    def _select_conflict(self, n: int) -> None:
        try:
            current_decision = self._merge_output.get_decision(n)
        except IndexError:
            return
        self._selected_conflict = n

        self._change_panes[0].set_change(current_decision.conflict.base, current_decision.conflict.a)
        self._change_panes[1].set_change(current_decision.conflict.base, current_decision.conflict.b)

        lines, _ = self._stdscr.getmaxyx()
        max_change_height = max(pane.preferred_height for pane in self._change_panes)
        new_hsplit = min(max_change_height, lines // 2)
        self._move_hsplit_to(new_hsplit)
        self._output_pane.scroll_to_conflict(self._selected_conflict)

    def _get_chunk_if_text(self, i: int) -> list[str]:
        try:
            chunk = self._merge_output.edited_chunk(i)
            if not isinstance(chunk, Decision):
                return chunk
        except IndexError:
            pass
        return []

    def _edit_selected_conflict(self) -> None:
        selected_decision = self._merge_output.get_decision(self._selected_conflict)
        decision_chunk_index = self._merge_output.decision_chunk_indices[self._selected_conflict]
        prelude = self._get_chunk_if_text(decision_chunk_index - 1)
        epilogue = self._get_chunk_if_text(decision_chunk_index + 1)
        conflicted_lines: Iterable[str]
        if selected_decision.resolution == Resolution.EDITED:
            conflicted_lines = selected_decision.edit
        else:
            conflicted_lines = selected_decision.conflict_lines()
        editor_lines = [*prelude, *conflicted_lines, *epilogue]
        editor = os.getenv('VISUAL', os.getenv('EDITOR', 'vi'))
        with NamedTemporaryFile('w+', delete_on_close=False, prefix='dlmerge-') as editor_file:
            editor_file.writelines(f'{line}\n' for line in editor_lines)
            editor_file.close()
            curses.def_prog_mode()
            curses.endwin()
            try:
                subprocess.run(
                    editor.split(' ')
                    + [f'+{len(prelude) + 1}', editor_file.name],
                    encoding='utf-8',
                    check=True
                )
            except subprocess.CalledProcessError:
                #TODO: error dialog
                pass
            finally:
                curses.reset_prog_mode()
            with open(editor_file.name, 'r') as f:
                newlines = f.read().splitlines()

        if editor_lines == newlines:
            return
        new_prelude = list(common_prefix(prelude, newlines))
        new_epilogue = [*reversed([*common_prefix(reversed(epilogue), reversed(newlines))])]
        edit_text = newlines[len(new_prelude):-len(new_epilogue) if new_epilogue else len(newlines)]
        edit = Edit(new_prelude, edit_text, new_epilogue)
        self._output_pane.resolve(self._selected_conflict, Resolution.EDITED, edit)

    def _change_a_dim(self) -> tuple[int, int, int, int]:
        return self._hsplit_row, self._vsplit_col, 0, 0

    def _change_b_dim(self) -> tuple[int, int, int, int]:
        _, cols = self._stdscr.getmaxyx()
        return self._hsplit_row, cols - self._vsplit_col - 1, 0, self._vsplit_col + 1

    def _output_dim(self) -> tuple[int, int, int, int]:
        lines, cols = self._stdscr.getmaxyx()
        return lines - self._hsplit_row, cols, self._hsplit_row, 0

    def _set_focus(self, n: int) -> None:
        self._focused = n
        for i, pane in enumerate(self._panes):
            pane.focus(i == n)

    def _pane_under_cell(self, row: int, col: int) -> tuple[int, Pane] | None:
        for i, pane in enumerate(self._panes):
            if pane.enclose(row, col):
                return i, pane
        return None

    def _banish_cursor(self) -> None:
        # On terminals where the cursor can't be hidden, move it someplace
        # out-of-the-way.
        rows, _ = self._stdscr.getmaxyx()
        self._stdscr.move(rows - 1, 0)

    def run(self) -> None:
        self._change_panes = [
            ChangePane(self._files[0], 'A', 'Current', ColorPair.A, *self._change_a_dim()),
            ChangePane(self._files[1], 'B', 'Incoming', ColorPair.B, *self._change_b_dim()),
        ]
        self._output_pane = OutputPane(
                self._files[2], self._outfile, ColorPair.BASE, ColorPair.UNRESOLVED, self._merge_output, *self._output_dim())

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
            self._banish_cursor()

            c = self._stdscr.getch()
            if c == ord('q'):
                return
            elif c == ord('\t'):
                self._set_focus((self._focused + 1) % 3)
            elif c == curses.KEY_BTAB:
                self._set_focus((self._focused - 1) % 3)
            elif c in (curses.KEY_UP, ord('k')):
                self._panes[self._focused].scroll_vert(-1)
            elif c in (curses.KEY_DOWN, ord('j')):
                self._panes[self._focused].scroll_vert(1)
            elif c in (curses.KEY_LEFT, ord('h')):
                self._panes[self._focused].scroll_horiz(-2)
            elif c in (curses.KEY_RIGHT, ord('l')):
                self._panes[self._focused].scroll_horiz(2)
            elif c in (curses.KEY_SR, ord('K'), ord('-')):
                self._move_hsplit(-1)
            elif c in (curses.KEY_SF, ord('J'), ord('='), ord('+')):
                self._move_hsplit(1)
            elif c in (curses.KEY_SLEFT, ord('H'), ord('[')):
                self._move_vsplit(-1)
            elif c in (curses.KEY_SRIGHT, ord('L'), ord(']')):
                self._move_vsplit(1)
            elif c in (curses.KEY_HOME, ord('<')):
                self._panes[self._focused].scroll_vert_to(0)
            elif c in (curses.KEY_END, ord('>')):
                pane = self._panes[self._focused]
                lines, _, _, _ = self._output_dim()
                pane.scroll_vert_to(pane.content_height - 1 - lines)
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
            elif c == ord('e'):
                self._edit_selected_conflict()
            elif c == ord('w'):
                self._save()
                return

            elif c == curses.KEY_RESIZE:
                curses.update_lines_cols()
                self._resized()
            elif c == curses.KEY_MOUSE:
                _, mcol, mrow, _, bstate = curses.getmouse()

                if bstate & curses.BUTTON1_PRESSED:
                    if mrow == self._hsplit_row:
                        self._dragging = 'hsplit'
                        self._set_focus(2)
                    elif mrow < self._hsplit_row and mcol == self._vsplit_col:
                        self._dragging = 'vsplit'
                    else:
                        t = self._pane_under_cell(mrow, mcol)
                        if t:
                            i, _ = t
                            self._set_focus(i)

                if self._dragging == 'hsplit':
                    self._move_hsplit_to(mrow)
                if self._dragging == 'vsplit':
                    self._move_vsplit_to(mcol)
                if bstate & curses.BUTTON1_RELEASED:
                    self._dragging = False

                if bstate & curses.BUTTON4_PRESSED:
                    if t := self._pane_under_cell(mrow, mcol):
                        _, pane = t
                        if bstate & (curses.BUTTON_SHIFT | curses.BUTTON_CTRL):
                            pane.scroll_horiz(-2)
                        else:
                            pane.scroll_vert(-1)
                if bstate & curses.BUTTON5_PRESSED:
                    if t := self._pane_under_cell(mrow, mcol):
                        _, pane = t
                        if bstate & (curses.BUTTON_SHIFT | curses.BUTTON_CTRL):
                            pane.scroll_horiz(2)
                        else:
                            pane.scroll_vert(1)

    def _save(self) -> None:
        outfile = self._outfile or self._files[2].filename
        if not outfile:
            raise ValueError('No output filename provided')
        with open(outfile, 'w') as f:
            #FIXME: Deal properly with files that don't have newlines
            f.writelines(f'{line}\n' for line in self._merge_output.lines())


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
    parser.add_argument('--label', '-L', action='append', default=[])
    parser.add_argument('--output', '-o')
    parser.add_argument('MYFILE')
    parser.add_argument('OLDFILE')
    parser.add_argument('YOURFILE')
    args = parser.parse_args()

    labels: list[str] = args.label
    label_iter = iter(labels)
    label_args = [arg for label in labels for arg in ['-L', label]]
    myfile = Revision(args.MYFILE, next(label_iter, None))
    oldfile = Revision(args.OLDFILE, next(label_iter, None))
    yourfile = Revision(args.YOURFILE, next(label_iter, None))
    outfile = args.output

    diff3_result = subprocess.run(
        'git merge-file -p --zdiff3 --'.split(' ')
        + label_args
        + [args.MYFILE, args.OLDFILE, args.YOURFILE],
        stdout=subprocess.PIPE,
        encoding='utf-8',
    )

    merge = MergeParser(diff3_result.stdout.splitlines()).parse()
    with DLMerge(myfile, yourfile, oldfile, merge, outfile=outfile) as dlmerge:
        dlmerge.run()

    exit()


if __name__ == '__main__':
    main()
