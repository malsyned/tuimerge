from __future__ import annotations

from abc import abstractmethod
import argparse
import curses
import curses.panel as panel
from dataclasses import dataclass, field
from enum import Enum, IntEnum, auto
from itertools import chain, repeat
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from tempfile import NamedTemporaryFile
import textwrap
from types import MappingProxyType
from typing import (
    IO,
    Callable,
    Concatenate,
    Generator,
    Iterable,
    Literal,
    NoReturn,
    Optional,
    Sequence,
    overload,
)

# display 3 windows:
#     top left = new A hunk with diff from original
#     top right = new B hunk with diff from original
#     bottom = merge result
# All 3 windows have titles containing both their filenames and their labels
#
# Key bindings
# (n) next conflict
# (p) previous conflict
# (shift+n)/(shift+p) next/previous unresolved conflict
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
# (shift+s) save as (will have to figure out how to integrate readline)
# (q, ^X) quit, prompt to save
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
# Add environment variables like TUIMERGE_EDITOR, TUIMERGE_PAGER, etc.

# Wishlist: Menus

def clamp[T: float](minimum: T, value: T, maximum: T) -> T:
    return max(minimum, min(value, maximum))


class Dialog:
    def __init__(self) -> None:
        self._win = curses.newwin(1, 1, 0, 0)
        self._pad = curses.newpad(1, 1)
        self._panel = panel.new_panel(self._win)
        self._panel.hide()
        self._color = ColorPair.DIALOG_INFO
        self._text = ' '
        self._prompt: str | None = None
        self._wide = False
        self._center = True

    def set_contents(self, text: str, prompt: Optional[str], center: bool = False, wide: bool = False) -> None:
        self._text = text
        self._prompt = prompt
        self._wide = wide
        self._center = center

    def set_color(self, color: ColorPair) -> None:
        self._color = color

    def resize(self, screen_nlines: int, screen_ncols: int) -> None:
        max_width = screen_ncols - 4 if self._wide else screen_ncols * 2 // 3
        lines = list(wrap(self._text, max_width))

        contents_height = (len(lines) + (2 if self._prompt else 0)) or 1
        contents_width = max([*map(len, lines), len(self._prompt or '')]) or 1
        height = min(contents_height + 2, screen_nlines)
        width = contents_width + 4
        row = (screen_nlines - height) // 2
        col = (screen_ncols - width) // 2

        self._win = curses.newwin(height, width, row, col)
        self._pad = curses.newpad(contents_height, contents_width)
        self._win.attron(self._color.attr)
        self._win.box()
        self._win.attroff(self._color.attr)
        for i, line in enumerate(lines):
            col = (contents_width - len(line)) // 2 if self._center else 0
            self._pad.addstr(i, col, line)
        if self._prompt:
            noerror(self._pad.addstr, contents_height - 1, contents_width - len(self._prompt), self._prompt)
        self._pad.overwrite(self._win, 0, 0, 1, 2, height - 2, width - 3)
        self._panel.replace(self._win)

    def show(self) -> None:
        self._panel.top()
        self._panel.show()

    def hide(self) -> None:
        self._panel.hide()

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
        gutter_width: int = 1
    ):
        self._color = color
        self._set_size_attrs(nlines, ncols, begin_line, begin_col)
        self._hscroll = 0
        self._vscroll = 0
        self._focused = False
        self._gutter_width = gutter_width

        self._win, title, gutter, content = self._create_wins()
        self._title_panel = panel.new_panel(title)
        self._gutter_panel = panel.new_panel(gutter)
        self._content_panel = panel.new_panel(content)
        self._gutter_pad = curses.newpad(1, self._gutter_width)
        self._content_pad = curses.newpad(1, 1)

    def _set_size_attrs(self, nlines: int, ncols: int, begin_line: int, begin_col: int) -> None:
        self._nlines = nlines
        self._ncols = ncols
        self._begin_line = begin_line
        self._begin_col = begin_col

    def _create_wins(self) -> tuple[curses.window, curses.window, curses.window, curses.window]:
        win = curses.newwin(self._nlines, self._ncols, self._begin_line, self._begin_col)
        title = win.derwin(1, self._ncols, 0, 0)
        gutter = win.derwin(self._nlines - 1, self._gutter_width, 1, 0)
        content = win.derwin(self._nlines - 1, self._ncols - self._gutter_width, 1, self._gutter_width)
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

    def scroll_page(self, n: int) -> None:
        content_height, _ = self._content_panel.window().getmaxyx()
        self.scroll_vert_to(self._vscroll + n * content_height)

    def _draw_titlebar(self) -> None:
        titlewin = self._title_panel.window()
        bg_attr = self._color.attr | curses.A_REVERSE
        if self._focused and curses.COLORS != 8:
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
        self._gutter_pad.resize(nlines, self._gutter_width)

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

            #TODO: Consider supporting non-unified diffs as well
            diff_output = do_diff2(
                tmporig.name, tmpnew.name, context=len(orig) + len(new))
        contents = [
            line for line in diff_output
            if (
                any(line.startswith(c) for c in '+-<> ')
                and not any(line.startswith(p) for p in['---', '+++'])
            )
        ]
        height = len(contents) or 1
        width = max(map(len, contents), default=1)
        self._hscroll = 0
        self._vscroll = 0
        self._gutter_pad.erase()
        self._content_pad.erase()
        self._resize_content(height, width)
        for i, line in enumerate(contents):
            prefix = line[0]
            data = line[1:]
            if prefix in ('+', '>'):
                attr = ColorPair.DIFF_ADDED.attr
                content_attr = curses.A_BOLD
            elif prefix in ('-', '<'):
                content_attr = curses.A_BOLD
                attr = ColorPair.DIFF_REMOVED.attr
            else:
                attr = curses.A_NORMAL
                content_attr = curses.A_NORMAL
            noerror(self._gutter_pad.addch, i, 0, prefix, attr)
            noerror(self._content_pad.addstr, i, 0, data, attr | content_attr)
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
        if self.decision_chunk_indices:
            selected_chunk = self.decision_chunk_indices[selected_conflict]
        else:
            selected_chunk = -1

        #FIXME: Deal properly with ^M and other control characters
        for i, e in enumerate(self.edited_chunks()):
            if isinstance(e, Decision):
                lineno = e.draw(pane, lineno, i == selected_chunk)
            else:
                for line in e:
                    noerror(pane.gutter.addch, lineno, 0, ' ')
                    noerror(pane.content.addstr, lineno, 0, line)
                    lineno += 1

    def lines(self, ignore_unresolved: bool = False) -> Generator[str]:
        for e in self.edited_chunks():
            if isinstance(e, Decision):
                yield from e.lines(ignore_unresolved=ignore_unresolved)
            else:
                yield from e

    def fully_resolved(self) -> bool:
        return not any(
            d.resolution == Resolution.UNRESOLVED
            for d in self.decisions()
        )

    def fully_unresolved(self) -> bool:
        return all(
            d.resolution == Resolution.UNRESOLVED
            for d in self.decisions()
        )


def noerror[**P, T](f: Callable[P, T], *args: P.args, **kwargs: P.kwargs) -> None:
    try:
        f(*args, **kwargs)
    except curses.error:
        pass


class ConflictVisibility(Enum):
    ABOVE_WINDOW = auto()
    PARTIALLY_ABOVE_WINDOW = auto()
    WITHIN_WINDOW = auto()
    OVERFLOWS_WINDOW = auto()
    PARTIALLY_BELOW_WINDOW = auto()
    BELOW_WINDOW = auto()

    @property
    def is_visible(self) -> bool:
        return self not in (
            ConflictVisibility.ABOVE_WINDOW,
            ConflictVisibility.BELOW_WINDOW
        )


class ScrollSnap(Enum):
    TOP = auto()
    CENTER = auto()
    BOTTOM = auto()


class OutputPane(Pane):
    def __init__(
        self,
        parent: TUIMerge,
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
        self._parent = parent
        self._file = file
        self._outfile = outfile
        self._merge_output = merge_output
        self._resolved_color = resolved_color
        self._unresolved_color = unresolved_color

        color = unresolved_color if merge_output.decision_chunk_indices else resolved_color

        super().__init__(color, nlines, ncols, begin_line, begin_col, gutter_width=2)
        self._resize_content(merge_output.height, merge_output.width)
        merge_output.draw(self)
        self._draw()

    @property
    def selected_conflict(self) -> int:
        return self._selected_conflict

    def select_conflict(self, conflict: int) -> None:
        self._selected_conflict = conflict
        self._draw_merge_output()

    def _selected_conflict_and_line(self, conflict: int) -> tuple[int, Decision]:
        selected_decision_chunk_index = self._merge_output.decision_chunk_indices[conflict]
        lineno = 0
        for i, c in enumerate(self._merge_output.edited_chunks()):
            if isinstance(c, Decision):
                if i < selected_decision_chunk_index:
                    lineno += c.linecount
                else:
                    return lineno, self._merge_output.get_decision(conflict)
            else:
                lineno += len(c)
        raise IndexError(f'No conflict found with index {conflict}')

    def scroll_to_conflict(
        self,
        conflict: int,
        scroll_hints: tuple[ScrollSnap, ScrollSnap],
    ) -> None:
        scroll_hint, scroll_hint_overflow = scroll_hints
        decision, lineno, _, visible_lines = self._conflict_position_info(conflict)
        pane_size = visible_lines.stop - visible_lines.start
        if visible_lines.stop - visible_lines.start < decision.linecount:
            scroll_hint = scroll_hint_overflow

        match scroll_hint:
            case ScrollSnap.TOP:
                self.scroll_vert_to(lineno)
            case ScrollSnap.CENTER:
                offset = (pane_size - decision.linecount) // 2
                self.scroll_vert_to(lineno - offset)
                pass
            case ScrollSnap.BOTTOM:
                self.scroll_vert_to(lineno + decision.linecount - pane_size)

    def _conflict_position_info(self, conflict: int) -> tuple[Decision, int, int, range]:
        content_window_height, _ = self._content_panel.window().getmaxyx()
        lineno, decision = self._selected_conflict_and_line(conflict)
        lastline = lineno + decision.linecount - 1
        first_line_past_window = self._vscroll + content_window_height
        visible_lines = range(self._vscroll, first_line_past_window)
        return decision, lineno, lastline, visible_lines

    def conflict_visibility(self, conflict: int) -> ConflictVisibility:
        _, lineno, lastline, visible_lines = self._conflict_position_info(conflict)
        if lineno in visible_lines and lastline in visible_lines:
            return ConflictVisibility.WITHIN_WINDOW
        if lineno in visible_lines:
            return ConflictVisibility.PARTIALLY_BELOW_WINDOW
        if lastline in visible_lines:
            return ConflictVisibility.PARTIALLY_ABOVE_WINDOW
        if lineno >= visible_lines.stop:
            return ConflictVisibility.BELOW_WINDOW
        if lastline < visible_lines.start:
            return ConflictVisibility.ABOVE_WINDOW
        return ConflictVisibility.OVERFLOWS_WINDOW

    def conflict_is_visible(self, conflict: int) -> bool:
        try:
            return self.conflict_visibility(conflict).is_visible
        except KeyError:
            return False

    def visible_conflicts(self) -> Generator[int]:
        for i in self._merge_output.decision_chunk_indices.keys():
            if self.conflict_is_visible(i):
                yield i

    def _fully_resolved(self) -> bool:
        return self._merge_output.fully_resolved()

    def swap_resolutions(self, conflict: int) -> None:
        decision = self._merge_output.get_decision(conflict)
        if decision.resolution == Resolution.USE_A_FIRST:
            self.resolve(conflict, Resolution.USE_B_FIRST)
        elif decision.resolution == Resolution.USE_B_FIRST:
            self.resolve(conflict, Resolution.USE_A_FIRST)
        elif decision.resolution == Resolution.USE_A:
            self.resolve(conflict, Resolution.USE_B)
        elif decision.resolution == Resolution.USE_B:
            self.resolve(conflict, Resolution.USE_A)

    def toggle_resolution(self, conflict: int, resolution: Resolution) -> None:
        decision = self._merge_output.get_decision(conflict)
        if resolution in decision.resolution:
            new_resolution = decision.resolution - resolution
        else:
            try:
                new_resolution = decision.resolution + resolution
            except ArithmeticError:
                new_resolution = resolution
        self.resolve(conflict, new_resolution)

    def resolve(self, conflict: int, resolution: Resolution, edit: Optional[Edit] = None) -> None:
        pre_visibility = self.conflict_visibility(conflict)
        decision_chunk_index = self._merge_output.decision_chunk_indices[conflict]
        decision = self._merge_output.get_decision(conflict)
        assert(isinstance(decision, Decision))

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
            # TODO: Detect when resolution is equal to one of the stock ones,
            # rewrite resolution to be one of those instead
        else:
            assert(edit is None)

            if decision.resolution == Resolution.EDITED:
                result = self._parent.show_dialog(
                    'Resolution has been edited externally.\n\n'
                    f'Discard edits and change resolution to "{resolution.value}"?',
                    '(Y)es/(N)o', 'yn',
                    color=ColorPair.DIALOG_WARNING
                )
                if result != 'y':
                    return

                try:
                    self._merge_output.edited_text_chunks[decision_chunk_index - 1] = None
                except IndexError:
                    pass
                try:
                    self._merge_output.edited_text_chunks[decision_chunk_index + 1] = None
                except IndexError:
                    pass

        decision.resolution = resolution
        self._resize_content(self._merge_output.height, self._merge_output.width)
        if self._fully_resolved():
            self._color = self._resolved_color
        else:
            self._color = self._unresolved_color
        self._draw_merge_output()
        post_visibility = self.conflict_visibility(conflict)
        scroll_hint = None
        if post_visibility.is_visible:
            pass
        elif pre_visibility.is_visible and not post_visibility.is_visible:
            scroll_hint = (ScrollSnap.TOP, ScrollSnap.BOTTOM)
        elif post_visibility == ConflictVisibility.ABOVE_WINDOW:
            scroll_hint = (ScrollSnap.CENTER, ScrollSnap.BOTTOM)
        elif post_visibility == ConflictVisibility.BELOW_WINDOW:
            scroll_hint = (ScrollSnap.CENTER, ScrollSnap.BOTTOM)

        if scroll_hint:
            self.scroll_to_conflict(conflict, scroll_hint)
        else:
            self._draw()

    def _draw_merge_output(self) -> None:
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
    UNRESOLVED = "Unresolved"
    USE_A = "Use changes from file A"
    USE_B = "Use changes from file B"
    USE_A_FIRST = "Combine changes, file A first"
    USE_B_FIRST = "Combine changes, file B first"
    USE_BASE = "Ignore changes from both A and B"
    EDITED = "Edited externally"

    def can_contain(self, item: Resolution) -> bool:
        if self in (Resolution.USE_A_FIRST, Resolution.USE_B_FIRST):
            if item in (Resolution.USE_A, Resolution.USE_B):
                return True
        return False

    def __contains__(self, item: Resolution) -> bool:
        if self.can_contain(item):
            return True
        if self == item:
            return True
        return False

    def __sub__(self, other: Resolution) -> Resolution:
        if self.can_contain(other):
            return Resolution.USE_B if other == Resolution.USE_A else Resolution.USE_A
        if self == other:
            return Resolution.UNRESOLVED
        raise ArithmeticError(f"Can't remove {other} from {self}")

    def __add__(self, other: Resolution) -> Resolution:
        if self == Resolution.USE_A and other == Resolution.USE_B:
            return Resolution.USE_A_FIRST
        if self == Resolution.USE_B and other == Resolution.USE_A:
            return Resolution.USE_B_FIRST
        if self == other or other in self:
            return self
        if self in other:
            return other
        if self in (Resolution.UNRESOLVED, Resolution.USE_BASE):
            return other
        raise ArithmeticError(f"Can't add {other} to {self}")

class ColorPair(IntEnum):
    DIFF_REMOVED = 1
    DIFF_ADDED = auto()
    A = auto()
    B = auto()
    BASE = auto()
    UNRESOLVED = auto()
    EDITED = auto()
    DIALOG_INFO = auto()
    DIALOG_WARNING = auto()
    DIALOG_ERROR = auto()

    @classmethod
    def init(cls) -> None:
        if not curses.has_colors():
            return
        # ordinary blue is garish on almost every terminal emulator I've tried,
        # so use bright blue everywhere if possible.
        bright = 8 if curses.COLORS > 8 else 0
        curses.init_pair(cls.DIFF_REMOVED, curses.COLOR_RED, -1)
        curses.init_pair(cls.DIFF_ADDED, curses.COLOR_GREEN, -1)
        curses.init_pair(cls.A, curses.COLOR_CYAN, -1)
        curses.init_pair(cls.B, curses.COLOR_BLUE + bright, -1)
        curses.init_pair(cls.BASE, -1, -1)
        curses.init_pair(cls.UNRESOLVED, curses.COLOR_MAGENTA, -1)
        curses.init_pair(cls.EDITED, curses.COLOR_YELLOW, -1)
        curses.init_pair(cls.DIALOG_INFO, curses.COLOR_BLUE + bright, -1)
        curses.init_pair(cls.DIALOG_WARNING, curses.COLOR_YELLOW, -1)
        curses.init_pair(cls.DIALOG_ERROR, curses.COLOR_RED, -1)

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
        start_chunk: bool = True,
        end_chunk: bool = True,
    ) -> int:
        gutter_attr = curses.A_STANDOUT if selected else curses.A_BOLD
        pane_attr = curses.A_BOLD if selected else 0
        for i, line in enumerate(text or ['']):  # loop at least once for empty chunk hardrule
            if i == 0:
                this_prefix = prefix
            else:
                this_prefix = ' '
            if start_chunk and end_chunk and len(text) <= 1:
                bracket = ord(':')
            elif i == 0 and start_chunk:
                bracket = curses.ACS_ULCORNER
            elif i == len(text) - 1 and end_chunk:
                bracket = curses.ACS_LLCORNER
            else:
                bracket = curses.ACS_VLINE
            noerror(pane.gutter.addch, lineno, 0, ord(this_prefix), color.attr | gutter_attr)
            noerror(pane.gutter.addch, lineno, 1, bracket, color.attr | gutter_attr)
            if text:
                noerror(pane.content.addstr, lineno, 0, line, color.attr | pane_attr)
            else:
                pane.content.attron(color.attr | pane_attr)
                pane.content.hline(lineno, 0, 0, pane.width)
                pane.content.attroff(color.attr | pane_attr)
            lineno += 1
        return lineno

    def _draw_a(self, pane: Pane, selected: bool, lineno: int, start_chunk: bool = True, end_chunk: bool = True) -> int:
        prefix = 'A'
        return self._draw_with_gutter(pane, self.conflict.a, ColorPair.A, prefix, selected, lineno, start_chunk=start_chunk, end_chunk=end_chunk)

    def _draw_b(self, window: Pane, selected: bool, lineno: int, start_chunk: bool = True, end_chunk: bool = True) -> int:
        prefix = 'B'
        return self._draw_with_gutter(window, self.conflict.b, ColorPair.B, prefix, selected, lineno, start_chunk=start_chunk, end_chunk=end_chunk)

    def _draw_base(self, window: Pane, color: ColorPair, p: str, selected: bool, lineno: int) -> int:
        return self._draw_with_gutter(window, self.conflict.base, color, p, selected, lineno)

    def _draw_edit(self, window: Pane, selected: bool, lineno: int) -> int:
        prefix = 'E'
        return self._draw_with_gutter(window, self.edit, ColorPair.EDITED, prefix, selected, lineno)

    def draw(self, window: Pane, lineno: int, selected: bool) -> int:
        match self.resolution:
            case Resolution.UNRESOLVED:
                lineno = self._draw_base(window, ColorPair.UNRESOLVED, '?', selected, lineno)
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
                lineno = self._draw_base(window, ColorPair.BASE, 'I', selected, lineno)
            case Resolution.EDITED:
                lineno = self._draw_edit(window, selected, lineno)
        return lineno

    def lines(self, ignore_unresolved: bool = False) -> Generator[str]:
        match self.resolution:
            case Resolution.UNRESOLVED:
                if ignore_unresolved:
                    yield from self.conflict.base
                else:
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
        if self.conflict.base:
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


def wrap(text: str, width: int) -> Generator[str]:
    for line in text.split('\n'):
        if line:
            yield from textwrap.wrap(line, width)
        else:
            yield line


@dataclass
class Revision:
    filename: Optional[str]
    label: Optional[str]


class TUIMerge:
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
        self._merge_output = MergeOutput(merge)
        self._vsplit = .5
        self._hsplit = .5
        self._dragging: bool | Literal['hsplit'] | Literal['vsplit'] = False

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
        if not self._has_conflicts:
            return
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
        self._dialog.resize(*self._stdscr.getmaxyx())

    @property
    def _selected_conflict(self) -> int:
        return self._output_pane.selected_conflict

    def _select_conflict(self, n: int, scroll_minimal: bool = False) -> None:
        if not self._has_conflicts:
            return
        try:
            pre_visibility = self._output_pane.conflict_visibility(n)
            current_decision = self._merge_output.get_decision(n)
        except (KeyError, IndexError):
            return
        self._output_pane.select_conflict(n)

        self._change_panes[0].set_change(current_decision.conflict.base, current_decision.conflict.a)
        self._change_panes[1].set_change(current_decision.conflict.base, current_decision.conflict.b)

        lines, _ = self._stdscr.getmaxyx()
        max_change_height = max(pane.preferred_height for pane in self._change_panes)
        old_hsplit = self._hsplit_row
        new_hsplit = min(max_change_height, lines // 2)
        hsplit_change = new_hsplit - old_hsplit
        self._output_pane.scroll_vert(hsplit_change)
        self._move_hsplit_to(new_hsplit)
        post_visibility = self._output_pane.conflict_visibility(n)

        if scroll_minimal:
            if post_visibility in (
                ConflictVisibility.WITHIN_WINDOW,
                ConflictVisibility.PARTIALLY_BELOW_WINDOW
            ):
                return
            scroll_snap = (ScrollSnap.TOP, ScrollSnap.TOP)
        else:
            if post_visibility == ConflictVisibility.WITHIN_WINDOW:
                return
            elif not pre_visibility.is_visible:
                scroll_snap = (ScrollSnap.CENTER, ScrollSnap.TOP)
            elif post_visibility == ConflictVisibility.PARTIALLY_BELOW_WINDOW:
                scroll_snap = (ScrollSnap.BOTTOM, ScrollSnap.TOP)
            else:
                scroll_snap = (ScrollSnap.TOP, ScrollSnap.TOP)

        self._output_pane.scroll_to_conflict(
            self._selected_conflict,
            scroll_hints=scroll_snap
        )

    def _select_next_or_page_down(self) -> None:
        if not self._has_conflicts:
            self._output_pane.scroll_page(1)
            return
        selected_seen = False
        next_conflict = self._selected_conflict + 1
        if self._output_pane.conflict_is_visible(next_conflict):
            self._select_conflict(next_conflict, scroll_minimal=True)
        else:
            visible_conflicts = [*self._output_pane.visible_conflicts()]
            selected_seen = self._selected_conflict in visible_conflicts
            if visible_conflicts and not selected_seen:
                self._select_conflict(visible_conflicts[0], scroll_minimal=True)
            else:
                self._output_pane.scroll_page(1)
                visible_conflicts = [*self._output_pane.visible_conflicts()]
                if selected_seen and self._selected_conflict in visible_conflicts:
                    return
                if visible_conflicts:
                    self._select_conflict(visible_conflicts[0], scroll_minimal=True)

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

        orig_lines: list[str] = []  # will be filled in on the first iteration
        while True:
            prelude = self._get_chunk_if_text(decision_chunk_index - 1)
            epilogue = self._get_chunk_if_text(decision_chunk_index + 1)
            editor_lines = [*prelude, *selected_decision.lines(), *epilogue]
            orig_lines = orig_lines or editor_lines
            editor_program = editor()
            with NamedTemporaryFile(
                'w+', delete_on_close=False,
                prefix=tempfile_prefix(self._outfile or self._files[2].filename),
                suffix='.diff3'
            ) as editor_file:
                editor_file.writelines(f'{line}\n' for line in editor_lines)
                editor_file.close()
                curses.def_prog_mode()
                curses.endwin()
                try:
                    subprocess.run(
                        editor_program.split(' ')
                        + [f'+{len(prelude) + 1}', editor_file.name],
                        encoding=sys.getdefaultencoding(),
                        check=True
                    )
                except subprocess.CalledProcessError:
                    #TODO: error dialog
                    pass
                finally:
                    curses.reset_prog_mode()
                with open(editor_file.name, 'r') as f:
                    edited_lines = f.read().splitlines()

            if edited_lines == orig_lines:
                # FIXME: BUG: If the file is edited twice, and the second time
                # is back to the original text, the first edit will be left as
                # the resolution, rather than discarding the resolution
                # entirely.
                #
                # Can this fix be subsumed by the edit-matches-other-resolution
                # improvement I've been contemplating?
                return

            # TODO: Should the new prelude and epilogue be calculated from the
            # original prelude and epilogue chunks instead of the edited ones?
            new_prelude = list(common_prefix(prelude, edited_lines))
            new_epilogue = [*reversed([*common_prefix(reversed(epilogue), reversed(edited_lines))])]
            edit_text = edited_lines[len(new_prelude):len(edited_lines) - len(new_epilogue)]
            edit = Edit(new_prelude, edit_text, new_epilogue)
            self._output_pane.resolve(self._selected_conflict, Resolution.EDITED, edit)

            conflict_markers = has_conflict_markers(edited_lines)
            if conflict_markers:
                conflict_text = ''.join(f'    {line}\n' for line in conflict_markers)
                dialog_result = self.show_dialog(
                    'The edited resolution contains conflict markers:\n'
                    f'{conflict_text}'
                    'This suggests the conflict is not fully resolved.\n\n'
                    'Keep editing?',
                    '(Y)es/(N)o', inputs='ynqe', color = ColorPair.DIALOG_WARNING,
                    esc=True, enter=False, center=False,
                )
                if dialog_result in ['y', 'e']:
                    continue
            break

    def _change_a_dim(self) -> tuple[int, int, int, int]:
        return self._hsplit_row, self._vsplit_col, 0, 0

    def _change_b_dim(self) -> tuple[int, int, int, int]:
        _, cols = self._stdscr.getmaxyx()
        return self._hsplit_row, cols - self._vsplit_col - 1, 0, self._vsplit_col + 1

    def _output_dim(self) -> tuple[int, int, int, int]:
        lines, cols = self._stdscr.getmaxyx()
        if not self._has_conflicts:
            return lines, cols, 0, 0
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

    def _update(self) -> None:
        panel.update_panels()
        curses.doupdate()
        self._banish_cursor()

    def _getch(self) -> int:
        while True:
            c = self._stdscr.getch()
            if c == curses.KEY_RESIZE:
                self._resized()
                self._update()
            else:
                return c

    def show_dialog(
        self,
        text: str,
        prompt: Optional[str],
        inputs: str,
        color: ColorPair = ColorPair.DIALOG_INFO,
        esc: bool = True,
        enter: bool = True,
        center: bool = True,
        wide: bool = False,
    ) -> str | bool:
        self._dialog.set_contents(text, prompt, center=center, wide=wide)
        self._dialog.set_color(color)
        self._dialog.resize(*self._stdscr.getmaxyx())
        self._dialog.show()
        self._update()

        waitfor = [ord(c) for c in inputs]
        escdelay = curses.get_escdelay()
        curses.set_escdelay(50)
        if esc:
            waitfor.append(27)
        if enter:
            waitfor.extend([ord(' '), ord('\n')])
        while (c := self._getch()) not in waitfor:
            pass
        self._dialog.hide()
        curses.set_escdelay(escdelay)
        if c == 27:
            return False
        if c in [ord('\n'), ord(' ')]:
            return inputs[0]
        return chr(c)

    @property
    def _has_conflicts(self) -> bool:
        return bool(self._merge_output.decision_chunk_indices)

    def run(self, stdscr: curses.window) -> None:
        self._stdscr = stdscr
        noerror(curses.use_default_colors)
        ColorPair.init()
        noerror(curses.curs_set, 0)
        curses.mousemask(curses.ALL_MOUSE_EVENTS | curses.REPORT_MOUSE_POSITION)
        curses.mouseinterval(0)

        if self._has_conflicts:
            self._change_panes = [
                ChangePane(self._files[0], 'A', 'Current', ColorPair.A, *self._change_a_dim()),
                ChangePane(self._files[1], 'B', 'Incoming', ColorPair.B, *self._change_b_dim()),
            ]
        else:
            self._change_panes = []
        self._output_pane = OutputPane(
            self,
            self._files[2],
            self._outfile,
            ColorPair.BASE,
            ColorPair.UNRESOLVED,
            self._merge_output,
            *self._output_dim()
        )

        self._panes: list[Pane] = [
            *self._change_panes,
            self._output_pane,
        ]
        self._dialog = Dialog()

        self._set_focus(len(self._panes) - 1)
        self._select_conflict(0)
        self._draw_borders()

        while True:
            self._update()
            c = self._getch()
            if c == ord('q'):
                dialog_text = 'Quit without saving?'
                if self._merge_output.fully_unresolved():
                    dialog_color = ColorPair.DIALOG_INFO
                else:
                    dialog_color = ColorPair.DIALOG_WARNING
                    dialog_text = 'Unsaved changes made.\n\n' + dialog_text
                result = self.show_dialog(
                    dialog_text,
                    '(Y)es/(N)o', 'yn',
                    color=dialog_color
                )
                if result == 'y':
                    exit(1)  # indicate to git that the merge wasn't completed
            elif c == ord('\t'):
                self._set_focus((self._focused + 1) % len(self._panes))
            elif c == curses.KEY_BTAB:
                self._set_focus((self._focused - 1) % len(self._panes))
            elif c in (curses.KEY_UP, ord('k')):
                self._panes[self._focused].scroll_vert(-1)
            elif c in (curses.KEY_DOWN, ord('j')):
                self._panes[self._focused].scroll_vert(1)
            elif c in (curses.KEY_LEFT, ord('h')):
                self._panes[self._focused].scroll_horiz(-2)
            elif c in (curses.KEY_RIGHT, ord('l')):
                self._panes[self._focused].scroll_horiz(2)
            elif c == curses.KEY_NPAGE:
                self._panes[self._focused].scroll_page(1)
            elif c == curses.KEY_PPAGE:
                self._panes[self._focused].scroll_page(-1)
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
            elif c == ord(' '):
                self._select_next_or_page_down()
            elif c == ord('p') and self._has_conflicts:
                self._select_conflict(self._selected_conflict - 1)
            elif c == ord('n') and self._has_conflicts:
                self._select_conflict(self._selected_conflict + 1)
            elif c == ord('a') and self._has_conflicts:
                self._output_pane.toggle_resolution(self._selected_conflict, Resolution.USE_A)
            elif c == ord('b') and self._has_conflicts:
                self._output_pane.toggle_resolution(self._selected_conflict, Resolution.USE_B)
            elif c == ord('A') and self._has_conflicts:
                self._output_pane.toggle_resolution(self._selected_conflict, Resolution.USE_A_FIRST)
            elif c == ord('B') and self._has_conflicts:
                self._output_pane.toggle_resolution(self._selected_conflict, Resolution.USE_B_FIRST)
            elif c == ord('x') and self._has_conflicts:
                self._output_pane.swap_resolutions(self._selected_conflict)
            elif c in (ord('i'), curses.KEY_DC) and self._has_conflicts:
                self._output_pane.toggle_resolution(self._selected_conflict, Resolution.USE_BASE)
            elif c in (ord('u'), curses.KEY_BACKSPACE) and self._has_conflicts:
                self._output_pane.resolve(self._selected_conflict, Resolution.UNRESOLVED)
            elif c == ord('e') and self._has_conflicts:
                self._edit_selected_conflict()
            elif c in (ord('d'), ord('v')):
                self._view_diff()
            elif c in (ord('?'), ord('/'), curses.KEY_F1):
                self._show_help()
            # elif c == ord('!'):
            #     self.show_dialog('Here is a big long chunk of text for the dialog box to display. How do you think it will do with it? Let\'s find out.\n\n--Love, Dennis', '(Y)es/(N)o/(C)ancel', 'ync')
            elif c in (ord('w'), ord('s')):
                if self._save():
                    exit(0)  # indicate to git that merge was successful

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

    def _show_help(self) -> None:
        # TODO: What if the help text is longer than the terminal?
        # TODO: Run actions associated with most keys
        help_text = '\n'.join([
            'N        Jump to next conflict',
            'P        Jump to previous conflict',
            'Space    Select visible conflict or page down',
            'A        ' + Resolution.USE_A.value,
            'B        ' + Resolution.USE_B.value,
            'Shift+A  ' + Resolution.USE_A_FIRST.value,
            'Shift+B  ' + Resolution.USE_B_FIRST.value,
            'I        ' + Resolution.USE_BASE.value,
            'U        Unresolve conflict',
            'E        Open conflict in external editor',
            'D        View diff between Base and Merge',
            'X        Swap resolution order',
            'S        Save changes and quit',
            'Q        Quit without saving changes',
            'Tab      Cycle focus between panes',
            'Arrows   Scroll focused pane',
            '?        Show this help',
        ])
        self.show_dialog(help_text, None, 'cq', center=False, wide=True)

    def _save(self) -> bool:
        outfile = self._outfile or self._files[2].filename
        if not outfile:
            raise ValueError('No output filename provided')

        if self._merge_output.fully_resolved():
            dialog_text = f'Save {outfile} and quit?'
            dialog_color = ColorPair.DIALOG_INFO
        else:
            dialog_text = (
                'Some conflicts are unresolved. Saved file will contain conflict markers.\n\n'
                f'Really save {outfile} and quit?'
            )
            dialog_color = ColorPair.DIALOG_WARNING
        result = self.show_dialog(
            dialog_text,
            '(Y)es/(N)o', 'yn',
            color=dialog_color
        )
        if result != 'y':
            return False
        with open(outfile, 'w') as f:
            #FIXME: Deal properly with files that don't have newlines
            f.writelines(f'{line}\n' for line in self._merge_output.lines())
        return True

    def _view_diff(self) -> None:
        orig_rev = self._files[2]
        if not orig_rev.filename:
            raise ValueError(f'No filename available for original file {orig_rev.label or ""}')
        orig_desc = orig_rev.label or orig_rev.filename or ''
        merge_desc = self._outfile or orig_desc
        if orig_desc:
            orig_desc += ' '
        if merge_desc:
            merge_desc += ' '
        orig_desc += '(Base)'
        merge_desc += '(Merge)'

        pager_program = pager()
        # TODO: Is this too cheeky?
        looks_like_less = 'less' in pager_program

        with (
            NamedTemporaryFile('w+', delete_on_close=False) as merged_file,
            NamedTemporaryFile(
                'w+', delete_on_close=False,
                prefix=tempfile_prefix(self._files[2].filename), suffix='.diff'
            ) as diff_file
        ):
            merged_file.writelines(f'{line}\n' for line in self._merge_output.lines(ignore_unresolved=True))
            merged_file.close()

            try:
                do_diff2(
                    orig_rev.filename, merged_file.name, diff_file,
                    labels=[orig_desc, merge_desc], color=looks_like_less
                )
            except subprocess.CalledProcessError:
                # TODO: error dialog
                pass

            try:
                #TODO: if diff is empty, show error dialog instead of pager
                do_pager(diff_file.name)
            except subprocess.CalledProcessError:
                #TODO: error dialog
                pass


def do_pager(file: str, pause_curses: bool = True) -> None:
    ## work around some common pager quirks ##
    pager_env = os.environ.copy()
    pager_env['LESS'] = ' '.join([
        *[less_env for less_env in [pager_env.get('LESS')] if less_env],
        '--+no-init',
        '--+quit-if-one-screen',  # scroll single-screen contents to top
        '--+quit-at-eof',
        '--RAW-CONTROL-CHARS',
        '--clear-screen',
        '--no-lessopen',
    ])
    # force GNU more to pause at EOF
    pager_env['POSIXLY_CORRECT'] = '1'

    if pause_curses:
        curses.def_prog_mode()
        curses.endwin()
    try:
        subprocess.run(
            [pager(), file],
            encoding=sys.getdefaultencoding(),
            check=True,
            env=pager_env
        )
    finally:
        if pause_curses:
            curses.reset_prog_mode()


def CTRL(c: str) -> int:
    return ord(c.upper()) - ord('A') + 1


def hunt_for_binary(*names: str) -> str | None:
    for prefix in ('', '/usr/local/bin/', '/usr/bin/', '/bin/'):
        for name in names:
            found = shutil.which(f'{prefix}{name}')
            if found:
                return found
    return None


def getenvtool(*vars: str, try_prefixes: Optional[bool] = True) -> str | None:
    if try_prefixes:
        prefixes = ['TUIMERGE_', 'MERGE_', '']
    else:
        prefixes = ['']
    for prefix in prefixes:
        for var in vars:
            found = os.getenv(f'{prefix}{var}')
            if found:
                return found
    return None


def editor() -> str:
    return (
        getenvtool('VISUAL', 'EDITOR')
        or hunt_for_binary('sensible-editor', 'vi', 'ed')
        or 'vi'  # shrug
    )


def pager() -> str:
    return (
        getenvtool('PAGER')
        or hunt_for_binary('sensible-pager', 'less', 'more')
        or 'more'  # shrug
    )


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


A_RE    = re.compile(         r'<<<<<<< (.*)')
BASE_RE = re.compile(re.escape('|||||||') + r' (.*)')
B_RE    = re.compile(         r'=======')
END_RE  = re.compile(         r'>>>>>>> (.*)')


def has_conflict_markers(lines: list[str]) -> Iterable[str]:
    regexes = (A_RE, BASE_RE, B_RE, END_RE)
    return [
        line for regex in regexes for line in lines
        if regex.fullmatch(line)
    ]


def tokenize_merge(lines: list[str]) -> Generator[Token]:

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
        # TODO: Treat missing bases as empty
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


def merge_from_diff(
    diff: list[str],
    mine_label: str, mine: list[str],
    yours_label: str, yours: list[str]
) -> list[list[str] | Conflict]:
    return list(_merge_from_diff(diff, mine_label, mine, yours_label, yours))


def _merge_from_diff(
    diff: list[str],
    mine_label: str, mine: list[str],
    yours_label: str, yours: list[str]
) -> Generator[list[str] | Conflict]:
    HUNK_RE = re.compile(r'^@@\s+-(\d+)(?:,(\d+))?\s+\+(\d+)(?:,(\d+))?\s+@@')
    next_mine_lineno = 0
    for line in diff:
        if (m := HUNK_RE.match(line)):
            mine_lineno_raw, mine_count, yours_lineno_raw, yours_count = [
                int(n) if n else 1 for n in m.group(1, 2, 3, 4)
            ]
            mine_lineno = mine_lineno_raw - int(bool(mine_count))
            yours_lineno = yours_lineno_raw - int(bool(yours_count))
            prelude_lines = mine[next_mine_lineno:mine_lineno]
            if prelude_lines:
                yield prelude_lines
            mine_lines = mine[mine_lineno:mine_lineno + mine_count]
            yours_lines = yours[yours_lineno:yours_lineno + yours_count]
            yield Conflict("!!!BUG!!!", [], mine_label, mine_lines, yours_label, yours_lines)
            next_mine_lineno = mine_lineno + mine_count
    rest = mine[next_mine_lineno:]
    if rest:
        yield rest


def diff3() -> str:
    envtool = getenvtool('TUIMERGE_DIFF3', try_prefixes=False)
    if envtool:
        return envtool
    git = hunt_for_binary('git')
    if git:
        return f'{git} merge-file -p --zdiff3'
    diff3 = hunt_for_binary('diff3') or 'diff'
    return f'{diff3} -m'


def do_diff3(
    myfile: str, oldfile: str, yourfile: str, labels: list[str] = []
) -> list[str]:
    diff3_prog = diff3()
    label_args = flag_list('-L', labels)
    diff3_result = subprocess.run(
        diff3_prog.split(' ')
        + label_args
        + ['--', myfile, oldfile, yourfile],
        stdout=subprocess.PIPE,
        encoding=sys.getdefaultencoding(),
    )
    return diff3_result.stdout.splitlines()


def diff2() -> str:
    return (
        getenvtool('TUIMERGE_DIFF', try_prefixes=False)
        or hunt_for_binary('diff')
        or 'diff'
    )


@overload
def do_diff2(
    myfile: str,
    yourfile: str,
    outfile: IO[str],
    context: Optional[int] = None,
    labels: list[str] = [],
    color: bool = False,
) -> None:
    ...
@overload
def do_diff2(
    myfile: str,
    yourfile: str,
    outfile: Optional[None] = None,
    context: Optional[int] = None,
    labels: list[str] = [],
    color: bool = False,
) -> list[str]:
    ...
def do_diff2(
    myfile: str,
    yourfile: str,
    outfile: Optional[IO[str]] = None,
    context: Optional[int] = None,
    labels: list[str] = [],
    color: bool = False,
) -> list[str] | None:
        diff2_prog = diff2().split(' ')
        color_opts = ['--color=always'] if color else []
        context_arg = '-u' if context is None else f'-U{context}'
        label_args = flag_list('--label', labels)
        stdout = outfile or subprocess.PIPE
        diff_result = subprocess.run(
            [
                *diff2_prog, context_arg, *label_args, *color_opts,
                '--', myfile, yourfile
            ],
            stdout=stdout, stderr=subprocess.PIPE,
            encoding=sys.getdefaultencoding()
        )
        if diff_result.returncode > 1:
            if outfile:
                outfile.seek(0)
            # Retry with only POSIX arguments
            diff_result = subprocess.run(
                [*diff2_prog, context_arg, myfile, yourfile],
                stdout=stdout, stderr=subprocess.PIPE,
                encoding=sys.getdefaultencoding()
            )
            if diff_result.returncode > 1:
                diff_result.check_returncode()
        if outfile:
            return None
        return diff_result.stdout.splitlines()


def flag_list(flag: str, args: list[str]) -> list[str]:
    '''return args with flag inserted before every element'''
    return list(chain.from_iterable(zip(repeat(flag), args)))


def wrapper[**P](
    func: Callable[Concatenate[curses.window, P], None],
    *args: P.args, **kwargs: P.kwargs
) -> None:
    def wrapping(stdscr: curses.window) -> None:
        try:
            term_enable_mouse_drag()
            func(stdscr, *args, **kwargs)
        finally:
            term_enable_mouse_drag(False)

    curses.wrapper(wrapping)


def tempfile_prefix(filename: Optional[str] = None) -> str:
    prefix = Path(filename).name if filename else 'tuimerge'
    return f'{prefix}-'


def main() -> None:
    parser = argparse.ArgumentParser(
        prog='tuimerge',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Terminal-based interactive 2-way/3-way merge tool'
    )
    #TODO: Add epilog about how to configure for git mergetool & pacdiff
    #TODO: Add --pacdiff to combine all pacdiff compatibility flags
    parser.add_argument('-o', '--output',
                        help='write merged result to the given file instead of BASE')
    parser.add_argument('-w', '--write-to-version', choices=['current', 'base', 'incoming'], default='base',
                        help='which input file to save output to; OUTPUT, if set, overrides this')
    parser.add_argument('-n', '--view-only', action='store_true',
                        help='display the raw diff or zdiff3 output instead of opening tuimerge')
    parser.add_argument('-x', '--exchange', action='store_true',
                        help='exchange CURRENT and INCOMING')
    parser.add_argument('-L', '--label', action='append', default=[],
                        help='use LABEL instead of the file name (can be repeated up to 3 times)')
    parser.add_argument('-3', '--three', action='store_true',
                        help='if called with fewer than 3 files, run with --view-only')
    parser.add_argument('CURRENT',
                        help='new local revision of a file, a.k.a. "MYFILE"')
    parser.add_argument('BASE', nargs='?',
                        help='original file from which CURRENT and INCOMING diverged, a.k.a. "OLDFILE"')
    parser.add_argument('INCOMING',
                        help='incoming revision of a file, a.k.a. "YOURFILE"')
    args = parser.parse_args()

    labels: list[str] = args.label
    label_iter = iter(labels)

    myfile = Revision(args.CURRENT, next(label_iter, None))
    oldfile = Revision(args.BASE, next(label_iter, None)) if args.BASE else None
    yourfile = Revision(args.INCOMING, next(label_iter, None))
    if args.exchange:
        myfile, yourfile = yourfile, myfile

    outfile = args.output
    if not outfile:
        if args.write_to_version == 'current':
            outfile = myfile.filename
        elif args.write_to_version == 'incoming':
            outfile = yourfile.filename
        # 'base' is indicated by outfile == None

    view_only = (
        args.view_only
        or not sys.stdout.isatty()
        or (args.three and not args.BASE)
    )

    if oldfile:
        # Promise the type system I know what I'm doing
        assert(myfile.filename and oldfile.filename and yourfile.filename)
        diff3 = do_diff3(
            myfile.filename, oldfile.filename, yourfile.filename, labels)
        if view_only:
            with NamedTemporaryFile(
                'w+', delete_on_close=False,
                prefix=tempfile_prefix(oldfile.filename), suffix='.diff3'
            ) as viewfile:
                viewfile.writelines(f'{line}\n' for line in diff3)
                viewfile.close()
                # TODO: Use color where applicable
                # TODO: Consider -F -X, unset POSIXLY_CORRECT
                do_pager(viewfile.name, pause_curses=False)
            exit()
        merge = MergeParser(diff3).parse()
    else:
        oldfile = myfile

        assert(myfile.filename and yourfile.filename)
        with (open(myfile.filename) as mf, open(yourfile.filename) as yf):
            mine = mf.read().splitlines()
            yours = yf.read().splitlines()

        diff2 = do_diff2(
            myfile.filename, yourfile.filename,
            labels=labels, context=None if view_only else 0
        )

        if view_only:
            with NamedTemporaryFile(
                'w+', delete_on_close=False,
                prefix=tempfile_prefix(myfile.filename), suffix='.diff'
            ) as viewfile:
                viewfile.writelines(f'{line}\n' for line in diff2)
                viewfile.close()
                # TODO: Use color where applicable
                # TODO: Consider -F -X, unset POSIXLY_CORRECT
                do_pager(viewfile.name, pause_curses=False)
            exit()
        mine_label = myfile.label or myfile.filename
        yours_label = yourfile.label or yourfile.filename
        assert(mine_label and yours_label)
        merge = merge_from_diff(diff2, mine_label, mine, yours_label, yours)

    tuimerge = TUIMerge(myfile, yourfile, oldfile, merge, outfile=outfile)
    wrapper(tuimerge.run)

    exit()


if __name__ == '__main__':
    main()
