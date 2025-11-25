import curses
import platform

from .util import clamp

cpython = platform.python_implementation() == 'CPython'

if cpython and hasattr(curses, 'ncurses_version'):
    from ._curses_extra import ffi, lib
    def pad_to_win(
        pad: curses.window, win: curses.window, sminline: int, smincol: int
    ) -> None:
        lib.py_pad_to_win(
            ffi.cast('void *', id(pad)),
            ffi.cast('void *', id(win)),
            sminline, smincol
        )
    define_key = lib.define_key
# TODO: pypy support
else:
    def pad_to_win(
        pad: curses.window, win: curses.window, sminline: int, smincol: int
    ) -> None:
        winlines, wincols = win.getmaxyx()
        padlines, padcols = pad.getmaxyx()
        copylines = clamp(0, winlines, padlines - sminline)
        copycols = clamp(0, wincols, padcols - smincol)
        win.erase()
        if copylines and copycols:
            pad.overwrite(
                win, sminline, smincol, 0, 0, copylines - 1, copycols - 1)

    def define_key(definition: bytes, keycode: int) -> None:
        raise NotImplementedError('Pure Python implementation of define_key() not possible')
