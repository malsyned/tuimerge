/*
Have to roll our own pad-to-window copy routine because ncurses copywin()
doesn't handle CJK full-width characters correctly.

> These  functions  are  described  in  X/Open Curses, Issue 4, which adds const
> qualifiers to the arguments.  It further specifies their  behavior in  the
> presence  of characters with multibyte renditions (not yet sup‐ ported in this
> implementation).
>
> curs_overlay(3X), ncurses 6.5

See also: https://mail.gnu.org/archive/html/bug-ncurses/2002-05/msg00070.html

Have to do it in C because Python's curses module doesn't expose in_wch().
*/

#include <Python.h>
#include <py_curses.h>

static int clamp(int min, int n, int max)
{
    if (n < min)
        return min;
    if (n > max)
        return max;
    return n;
}

static int getcchar_info(
    const cchar_t *wch, int *width, attr_t *attrs, short *color_pair, void *opts)
{
    int wstrlen = getcchar(wch, NULL, NULL, NULL, NULL);
    wchar_t *wstr = malloc(sizeof(wchar_t) * wstrlen);
    if (!wstr)
        return ERR;
    int r = getcchar(wch, wstr, attrs, color_pair, opts);
    *width = wcswidth(wstr, wstrlen - 1);
    free(wstr);
    return r;
}

int curses_pad_to_win(WINDOW *pad, WINDOW *win, int sminline, int smincol)
{
    const wchar_t sub_rhalf[] = L"]";
    const wchar_t sub_lhalf[] = L"[";
    int padlines, padcols, winlines, wincols, copylines, copycols;
    cchar_t wch;
    attr_t attrs;
    short color_pair;
    int wcolor_pair;
    int width;
    cchar_t wch2, wch3;
    int r = OK;

    getmaxyx(pad, padlines, padcols);
    getmaxyx(win, winlines, wincols);

    copylines = clamp(0, winlines, padlines - sminline);
    copycols = clamp(0, wincols, padcols - smincol);
    if (!copylines || !copycols)
        return 0;

    if (ERR == werase(win))
        r = ERR;

    int smaxcol = smincol + copycols;

    for (int i = 0; i < copylines; i++) {
        for (int j = 0; j < padcols; j++) {
            if (ERR == mvwin_wch(pad, sminline + i, j, &wch))
                r = ERR;
            if (ERR == getcchar_info(&wch, &width, &attrs, &color_pair, &wcolor_pair))
                r = ERR;
            if (j >= smincol) {
                if (j + width <= smaxcol) {
                    if (ERR == mvwadd_wch(win, i, j - smincol, &wch))
                        r = ERR;
                } else if (j < smaxcol) {
                    if (ERR == setcchar(&wch2, sub_lhalf, attrs | WA_REVERSE, color_pair, &wcolor_pair))
                        r = ERR;
                    if (ERR == mvwadd_wch(win, i, j - smincol, &wch2))
                        r = ERR;
                }
            }
            while (--width) {
                if (ERR == setcchar(&wch3, sub_rhalf, attrs | WA_REVERSE, color_pair, &wcolor_pair))
                    r = ERR;
                j++;
                if (j == smincol) {
                    if (ERR == mvwadd_wch(win, i, j - smincol, &wch3))
                        r = ERR;
                }
            }
        }
    }
    return r;
}

int py_pad_to_win(PyObject *pypad, PyObject *pywin, int sminline, int smincol)
{
    PyCursesWindowObject *pobj = (PyCursesWindowObject *)pypad;
    PyCursesWindowObject *wobj = (PyCursesWindowObject *)pywin;
    WINDOW *pad, *win;

    // Work around an unused variable warning
    (void)&PyCursesWindow_Type;


    if (!pobj || !wobj)
        return ERR;

    pad = pobj->win;
    win = wobj->win;

    if (!pad || !win)
        return ERR;

    return curses_pad_to_win(pad, win, sminline, smincol);
}
