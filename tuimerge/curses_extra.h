#ifndef CURSES_EXTRA_H
#define CURSES_EXTRA_H

#include <Python.h>
#include <curses.h>

int curses_pad_to_win(WINDOW *pad, WINDOW *win, int sminline, int smincol);
int py_pad_to_win(PyObject *pypad, PyObject *pywin, int sminline, int smincol);

#endif
