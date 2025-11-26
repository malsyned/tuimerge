import os
import shlex
import sysconfig
from cffi import FFI
ffibuilder = FFI()


this_dir = os.path.dirname(__file__)
cflags = shlex.split(sysconfig.get_config_var('MODULE__CURSES_CFLAGS') or '')
ldflags = shlex.split(sysconfig.get_config_var('MODULE__CURSES_LDFLAGS') or '')


ffibuilder.set_source(
    'tuimerge._curses_extra',
    '#include "curses_extra.h"',
    sources=[os.path.join(this_dir, 'curses_extra.c')],
    include_dirs=[this_dir],
    extra_compile_args=cflags,
    extra_link_args=ldflags,
)


ffibuilder.cdef(
    'int py_pad_to_win(void *pypad, void *pywin, int sminline, int smincol);'
)


ffibuilder.cdef(
    'int define_key(const char *definition, int keycode);'
)


if __name__ == '__main__':
    ffibuilder.compile(verbose=True)
