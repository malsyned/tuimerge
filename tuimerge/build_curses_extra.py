import os
import shlex
import sysconfig
from cffi import FFI
ffibuilder = FFI()


this_dir = os.path.dirname(__file__)


ffibuilder.set_source(
    'tuimerge._curses_extra',
    '#include "curses_extra.h"',
    sources=[os.path.join(this_dir, 'curses_extra.c')],
    include_dirs=[this_dir],
    extra_compile_args=shlex.split(sysconfig.get_config_var('MODULE__CURSES_CFLAGS')),
    extra_link_args=shlex.split(sysconfig.get_config_var('MODULE__CURSES_LDFLAGS')),
)


ffibuilder.cdef(
    'int py_pad_to_win(void *pypad, void *pywin, int sminline, int smincol);'
)


ffibuilder.cdef(
    'int define_key(const char *definition, int keycode);'
)


if __name__ == '__main__':
    ffibuilder.compile(verbose=True)
