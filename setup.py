from setuptools import setup

setup(
    cffi_modules=["tuimerge/build_curses_extra.py:ffibuilder"]
)
