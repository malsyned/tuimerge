import argparse
import curses
import os
import sys
from tempfile import NamedTemporaryFile
from typing import (
    Callable,
    Concatenate,
)
from . import (
    MergeParser,
    Revision,
    TUIMerge,
    do_diff2,
    do_diff3,
    do_pager,
    merge_from_diff,
    tempfile_prefix,
)

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
