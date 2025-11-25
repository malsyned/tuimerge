from typing import Protocol
import _cffi_backend

ffi: _cffi_backend.FFI

class CData(Protocol): ...

class _Lib:
    def py_pad_to_win(
            self, pypad: CData, pywin: CData, sminline: int, smincol: int
    ) -> None: ...

    def define_key(self, definition: bytes, keycode: int) -> None: ...


lib: _Lib
