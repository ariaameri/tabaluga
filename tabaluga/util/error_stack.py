# inspired by Rust crate, error_stack

import inspect
from dataclasses import dataclass
from typing import Optional, List

import colored

from tabaluga.util.result import Result, Ok, Err


class ErrorStack(Exception):
    @dataclass(frozen=True)
    class __FrameInfo:
        file_name: str
        function_name: str
        line_no: int

    def __init__(self, error: BaseException):

        self.__error: BaseException = error
        self.__child: Optional['ErrorStack'] = None
        self.__fi = self.__get_frame_info()
        self.__printables: List[str] = []

    def __get_frame_info(self):

        # method has to be called from the depth of 1

        caller_frame_info = inspect.getouterframes(inspect.currentframe(), 3)[2]  # get the frame of 2 above!

        fi = self.__FrameInfo(
            file_name=caller_frame_info.filename,
            function_name=caller_frame_info.function,
            line_no=caller_frame_info.lineno,
        )

        return fi

    def __set_child(self, child: 'ErrorStack'):
        self.__child = child
        return self

    def change_context(self, error: BaseException) -> 'ErrorStack':
        es = (
            ErrorStack(error)
            .__set_child(self)
        )

        # correct the frame info
        es.__fi = self.__get_frame_info()

        return es

    def attach_printable(self, printable: str) -> Result['ErrorStack', Exception]:
        try:
            self.__printables.append(str(printable))
            return Ok(self)
        except Exception as e:
            return Err(e)

    def __self_str(self, indent: int = 0) -> str:

        ind = '    ' * indent
        printables = ""
        if self.__printables:
            printables = f"{ind}\u251c " + f"\n{ind}\u251c ".join(self.__printables) + "\n"
        msg = (
                  f"{ind}{colored.attr('bold')}{self.__error}{colored.attr('reset')}\n"
                  f"{ind}\u251c at {self.__fi.file_name}:{self.__fi.function_name}:{self.__fi.line_no}\n"
                  f"{printables}"
              ) + (f"{ind}\u2502\n" if self.__child is not None else "")

        return msg

    def __error_str(self) -> str:
        child_str = "" if self.__child is None else self.__child.__self_str(1)

        error_str = self.__self_str(0) + child_str

        return error_str

    def __str__(self):

        return self.__error_str()

    def print(self) -> 'ErrorStack':

        print(self.__error_str())

        return self
