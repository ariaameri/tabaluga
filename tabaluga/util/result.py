"""
Result represents the result of an operation that can either be successful and, potentially, have the result, or can be
unsuccessful and have an error. Thus, Result is mostly a wrapper. It helps with proper propagation and handling of
errors.
"""

from abc import ABC, abstractmethod
from typing import Any, Type, Callable, TypeVar, Generic
from .option import Some, nothing, Option

T = TypeVar('T')  # Ok type
S = TypeVar('S')  # Err type


class Result(Generic[T, S], ABC):
    """Abstract Result class. It represents the result of an operation that can either succeed and have a value or
    can fail and have an error. it is abstract and its has two subclasses, OK and Err, to represent the two states."""

    @staticmethod
    def from_func(func: Callable[[Any], Any], *args, **kwargs) -> 'Result':
        """
        Builds a Result from running the given function.

        In case of throwing error by the function, an Err will be returned. In case of successful operation of the
        function, Ok will be returned.

        Parameters
        ----------
        func : Callable[[], Any]
            the function to call
        args
            the arguments to be passed to the function
        kwargs
            the key arguments to be passed to the function

        Returns
        -------
        Result
            Result of the application of the function,
        """

        try:
            return Ok(func(*args, **kwargs))
        except BaseException as error:
            return Err(error)

    @abstractmethod
    def get(self) -> Any:
        """Returns the value or error."""

        raise NotImplementedError

    @abstractmethod
    def get_err(self) -> BaseException:
        """Returns the error or raises exception."""

        raise NotImplementedError

    @abstractmethod
    def get_or_else(self, default: Any) -> Any:
        """
        Returns the value or the default value provided.

        Parameters
        ----------
        default : Any
            the default value to use

        Returns
        -------
        Any
            Returns the value it contains or the default value in case of Err

        """

        raise NotImplementedError

    @abstractmethod
    def is_ok(self) -> bool:
        """Returns true if Ok."""

        raise NotImplementedError

    @abstractmethod
    def is_err(self) -> bool:
        """Returns true if Err."""

        raise NotImplementedError

    @abstractmethod
    def contains(self, x: Any) -> bool:
        """
        Returns true if Ok contains `x`.

        Parameters
        ----------
        x : Any
            value to check

        Returns
        -------
        bool
            whether the instance has `x`
        """

        raise NotImplementedError

    @abstractmethod
    def contains_err(self, err: BaseException) -> bool:
        """
        Returns true if Err contains the given error

        Parameters
        ----------
        err : BaseException
           exception to check

        Returns
        -------
        bool
           whether the instance has the error
        """
        raise NotImplementedError

    @abstractmethod
    def ok(self) -> Option:
        """Converts to Option."""

        raise NotImplementedError

    @abstractmethod
    def err(self) -> Option:
        """Converts to Option."""

        raise NotImplementedError

    @abstractmethod
    def map(self, func: Callable[[Any], Any]) -> 'Result':
        """
        Apply a function to the value in case of OK and ignore in case of Err.

        Parameters
        ----------
        func : Callable[[Any], Any]
            function to be called on the value

        Returns
        -------
        Result
            new instance of Result
        """

        raise NotImplementedError

    @abstractmethod
    def flat_map(self, func: Callable[[Any], 'Result']) -> 'Result':
        """
        Apply a function to the value in case of OK and ignore in case of Err.

        Parameters
        ----------
        func : Callable[[Any], Result]
            function to be called on the value

        Returns
        -------
        Result
            new instance of Result
        """

        raise NotImplementedError

    @abstractmethod
    def map_err(self, func: Callable[[BaseException], BaseException]) -> 'Result':
        """
        Apply a function to the error in case of Err and ignore in case of Ok.

        Parameters
        ----------
        func : Callable[[BaseException], BaseException]
            function to be called on the error

        Returns
        -------
        Result
            new instance of Result
        """

        raise NotImplementedError

    @abstractmethod
    def expect(self, msg: str) -> Any:
        """Returns the value or raises error with the provided msg."""

        raise NotImplementedError


class Ok(Result):
    """Ok Result class. This class represents an operation that has succeeded and potentially has a value."""

    def __init__(self, value: Any):
        """
        Initializer

        Parameters
        ----------
        value : Any
            the value that this instance should hold.
        """

        self._value: Any = value

    def get(self) -> Any:
        """
        Returns the value within. If no value is provided, then raises error.

        Returns
        -------
        Any
            stored value

        """

        return self._value

    def get_err(self) -> BaseException:
        """Raises error as this does not contain any error."""

        raise ValueError(f"this instance of {self.__class__.__name__} contains no error")

    def get_or_else(self, default: Any) -> Any:
        """
        Returns the value.

        Parameters
        ----------
        default : Any
            the default value to use, unused

        Returns
        -------
        Any
            Returns the value

        """

        return self._value

    def is_ok(self) -> bool:
        """Returns true."""

        return True

    def is_err(self) -> bool:
        """Returns False."""

        return False

    def contains(self, x: Any) -> bool:
        """
        Returns true if the value within is `x`.

        Parameters
        ----------
        x : Any
            value to check

        Returns
        -------
        bool
            whether the instance has `x`
        """

        return self._value == x

    def contains_err(self, err: BaseException) -> bool:
        """
        Returns false

        Parameters
        ----------
        err : BaseException
           exception to check, ignored

        Returns
        -------
        bool
           false as we do not contain an error
        """

        return False

    def ok(self) -> Option:
        """
        Converts to Some.

        Returns
        -------
        Some
            Some value containing the value within
        """

        return Some(self._value)

    def err(self) -> Option:
        """
        Converts to Nothing.

        Returns
        -------
        Nothing
            Nothing value as this instance is not Err
        """

        return nothing

    def map(self, func : Callable[[Any], Any]) -> 'Result':
        """
        Apply a function to the value within,

        Parameters
        ----------
        func : Callable[[Any], Any]
            function to be called on the value

        Returns
        -------
        Result
            new instance of
        """

        return Ok(func(self._value))

    def flat_map(self, func: Callable[[Any], 'Result']) -> 'Result':
        """
        Apply a function to the value in case of OK and ignore in case of Err.

        Parameters
        ----------
        func : Callable[[Any], Result]
            function to be called on the value

        Returns
        -------
        Result
            new instance of Result
        """

        return func(self._value)

    def map_err(self, func : Callable[[BaseException], BaseException]) -> 'Result':
        """
        Just ignore.

        Parameters
        ----------
        func : Callable[[BaseException], BaseException]
            function to be called on the error, which in this case does not exist

        Returns
        -------
        Result
            same instance
        """

        return self

    def expect(self, msg: str) -> Any:
        """Returns the value or raises error with the provided msg."""

        return self.get()


class Err(Result):
    """Err Result class. This class represents an operation that has failed and contains an error."""

    def __init__(self, error: BaseException):
        """
        Initializer

        Parameters
        ----------
        error : BaseException
            the error of the operation that failed
        """

        self._error: BaseException = error

    def get(self) -> Any:
        """Raises an error."""

        raise ValueError(f"this instance of {self.__class__.__name__} contains no value")

    def get_err(self) -> BaseException:
        """Returns the error."""

        return self._error

    def get_or_else(self, default: Any) -> Any:
        """
        Returns the default value provided.

        Parameters
        ----------
        default : Any
            the default value to use

        Returns
        -------
        Any
            Returns the default value

        """

        return default

    def is_ok(self) -> bool:
        """Returns false."""

        return False

    def is_err(self) -> bool:
        """Returns true."""

        return True

    def contains(self, x: Any) -> bool:
        """
        Returns false as we do not contain a value

        Parameters
        ----------
        x : Any
            value to check, ignored

        Returns
        -------
        bool
            false, as we do not contain a value
        """

        return False

    def contains_err(self, err: BaseException) -> bool:
        """
        Returns true if we contain the given error

        Parameters
        ----------
        err : BaseException
           exception to check

        Returns
        -------
        bool
           whether the instance has the error
        """

        return self._error.__class__ == err.__class__ and str(err) == str(self._error)

    def ok(self) -> Option:
        """
        Converts to Nothing.

        Returns
        -------
        Nothing
            Nothing value as this instance is not Ok
        """

        return nothing

    def err(self) -> Option:
        """
        Converts to Some.

        Returns
        -------
        Some
            Some value containing the error within
        """

        return Some(self._error)

    def map(self, func: Callable[[Any], Any]) -> 'Result':
        """
        Just ignore.

        Parameters
        ----------
        func : Callable[[Any], Any]
            function to be called on the value, which does not exist in this case

        Returns
        -------
        Result
            same instance
        """

        return self

    def flat_map(self, func: Callable[[Any], 'Result']) -> 'Result':
        """
        Apply a function to the value in case of OK and ignore in case of Err.

        Parameters
        ----------
        func : Callable[[Any], Result]
            function to be called on the value

        Returns
        -------
        Result
            new instance of Result
        """

        return self

    def map_err(self, func: Callable[[BaseException], BaseException]) -> 'Result':
        """
        Apply a function to the error

        Parameters
        ----------
        func : Callable[[BaseException], BaseException]
            function to be called on the error

        Returns
        -------
        Result
            new instance of Result
        """

        return Err(func(self._error))

    def expect(self, msg: str) -> Any:
        """Returns the value or raises error with the provided msg."""

        raise ValueError(msg)