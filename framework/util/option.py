from __future__ import annotations
from abc import ABC, abstractmethod
from types import FunctionType
from typing import Any, Type, Callable, TypeVar, Generic

T = TypeVar('T')
S = TypeVar('S')


class Option(Generic[T], ABC):
    """A class to hold an optional value.

    The internal value is of type T.

    """

    @abstractmethod
    def flatten(self) -> Option[Any]:
        """Returns the nested Option value within."""

        pass

    @abstractmethod
    def map(self, function: Callable[[T], Any]) -> Option[Any]:
        """Method to apply the function to the internal value.

        Parameters
        ----------
        function : FunctionType
            Function to apply to the item within

        Returns
        -------
        An Option instance of the result

        """

        pass

    @abstractmethod
    def flat_map(self, function: Callable[[T], Option[Any]]) -> Option[Any]:
        """Method to apply the function to the internal value and return its result.

        It differs from map in that `function` in this method returns an Option itself.

        Parameters
        ----------
        function : FunctionType
            Function to apply to the item within that returns an Option

        Returns
        -------
        An Option instance of the result

        """

        pass

    def fold(self, function: Callable[[T], Any], default_value: Any) -> Any:
        """Method to apply the function to the internal value and return its result
        or returns `default_value` if the Option is empty.

        Parameters
        ----------
        function : FunctionType
            Function to apply to the item within
        default_value : Any
            The default value to return in case of empty Option

        Returns
        -------
        The result

        """

        return self.map(function).get_or_else(default_value)

    @abstractmethod
    def filter(self, function: Callable[[T], bool]) -> Option[T]:
        """Method to apply the filter function to the internal value.

        Parameters
        ----------
        function : FunctionType
            Function to filter, must return boolean

        Returns
        -------
        An Option instance of the result

        """

        pass

    def filter_not(self, function: Callable[[T], bool]) -> Option[T]:
        """Method to apply the filter function to the internal value and returns true when the Option value is non-empty
            and the predicate function is not satisfied.

        Parameters
        ----------
        function : FunctionType
            Function to filter, must return boolean

        Returns
        -------
        An Option instance of the result

        """

        return self.filter(lambda x: not function(x))

    @abstractmethod
    def exist(self, function: Callable[[T], bool]) -> bool:
        """Returns true if the Option is non-empty and predicate function is satisfied, false otherwise.

        Parameters
        ----------
        function : FunctionType
            Function predicate to test

        Returns
        -------
        A boolean containing the result

        """

        pass

    @abstractmethod
    def for_all(self, function: Callable[[T], bool]) -> bool:
        """Returns true if the predicate function is satisfied or the Option is empty, false otherwise.

        Parameters
        ----------
        function : FunctionType
            Function predicate to test

        Returns
        -------
        A boolean containing the result

        """

        pass

    @abstractmethod
    def for_each(self, function: Callable[[T], None]) -> None:
        """Applies the function `function` to the internal value if it exists.

        The function should have side-effects and not return anything.

        Parameters
        ----------
        function : FunctionType
            Function to apply

        """

        pass

    def or_else(self, default_value: Option[Any]) -> Option[Any]:
        """Returns the Option itself if non-empty or the default value that is an Option.

        Parameters
        ----------
        default_value : Option
            Default value to return in case of non-existence internal value

        Returns
        -------
        Self in case of non-empty Option or default_value

        """

        return default_value if self.is_empty() else self

    def or_else_func(self, default_func: Callable[[], Option[Any]]) -> Option[Any]:
        """Returns the Option itself if non-empty or runs the default function that returns an option.

        Parameters
        ----------
        default_func : Callable[[], Option[Any]]
            Default function to call and return in case of non-existence internal value

        Returns
        -------
        Self in case of non-empty Option or the result of running default_func

        """

        return default_func() if self.is_empty() else self

    @abstractmethod
    def get_or_else(self, default_value: Any) -> Any:
        """Returns the internal value or default_value if non-existence.

        Parameters
        ----------
        default_value : Any
            Default value to return in case of non-existence internal value

        Returns
        -------
        The internal value or the default_value

        """

        pass

    @abstractmethod
    def get(self) -> T:
        """Returns the internal value or raises an error if non-existence.

        Returns
        -------
        The internal value (or raises and error)

        """

        pass

    @abstractmethod
    def is_empty(self) -> bool:
        """Returns whether or not an internal value exist.

        Returns
        -------
        A boolean indicating whether an internal value does not exist

        """

        pass

    def is_defined(self) -> bool:
        """Returns whether or not an internal value exist.

        Returns
        -------
        A boolean indicating whether an internal value exist

        """

        return not self.is_empty()

    @abstractmethod
    def contains(self, x: Any) -> bool:
        """Tests whether the option contains a given value as an element.

        Parameters
        ----------
        x : Any
            The value to check against the internal value

        Returns
        -------
        A boolean result of whether `x` is equal to internal value

        """

        pass

    def expect(self, msg: str) -> Any:
        """
        Returns the value in case of some or raise an error with the given message in case of nothing.

        Parameters
        ----------
        msg : str
            the error message to use

        Returns
        -------
        Any
            the wrapped value

        """

        if self.is_empty():
            raise ValueError(msg)

        return self.get()

    @abstractmethod
    def zip(self, that: Option[S]) -> Option[(T, S)]:
        """
        Zips this and that option and returns the result.

        Parameters
        ----------
        that : Option[S]
            the other option

        Returns
        -------
        Option[(T, S)]
            the zipped option

        """

        raise NotImplementedError


class Some(Option):
    """A subclass of the Option class that holds a value."""

    def __init__(self, value: T):
        """Initializer to the instance.

        Parameters
        ----------
        value : Any
            Any value to store

        """

        self._value: T = value

    def get(self) -> T:
        """Returns the internal value

        Returns
        -------
        The internal value

        """

        return self._value

    def get_or_else(self, default_value: Any) -> T:
        """Returns the internal value.

        Parameters
        ----------
        default_value : Any
            Default value to return in case of non-existence internal value, does not work here

        Returns
        -------
        The internal value

        """

        return self._value

    def exist(self, function: Callable[[T], bool]) -> bool:
        """Returns the result of applying the predicate function to the internal value.

        Parameters
        ----------
        function : FunctionType
            Function predicate to test

        Returns
        -------
        A boolean containing the result

        """

        return function(self._value)

    def for_all(self, function: Callable[[T], bool]) -> bool:
        """Returns true if the predicate function is satisfied.

        Parameters
        ----------
        function : FunctionType
            Function predicate to test

        Returns
        -------
        A boolean containing the result

        """

        return function(self._value)

    def for_each(self, function: Callable[[T], None]) -> None:
        """Applies the function `function` to the internal value.

        The function should have side-effects and not return anything.


        Parameters
        ----------
        function : FunctionType
            Function to apply

        """

        function(self._value)

    def flatten(self) -> Option[Any]:
        """Returns the nested Option value within."""

        return self._value if isinstance(self._value, Some) else self

    def map(self, function: Callable[[T], Any]) -> Some:
        """Method to apply the function to the internal value.

        Parameters
        ----------
        function : FunctionType
            Function to apply to the item within

        Returns
        -------
        An Some instance of the result

        """

        return Some(function(self._value))

    def flat_map(self, function: Callable[[T], Option[Any]]) -> Option[Any]:
        """Method to apply the function to the internal value and return its result.

        It differs from map in that `function` in this method returns an Option itself.

        Parameters
        ----------
        function : FunctionType
            Function to apply to the item within that returns an Option

        Returns
        -------
        An Option instance of the result

        """

        return function(self._value)

    def filter(self, function: Callable[[T], bool]) -> Option[T]:
        """Method to apply the filter function to the internal value.

        Parameters
        ----------
        function : FunctionType
            Function to filter, must return boolean

        Returns
        -------
        An Option instance of the result

        """

        return self if function(self._value) is True else nothing

    def is_empty(self) -> bool:
        """Returns whether or not an internal value exist.

        Returns
        -------
        False because an internal value exists

        """

        return False

    def contains(self, x: Any) -> bool:
        """Tests whether the option contains a given value as an element.

        Parameters
        ----------
        x : Any
            The value to check against the internal value

        Returns
        -------
        A boolean result of whether `x` is equal to internal value

        """

        return self._value == x

    def zip(self, that: Option[S]) -> Option[(T, S)]:
        """
        Zips this and that option and returns the result. If that option is nothing, it will return nothing, else, will
        return a Some containing a tuple of both the values.

        Parameters
        ----------
        that : Option[S]
            the other option

        Returns
        -------
        Option[(T, S)]
            the zipped option

        """

        if that.is_defined():
            return Some((self._value, that._value))
        else:
            return nothing


class Nothing(Option):

    def __init__(self):
        """Initializer to the instance."""

        pass

    def get(self) -> None:
        """Raises an error as there is no internal value."""

        raise AttributeError

    def get_or_else(self, default_value: Any) -> Any:
        """Returns default_value.

        Parameters
        ----------
        default_value : Any
            Default value to return

        Returns
        -------
        The default_value

        """

        return default_value

    def exist(self, function: Callable[[T], bool]) -> bool:
        """Returns false as there is no element to satisfy the predicate.

        Parameters
        ----------
        function : FunctionType
            Function predicate to test

        Returns
        -------
        false

        """

        return False

    def for_all(self, function: Callable[[T], bool]) -> bool:
        """Returns true as the predicate is satisfied.

        Parameters
        ----------
        function : FunctionType
            Function predicate to test

        Returns
        -------
        A boolean containing the result

        """

        return True

    def for_each(self, function: Callable[[T], None]) -> None:
        """Does not do anything!

        The function passed should have side-effects and not return anything.


        Parameters
        ----------
        function : FunctionType
            Function to apply

        """

        pass

    def flatten(self) -> Nothing:
        """Returns the nested Option value within."""

        return self

    def map(self, function: Callable[[T], Any]) -> Nothing:
        """Method to apply the function to the internal value.

        Parameters
        ----------
        function : FunctionType
            Function to apply to the item within

        Returns
        -------
        The instance of Nothing

        """

        return self

    def flat_map(self, function: Callable[[T], Option[Any]]) -> Nothing:
        """Method to apply the function to the internal value and return its result.

        It differs from map in that `function` in this method returns an Option itself.

        Parameters
        ----------
        function : FunctionType
            Function to apply to the item within that returns an Option

        Returns
        -------
        An Option instance of the result

        """

        return self

    def filter(self, function: Callable[[T], bool]) -> Nothing:
        """Method to apply the filter function to the value.

        Parameters
        ----------
        function : FunctionType
            Function to filter, must return boolean

        Returns
        -------
        This instance of the class

        """

        return self

    def is_empty(self) -> bool:
        """Returns whether or not an internal value exist.

        Returns
        -------
        False because no internal values exist

        """

        return True

    def contains(self, x: Any) -> bool:
        """Tests whether the option contains a given value as an element.

        Parameters
        ----------
        x : Any
            The value to check against the internal value

        Returns
        -------
        A boolean result of whether `x` is equal to internal value

        """

        return False

    def zip(self, that: Option[S]) -> Option[(T, S)]:
        """
        returns nothing

        Parameters
        ----------
        that : Option[S]
            the other option

        Returns
        -------
        Option[(T, S)]
            nothing

        """

        return nothing


# An instance that should be passed around and imported elsewhere
nothing = Nothing()
