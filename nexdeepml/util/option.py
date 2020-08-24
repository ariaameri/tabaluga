from __future__ import annotations
from abc import ABC, abstractmethod
from types import FunctionType
from typing import Any, Type, Callable


class Option(ABC):
    """A class to hold an optional value."""

    @abstractmethod
    def flatten(self) -> Type[Option]:
        """Returns the nested Option value within."""

        pass

    @abstractmethod
    def map(self, function: Callable[[Any], Any]) -> Type[Option]:
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
    def flat_map(self, function: Callable[[Any], Type[Option]]) -> Type[Option]:
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

    @abstractmethod
    def filter(self, function: Callable[[Any], bool]) -> Type[Option]:
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

    @abstractmethod
    def exist(self, function: Callable[[Any], bool]) -> bool:
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
    def get(self) -> Any:
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
        A boolean indicating whether an internal value exist

        """

        pass


class Some(Option):
    """A subclass of the Option class that holds a value."""

    def __init__(self, value: Any):
        """Initializer to the instance.

        Parameters
        ----------
        value : Any
            Any value to store

        """

        self._value = value

    def get(self) -> Any:
        """Returns the internal value

        Returns
        -------
        The internal value

        """

        return self._value

    def get_or_else(self, default_value: Any) -> Any:
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

    def exist(self, function: Callable[[Any], bool]) -> bool:
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

    def flatten(self) -> Type[Option]:
        """Returns the nested Option value within."""

        return self._value if isinstance(self._value, Some) else Nothing()

    def map(self, function: Callable[[Any], Any]) -> Some:
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

    def flat_map(self, function: Callable[[Any], Type[Option]]) -> Type[Option]:
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

    def filter(self, function: Callable[[Any], bool]) -> Type[Option]:
        """Method to apply the filter function to the internal value.

        Parameters
        ----------
        function : FunctionType
            Function to filter, must return boolean

        Returns
        -------
        An Option instance of the result

        """

        return self if function(self._value) is True else Nothing()

    def is_empty(self) -> bool:
        """Returns whether or not an internal value exist.

        Returns
        -------
        False because an internal value exists

        """

        return False


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

    def exist(self, function: Callable[[Any], bool]) -> bool:
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

    def flatten(self) -> Nothing:
        """Returns the nested Option value within."""

        return self

    def map(self, function: Callable[[Any], Any]) -> Nothing:
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

    def flat_map(self, function: Callable[[Any], Type[Option]]) -> Nothing:
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

    def filter(self, function: Callable[[Any], bool]) -> Nothing:
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
