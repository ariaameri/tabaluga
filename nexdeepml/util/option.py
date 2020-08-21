from __future__ import annotations
from abc import ABC, abstractmethod
from types import FunctionType
from typing import Any, Type


class Option(ABC):
    """A class to hold an optional value."""

    @abstractmethod
    def map(self, function: FunctionType) -> Type[Option]:
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
    def filter(self, function: FunctionType) -> Type[Option]:
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

    def get_or_else(self, default_value) -> Any:
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

    def map(self, function) -> Some:
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

    def filter(self, function) -> Type[Option]:
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

    def map(self, function) -> Nothing:
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

    def filter(self, function) -> Nothing:
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
