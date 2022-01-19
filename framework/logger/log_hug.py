from ..util.data_muncher import DataMuncher, DataMuncherLeaf
from ..util.data_muncher import FILTER_MODIFIERS, FILTER_OPERATIONS, FM, FO
from ..util.data_muncher import UPDATE_MODIFIERS, UPDATE_OPERATIONS, UPDATE_CONDITIONALS, UM, UO, UC
from ..util.symbols_unicode import SYMBOL_UNICODE_CONFIG as SUC
import re
import colored


class LogHug(DataMuncher):

    # Set static variables
    item_begin_symbol = lambda _: \
        f'{colored.fg("deep_sky_blue_4c")}{SUC.right_facing_armenian_eternity_sign}{colored.attr("reset")}'
    item_color = lambda _: colored.fg("deep_sky_blue_3b")
    after_item_symbol = lambda _: ''

    vertical_bar_with_color = lambda _: f''

    def __init__(self, log_dict: dict = None):
        """Initializes the class based on the input data dictionary.

        Parameters
        ----------
        log_dict : Dict
            A dictionary containing all the data
        """

        super().__init__(log_dict, LogHugLeaf)

    def str_representation_with_title(self, title: str, depth: int = -1) -> str:
        """Helper function to create a string representation of the instance.

        Parameters
        ----------
        title : str
            The title of the current instance to be printed at the top level
        depth : int, optional
            The depth until which the string representation should go down the configuration

        Returns
        -------
        String containing the representation of the configuration given

        """

        # Get the original string representation
        out_string = self.str_representation(name='', depth=depth)

        # Remove the first line and replace it
        index_of_end_first_line = out_string.index(f'\n')
        out_string = title + out_string[index_of_end_first_line:]

        return out_string

    def str_representation(self, name: str, depth: int = -1) -> str:
        """Helper function to create a string representation of the instance.

        Parameters
        ----------
        name : str
            The name of the current instance
        depth : int, optional
            The depth until which the string representation should go down the configuration

        Returns
        -------
        String containing the representation of the configuration given

        """

        # Check if we have reach the root of the recursion, i.e. depth is zero
        if depth == 0:
            return ''

        # Create the resulting string
        out_string = ''
        out_string += self._identity_str_representation(name)
        out_string += self.after_item_symbol() if depth != 1 and name != '' else ''  # Only add after_item_symbol if we want to print anything in front
        out_string += f'\n'

        # Create the string from all the children
        # First process the leaves and then the branches
        out_substring = \
            ''.join(
                item.str_representation(name=key, depth=depth-1)
                for key, item
                in sorted(self._parameters.items())
                if item.is_leaf()
            ) \
            + \
            ''.join(
                item.str_representation(name=key, depth=depth-1)
                for key, item
                in sorted(self._parameters.items())
                if item.is_branch()
            )

        # Indent the children result and add to the result
        out_string += re.sub(
            r'(^|\n)(?!$)',
            r'\1' + f'{self.vertical_bar_with_color()}' + r'\t',
            out_substring
        )

        return out_string


class LogHugLeaf(DataMuncherLeaf):

    # Set static variables
    item_begin_symbol = lambda _: SUC.horizontal_bar
    item_color = lambda _: colored.fg("light_salmon_3b")
    item_value_color = lambda _: colored.fg("orange_3")
    begin_list_symbol = lambda _: f'\x1b[38;5;81m-\033[0m'

    def _item_str_representation(self) -> str:

        return f'{self.item_value_color()}{self._value}{colored.attr("reset")}\n'
