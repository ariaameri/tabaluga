from .panacea import Panacea, PanaceaLeaf
import sys
import colored


cursor_dict = {
            "hide": '\033[?25l',
            "show": '\033[?25h',
            "clear_line": '\033[2K\r',
            "clear_until_end": '\033[0J',
            "up": ['\033[', 'A'],
            "down": ['\033[', 'B'],
            "right": ['\033[', 'C'],
            "left": ['\033[', 'D'],
            "begin_next_line": ['\033[', 'E'],
            "begin_previous_line": ['\033[', 'F'],
            "save_location": '\033[s',
            "restore_location": '\033[u'
        }


class CursorModifier(Panacea):

    def __init__(self, config_dict):

        super().__init__(config_dict=config_dict, leaf_class=CursorModifierLeaf)

    def get(self, item: str, add=''):
        """Get a cursor modifying string from the this class and add 'add' if possible.

        Parameters
        ----------
        item : str
            The description of the cursor modifier to be returned
        add : Union[str, int], optional
            String to be added in the middle of the modifier, if possible

        Returns
        -------
        ANSI escape sequence corresponding to the modifier demanded

        """

        if not sys.stdout.isatty():
            return ''

        esc_sequence = super().get(item)

        if type(esc_sequence) is list:
            esc_sequence = str(add).join(esc_sequence)

        return esc_sequence


class CursorModifierLeaf(PanaceaLeaf):

    def _item_str_representation(self) -> str:

        return f'{self._value}This is some sample text{colored.attr("reset")}'


CONSOLE_CURSOR_MODIFIER = CursorModifier(cursor_dict)
