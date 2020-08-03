from .config import ConfigParser
from typing import Dict


symbols_unicode = {
    "bullet": f'\u2022',
    "broken_bar": f'\u00a6',
    "copyright_sign": f'\u00a9',
    "registered_sign": f'\u00ae',
    "degree_sign": f'\u00b0',
    "cyrillic_thousands_sign": f'\u0482',
    "right_facing_armenian_eternity_sign": f'\u058d',
    "left_facing_armenian_eternity_sign": f'\u058e',
    "horizontal_bar": f'\u2015',
    "Box_drawings_light_triple_dash_horizontal": f'\u2504',
    "Box_drawings_light_triple_dash_vertical": f'\u2506',
    "Box_drawings_light_quadruple_dash_horizontal": f'\u2508',
    "Box_drawings_light_quadruple_dash_vertical": f'\u250a',
    "heavy_teardrop_spoked_asterisk": f'\u273d',
    "sixteen_pointed_asterisk": f'\u273a',
    "eight_spoked_asterisk": f'\u2733',
    "open_centre_teardrop_spoked_asterisk": f'\u273c',
    "black_florette": f'\u273f',
    "eight_petalled_outlined_black_florette": f'\u2741',
    "snowflake": f'\u2744',
    "tight_trifoliate_snowflake": f'\u2745',
    "heavy_chevron_snowflake": f'\u2746',
    "balloon_spoked_asterisk": f'\u2749',
    "rightwards_arrow_with_tail": f'\u21a3',
    "rightwards_arrow_to_bar": f'\u21e5'
}


def symbols_unicode_modifier(symbols_unicode_dict: Dict) -> Dict:
    """Function to add \\u before all the entries in the unicode symbols input dictionary.

    Parameters
    ----------
    symbols_unicode_dict : Dict
        The dictionary containing the unicode symbols data

    Returns
    -------
    A dictionary with the correct encoding strings

    """

    def symbols_unicode_modifier_helper(value, prefix=''):
        """Helper function to create the dictionary"""

        if type(value) == dict:
            out = {}
            for key, item in value.items():
                prefix = value['_prefix'] if '_prefix' in value.keys() else ''
                out[key] = symbols_unicode_modifier_helper(item, prefix)
        else:
            out = f'{prefix}{value}'

        return out

    # Create the dictionary
    out_dict = symbols_unicode_modifier_helper(symbols_unicode_dict)

    return out_dict


SymbolParser = ConfigParser

SYMBOL_UNICODE_CONFIG = SymbolParser(symbols_unicode_modifier(symbols_unicode))

