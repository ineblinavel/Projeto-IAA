# Copyright (c) VCIP-NKU. All rights reserved.

__version__ = '0.1.0'

from typing import Tuple

short_version = __version__


def parse_version_info(version_str: str) -> Tuple:
    """Parse version info of YOLOMS."""
    version_info = []
    for x in version_str.split('.'):
        if x.isdigit():
            version_info.append(int(x))
        elif x.find('rc') != -1:
            patch_version = x.split('rc')
            version_info.append(int(patch_version[0]))
            version_info.append(f'rc{patch_version[1]}')
    return tuple(version_info)


version_info = parse_version_info(__version__)
