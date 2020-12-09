__version__ = '0.1.0'
__name__ = 'onlinehd'

import os
from pathlib import Path
from torch.utils.cpp_extension import load
_fasthd = load(
    name='fasthd',
    extra_cflags=['-O3'],
    is_python_module=True,
    sources=[
        os.path.join(Path(__file__).parent, 'fasthd', 'onlinehd.cpp')
    ]
)

from . import fasthd
from . import spatial
from .encoder import Encoder
from .onlinehd import OnlineHD
