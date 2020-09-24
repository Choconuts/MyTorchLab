import os
import time
import datetime
import json
import pprint
import types
from typing import Any

import sys
import os
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

from dnnlib.submission.submit import *


def fun(submit_config, i):
    print('hello dnnlib: %d' % i)


submit_config = SubmitConfig()
submit_config.run_desc = 'abc'
submit_config.run_dir_ignore = ['dnnlib', 'style_gan', 'tutorials']
submit_config.run_dir_root = '../tutorial/result'
submit_config.submit_target = SubmitTarget.LOCAL


if __name__ == '__main__':
    pass
    # submit_run(submit_config, 'utils.context.fun', i=123)
    print(os.path.split(curPath))

