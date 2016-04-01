# Copyright (c) Jupyter Development Team.
from jupyter_core.paths import jupyter_data_dir
import subprocess
import os
import errno
import stat
import sys
sys.path.append('/usr/lib/pygplates/revision12/')

c = get_config()
c.NotebookApp.ip = '*'
c.NotebookApp.port = 8887
c.NotebookApp.open_browser = False
c.NotebookApp.password = u'sha1:7f774ef7dbb3:072f298f9fc74728c80ff6c5ac96db4fe2ab5e01'
