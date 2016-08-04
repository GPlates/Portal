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
#c.NotebookApp.password = u'sha1:c4e182f257d2:7c773a0df126dc50a366b533d29aebbe8506d40b'

# Set a password if PASSWORD is set
if 'PASSWORD' in os.environ:
    from IPython.lib import passwd
    c.NotebookApp.password = passwd(os.environ['PASSWORD'])
    del os.environ['PASSWORD']

