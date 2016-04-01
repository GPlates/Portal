#!/bin/bash

if [ $UID == 0 ] ; then
    # Start the notebook server
    su $NB_USER -c "env PATH=$PATH jupyter notebook $*"
else
    # Otherwise just exec the notebook
    jupyter notebook $*
fi

