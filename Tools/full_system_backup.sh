#!/bin/bash
backdest=/geoframe/mchin/backups
date=$(date +%Y_%m_%d_%H_%M_%S)
backupfile="$backdest/$date.tar.gz"
tar --exclude-from=exclude.txt -czpvf $backupfile /