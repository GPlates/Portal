#!/bin/bash
backdest=/home/mchin/backup/$(date +%Y_%m_%d_%H_%M_%S)
echo $backdest
mkdir $backdest
/usr/local/bin/svnadmin dump /var/apache2_data/EByteDeformingModels/ --deltas > $backdest/EByteDeformingModels_deltas.dump
/usr/local/bin/svnadmin dump /var/apache2_data/EByteRigidModels/ --deltas > $backdest/EByteRigidModels_deltas.dump
/usr/local/bin/svnadmin dump /var/svn/projects/ --deltas > $backdest/gplates_deltas.dump
pg_dump jira > $backdest/jira.sql
tar -cvzf $backdest/jira_data.tgz /var/atlassian/application-data/jira/data/
/usr/local/bin/trac-admin /var/trac/gplates/ hotcopy $backdest/trac
cp /var/atlassian/application-data/confluence/backups/`ls /var/atlassian/application-data/confluence/backups/ | tail -1` $backdest/confluence.zip
cp /var/atlassian/application-data/crowd-home/backups/`ls /var/atlassian/application-data/crowd-home/backups/ | tail -1` $backdest/crowd.zip
/opt/atlassian/fecru-2.9.1/bin/fisheyectl.sh backup -f $backdest/fisheye.zip
cp /etc/httpd/conf/httpd.conf $backdest/
cp /usr/local/apache2/conf/httpd.conf $backdest/apache2_httpd.conf
cp /home/mchin/htdigest $backdest/
cp /home/mchin/.pgpass $backdest/pgpass
cp -rf /var/apache2_data/EByteDeformingModels/hooks $backdest/hooks
cp /var/apache2_data/EByteDeformingModels/conf/svn-auth-file $backdest/
cp /home/mchin/pg_hba.conf $backdest/
scp -r $backdest/ 130.56.249.211:/mnt/backups
rm -rf $backdest/
#psql -d gplates -t -P format=unaligned -c 'show hba_file';

