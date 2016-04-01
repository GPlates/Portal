We use Docker container to run ipython notebooks because of security concerns.

sudo service docker start

build docker image:
	docker build -t pygplates .
run docker container and login as root:
	docker run -t -i --user root -v /mnt/data/ipython_notebooks/sandbox:/home/ipython/work/sandbox pygplates
run ipython notebook:
	docker run -d -p 8888:8887 -v /mnt/data/ipython_notebooks/sandbox:/home/ipython/work/sandbox pygplates /bin/bash ipython_notebook.sh
	
In case you get the following error in docker service start log (/var/log/upstart/docker.log) on Ubuntu Linux (14.04),
[graphdriver] prior storage driver "aufs" failed: driver not supported
You can install linux-image-extra using the following command to fix this issue.
$ sudo apt-get install linux-image-extra-$(uname -r)