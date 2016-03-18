We use Docker container to run ipython notebooks because of security concerns.

build docker image:
	docker build -t mchin/simon_docker .
run docker container and login as root:
	docker run -t -i --user root -v /mnt/data/ipython_notebooks/sandbox:/home/ipython/work/sandbox mchin/simon_docker
run ipython notebook:
	docker run -d -p 8888:8887 -v /mnt/data/ipython_notebooks/sandbox:/home/ipython/work/sandbox mchin/simon_docker /bin/bash ipython_notebook.sh
