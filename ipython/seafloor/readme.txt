docker run -t -i mchin/seafloor_litho /bin/bash
docker run -d -p 8889:8888 mchin/seafloor_litho
docker build -t mchin/seafloor_litho .

sudo docker login --username=mchin --email=michael.chin@sydney.edu.au
sudo docker push gplates/seafloor_lithology
sudo docker pull gplates/seafloor_litholog

The docker folder:
	130.56.249.211:~/workspace/seabed_lithology/docker
	windows: D:\workspaces\sea floor big data\docker\image

Troubleshooting:
	If the docker daemon cannot start, try sudo apt-get install linux-image-extra-$(uname -r)
	Use (https://github.com/NICTA/revrand) revrand commit with hash b5f23b7eac6e84e22a9d223ad9556630b618268d 
