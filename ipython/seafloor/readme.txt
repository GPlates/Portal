docker run -t -i mchin/seafloor_litho /bin/bash
docker run -d -p 8889:8888 mchin/seafloor_litho
docker build -t mchin/seafloor_litho .

sudo docker login --username=mchin --email=michael.chin@sydney.edu.au
sudo docker push gplates/seafloor_lithology
sudo docker pull gplates/seafloor_litholog

Troubleshooting:
	If the docker daemon cannot start, try sudo apt-get install linux-image-extra-$(uname -r)