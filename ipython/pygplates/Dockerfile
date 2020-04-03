FROM ubuntu:bionic
MAINTAINER Michael Chin

#Install ubuntu libraires and packages
RUN apt-get update -y
RUN apt-get -y install python3

#Download pygplates and install it
ADD pygplates_2.0_amd64.deb .
RUN apt-get install -y ./pygplates_2.0_amd64.deb
RUN rm pygplates_2.0_amd64.deb

#Set the path so python can find pygplates
ENV PYTHONPATH ${PYTHONPATH}:/usr/lib:/usr/lib/pygplates/revision26/
