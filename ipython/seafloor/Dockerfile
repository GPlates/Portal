FROM jupyter/notebook

RUN apt-get update
RUN apt-get install -yq python3-numpy python3-scipy \ 
python3-matplotlib python3-pandas libatlas-base-dev gfortran 

Run pip3 install --upgrade pip
#RUN pip3 install numpy
#RUN pip3 install scipy
RUN pip3 install --upgrade scipy
RUN pip3 install scikit-learn
RUN pip3 install parmap
RUN pip3 install six
RUN pip3 install --upgrade six

RUN cd /tmp/ && curl -O http://ab-initio.mit.edu/nlopt/nlopt-2.4.2.tar.gz && \
    tar -zxf nlopt-2.4.2.tar.gz && \
    cd nlopt-2.4.2 && \
    ./configure --enable-shared PYTHON=/usr/bin/python3 && \
    sudo make install 
    
ENV PYTHONPATH=$PYTHONPATH:/usr/local/lib/python3.4/site-packages/
ENV LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/lib/python3.4/site-packages/:/usr/local/lib

ADD . /notebooks  

