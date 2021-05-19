FROM kundajelab/cuda-anaconda-base:latest



RUN apt update
RUN apt-get install -y --fix-missing \
    build-essential \
    cmake \
    gfortran \
    git \
    wget \
    curl \
    graphicsmagick \
    libgraphicsmagick1-dev \
    libatlas-base-dev \
    libavcodec-dev \
    libavformat-dev \
    libgtk2.0-dev \
    libjpeg-dev \
    liblapack-dev \
    libswscale-dev \
    pkg-config \
    python3-dev \
    python3-numpy \
    software-properties-common \
    zip \
    && apt-get clean && rm -rf /tmp/* /var/tmp/*

RUN pip install dlib

RUN conda update -n base conda

RUN conda install --channel conda-forge --channel defaults \
  _libgcc_mutex=0.1=main \
  ca-certificates=2020.4.5.1=hecc5488_0 \
  certifi=2020.4.5.1=py36h9f0ad1d_0 \
  libedit=3.1.20181209=hc058e9b_0 \
  libffi=3.2.1=hd88cf55_4 \
  libgcc-ng=9.1.0=hdf63c60_0 \
  libstdcxx-ng=9.1.0=hdf63c60_0 \
  ncurses=6.2=he6710b0_1 \
  ninja=1.10.0=hc9558a2_0 \
  openssl=1.1.1g=h516909a_0 \
  pip=20.0.2=py36_3 \
  python=3.6.7=h0371630_0 \
  python_abi=3.6=1_cp36m \
  readline=7.0=h7b6447c_5 \
  setuptools=46.4.0=py36_0 \
  sqlite=3.31.1=h62c20be_1 \
  tk=8.6.8=hbc83047_0 \
  wheel=0.34.2=py36_0 \
  xz=5.2.5=h7b6447c_0 \
  zlib=1.2.11=h7b6447c_3



RUN pip install \
    absl-py==0.9.0 \
    cachetools==4.1.0 \
    chardet==3.0.4 \
    cycler==0.10.0 \
    decorator==4.4.2 \
    future==0.18.2 \
    google-auth==1.15.0 \
    google-auth-oauthlib==0.4.1 \
    grpcio==1.29.0 \
    idna==2.9 \
    imageio==2.8.0 \
    importlib-metadata==1.6.0 \
    kiwisolver==1.2.0 \
    markdown==3.2.2 \
    matplotlib==3.2.1 \
    mxnet==1.6.0

RUN pip install \
    networkx==2.4 \
    numpy==1.18.4 \
    oauthlib==3.1.0 \
    opencv-python==4.2.0.34 \
    pillow==7.1.2 \
    protobuf==3.12.1 \
    pyasn1==0.4.8 \
    pyasn1-modules==0.2.8 \
    pyparsing==2.4.7 \
    python-dateutil==2.8.1 \
    pytorch-lightning==0.7.1 \
    pywavelets==1.1.1 \
    requests==2.23.0 \
    requests-oauthlib==1.3.0 \
    rsa==4.0 \
    scikit-image==0.17.2 \
    scipy==1.4.1 \
    six==1.15.0 \
    tensorboard==2.2.1 \
    tensorboard-plugin-wit==1.6.0.post3 \
    tensorboardx==1.9 \
    tifffile==2020.5.25

RUN pip install \
    torch==1.6.0 \
    torchvision==0.7.0 \
    tqdm==4.46.0 \
    urllib3==1.25.9 \
    werkzeug==1.0.1 \
    zipp==3.1.0 \
    pyaml==5.4.1

RUN pip install \
    dlib==19.21.1 \
    Flask==1.1.2 \
    pillow==8.1.2

WORKDIR /app