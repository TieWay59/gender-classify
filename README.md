# gender classify assignment

## quick start

Install dependencies manually or by conda(see below).

```shell
python -m flask run
```

visit localhost http://127.0.0.1:5000

## dev env

I use `conda 4.12.0` to manage the `python 3.7` environment and the dependencies file below is generated
by `conda list --export`.

You can install the dependencies by following the instructions.

However, this file includes a large number of `jupyter_*` packages, which may not be necessary for your run. You can
also add dependencies one by one in the IDE, as most of them are imported in the `app.py` file.

```txt
# This file may be used to create an environment using:
# $ conda create --name <env> --file <this file>
# platform: win-64
anyio=3.5.0=py37haa95532_0
argon2-cffi=20.1.0=py37h2bbff1b_1
attrs=22.1.0=py37haa95532_0
babel=2.11.0=py37haa95532_0
backcall=0.2.0=pyhd3eb1b0_0
beautifulsoup4=4.11.1=py37haa95532_0
blas=1.0=mkl
bleach=4.1.0=pyhd3eb1b0_0
bottleneck=1.3.5=py37h080aedc_0
brotlipy=0.7.0=py37h2bbff1b_1003
ca-certificates=2023.01.10=haa95532_0
certifi=2022.12.7=py37haa95532_0
cffi=1.15.1=py37h2bbff1b_3
charset-normalizer=2.0.4=pyhd3eb1b0_0
click=8.0.4=py37haa95532_0
colorama=0.4.6=py37haa95532_0
cryptography=39.0.1=py37h21b164f_0
debugpy=1.5.1=py37hd77b12b_0
decorator=5.1.1=pyhd3eb1b0_0
defusedxml=0.7.1=pyhd3eb1b0_0
entrypoints=0.4=py37haa95532_0
fftw=3.3.9=h2bbff1b_1
flask=2.2.2=py37haa95532_0
giflib=5.2.1=h8cc25b3_3
glib=2.69.1=h5dc1a3c_2
gst-plugins-base=1.18.5=h9e645db_0
gstreamer=1.18.5=hd78058f_0
icc_rt=2022.1.0=h6049295_2
icu=58.2=ha925a31_3
idna=3.4=py37haa95532_0
importlib-metadata=4.11.3=py37haa95532_0
importlib_metadata=4.11.3=hd3eb1b0_0
importlib_resources=5.2.0=pyhd3eb1b0_1
intel-openmp=2021.4.0=haa95532_3556
ipykernel=6.15.2=py37haa95532_0
ipython=7.31.1=py37haa95532_1
ipython_genutils=0.2.0=pyhd3eb1b0_1
ipywidgets=7.6.5=pyhd3eb1b0_1
itsdangerous=2.0.1=pyhd3eb1b0_0
jedi=0.18.1=py37haa95532_1
jinja2=3.1.2=py37haa95532_0
joblib=1.1.1=py37haa95532_0
jpeg=9e=h2bbff1b_1
json5=0.9.6=pyhd3eb1b0_0
jsonschema=4.17.3=py37haa95532_0
jupyter=1.0.0=py37haa95532_8
jupyter_client=7.4.9=py37haa95532_0
jupyter_console=6.4.4=py37haa95532_0
jupyter_core=4.11.2=py37haa95532_0
jupyter_server=1.23.4=py37haa95532_0
jupyterlab=3.5.3=py37haa95532_0
jupyterlab_pygments=0.1.2=py_0
jupyterlab_server=2.19.0=py37haa95532_0
jupyterlab_widgets=1.0.0=pyhd3eb1b0_1
krb5=1.19.4=h5b6d351_0
lerc=3.0=hd77b12b_0
libclang=14.0.6=default_hb5a9fac_1
libclang13=14.0.6=default_h8e68704_1
libdeflate=1.17=h2bbff1b_0
libffi=3.4.2=hd77b12b_6
libiconv=1.16=h2bbff1b_2
libogg=1.3.5=h2bbff1b_1
libpng=1.6.39=h8cc25b3_0
libsodium=1.0.18=h62dcd97_0
libtiff=4.5.0=h6c2663c_2
libvorbis=1.3.7=he774522_0
libwebp=1.2.4=hbc33d0d_1
libwebp-base=1.2.4=h2bbff1b_1
libxml2=2.10.3=h0ad7f3c_0
libxslt=1.1.37=h2bbff1b_0
lz4-c=1.9.4=h2bbff1b_0
markupsafe=2.1.1=py37h2bbff1b_0
matplotlib-inline=0.1.6=py37haa95532_0
mistune=0.8.4=py37hfa6e2cd_1001
mkl=2021.4.0=haa95532_640
mkl-service=2.4.0=py37h2bbff1b_0
mkl_fft=1.3.1=py37h277e83a_0
mkl_random=1.2.2=py37hf11a4ad_0
nbclassic=0.5.2=py37haa95532_0
nbclient=0.5.13=py37haa95532_0
nbconvert=6.4.4=py37haa95532_0
nbformat=5.7.0=py37haa95532_0
nest-asyncio=1.5.6=py37haa95532_0
notebook=6.5.2=py37haa95532_0
notebook-shim=0.2.2=py37haa95532_0
numexpr=2.8.4=py37h5b0cc5e_0
numpy=1.21.5=py37h7a0a035_3
numpy-base=1.21.5=py37hca35cd5_3
openssl=1.1.1t=h2bbff1b_0
packaging=22.0=py37haa95532_0
pandas=1.3.5=py37h6214cd6_0
pandocfilters=1.5.0=pyhd3eb1b0_0
parso=0.8.3=pyhd3eb1b0_0
pcre=8.45=hd77b12b_0
pickleshare=0.7.5=pyhd3eb1b0_1003
pip=22.3.1=py37haa95532_0
pkgutil-resolve-name=1.3.10=py37haa95532_0
ply=3.11=py37_0
prometheus_client=0.14.1=py37haa95532_0
prompt-toolkit=3.0.36=py37haa95532_0
prompt_toolkit=3.0.36=hd3eb1b0_0
psutil=5.9.0=py37h2bbff1b_0
pycparser=2.21=pyhd3eb1b0_0
pygments=2.11.2=pyhd3eb1b0_0
pyopenssl=23.0.0=py37haa95532_0
pyqt=5.15.7=py37hd77b12b_0
pyqt5-sip=12.11.0=py37hd77b12b_0
pyrsistent=0.18.0=py37h196d8e1_0
pysocks=1.7.1=py37_1
python=3.7.16=h6244533_0
python-dateutil=2.8.2=pyhd3eb1b0_0
python-fastjsonschema=2.16.2=py37haa95532_0
pytz=2022.7=py37haa95532_0
pywin32=305=py37h2bbff1b_0
pywinpty=2.0.10=py37h5da7b33_0
pyzmq=23.2.0=py37hd77b12b_0
qt-main=5.15.2=he8e5bd7_8
qt-webengine=5.15.9=hb9a9bb5_5
qtconsole=5.4.0=py37haa95532_0
qtpy=2.2.0=py37haa95532_0
qtwebkit=5.212=h2bbfb41_5
requests=2.28.1=py37haa95532_0
scikit-learn=1.0.2=py37hf11a4ad_1
scipy=1.7.3=py37h7a0a035_2
send2trash=1.8.0=pyhd3eb1b0_1
setuptools=65.6.3=py37haa95532_0
sip=6.6.2=py37hd77b12b_0
six=1.16.0=pyhd3eb1b0_1
sniffio=1.2.0=py37haa95532_1
soupsieve=2.3.2.post1=py37haa95532_0
sqlite=3.41.2=h2bbff1b_0
terminado=0.17.1=py37haa95532_0
testpath=0.6.0=py37haa95532_0
threadpoolctl=2.2.0=pyh0d69192_0
toml=0.10.2=pyhd3eb1b0_0
tomli=2.0.1=py37haa95532_0
tornado=6.2=py37h2bbff1b_0
traitlets=5.7.1=py37haa95532_0
typing_extensions=4.3.0=py37haa95532_0
urllib3=1.26.14=py37haa95532_0
vc=14.2=h21ff451_1
vs2015_runtime=14.27.29016=h5e58377_2
wcwidth=0.2.5=pyhd3eb1b0_0
webencodings=0.5.1=py37_1
websocket-client=0.58.0=py37haa95532_4
werkzeug=2.2.2=py37haa95532_0
wheel=0.38.4=py37haa95532_0
widgetsnbextension=3.5.2=py37haa95532_0
win_inet_pton=1.1.0=py37haa95532_0
wincertstore=0.2=py37haa95532_2
winpty=0.4.3=4
xz=5.4.2=h8cc25b3_0
zeromq=4.3.4=hd77b12b_0
zipp=3.11.0=py37haa95532_0
zlib=1.2.13=h8cc25b3_0
zstd=1.5.5=hd43e919_0
```

## credits

The model is highly inspired
by https://www.kaggle.com/code/santosh012/gender-classification-with-100-accuracy, and
also I adopted a similar data process workflow.