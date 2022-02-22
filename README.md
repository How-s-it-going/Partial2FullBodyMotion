# Full-Body Motion from a Single Head-Mounted Device
Generates articulated poses of a human skeleton based on noisy streams of head and hand pose.

![Untitled](https://user-images.githubusercontent.com/18856994/155125242-84633290-812e-4a7e-921c-298102c397fa.png)

Implementation of https://www.microsoft.com/en-us/research/uploads/prod/2021/10/full_body_prediction.pdf

Youtube: https://www.youtube.com/watch?v=Gj5MBR3B5i8

## Usage
### Download AMASS Dataset
1. Access to [AMASS](https://amass.is.tue.mpg.de/download.php) and download datasets below (You may have to register).
- __KIT__
- __MPI_HDM05__
- __CMU__

  You have to choose __SMPL+H body__ data.

2. Access to [MANO](https://mano.is.tue.mpg.de/download.php) and download body data (You may have to register).
  
    Choose __Extended SMPL+H model__

### Clone this repository
~~~sh
$ git clone https://github.com/How-s-it-going/Partial2FullBodyMotion.git
~~~

### Build docker image and start up container
~~~sh
$ cd Partial2FullBodyMotion/
$ docker build -t p2fbody:latest .
$ docker run -it --gpus all -v $PWD:/workspace p2fbody:latest bash
~~~

### Run training script
~~~sh
$ python3 train/train.py
~~~

You can also use .ipynb instead of train.py.
