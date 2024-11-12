# The Development of a Computer Vision Model for Automated Kidney Stone Segmentation and an Evaluation against Expert Surgeons  

---

Ekamjit S. Deol (co)<sup>1</sup>, Daiwei Lu (co)<sup>2</sup>, Tatsuki Koyama<sup>3</sup>, Ipek Oguz<sup>2</sup>, Nicholas L Kavoussi<sup>4</sup> 

<sup>1</sup> Saint Louis University School of Medicine, St Louis, MO, USA 

<sup>2</sup> Department of Electrical Engineering and Computer Science, Vanderbilt University, Nashville, TN, USA 

<sup>3</sup> Department of Biostatistics, Vanderbilt University, Nashville, TN, USA 

<sup>4</sup> Department of Urology, Vanderbilt University Medical Center, Nashville, TN, USA 

Submitted to JEndourology 2024


---

## Install & Requirements
This project should be operated in a conda environment. Otherwise, you will run into a slew of problems, particularly with OpenCV. 

Required install commands: 
- conda install -c conda-forge opencv 

- conda install pytorch torchvision torchaudio -c pytorch
     - *You should go to the [PyTorch website](https://pytorch.org) and perform the generated install command for conda on your machine.*

- conda install -c conda-forge seaborn

- conda install -c conda-forge pandas

- pip install comet_ml

- conda install -c conda-forge tensorboard 

- conda install -c conda-forge scikit-learn

- pip install tqdm

- pip install scikit-image

- pip install segmentation-models-pytorch

- pip install albumentations

---

### Comet Experiments

Our codebase uses [Comet](https://www.comet.com/) to log our experiments. Create an account and workspace, then follow instructions to get an api key. Create a config.yaml file with the following structure:

```
api_key: <KEY>
project_name: <PROJECT_NAME>
workspace: <WORKSPACE>
```

---

### Data
Data is presumed to be loaded in the form of cropped endoscopy images (.png/.jpg/etc) with corresponding 0-1 image segmentation masks in the following structure:

```
root
-data
--train
---images
----video1folder
-----image1.png
-----image2.png
----video2folder
-----image1.png
-----image2.png
----video2folder
-----...
---masks
----video1folder
-----image1.png
-----image2.png
----video2folder
-----image1.png
-----image2.png
--val
---...
--test
---...
```

Video folders and frame names are assumed to be constant between image and mask folders.

---

### Scripts
1. Run train.py or test.py like so

Arguments can be found in util/\_\_init\_\_.py

```
python ../<phase>.py [--<argument name> <arg value> ...]
```

*If running from root, you should uncomment any `os.chdir('..')` instructions in main script flow; the default behavior is to call from slurm dir for running on a cluster.*

2. From the project root, input the command:

```
python scripts/<script>.py [--<argument name> <arg value> ...]
```

---

## Troubleshooting
-  If the model trains on normalized inputs, then inputs for testing and synthesis must also be normalized for the model.** 


