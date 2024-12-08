# Breast datasets preparation

This repo aims to prepare the some of the main datasets for breast anomaly classification tasks.
It includes different ways to prepare the following datasets : 

- Vindr Mammo
- CBIS-DDSM
- INbreast

## 1. Requirements

The pip dependencies for this project can also be downloaded again using :

```bash
pip install -r requirements.txt
```

## 2. Installation

First, you need to download the datasets :

- [CBIS-DDSM](https://www.cancerimagingarchive.net/collection/cbis-ddsm/)
- [INBreast](https://www.kaggle.com/datasets/tommyngx/inbreast2012)
- [Vindr-Mammo](https://physionet.org/content/vindr-mammo/)

Then just clone this repository.

## 3. Usage

You can run the preparation script with the following command and given flags : 

```bash
python run.py --name inbreast --data_dir ./data/INBreast --out_dir ./data --task lesions
```

| Flag                  | Description                                                                                                       | Default Value   |
|-----------------------|-------------------------------------------------------------------------------------------------------------------|-----------------|
| --data_dir            | The folder where the CBIS-DDSM dataset is stored                                                                  | None            |
| --out_dir             | The folder where the prepared dataset will be stored                                                              | ./data          |
| --task                | The task for which the dataset will be prepared                                                                   | 'lesions'       |
| --name                | The name of the dataset to prepare                                                                                | inbreast        |

### 3.1. Dataset task

We implemented different ways to prepare each datasets depending on the targetted classification system development.

#### CBIS-DDSM
- ```lesions```: It's the original class split, it separates the image dataset "calc" and "mass" classes.
- ```lesions-severity```: This task separates both calc and mass datasets into "benign" and "malignant".

#### INbreast
- ```lesions```: Separates the dataset in binary classes "normal" and "abnormal".
- ```birads```: Separates the dataset according to the bi-rads assessments.

#### Vindr-Mammo
- ```lesions```: Separates the dataset using the finding annotation labels, namely "no_finding", "mass" and "suspicious_calcification".
- ```birads```: Separates the dataset according to the bi-rads assessments.

### 3.2. Preprocessing Pipeline

Each dataset follows the same processing pipeline defined by :

```python
class BreastImageProcessingPipeline(BasePipeline):
    def __init__(self):
        super().__init__()
        self.operations = [
            read_dicom,
            crop_to_roi,
            truncate_normalization,
            resize_square,
            normalize_int8
        ]
```

A ```BasePipeline``` instance can be used to also fully customize the pipeline to transform the images.
The folder ```src.operations``` contains examples of functions that can be used inside the pipeline. 

### 3.3. File structure

For each dataset and task the script will create training and testing sets based on the original datasets splits. For a given task the file structure will then look like :

- 📂 data/
    - 📂 dataset_name/
        - 📂 task_name/
            - 📂 train/
                - 📂 label_0/
                    - 📄 01.png
                    - 📄 02.png
                - 📂 label_1/
                    - 📄 01.png
                    - 📄 02.png
            - 📂 test/
                - 📂 label_0/
                    - 📄 01.png
                    - 📄 02.png
                - 📂 label_1/
                    - 📄 01.png
                    - 📄 02.png

This structure can then be used by a PyTorch ImageFolder dataset in any Deep Learning project.
