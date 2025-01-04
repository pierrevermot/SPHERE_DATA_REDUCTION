# NGC 1068 Data Reduction Code

This repository contains code designed to reduce imaging data obtained with SPHERE/IRDIS for the galaxy NGC 1068. While the code was initially developed for this specific dataset, it is implemented in a **generalized manner** and should be applicable to a **wider range of applications** involving similar data processing tasks.

---

## Table of Contents
1. [Features](#features)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Dependencies](#dependencies)
5. [Data Organization](#data-organization)
6. [Acknowledgments](#acknowledgments)
7. [License](#license)

---

## Features
- Generalized pipeline for reducing imaging data.
- Initial focus on SPHERE/IRDIS data for NGC 1068.
- Modular structure for adaptability to other datasets.

---

## Installation
### 1. Clone the Repository:
```
git clone https://github.com/pierrevermot/SPHERE_DATA_REDUCTION.git
cd SPHERE_DATA_REDUCTION
```

### 2. Set Up the Environment:
The file `./env.yml` contains the **exact environment** in which this code was developed. However, the dependencies are standard libraries, and the code is likely to work with minor or no modifications in other environments.

To create the environment:
```
conda env create -f env.yml
conda activate sphere_env
```

---

## Usage
### Run the Pipeline:
1. Ensure data is downloaded and organized as described in the [Data Organization](#data-organization) section.
2. Execute the main script:
```
python scripts/main.py
```
3. Output files will be stored in the results directory.

---

## Dependencies
- The code relies on standard Python libraries.
- Full dependency details can be found in `env.yml`.
- Likely compatible with Python 3.8+.

---

## Data Organization
Due to size limitations, the raw data and resulting files could not be uploaded to GitHub. Instead, a `tree.txt` file is provided in the `data/` directory, describing the folder structure and filenames.

### Downloading Data:
- Data for NGC 1068 can be retrieved from the ESO Archive:
  [ESO Data Retrieval](http://archive.eso.org/cms/eso-data/eso-data-direct-retrieval.html)
- Follow the folder structure specified in `data/tree.txt` to get the exact filenames and organize the data correctly.

---

## Acknowledgments
Special thanks to the European Southern Observatory (ESO) for providing access to the SPHERE/IRDIS dataset. Documentation of the code heavily relies on the use of **ChatGPT 4o**.

---

## License
This project is licensed under the **MIT License**. See the `LICENSE` file for details.

