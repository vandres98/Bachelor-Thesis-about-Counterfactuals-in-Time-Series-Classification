# Bachelor Thesis: Generating Counterfactual Explanations for Electrocardiography Classification with Native Guide
This repository is accompanying the bachelor thesis "Generating Counterfactual Explanations for Electrocardiography Classification with Native Guide", which investigates the Native Guide method from [Instance-based Counterfactual Explanations for Time Series Classification](https://arxiv.org/pdf/2009.13211.pdf) using the [PTB-XL dataset](https://www.nature.com/articles/s41597-020-0495-6). 
This  thesis  applies  the  method  on  the  explanation of  electrocardiogram  (ecg)  classification. Synchronization of the data was shown to be the most important contribution  to  the  method  that  enabled  the  generation  of  plausible  counterfactuals. 

To produce counterfactuals for the given dataset, please clone this repository and follow the instructions:

## Setup

### Install dependencies
This code runs on a linux environment with conda.

Install the dependencies to execute the code by creating a conda environment:

    conda env create -f native_guide_environment.yml
    conda activate ng_env

Use this environment for the next step.

If you don't have conda, you first need to install it on Linux using the terminal:

Download the latest shell script,
make the miniconda installation script executable and
run miniconda installation script:
    
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    chmod +x Miniconda3-latest-Linux-x86_64.sh
    ./Miniconda3-latest-Linux-x86_64.sh

## Produce counterfactuals for PTB-XL data
To generate the counterfactuals for the downloaded PTB-XL data, please run all the cells in the provided jupiter notebook [NativeGuide.ipynb](https://git.imp.fu-berlin.de/viktoa98/bachelor-thesis-about-counterfactuals-in-time-series-classification/-/blob/master/NativeGuide.ipynb) in given order. Usually you can just click on "Run All". The last cell produces the counterfactuals. To get the counterfactual of a specific query time series from the testing data you have to provide a X_test_index in the first cell. Furthermore, you can specify the class that the query should have. If both are not provided, a random query ECG sample is selected every time running the last cell.

# References

For the Native Guide method, please cite

    @article{Delany:2020Counterfactual,
    author    = {Eoin Delaney and
                Derek Greene and
                Mark T. Keane},
    title     = {Instance-Based Counterfactual Explanations for Time Series Classification},
    journal   = {CoRR},
    volume    = {abs/2009.13211},
    year      = {2020},
    url       = {https://arxiv.org/abs/2009.13211},
    archivePrefix = {arXiv},
    eprint    = {2009.13211},
    timestamp = {Wed, 30 Sep 2020 16:16:22 +0200},
    biburl    = {https://dblp.org/rec/journals/corr/abs-2009-13211.bib},
    bibsource = {dblp computer science bibliography, https://dblp.org}
    }

For the PTB-XL dataset, please cite

    @article{Wagner:2020PTBXL,
    doi = {10.1038/s41597-020-0495-6},
    url = {https://doi.org/10.1038/s41597-020-0495-6},
    year = {2020},
    publisher = {Springer Science and Business Media {LLC}},
    volume = {7},
    number = {1},
    pages = {154},
    author = {Patrick Wagner and Nils Strodthoff and Ralf-Dieter Bousseljot and Dieter Kreiseler and Fatima I. Lunze and Wojciech Samek and Tobias Schaeffter},
    title = {{PTB}-{XL},  a large publicly available electrocardiography dataset},
    journal = {Scientific Data}
    }

    @misc{Wagner2020:ptbxlphysionet,
    title={{PTB-XL, a large publicly available electrocardiography dataset}},
    author={Patrick Wagner and Nils Strodthoff and Ralf-Dieter Bousseljot and Wojciech Samek and Tobias Schaeffter},
    doi={10.13026/qgmg-0d46},
    year={2020},
    journal={PhysioNet}
    }

    @article{Goldberger2020:physionet,
    author = {Ary L. Goldberger  and Luis A. N. Amaral  and Leon Glass  and Jeffrey M. Hausdorff  and Plamen Ch. Ivanov  and Roger G. Mark  and Joseph E. Mietus  and George B. Moody  and Chung-Kang Peng  and H. Eugene Stanley },
    title = {{PhysioBank, PhysioToolkit, and PhysioNet}},
    journal = {Circulation},
    volume = {101},
    number = {23},
    pages = {e215-e220},
    year = {2000},
    doi = {10.1161/01.CIR.101.23.e215}
    }
    