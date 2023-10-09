# 🚀 A Root-based SiPM Signal Extractor and Fitter

[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](#license)

Extracting and fitting SiPM charge spectrum with ease using a ROOT-based workflow, this is a dedicated framework of the SiPM mass testing for the Taishan Anti-neutrino Observatory (TAO) experiment. We are adopting 4024 SiPM tiles (= 64384 SiPM units) covering an area of 10$m^2$ of the sphere central detector operating at -50&deg;.

## 📋 Table of Contents

- [Usage](#usage)
- [A Simple Test](#a-simple-test)
- [Running Massive Jobs](#running-massive-jobs)
- [Automatic Gaussian Peaks Finder and Fitter](#automatic-gaussian-peaks-finder-and-fitter)
- [Perform Several Fits with One Script](#perform-several-fits-with-one-script)
- [Perform the DCR Fit](#perform-the-dcr-fit)
- [Calculate the Gain and PDE](#calculate-the-gain-and-pde)
- [Requirements](#requirements)
- [License](#license)

---

## 🛠 Installation and Environment Setting

```bash
git clone git@code.ihep.ac.cn:wanghanwen/sipm-massive.git
cd sipm-massive
. env_lcg.sh
mkdir build
cd build
cmake ..
make -j$(nproc)
cd -
```

Set the EOS and HEP job management environment:

```bash
export EOS_MGM_URL="root://junoeos01.ihep.ac.cn"
export PATH=/afs/ihep.ac.cn/soft/common/sysgroup/hep_job/bin:$PATH
```

## 💻 Run the analysis

There are two major scripts for the job manipulation,   
`script/prepare_all_jobs.py` and `script/check_all_jobs.py`.  
Allowed analysis types:   
- [🔳main] Convert the binary data files from EOS to root tree files at target path
- [💡light] Convert the binary data files from EOS to root tree files at target path
- [🔳signal-fit] Run after `main` analysis, perform the charge spectrum fit
- [🔳dcr] Run after `signal-fit` analysis, convert the binary data files from EOS to root tree files at target path
- [🔳dcr-fit] Run after `dcr` analysis, perform the charge spectrum fit of DCR events
- [💡light-fit] Extract the reference mu from the root trees of light runs (outdated, replaced by `light-match`)
- [🔳mainrun-light-fit] Extract the reference mu from the root trees of main runs (outdated, replaced by `main-reff` and `main-match`)
- [🔳vbd] Run after `signal-fit` analysis, perform the linear regression of the charge gains to derive the breakdown voltages
- [🔳harvest] Run after all of the analyses above being finished, generate plots and tarball everything into ROOT format
- [💡light-match] Standalone analysis for extracting reference mu values and match them to the light map
- [💡light-match-bootstrap] Uncertainty analysis for `light-match`
- [🔳main-reff] Convert the binary data files from EOS directly to reference mu values
- [🔳main-match] Perform the position matching of the reference mu in main runs to the light map
- [💡decoder] Extract decoder logs from EOS (outdated, replaced by `test/getLightLogs.py`)

🔳: should use the main run list  
💡: should use the light run list  

To prepare the job scripts for the light run:
```
python script/prepare_all_jobs.py config/simple_light_run.yaml <target-path-on-junofs> <analysis-type-from-above>
```

To prepare the job scripts for the main run:
```
python script/prepare_all_jobs.py config/simple_main_run.yaml <target-path-on-junofs> <analysis-type-from-above>
```
Then you can check a single job script at `<target-path-on-junofs>/<analysis-type-from-above>/<type_run_number>/jobs`  
To submit the jobs, run:  
```
python script/check_all_jobs.py config/simple_light_run.yaml <target-path-on-junofs> <analysis-type-from-above>
```

---

## 📦 Requirements

To set up the environment, you'll need:

### Python 3.8+
- [NumPy](https://numpy.org/)
- [PyYAML](https://pyyaml.org/)
### C++ 17
- [yaml-cpp](https://github.com/jbeder/yaml-cpp)
- [boost](https://www.boost.org/)
- [ROOT](https://root.cern/)

### 🍏 MacOS

Please refer to https://code.ihep.ac.cn/wanghanwen/light-match
<!-- MacOS installation steps go here -->

### 🐧 Ubuntu

Please refer to https://code.ihep.ac.cn/wanghanwen/light-match
<!-- Ubuntu installation steps go here -->

### 📡  Linux server (IHEP cluster)

To compile the codes, you need source the `LCG` environment once with
```
. env_lcg.sh
```
After this, you will not need the `LCG` environment locally.
For requirements of Python modules on the cluster, please refer to  
https://juno.ihep.ac.cn/~offline/Doc/user-guide/appendix/anaconda.html  
when using `conda` environment on the cluster, please remember to create a symbolic link
```
cd /junofs/users/<your-account-name>/
mkdir .local
ln -s $(pwd)/.local ~/.local
ln -s $(pwd)/.conda ~/.conda
cd -
```
to avoid limitation of the disk quota of the home path. Then use
```
source /cvmfs/juno.ihep.ac.cn/sw/anaconda/Anaconda3-2020.11-Linux-x86_64/bin/activate root622
```

---

## 📜 License

This project is licensed under the terms of the MIT License. See [LICENSE](https://code.ihep.ac.cn/wanghanwen/sipm-massive/-/raw/main/LICENSE) file for details.
