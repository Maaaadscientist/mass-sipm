# A Root-based SiPM signal extractor and fitter
## Usage
```
git clone git@code.ihep.ac.cn:wanghanwen/sipm-massive.git
cd sipm-massive
. env_lcg.sh
mkdir build
cd build
cmake ..
make -j$(nproc)
cd -
```
## A simple test
Set the EOS environment
```
export EOS_MGM_URL="root://junoeos01.ihep.ac.cn"
eos cp /eos/juno/groups/TAO/taoprod/mass_test/run_data/main_run_0064/main_run_0064_ov_2.00_sipmgr_02_tile_0cd96b60_20230507_202012.data .
./bin/scan -i main_run_0064_ov_2.00_sipmgr_02_tile_0cd96b60_20230507_202012.data -c test.yaml -r 64 -v 2 -t main_tile_02 -m 200 -o test.root
```
where
- `-i, --input` : Root file as an input
- `-c, --config` : The configuration file (mandatory)
- `-r, --run` : Run number 
- `-v, --voltage` : The over voltage
- `-t, --type` : The run type, SiPM type and channel suffix 
- `-m, --maxEvents` : Maximum events to skim 
- `-o, --output` : The output file path(name) 

## Running massive jobs
Use the bash script `get_datasets.sh` to get lines for the dataset file.
```
./get_datasets.sh
```
This will create a `datasets` directory at current localtion (if it doesn't exist) and look for all the newly uploaded runs to make a scan and record all the path with `*.data` in each line and save as one .txt file per run.

Then use
```
python3 prepare_jobs datasets/main_run_0064.txt ../results/main_run_0064
```
This will create incursively a `../results/main_run_0064` directory containing all the things for a massive jobs-submission.
But there is an issue regarding the `LCG` environment and the IHEP HTCondor environment, you should open another terminal label and login to the server.
Go to the `../results/main_run_0064` directory, and do
```
./submit_jobs.sh
```
Then all the jobs are submitted to the cluster. Later you may like to check if some jobs fail and one can use
```
ls -lrth
```
to check if some output files are of extremely small size (hundreds of bytes). Use
```
./check_failed_jobs.sh 3
```
to check the `*.root` file whose size is smaller than 3 MB and resubmit corresponding jobs.

## Automatic Gaussian peaks finder and fitter
`fit_peaks.py` is a script reading the values from a TTree and quickly finding the mean values and widths of all Gaussian peaks. Simply use with
```
python3 fit_peaks.py <input_file> <tree_name> <variable_name> <num_bins> <minRange> <maxRange> <output_path>
```
The args are defined in a similar way as those with the `draw_histogram.py`.
- `<input_file>` : ROOT file as an input
- `<tree_name>` : The name of TTree
- `<variable_name>` : The name of the observable in the tree
- `<num_bins>` : Number of bins to initialize the histogram
- `<minRange>` : The lower edge of all bins to initialize the histogram
- `<maxRange>` : The upper edge of all bins to initialize the histogram 
- `<output_path>` : The output directory path that will store all the output plots and CSV files 

## Perform several fits with one script
Use
```
python3 prepare_all_jobs.py ../results/main_run_0064 ../fit_results/main_run_0064
```
This will create a `../fit_results/main_run_0064` directory that contains everything for a massive fit jobs-submission.
Similarly, use another ssh client and go to this directory to submit the jobs, by default (with no arguments)
```
./submit_fit_jobs.sh
```
will submit all the `fit*.sh` and `hist*.sh` under `jobs` directory.

## Perform the DCR fit
Once all the jobs finish, one can submit the DCR jobs (because DCR need the peak positions from the signal fit). Use the client with `LCG` environment and execute
```
python3 combine_csv.py fit_results
```
and then use the client with default server environment and do
```
./submit_fit_jobs.sh dcr
```
This will submit all the DCR fit jobs to the cluser. To examine a single DCR fit, one can use
`get_dcr.py` similarly as `fit_peaks.py` with 
```
python3 get_dcr.py <input_file> <tree_name> <variable_name> <num_bins> <minRange> <maxRange> <output_path>
```
Currently, this requires that a `fit_results_plots.csv` file is under the same directory with `get_dcr.py` and the script will read all the peak information from the signal fit to perform a DCR fit with fixed peak position and range.

## Calculate the gain and PDE
Use `get_pde.py` to derive the results with built-in functions.
By default, use
```
python get_pde.py config/parameters.yaml
```
The input YAML file is mandatory, which specifies the input parameters and SiPM SNs.
If the script gets one parameter of the serial number of the SiPM, it will perform the calculation of the specified one.
```
python get_pde.py config/parameters.yaml 11670
```
In this case, you need to make sure that the SiPM SN e.g. 11670 is in the list of provided YAML file.
