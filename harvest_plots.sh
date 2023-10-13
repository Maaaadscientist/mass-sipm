#! /bin/bash

target_root=$1

#/usr/local/bin/python plotter/pde_draw.py $target_root
#/usr/local/bin/python plotter/dcr_draw.py $target_root
#/usr/local/bin/python plotter/gain_draw.py $target_root
#/usr/local/bin/python plotter/pct_draw.py $target_root
#/usr/local/bin/python plotter/vbd_diff_draw.py $target_root
python3 plotter/pde_draw.py $target_root
python3 plotter/dcr_draw.py $target_root
python3 plotter/gain_draw.py $target_root
python3 plotter/pct_draw.py $target_root
python3 plotter/vbd_diff_draw.py $target_root
