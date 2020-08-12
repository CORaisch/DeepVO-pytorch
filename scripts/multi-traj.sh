#!/bin/bash

evo_traj kitti -p --plot_mode xz --ref=${1}/${4}.txt ${2}/out_${4}.txt ${3}/out_${4}.txt
