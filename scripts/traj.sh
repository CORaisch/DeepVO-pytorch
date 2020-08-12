#!/bin/bash

evo_traj kitti -p --plot_mode xz --ref=${1}/${3}.txt ${2}/out_${3}.txt
