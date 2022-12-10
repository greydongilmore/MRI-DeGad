#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  9 20:01:15 2022

@author: greydon
"""
from nilearn import image
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
from glob import glob
import pandas as pd
import os


# gad_bids_path='/home/ROBARTS/fogunsanya/graham/scratch/degad/bids/degad_bids_gad'
# nongad_bids_path='/home/ROBARTS/fogunsanya/graham/scratch/degad/bids/degad_bids_nongad'
# output_dir = '/home/ROBARTS/fogunsanya/graham/scratch/degad/derivatives/resampled'

gad_bids_path='/media/data/data/degad/bids'
output_dir = '/media/data/data/degad/derivatives/resampled'

subject_dirs = sorted(glob(f"{gad_bids_path}/sub*"))
for index, idir in enumerate(subject_dirs):
	subject = os.path.basename(idir)
	gad_img_path=f'{gad_bids_path}/{subject}/ses-pre/anat/{subject}_ses-pre_acq-gad_run-01_T1w.nii.gz'
	nongad_img_path=f'{gad_bids_path}/{subject}/ses-pre/anat/{subject}_ses-pre_acq-nongad_run-01_T1w.nii.gz'
	
	if all(os.path.exists(x) for x in {gad_img_path,nongad_img_path}):#ensure both image files exist
		output_dir_sub=os.path.join(output_dir,subject)
		if not os.path.exists(output_dir_sub):#if path does not exist, create it
			os.makedirs(output_dir_sub)
		
		gad_orig_img=nib.load(gad_img_path)
		gad_orig_resample = image.resample_img(gad_orig_img, target_affine=np.eye(3), interpolation='nearest') #resampling to 1mm x 1 mm x 1 mm resolution
		nib.save(gad_orig_resample, f'{output_dir_sub}/{subject}_acq-gad_resampled_T1w.nii.gz')
		
		nongad_orig_img=nib.load(nongad_img_path)
		nongad_orig_resample = image.resample_img(nongad_orig_img, target_affine=np.eye(3), interpolation='nearest')
		nib.save(nongad_orig_resample, f'{output_dir_sub}/{subject}_acq-nongad_resampled_T1w.nii.gz')
		