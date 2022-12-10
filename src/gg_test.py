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
import subprocess

from helpers import output_html


def run_command(cmdLineArguments):
	subprocess.run(cmdLineArguments, stdout=subprocess.PIPE,stderr=subprocess.STDOUT, shell=True)

def xfm_txt_to_tfm(xfm_fname):
	transformMatrix = np.loadtxt(xfm_fname)
	lps2ras=np.diag([-1, -1, 1, 1])
	ras2lps=np.diag([-1, -1, 1, 1])
	transform_lps=np.dot(ras2lps, np.dot(transformMatrix,lps2ras))
	
	Parameters = " ".join([str(x) for x in np.concatenate((transform_lps[0:3,0:3].reshape(9), transform_lps[0:3,3]))])
	
	tfm_fname=os.path.splitext(xfm_fname)[0] + '.tfm'
	with open(tfm_fname, 'w') as fid:
		fid.write("#Insight Transform File V1.0\n")
		fid.write("#Transform 0\n")
		fid.write("Transform: AffineTransform_double_3_3\n")
		fid.write("Parameters: " + Parameters + "\n")
		fid.write("FixedParameters: 0 0 0\n")
	
# gad_bids_path='/home/ROBARTS/fogunsanya/graham/scratch/degad/bids/degad_bids_gad'
# nongad_bids_path='/home/ROBARTS/fogunsanya/graham/scratch/degad/bids/degad_bids_nongad'
# output_dir = '/home/ROBARTS/fogunsanya/graham/scratch/degad/derivatives/resampled'

gad_bids_path='/media/data/data/degad/bids'
output_dir = '/media/data/data/degad/derivatives'
greedy_bin=r'/opt/itksnap-4.0.0-alpha/bin/greedy'
output_html=r'output_dir'

subject_dirs = sorted(glob(f"{gad_bids_path}/sub*"))
for index, idir in enumerate(subject_dirs):
	subject = os.path.basename(idir)
	gad_img_path=f'{gad_bids_path}/{subject}/ses-pre/anat/{subject}_ses-pre_acq-gad_run-01_T1w.nii.gz'
	nongad_img_path=f'{gad_bids_path}/{subject}/ses-pre/anat/{subject}_ses-pre_acq-nongad_run-01_T1w.nii.gz'
	
	if all(os.path.exists(x) for x in {gad_img_path,nongad_img_path}):#ensure both image files exist
		output_dir_resample=os.path.join(output_dir,'resampled',subject)
		if not os.path.exists(output_dir_resample):#if path does not exist, create it
			os.makedirs(output_dir_resample)
		
		gad_resample_fname=f'{output_dir_resample}/{subject}_acq-gad_resampled_T1w.nii.gz'
		nongad_resample_fname=f'{output_dir_resample}/{subject}_acq-nongad_resampled_T1w.nii.gz'
		
		gad_orig_img=nib.load(gad_img_path)
		gad_orig_resample = image.resample_img(gad_orig_img, target_affine=np.eye(3), interpolation='linear') #resampling to 1mm x 1 mm x 1 mm resolution
		nib.save(gad_orig_resample, gad_resample_fname)
		
		nongad_orig_img=nib.load(nongad_img_path)
		nongad_orig_resample = image.resample_img(nongad_orig_img, target_affine=np.eye(3), interpolation='linear')
		nib.save(nongad_orig_resample, nongad_resample_fname)
		
		output_dir_rigid=os.path.join(output_dir,'rigid',subject)
		if not os.path.exists(output_dir_rigid):#if path does not exist, create it
			os.makedirs(output_dir_rigid)
		
		output_dir_affine=os.path.join(output_dir,'affine',subject)
		if not os.path.exists(output_dir_affine):#if path does not exist, create it
			os.makedirs(output_dir_affine)
		
		rigid_xfm_fname=os.path.join(output_dir_rigid, f"{subject}_desc-rigid_from-nongad_to-gad_xfm.txt")
		rigid_nii_fname=os.path.join(output_dir_rigid, f"{subject}_acq-nongad_desc-rigid_resliced_T1w.nii.gz")
		
		affine_xfm_fname=os.path.join(output_dir_affine, f"{subject}_desc-affine_from-nongad_to-gad_xfm.txt")
		affine_nii_fname=os.path.join(output_dir_affine, f"{subject}_acq-nongad_desc-affine_resliced_T1w.nii.gz")
		
		rigid_cmd = ' '.join([
			f'{greedy_bin} -d 3 -threads 4',
			"-a -dof 6 -ia-image-centers",
			f"-m MI",
			f'-i "{gad_resample_fname}" "{nongad_resample_fname}"',
			f'-o "{rigid_xfm_fname}"',
			f"-n 100x50x25"
		])

		apply_rigid_cmd = ' '.join([
			f"{greedy_bin} -d 3 -threads 4",
			f'-rf "{gad_resample_fname}"',
			f'-rm "{nongad_resample_fname}" "{rigid_nii_fname}"',
			f'-r "{rigid_xfm_fname}"'
		])

		reg_rigid_cmd = rigid_cmd + '&&' + apply_rigid_cmd
		run_command(reg_rigid_cmd)
		xfm_txt_to_tfm(rigid_xfm_fname)
	
		affine_cmd = ' '.join([
			f'{greedy_bin} -d 3 -threads 4',
			"-a -dof 12 -ia-image-centers",
			f"-m MI",
			f'-i "{gad_resample_fname}" "{nongad_resample_fname}"',
			f'-o "{affine_xfm_fname}"',
			f"-n 100x50x25"
		])

		apply_affine_cmd = ' '.join([
			f"{greedy_bin} -d 3 -threads 4",
			f'-rf "{gad_resample_fname}"',
			f'-rm "{nongad_resample_fname}" "{affine_nii_fname}"',
			f'-r "{affine_xfm_fname}"'
		])

		reg_affine_cmd = affine_cmd + '&&' + apply_affine_cmd
		run_command(reg_affine_cmd)
		xfm_txt_to_tfm(affine_xfm_fname)

subject_dirs = sorted(glob(f"{output_dir}/resampled/sub*"))
html_list=[]
for index, idir in enumerate(subject_dirs):
	
	foreground_nii=[]
	plot_title=[]
	isub=os.path.basename(idir)
	
	background_fname=os.path.join(output_dir, 'resampled', isub, f"{isub}_acq-gad_resampled_T1w.nii.gz")
	rigid_nii_fname=os.path.join(output_dir, 'rigid', isub, f"{isub}_acq-nongad_desc-rigid_resliced_T1w.nii.gz")
	affine_nii_fname=os.path.join(output_dir, 'affine', isub, f"{isub}_acq-nongad_desc-affine_resliced_T1w.nii.gz")
	
	if os.path.exists(rigid_nii_fname):
		foreground_nii.append(rigid_nii_fname)
		plot_title.append('Rigid Transform: Nongad to Gad')
	
	if os.path.exists(affine_nii_fname):
		foreground_nii.append(affine_nii_fname)
		plot_title.append('Affine Transform: Nongad to Gad')
	
	html_out=output_html(background_fname,foreground_nii,plot_title)
	html_list.append(html_out)

html_string = "".join(sum(html_list,[]))
message = f"""<html>
		<head></head>
		<body>{html_string}</body>
		</html>"""

output_html=os.path.join(output_dir,'registration_QC.html')
with open(output_html, "w") as fid:
	fid.write(message)