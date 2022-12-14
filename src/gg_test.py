#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  9 20:01:15 2022

@author: greydon
"""

from nilearn import image
import numpy as np
import nibabel as nib
import glob
import os
import pandas as pd

os.chdir('/home/greydon/Documents/GitHub/MRI-DeGad/src')
from helpers.helpers import output_html,run_command,xfm_txt_to_tfm


# gad_bids_path='/home/ROBARTS/fogunsanya/graham/scratch/degad/bids/degad_bids_gad'
# nongad_bids_path='/home/ROBARTS/fogunsanya/graham/scratch/degad/bids/degad_bids_nongad'
# output_dir = '/home/ROBARTS/fogunsanya/graham/scratch/degad/derivatives/resampled'

gad_bids_path='/home/greydon/Documents/data/degad/bids'
output_dir = '/home/greydon/Documents/data/degad/derivatives'
greedy_bin=r'/usr/local/itksnap-nightly-vtk9qt6-Linux-gcc64/itksnap-4.0.0-alpha.2-20220118-Linux-gcc64/bin/greedy'


#%%


subject_dirs = sorted(glob.glob(f"{gad_bids_path}/sub*"))
for idir in subject_dirs:
	subject = os.path.basename(idir)
	gad_img_path=f'{gad_bids_path}/{subject}/ses-pre/anat/{subject}_ses-pre_acq-gad_run-01_T1w.nii.gz'
	nongad_img_path=f'{gad_bids_path}/{subject}/ses-pre/anat/{subject}_ses-pre_acq-nongad_run-01_T1w.nii.gz'
	
	if all(os.path.exists(x) for x in {gad_img_path,nongad_img_path}):#ensure both image files exist
		output_dir_resample=os.path.join(output_dir,'resampled',subject)
		if not os.path.exists(output_dir_resample):#if path does not exist, create it
			os.makedirs(output_dir_resample)
		
		gad_resample_fname=f'{output_dir_resample}/{subject}_acq-gad_resampled_T1w.nii.gz'
		nongad_resample_fname=f'{output_dir_resample}/{subject}_acq-nongad_resampled_T1w.nii.gz'
		
		if not os.path.exists(gad_resample_fname):
			gad_orig_img=nib.load(gad_img_path)
			gad_orig_resample = image.resample_img(gad_orig_img, target_affine=np.eye(3), interpolation='linear') #resampling to 1mm x 1 mm x 1 mm resolution
			nib.save(gad_orig_resample, gad_resample_fname)
		
		if not os.path.exists(nongad_resample_fname):
			nongad_orig_img=nib.load(nongad_img_path)
			nongad_orig_resample = image.resample_img(nongad_orig_img, target_affine=np.eye(3), interpolation='linear')
			nib.save(nongad_orig_resample, nongad_resample_fname)
		
		output_dir_rigid=os.path.join(output_dir,'rigid',subject)
		if not os.path.exists(output_dir_rigid):#if path does not exist, create it
			os.makedirs(output_dir_rigid)
		
		output_dir_affine=os.path.join(output_dir,'affine',subject)
		if not os.path.exists(output_dir_affine):#if path does not exist, create it
			os.makedirs(output_dir_affine)
		
		output_dir_deform=os.path.join(output_dir,'deform',subject)
		if not os.path.exists(output_dir_deform):#if path does not exist, create it
			os.makedirs(output_dir_deform)
		
		rigid_xfm_fname=os.path.join(output_dir_rigid, f"{subject}_desc-rigid_from-nongad_to-gad_xfm.txt")
		rigid_nii_fname=os.path.join(output_dir_rigid, f"{subject}_acq-nongad_desc-rigid_resliced_T1w.nii.gz")
		
		affine_xfm_fname=os.path.join(output_dir_affine, f"{subject}_desc-affine_from-nongad_to-gad_xfm.txt")
		affine_nii_fname=os.path.join(output_dir_affine, f"{subject}_acq-nongad_desc-affine_resliced_T1w.nii.gz")
		
		deform_xfm_fname=os.path.join(output_dir_deform, f"{subject}_desc-deform_from-nongad_to-gad_xfm.nii.gz")
		deformInv_xfm_fname=os.path.join(output_dir_deform, f"{subject}_desc-deformInv_from-nongad_to-gad_xfm.nii.gz")
		deform_nii_fname=os.path.join(output_dir_deform, f"{subject}_acq-nongad_desc-deform_resliced_T1w.nii.gz")
		
		if not os.path.exists(rigid_xfm_fname):
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
		
		if not os.path.exists(affine_xfm_fname):
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
		
		if not os.path.exists(deform_xfm_fname):
			deform_cmd = ' '.join([
				f'{greedy_bin} -d 3 -threads 4',
				"-m MI",
				f'-i "{gad_resample_fname}" "{nongad_resample_fname}"',
				f'-it "{affine_xfm_fname}"',
				f'-o "{deform_xfm_fname}"',
				f'-oinv "{deformInv_xfm_fname}"',
				"-n 100x50x25",
				"-s 4.0vox 1.4vox"
			])
			
			warp_cmd = ' '.join([
				f'{greedy_bin} -d 3',
				f'-rf "{gad_resample_fname}"',
				f'-rm "{nongad_resample_fname}" "{deform_nii_fname}"',
				f'-r "{deform_xfm_fname}" "{affine_xfm_fname}"'
			])
			
			reg_deform_cmd = deform_cmd + '&&' + warp_cmd
			run_command(reg_deform_cmd)
			

#%%

participants=pd.read_excel(r'/home/greydon/graham/projects/ctb-akhanf/cfmm-bids/Khan/clinical_imaging/degad/participants.xlsx')
subject_dirs = sorted(glob.glob(f"{output_dir}/resampled/sub*"))
html_final=[]

for index, idir in enumerate(subject_dirs):
	
	foreground_nii=[]
	fig_title=[]
	isub=os.path.basename(idir)
	days_btw=participants[participants['subject']==isub]['days_btw_scans'].values[0].astype(int)
	background_nii=os.path.join(output_dir, 'resampled', isub, f"{isub}_acq-gad_resampled_T1w.nii.gz")
	rigid_nii_fname=os.path.join(output_dir, 'rigid', isub, f"{isub}_acq-nongad_desc-rigid_resliced_T1w.nii.gz")
	affine_nii_fname=os.path.join(output_dir, 'affine', isub, f"{isub}_acq-nongad_desc-affine_resliced_T1w.nii.gz")
	deform_nii_fname=os.path.join(output_dir, 'deform', isub, f"{isub}_acq-nongad_desc-deform_resliced_T1w.nii.gz")
	
# 	if os.path.exists(rigid_nii_fname):
# 		foreground_nii.append(rigid_nii_fname)
# 		fig_title.append(f'{isub}: Rigid Transform - Nongad to Gad')
# 	
	if os.path.exists(affine_nii_fname):
		foreground_nii.append(affine_nii_fname)
		fig_title.append(f'{isub}: Affine Transform - Nongad to Gad ({days_btw} days btw scans)')
	
	if os.path.exists(deform_nii_fname):
		foreground_nii.append(deform_nii_fname)
		fig_title.append(f'{isub}: Deformable Transform - Nongad to Gad ({days_btw} days btw scans)')
	
	html_final.append(output_html(background_nii,foreground_nii,fig_title))
	
	print(f"Done plotting {isub}")


html_string=''.join(sum(html_final,[]))
message = f"""<html>
		<head></head>
		<body>{html_string}</body>
		</html>"""

output_html_fname=os.path.join(output_dir,'registration_QC.html')
with open(output_html_fname, "w") as fid:
	fid.write(message)
	
