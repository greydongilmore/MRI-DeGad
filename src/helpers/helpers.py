#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4
import matplotlib
import nibabel as nib
import numpy as np
from nilearn import plotting
from svgutils.compose import Unit
from svgutils.transform import GroupElement, SVGFigure, fromstring
import subprocess

matplotlib.use('Agg')

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

def svg2str(display_object, dpi):
	"""Serialize a nilearn display object to string."""

	image_buf = StringIO()
	display_object.frame_axes.figure.savefig(
		image_buf, dpi=dpi, format="svg", facecolor="k", edgecolor="k"
	)
	return image_buf.getvalue()


def extract_svg(display_object, dpi=250):
	"""Remove the preamble of the svg files generated with nilearn."""
	image_svg = svg2str(display_object, dpi)

	image_svg = re.sub(' height="[0-9]+[a-z]*"', "", image_svg, count=1)
	image_svg = re.sub(' width="[0-9]+[a-z]*"', "", image_svg, count=1)
	image_svg = re.sub(
		" viewBox", ' preseveAspectRation="xMidYMid meet" viewBox', image_svg, count=1
	)
	start_tag = ""
	start_idx = image_svg.find(start_tag)
	end_tag = ""
	end_idx = image_svg.rfind(end_tag)

	# rfind gives the start index of the substr. We want this substr
	# included in our return value so we add its length to the index.
	end_idx += len(end_tag)

	return image_svg[start_idx:end_idx]


def clean_svg(fg_svgs, bg_svgs, ref=0):
	# Find and replace the figure_1 id.
	svgs = bg_svgs + fg_svgs
	roots = [f.getroot() for f in svgs]

	sizes = []
	for f in svgs:
		viewbox = [float(v) for v in f.root.get("viewBox").split(" ")]
		width = int(viewbox[2])
		height = int(viewbox[3])
		sizes.append((width, height))
	nsvgs = len([bg_svgs])

	sizes = np.array(sizes)

	# Calculate the scale to fit all widths
	width = sizes[ref, 0]
	scales = width / sizes[:, 0]
	heights = sizes[:, 1] * scales

	# Compose the views panel: total size is the width of
	# any element (used the first here) and the sum of heights
	fig = SVGFigure(Unit(f"{width}px"), Unit(f"{heights[:nsvgs].sum()}px"))

	yoffset = 0
	for i, r in enumerate(roots):
		r.moveto(0, yoffset, scale_x=scales[i])
		if i == (nsvgs - 1):
			yoffset = 0
		else:
			yoffset += heights[i]

	# Group background and foreground panels in two groups
	if fg_svgs:
		newroots = [
			GroupElement(roots[:nsvgs], {"class": "background-svg"}),
			GroupElement(roots[nsvgs:], {"class": "foreground-svg"}),
		]
	else:
		newroots = roots

	fig.append(newroots)
	fig.root.attrib.pop("width", None)
	fig.root.attrib.pop("height", None)
	fig.root.set("preserveAspectRatio", "xMidYMid meet")

	with TemporaryDirectory() as tmpdirname:
		out_file = Path(tmpdirname) / "tmp.svg"
		fig.save(str(out_file))
		# Post processing
		svg = out_file.read_text().splitlines()

	# Remove 
	if svg[0].startswith(""):
		svg = svg[1:]

	# Add styles for the flicker animation
	if fg_svgs:
		svg.insert(
			2,
			"""\
<style type="text/css">
@keyframes flickerAnimation%s { 0%% {opacity: 1;} 100%% { opacity:0; }}
.foreground-svg { animation: 1s ease-in-out 0s alternate none infinite running flickerAnimation%s;}
.foreground-svg:hover { animation-play-state: running;}
</style>"""
			% tuple([uuid4()] * 2),
		)

	return svg

def sorted_nicely(data, reverse=False):
	convert = lambda text: int(text) if text.isdigit() else text
	alphanum_key = lambda key: [convert(c) for c in re.split("([0-9]+)", key)]

	return sorted(data, key=alphanum_key, reverse=reverse)

def output_html(background_nii, foreground_nii, fig_title):
		
	isub = os.path.basename(background_nii).split("_")[0]
	
	if not isinstance(foreground_nii,list):
		foreground_nii=list(foreground_nii)
	
	if not isinstance(fig_title,list):
		fig_title=list(fig_title)
	
	assert len(fig_title) == len(foreground_nii)
	
	html_list=[]
	
	#displaying gad image as background 
	gad_img = nib.load(background_nii)
	gad_img= nib.Nifti1Image(
		gad_img.get_fdata().astype(np.float32),
		header=gad_img.header,
		affine=gad_img.affine,
	)
	
	for icnt,ifore in enumerate(foreground_nii):
		nongad_rigid = nib.load(ifore)
		nongad_rigid = nib.Nifti1Image(
			nongad_rigid.get_fdata().astype(np.float32),
			header= nongad_rigid.header,
			affine=nongad_rigid.affine,
		)
		
		plot_args_ref = {"dim": -0.5}
		
		html_list.append([f"""<center>
<h3 style="font-size:42px">{fig_title[icnt]}</h3>
</center>"""])
		
		for icut in ('x','y','z'):
			display = plotting.plot_anat( #class that can extract vector graphics from image: plotting gad image
				nongad_rigid, #nongad rigid image 
				display_mode=icut,
				draw_cross=False,
				annotate=False,
				cut_coords=(-60,-40,0,20,40,60), #taking slice close to centre, coronal, sagittal and frontal
				**plot_args_ref, # ** upacks the dict
			)
			fg_svgs = [fromstring(extract_svg(display, 300))] #rescaling for nongad rigid 
			display.close()
			
			display = plotting.plot_anat( #class that can extract vector graphics from image: plotting gad image
				gad_img, #nongad rigid image 
				display_mode=icut,
				draw_cross=False,
				annotate=False,
				cut_coords=(-60,-40,0,20,40,60), #taking slice close to centre, coronal, sagittal and frontal
				**plot_args_ref, # ** upacks the dict
			)
			bg_svgs = [fromstring(extract_svg(display, 300))] #rescaling for nongad rigid 
			display.close()
			
			final_svg= "\n".join(clean_svg(fg_svgs, bg_svgs))
			html_list.append([f"""<center>
<p style="margin : 0; padding-top:0;">{final_svg}</p>
</center>"""])
		
		if icnt != len(foreground_nii)-1:
			html_list.append(["""<center>
<hr style="height:3px;border-width:0;color:black;background-color:black;margin:15px;">
</center>"""])
	
	html_list.append(["""<center>
<hr style="height:6px;border-width:0;color:black;background-color:black;margin:30px;">
<hr style="height:6px;border-width:0;color:black;background-color:black;margin:30px;">
</center>"""])
	
	return sum(html_list,[])