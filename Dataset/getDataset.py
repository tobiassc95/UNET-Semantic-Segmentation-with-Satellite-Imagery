"""
Filename: getDataset.py
Function:  Downloads the Massachusetts Roads Dataset and the Massachusetts Buildings Dataset.
Author: Tobias Scala
"""
import os
import urllib.request
from tqdm import tqdm
import time
import math

def download_images(input_path, output_path, dataset_split=0.1):
	dataset_paths = [os.path.join(output_path, "Train"), os.path.join(output_path, "Test")]
	dataset_types = ["Images", "Targets"]

	for dataset_path in dataset_paths:
		if not os.path.exists(dataset_path):
			os.mkdir(dataset_path)
		
		dataset_split = 1-dataset_split
		for dataset_type in dataset_types:
			input_path = os.path.join(input_path, dataset_type + ".txt")
			output_path = os.path.join(dataset_path, dataset_type)
			if not os.path.exists(output_path):
				os.mkdir(output_path)

			print("Downloading to " + output_path + '.')
			counter = 0
			with open(input_path, 'r') as input_files:
				input_files = input_files.readlines()
				if(dataset_split > 1-dataset_split):
					input_files = input_files[:math.ceil(dataset_split*len(input_files))] # Trainset
				else:
					input_files = input_files[math.floor((1-dataset_split)*len(input_files)):] # Testset

			for input_file in tqdm(input_files, total=len(input_files)):
				if(input_file[-1] == '\n'):
					input_file = input_file[:-1] # The EOL (end of line) is removed.
				urllib.request.urlretrieve(input_file, os.path.join(output_path, os.path.basename(input_file)))
				counter += 1

			input_path = os.path.dirname(input_path) # Head of the path.
			output_path = os.path.dirname(output_path) # Head of the path.
			print("Elements downloaded: {}.".format(counter))


if __name__ == '__main__':
	dataset_names = ["MassachusettsRoads", "MassachusettsBuildings"]

	for dataset_name in dataset_names:
		input_path = "Dataset/Links/{}".format(dataset_name)
		output_path = "Dataset/{}".format(dataset_name)

		if not os.path.exists(output_path):
			os.mkdir(output_path)
		
		start_time = time.time()
		download_images(input_path, output_path)
		print("Downloaded " + dataset_name + "Dataset. Total time: {} minutes.".format(round((time.time() - start_time)/60, 2)))