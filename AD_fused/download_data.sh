#!/usr/bin/env bash
cd path # path to the dataset storage
mkdir data
cd data

echo "This script downloads datasets necessary for the hybrid model."

# Download 14_weathers_data
mkdir 14_weathers_data
cd 14_weathers_data
for town in Town01 Town02 Town03 Town04 Town05 Town06 Town07 Town10
do
	wget https://s3.eu-central-1.amazonaws.com/avg-projects/transfuser/data/14_weathers_data/${town}.zip
	unzip -q ${town}.zip
	rm ${town}.zip
done