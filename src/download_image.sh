#!/bin/bash
cd /Volumes/Storage/IIIT-H/DIP_Project_data
git clone https://github.com/ultralytics/google-images-download
cd google-images-download
for x in "beer labels" "beer labels old"
do
  python3 bing_scraper.py --search "$x" --download --chromedriver /Volumes/Storage/IIIT-H/DIP_Project_data/chromedriver
done
