#!/bin/bash

while :
do
    rsync -avP rpi@192.168.0.122:~/bio_cam_test/images /mnt/d/MRT/scripts/fetched_images
    sleep 20
done