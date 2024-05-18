#!/bin/bash

filename=$(date +%H_%M_%S_%d-%m-%Y__$RANDOM)

rot_str=`rostopic echo -n 1 /sbg/ekf_euler | grep 'z'`
read rot rot_acc <<<${rot_str//[^0-9\.]/ }

gps_str=`rostopic echo -n 1 /sbg/ekf_nav`
lat=`echo $gps_str | grep latitude | tr -d -c 0-9.-`
lon=`echo $gps_str | grep longitude | tr -d -c 0-9.-`
elev=`echo $gps_str | grep altitude | tr -d -c 0-9.-`

acc_str=`echo $gps_str | grep position_accuracy -A 3`
lat_acc=`echo $gps_str | grep x | tr -d -c 0-9.-`
lon_acc=`echo $gps_str | grep y | tr -d -c 0-9.-`
elev_acc=`echo $gps_str | grep z | tr -d -c 0-9.-`


echo "{
  "rotation_north": $rot, # degrees
  "gps": {
    "lat": $lat,
    "lon": $lon,
    "elevation": $elev
  },
  "acc_range": {
    "rotation_north": $rot_acc,
    "lat": $lat_acc,
    "lon": $lon_acc,
    "elevation": $elev_acc
  }
}" > $filename.json

cat $filename.json
