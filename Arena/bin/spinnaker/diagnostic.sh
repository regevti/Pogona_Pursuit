#!/bin/bash

echo lsb_release -a:
echo
lsb_release -a
echo
echo

echo uname -a:
echo
uname -a
echo
echo

echo lspci:
echo
lspci
echo
echo

echo lsusb:
echo
lsusb
echo
echo


echo lsusb -t:
echo
lsusb -t
echo
echo


echo lsusb -v:
echo
lsusb -v
echo
echo

echo USBFS:
echo
cat /sys/module/usbcore/parameters/usbfs_memory_mb
echo
echo

echo ifconfig -a:
echo
ifconfig -a
echo
echo

echo ls -l /usr/lib/libfly*:
echo
ls -l /usr/lib/libfly*
echo
echo

echo ls -l /usr/lib/libSpin*:
echo
ls -l /usr/lib/libSpin*
