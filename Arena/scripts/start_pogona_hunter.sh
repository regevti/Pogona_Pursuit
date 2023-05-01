#!/bin/bash

params=( "--disable-session-crashed-bubble" "--disable-infobars" "--disable-component-update" \
         "--disable-pinch" "--chrome-frame" "--window-size=$2"  "--window-position=$4,001" \
         "--remember-cert-error-decisions" "--ignore-certificate-errors" \
         "--ignore-urlfetcher-cert-requests" "--allow-running-insecure-content" \
         "--display=$3" )

case "$*" in
(*--kiosk*) params+=( "--kiosk" );;
esac

echo "${params[@]}"
/opt/google/chrome/google-chrome "${params[@]}" http://localhost:8080/#/$1 &
