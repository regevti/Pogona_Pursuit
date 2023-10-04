#!/bin/bash

# https://peter.sh/experiments/chromium-command-line-switches/

params=( "--disable-session-crashed-bubble" "--disable-infobars" "--disable-component-update" \
         "--disable-pinch" "--chrome-frame" "--window-size=$2"  "--window-position=$4,001" \
         "--remember-cert-error-decisions" "--ignore-certificate-errors" \
         "--ignore-urlfetcher-cert-requests" "--allow-running-insecure-content" \
         '--simulate-outdated-no-au="01 Jan 2199"' '--disk-data-dir="/tmp/chromium"'\
         "--display=$3" )

case "$*" in
(*--kiosk*) params+=( "--kiosk" );;
esac

/opt/google/chrome/google-chrome "${params[@]}" http://localhost:8080/#/$1 &
echo google-chrome "${params[@]}" http://localhost:8080/#/$1