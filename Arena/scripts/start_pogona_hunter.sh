#!/bin/bash

params=( "--disable-session-crashed-bubble" "--disable-infobars" "--disable-component-update" \
         "--disable-pinch" "--chrome-frame" "--window-size=1920,1080" "--window-position=2025,001" \
         "--remember-cert-error-decisions" "--ignore-certificate-errors" \
         "--ignore-urlfetcher-cert-requests" "--allow-running-insecure-content" \
         "--display=:0.0" )

case "$*" in
(*--kiosk*) params+=( "--kiosk" );;
esac

echo "${params[@]}"
/opt/google/chrome/google-chrome "${params[@]}" http://localhost:8080/#/$1 &