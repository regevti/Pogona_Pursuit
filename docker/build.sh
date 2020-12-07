app_tag=v1.0.0
#!/bin/bash


SERVICES=`grep -oE "[a-z]+_tag" .env | xargs`
service=$1
if [ "$#" -ne 1 ]; then
    echo "Illegal number of parameters. Must set <service_name>" >&2; exit 1
fi
if [[ ! "${SERVICES[@]}" =~ "$1" ]]; then
  echo "unknown service: $1"
  exit 1
fi

git_version=`git describe --tags`
re='^v[0-9]+\.[0-9]+$'
if ! [[ $git_version =~ $re ]] ; then
   echo "error: bad git version: $git_version" >&2; exit 1
fi

last_octet=`grep -oE "'"$service"'_tag=v[0-9\.]+" .env | cut -d= -f2 | cut -d. -f3`
new_version=`echo "$git_version.$(($last_octet+1))"`
echo "new_version=$new_version"
sed -E 's/('"$service"'_tag=)v[0-9\.]+/\1'"$new_version"'/' .env > .env

dc build $service
dc up -d $service

echo "$(date) - $service - $new_version" >> ./deployments.log