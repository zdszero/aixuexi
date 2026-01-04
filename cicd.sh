#! /bin/bash

[[ -z $USER ]] && echo "USER environment cannot be empty" && exit 1
[[ -z $HOST ]] && echo "HOST environment cannot be empty" && exit 1

echo "Push local repo to github"
git push origin master

echo
echo "*** Running commands on remote $USER@$HOST ***"
echo


ssh -p 22 -T $USER@$HOST <<'EOL'
	cd cs-kaoyan-grocery || exit
	git pull origin master || { echo "Failed to pull from GitHub"; exit 1; }
	npm run build:production || { echo "Failed to build the project"; exit 1; }
	rm -rf /var/www/csgraduates.com/*
	cp -r public/* /var/www/csgraduates.com
EOL

echo "Finish deployment on remote server"
