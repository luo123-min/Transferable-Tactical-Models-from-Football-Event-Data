#!/bin/bash
cd "G:/Discover Artificial Intelligence/Topic3_TacticalModeling/data/events"
BASE="https://raw.githubusercontent.com/statsbomb/open-data/master/data/events"
find . -maxdepth 1 -name '*.json' -size -1k -delete 2>/dev/null
cat ../ids.txt | tr -d '\r' | grep -v '^$' | xargs -P 8 -I{} sh -c 'f="{}"; if [ ! -s "$f.json" ]; then curl -sL --max-time 90 -f -o "$f.json" "'"$BASE"'/$f.json" || echo "FAIL $f"; fi'
echo "DONE files=$(ls -1 *.json | wc -l)"
