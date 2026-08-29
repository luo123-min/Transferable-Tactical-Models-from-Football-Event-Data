#!/bin/bash
cd "G:/Discover Artificial Intelligence/Topic3_TacticalModeling/data/events"
BASE="https://raw.githubusercontent.com/statsbomb/open-data/master/data/events"
need=""
while IFS=$'\r' read -r id; do
  id=$(echo "$id" | tr -d '\r'); [ -z "$id" ] && continue
  if [ -s "$id.json" ] && "C:/Users/HP/.workbuddy/binaries/python/envs/default/Scripts/python.exe" -c "import json,sys; json.load(open('$id.json'))" 2>/dev/null; then
    :
  else
    need="$need $id"
  fi
done < ../ids.txt
echo "need redownload count: $(echo $need | wc -w)"
for id in $need; do
  ok=0
  for t in 1 2 3 4 5 6 7 8; do
    curl -sL --max-time 120 -f -o "_t_$id.json" "$BASE/$id.json"
    if "C:/Users/HP/.workbuddy/binaries/python/envs/default/Scripts/python.exe" -c "import json,sys; json.load(open('_t_$id.json'))" 2>/dev/null; then
      mv -f "_t_$id.json" "$id.json"; ok=1; break
    else
      rm -f "_t_$id.json"
    fi
  done
  [ $ok -eq 0 ] && echo "STILLFAIL $id"
done
echo "DONE files=$(ls -1 *.json | wc -l)"
