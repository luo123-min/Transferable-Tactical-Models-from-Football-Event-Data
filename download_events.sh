#!/bin/bash
cd "G:/Discover Artificial Intelligence/Topic3_TacticalModeling/data/events"
BASE="https://raw.githubusercontent.com/statsbomb/open-data/master/data/events"
# remove any partial/empty files
find . -maxdepth 1 -name '*.json' -size -1k -delete 2>/dev/null
ok=0; fail=0
while IFS=$'\r' read -r id; do
  id=$(echo "$id" | tr -d '\r')
  [ -z "$id" ] && continue
  if [ -s "$id.json" ]; then ok=$((ok+1)); continue; fi
  if curl -sL --max-time 60 -f -o "$id.json" "$BASE/$id.json"; then
    ok=$((ok+1))
  else
    echo "FAIL $id"; fail=$((fail+1))
  fi
done < ../ids.txt
echo "DONE ok=$ok fail=$fail"
