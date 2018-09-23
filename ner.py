from glob import glob
import requests
import sys
import os
import json

server = "http://nametag.dev.dszn.cz:4224/recognize"
params = {
    "output": "vertical",
    "model": "model_2018-04_annotated_not-merged_shuffled.ntg",
}

for path in ["~/data/text/ham/*", "~/data/text/phish/*"]:
    results = {}
    files = glob(os.path.expanduser(path))
    cnt = 0
    for pth in files:
        results[pth] = []
        with open(pth, "r") as f:
            content = "\n".join(f.readlines())
            # params["data"] = content
            res = None
            try:
                res = requests.post(server,
                                    data={"data": content},
                                    params=params)
                res = res.json().get("result", "")
                cnt += 1
                print("{}/{}".format(cnt, len(files)))
            except Exception as e:
                print(pth)
                print("err response {}".format(e), file=sys.stderr)
                print(res.text)
                continue
            if res:
                for t in res.split('\n'):
                    line = t.split('\t')
                    if not line[0]:
                        continue
                    tag = line[1]
                    # i - organizations
                    # p - people
                    # g - geographic names
                    if tag[0] in {'i', 'p', 'g'}:
                        results[pth].append(line[1:])
    cls = "ham" if "ham" in path else "phish"
    json.dump(results, open("ner-{}.json".format(cls), "w"))
