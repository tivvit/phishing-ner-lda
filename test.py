import pickle
import json
import random
import requests

max_ner_features = 20

ner_ham = json.load(open("ner-ham.json", "r"))
ner_phish = json.load(open("ner-phish.json", "r"))

vect = pickle.load(open("vectorizer.pkl", "rb"))
lda = pickle.load(open("lda-model.pkl", "rb"))

ner_types = pickle.load(open("ner_types.pkl", "rb"))
ner_values = pickle.load(open("ner_values.pkl", "rb"))

topics = {}
files = []
data = []
for pth in list(ner_ham.keys()) + list(ner_phish.keys()):
    with open(pth, "r") as f:
        content = "\n".join(f.readlines())
        files.append(pth)
        data.append(content)

data_vect = vect.transform(data)
lda_prob = lda.transform(data_vect)

features = []
for i in list(zip(files, lda_prob)):
    cls = 0 if "/ham/" in i[0] else 1
    ner = ner_phish[i[0]] if cls else ner_ham[i[0]]
    # print(len(i[1]), ner, cls)
    features.append((i[1], ner, i[0], cls))


def vectorize(d, v):
    return d.get(v, 0)


feat = []
for f in features:
    ner_ft = [0 for i in range(max_ner_features * 2)]
    if len(f[1]) > max_ner_features:
        print("more then {} ner features".format(max_ner_features))
    for i, ft in enumerate(f[1][:max_ner_features]):
        pos = 2 * i
        ner_ft[pos] = vectorize(ner_types, ft[0])
        ner_ft[pos + 1] = vectorize(ner_values, ft[1])
    feat.append((list(f[0]) + ner_ft, f[2], f[3]))

random.shuffle(feat)

for i in feat:
    clf = pickle.load(open("model.pkl", "rb"))
    # if clf.predict([i[0]])[0] == 0:
    #     print(i)
    #     break
    local_pred = clf.predict([i[0]])[0]
    with open(i[1], "r") as f:
        d = {
            "content": "\n".join(f.readlines()),
            "tags": ner_phish.get(i[1], []) if i[2] else ner_ham.get(i[1], []),
        }
        remote_pred = requests.post("http://localhost:2221/detect",
                                    json=d).json().get("result", None)
    if local_pred != remote_pred:
        print("bad pred !!!")
    else:
        print("ok {} {}".format(local_pred, remote_pred))

fl = "/Users/vit.listik/data/text/ham/00001.1a31cc283af0060967a233d26548a6ce"
with open(fl, "r") as f:
    d = {
        "content": "\n".join(f.readlines()),
        "tags": ner_ham[fl],
    }
    print(d)
    print(requests.post("http://localhost:2221/detect", json=d).text)
