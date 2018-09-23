import pickle
import json
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

max_ner_features = 20

ner_ham = json.load(open("ner-ham.json", "r"))
ner_phish = json.load(open("ner-phish.json", "r"))

vect = pickle.load(open("vectorizer.pkl", "rb"))
lda = pickle.load(open("lda-model.pkl", "rb"))

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
    features.append((i[1], ner, cls))

ner_types = {}
ner_values = {}


def vectorize(d, v):
    if v not in d:
        d[v] = len(d) + 1
    return d[v]


feat = []
for f in features:
    ner_ft = [0 for i in range(max_ner_features * 2)]
    if len(f[1]) > max_ner_features:
        print("more then {} ner features".format(max_ner_features))
    for i, ft in enumerate(f[1][:max_ner_features]):
        pos = 2 * i
        ner_ft[pos] = vectorize(ner_types, ft[0])
        ner_ft[pos + 1] = vectorize(ner_values, ft[1])
    feat.append((list(f[0]) + ner_ft, f[2]))

X = [i[0] for i in feat]
y = [i[1] for i in feat]
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=42)
print("train size: {} test size: {}".format(len(X_train), len(X_test)))

clf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
report = classification_report(y_test, y_pred, target_names=["ham", "phish"])
print(report)

with open("report.txt", "w") as rf:
    rf.write(report)

pickle.dump(ner_types, open("ner_types.pkl", "wb"))
pickle.dump(ner_values, open("ner_values.pkl", "wb"))
pickle.dump(clf, open("model.pkl", "wb"))
