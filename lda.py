from glob import glob
import os
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import random
import pickle

n_features = 1000
n_components = 200


def load_data():
    data = []
    for path in ["~/data/text/ham/*", "~/data/text/phish/*"]:
        for pth in glob(os.path.expanduser(path)):
            with open(pth, "r") as f:
                content = "\n".join(f.readlines())
                data.append(content)
    return data


data = load_data()
print("data loaded")

random.seed(42)
random.shuffle(data)
test_boundary = round(len(data) * 0.1)
train, test = data[test_boundary:], data[:test_boundary]
print("test size {}".format(len(test)))
print("data size {}".format(len(train)))

vectorizer = CountVectorizer(max_df=0.95,
                             min_df=2,
                             max_features=n_features,
                             stop_words='english')
tf = vectorizer.fit_transform(train)
print("features ready")

lda = LatentDirichletAllocation(n_components=n_components,
                                max_iter=5,
                                learning_method='online',
                                learning_offset=50.,
                                random_state=0)
model = lda.fit(tf)
print("learned")

tst_vect = vectorizer.transform(test)
print("perplexity: {}".format(model.perplexity(tst_vect)))
# print(lda.transform(tst_vect[0])[0])

pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))
pickle.dump(lda, open("lda-model.pkl", "wb"))

