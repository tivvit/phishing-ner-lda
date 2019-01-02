# This is a support repository for training Phishing NER LDA model

The repository does not contain training data

It contains all the trained models

* `lda.py` - train LDA model
* `ner.py` - extract NER features (using Nametag server with custom model) 
the result is stored in this repository
* `phish.py` - train phishing detector
* `test.py` - validate predictions

This repository is mostly documentation of how was the model trained. It does 
not aim for comfortable training on custom data or to reproduce the research 
with one command. 
If you want to use it contact me and I will refactor the 
code and prepare it for simpler use.