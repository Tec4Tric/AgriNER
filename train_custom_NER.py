import os
import json
import spacy
from spacy.tokens import DocBin




def train(train, test):
    try:
        f = open(train)
        train_data = json.load(f)
        f = open(test)
        test_data = json.load(f)
        nlp = spacy.blank("en")  # load a new spacy model
        doc_bin = DocBin()

        def create_testing(test_data):
            db = DocBin()
            total_count = 0
            for i, annot in (enumerate(test_data)):
                text = annot['text']
                doc = nlp.make_doc(text)
                ents = []
                bad_count = 0
                good_count = 0
                for start, end, label, _ in annot["entities"]:
                    span = doc.char_span(start, end, label=label, alignment_mode="contract")
                    if span is None:
                        print("Skipping entity" + "   " + label + "   " + str(start) + "   " + str(end) + "   " + annot[
                            'file'])
                        bad_count += 1
                    else:
                        ents.append(span)
                        good_count += 1
                doc.ents = ents
                db.add(doc)
                total_count += good_count
            return db, total_count

        def create_training(train_data):
            db = DocBin()
            total_count = 0
            for i, annot in (enumerate(train_data)):
                text = annot['text']
                doc = nlp.make_doc(text)
                ents = []
                bad_count = 0
                good_count = 0
                for start, end, label, _ in annot["entities"]:
                    span = doc.char_span(start, end, label=label, alignment_mode="contract")
                    if span is None:
                        print("Skipping entity" + "   " + label + "   " + str(start) + "   " + str(end) + "   " + annot[
                            'file'])
                        bad_count += 1
                    else:
                        ents.append(span)
                        good_count += 1
                doc.ents = ents
                db.add(doc)
                total_count += good_count
            return db, total_count

        db, _ = create_training(train_data)
        db_dev, _ = create_testing(test_data)
        db.to_disk("./AgriNer_Code/train.spacy")
        db_dev.to_disk("./AgriNer_Code/test.spacy")
        return "OK"
    except Exception:
        # return "Failed"
        return "Failed to create train and test file"
