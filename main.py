import spacy
from spacy.training.example import Example

TRAIN_DATA = [
    ('what is the price of polo?', {'entities': [(21, 25, 'PrdName')]}),
    ('what is the price of ball?', {'entities': [(21, 25, 'PrdName')]})
]

nlp = spacy.blank("en")


ner = nlp.add_pipe("ner")


ner.add_label("PrdName")

# Training loop
for itn in range(100):
    losses = {}
    examples = []
    for text, annotations in TRAIN_DATA:
        doc = nlp.make_doc(text)
        example = Example.from_dict(doc, annotations)
        examples.append(example)
    nlp.update(examples, drop=0.5, losses=losses)

nlp.to_disk("place our trained model here")


prdnlp = spacy.load("your_trained_model")


test_text = "What is the price of jeans?"
doc = prdnlp(test_text)
for ent in doc.ents:
    print(ent.text, ent.start_char, ent.end_char, ent.label_)
