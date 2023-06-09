from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")

nlp = pipeline("ner", model=model, tokenizer=tokenizer)
example = "My name is Wolfgang and I live in Berlin"

ner_results = nlp(example)
print(ner_results)

import json
data_path = "/zecheng/submission/data/Alpaca-CoT/alpaca/alpaca_data_cleaned.json"
data = None
with open(data_path,'r') as f:
    data = json.load(f)
import pdb
pdb.set_trace()

