from dont_patronize_me import DontPatronizeMe
import pandas as pd
import sys
from simpletransformers.classification import ClassificationModel, ClassificationArgs
from pathlib import Path

# helper function to save predictions to an output file
def labels2file(p, outf_path):
    with open(outf_path, 'w') as outf:
        for pi in p:
            outf.write(','.join([str(k) for k in pi]) + '\n')

# Parse path to model directory as command line argument 
model_dir = sys.argv[1]

model = ClassificationModel("roberta", model_dir, use_cuda=True)

test_path = "./datasets/submission_test.tsv"

with open(test_path) as f:
	test_data = f.readlines()

preds_task1, _ = model.predict(test_data)

labels2file([[k] for k in preds_task1], 'task1.txt')

assert(len(test_data) == 3832)
