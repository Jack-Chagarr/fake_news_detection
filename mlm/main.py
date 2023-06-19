# import mlm
# from mlm.scorers import MLMScorer, MLMScorerPT, LMScorer
# from mlm.models import get_pretrained
# import mxnet as mx
# ctxs = [mx.cpu()] # or, e.g., [mx.gpu(0), mx.gpu(1)]
 
# model, vocab, tokenizer = get_pretrained(ctxs, 'bert-base-en-uncased') # bert-large-en-uncased
# scorer = MLMScorer(model, vocab, tokenizer, ctxs)

# print(scorer.score_sentences(["Hello world!"]))

# # for line in all_lines:
# #     print(scorer.score_sentences(["Hello world!"]))
# # # >> [-12.410664200782776]
# # # print(scorer.score_sentences(["Hello world!"], per_token=True))
from src.mlm.scorers import MLMScorer, MLMScorerPT, LMScorer
from src.mlm.models import get_pretrained
import mxnet as mx
import numpy as np
import argparse
import jsonlines
from tqdm import tqdm
import torch

import pandas as pd

def fever_data_cleaning(sent):
    sent = sent.replace('-LRB-', '(')
    sent = sent.replace('-RRB-', ')')
    sent = sent.replace('-LSB-', '[')
    sent = sent.replace('-RSB-', ']')
    return sent

def prepare_data(file_path):
    test_set = []
    with jsonlines.open(file_path) as reader:
        objs = [obj for obj in reader]

    for obj in objs:
        if 'liar_plus' in file_path:
            claim = obj['claim'].lower().strip()
            test_set.append(claim)
        else:
            claim = obj['claim'].lower().strip()
            evs_line = fever_data_cleaning(obj['evidences'][0][0]).lower().strip()
            test_sent = " ".join([evs_line, claim])
            test_set.append(test_sent)        
    return test_set

models_mapping = {
    'bert-large': 'bert-large-en-uncased',
    'bert-base':'bert-base-en-uncased',
    'ernie' : 'ernie-base-en-uncased'
}
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_data_file", default=None, type=str, required=False, help=""
    )
    parser.add_argument(
        "--output_eval_file", default=None, type=str, required=False, help=""
    )
    parser.add_argument(
        "--model_name", default=None, type=str, required=False, help=""
    )
    args = parser.parse_args()

    #modelName = models_mapping[args.model_name]
    ctxs = [mx.cpu()] # or, e.g., [mx.gpu(0), mx.gpu(1)]
    model, vocab, tokenizer = get_pretrained(ctxs, args.model_name) # bert-base-en-uncased bert-large-en-uncased
    scorer = MLMScorerPT(model, vocab, tokenizer, ctxs)

    ppl_results=[]
    file_path = args.train_data_file

    print("Evaluating ", file_path)

    with jsonlines.open(file_path) as reader:
        objs = [obj for obj in reader]

    for i in tqdm(range(len(objs))):
        torch.cuda.empty_cache() 
        obj = objs[i]
        claim = fever_data_cleaning(obj['claim'].lower().strip())
        ppl = {'perplexity': scorer.score_sentences([claim])[0]}
        ppl_results.append(ppl)
        if i == 0:
            with jsonlines.open(args.output_eval_file.replace(".npy", ".jsonl"), 'w') as writer:
                writer.write(ppl)
        else:          
            with jsonlines.open(args.output_eval_file.replace(".npy", ".jsonl"), 'a') as writer:
                writer.write(ppl)
        
    # ppl_results = scorer.score_sentences(test_lines)
    np.save(args.output_eval_file, ppl_results)
