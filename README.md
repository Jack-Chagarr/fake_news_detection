# Fake News Detection with Knowledge Based Model via Perplexity

## Team 5 (Junyoung Chegal, Elias Tranchant, Jung Cheng Lin)

<br>
<br>

# 1. Introduction

We found the prior research, _"Evaluation of Fake News Detection with Knowledge-Enhanced Language Models"_ which tried to detect fake news by knowledge-enhanced language models, such as ERNIE, KnowBert, KEPLER, K-ADAPTER.

Their results showed that they got better detection accuracy than Pretrained language models, and we tried to improve the detection accuracy by using knowledge-enhanced language models as base models, with different approach other than just fine tuning.

We found another prior research in related field 'fact-checking', _"Towards Few-Shot Fact-Checking via Perplexity"_. They used 'pseudo-perplexity' which is similar to perplexity, but more fit to PLMs, to classify between real and fake claims. They did few shot learning(2, 10, 50 shot settings), and got better accuracy than finetuned PLMs.
So we tried to combine this approach to our based models.

# 2. Methodology

'Pseudo perplexity' is getting perplexity score of Masked Language Models. Since those models are trained by predicting masked tokens in the sentence, 'pseudo perplexity' is computed by summing all the log probabilities obtained by sequentially masking each token in the input sentence. In prior research and our task, we combined evidence and claim in our dataset to use it as input sentence. Since this idea is given by _"Masked Language Model Scoring"_, we modified their source code to calculate our models' 'pseudo perplexity'. In the source code, they call it as 'mlm score'.

Although, this source code is written with 'gluonnlp', 'transformers' packages, we tried to put some other models, which are not in those packages, to source code and calculate 'pseudo perplexity', but we failed. So we could only calculate 'pseudo perplexity' of ERNIE model which is in 'transformers' package.

# 3. Dataset

Our main dataset is 'Liar-Plus', justification added version of 'Liar' dataset, which is most popular dataset in 'fake news detection' task. <br><br>
We only used test dataset of it. <br><br>
Their original labels are 6, “pants-fire”, “false”, “barely-true”, “half-true”, “mostly true”, and “true”, and we changed first 3 as "false", and last 3 as "true".<br><br>
Also, we combined justification and statements to use it as input sentence to calculate 'pseudo perplexity' of our base models. Their were some data which do not have justification, we ignored them, and we also ignored some input sentences which contain more than 800 alphabets(+ space).

<br><br>
We also used 'fever' test dataset to compare accuracy of ERNIE version with GPT-2, BERT version.

<br><br>

# 4. What we made is

We did our task with colab. When you clone git, and execute 'fake_news_detection.ipynb' in colab, it will work.

In 'fake_news_detection.ipynb, what we did is <br>

1. Preprocessing 'Liar Plus' test dataset
2. Got 'pseudo perplexity' scores of ERNIE by using 'Liar Plus' test dataset. Those are saved in '.ppl_results'.
3. Got 'optimal' thershold and accuracy, f1 score, recall, ... of ERNIE by using 'pseudo perplexity' and labels. We checked 5 times and all the results are saved in '.results'.
4. Got 'pseudo perplexity' scores of ERNIE by using 'fever' test dataset. Those are saved in '.ppl_results'.
5. Got 'optimal' thershold and accuracy, f1 score, recall, ... of ERNIE by using 'pseudo perplexity' and labels. We checked 5 times and all the results are saved in '.results'.

To calculate "pseudo perplexity" of ERNIE, we added some lines to the files in .mlm directory.

1. ./mlm/src/mlm/models/bert.py , line 146 - 224
   Add ErnieForMaskedLMOptimized class

2. ./mlm/src/mlm/models/**init**.py, line 145 - 160
   Get Pretrained Ernie, and tokenizer of Ernie

3. ./mlm/srcmlm/scorers.py, line 749 - 756
   Calculate 'pseudo perplexity' of Ernie with input_sentence
   <br><br>

# 5. Result

## Dataset: Liar or Liar Plus <br>

PPL_ERNIE_2_shot: 45.20%<br>
PPL_ERNIE_10_shot: 46.89%%<br>
PPL_ERNIE_50_shot: 48.90%%<br>
Fine-tuned ERNIE: 27.53% (from "Evaluation of Fake News Detection with Knowledge-Enhanced Language Models")<br>
<br><br>

## Dataset: Fever test<br>

### Best result from n-shot results<br>

PPL_ERNIE: 47.77%(2-shot)<br>
PPL_BERT: 57.59% (from "Towards Few-Shot Fact-Checking via Perplexity")<br>
PPL_GPT2: 67.48% (from "Towards Few-Shot Fact-Checking via Perplexity")<br>

# 6. Conclusion

PPL_ERNIE got better accuracy than Fine-tuned ERNIE. However, PPL_BERT, and PPL_GPT2 got better accuracy than PPL_ERNIE, it seems that ERNIE's knowledge encoding did not give positive impact to our task. Also, all the result got lower than 50% accuracy, which is lower than random probability 50%, this approach seems not that valid for now.

# 7. Reference

1. Lee, N., Bang, Y., Madotto, A., Khabsa, M., & Fung, P. (2021). Towards few-shot fact-checking via perplexity. arXiv preprint arXiv:2103.09535.

2. Whitehouse, C., Weyde, T., Madhyastha, P., & Komninos, N. (2022). Evaluation of Fake News Detection with Knowledge-Enhanced Language Models. Proceedings of the International AAAI Conference on Web and Social Media, 16(1), 1425-1429. https://doi.org/10.1609/icwsm.v16i1.19400

3. Julian Salazar, Davis Liang, Toan Q. Nguyen, and Katrin Kirchhoff. 2020. Masked Language Model Scoring. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 2699–2712, Online. Association for Computational Linguistics. <br> github: https://github.com/awslabs/mlm-scoring
