TEST_DATA_PATH= ./data/fever_test.jsonl # e.g. covid_scientific.jsonl
EXP_NAME="fever_test"

LM_MODEL_TYPE=bert-base # bert-large
python main.py \
    --train_data_file=$TEST_DATA_PATH \
    --output_eval_file=/content/drive/MyDrive/FakeNewsDetection/ppl_results/$LM_MODEL_TYPE.$EXP_NAME.npy \
    --model_name=$LM_MODEL_TYPE