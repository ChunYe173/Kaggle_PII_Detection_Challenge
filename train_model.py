import os
import json
import random
from itertools import chain
from functools import partial

from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments, DataCollatorForTokenClassification
from tokenizers import AddedToken
import evaluate
from datasets import Dataset
import numpy as np

# class my_class(object):
#     pass

def filter_no_pii(example, percent_allow=0.2):
    # Return True if there is PII
    # Or 20% of the time if there isn't
    # To remove 80% of entry that only has "O" in the labels
    import random
    has_pii = set("O") != set(example["provided_labels"])

    return has_pii or (random.random() < percent_allow)

def tokenize(example, tokenizer, label2id, max_length):
    import numpy as np
    from tokenizers import AddedToken
    text = []
    labels = []

    for t, l, ws in zip(example["tokens"], example["provided_labels"], example["trailing_whitespace"]):

        text.append(t)
        labels.extend([l]*len(t))
        if ws:
            text.append(" ")
            labels.append("O")

    tokenized = tokenizer("".join(text), return_offsets_mapping=True, max_length=max_length)

    labels = np.array(labels)

    text = "".join(text)
    token_labels = []

    for start_idx, end_idx in tokenized.offset_mapping:

        # CLS token
        if start_idx == 0 and end_idx == 0:
            token_labels.append(label2id["O"])
            continue

        # case when token starts with whitespace
        if text[start_idx].isspace():
            start_idx += 1

        while start_idx >= len(labels):
            start_idx -= 1

        token_labels.append(label2id[labels[start_idx]])

    length = len(tokenized.input_ids)

    return {
        **tokenized,
        "labels": token_labels,
        "length": length
    }

def compute_metrics(p, metric, all_labels):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [all_labels[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [all_labels[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)

    # Unpack nested dictionaries
    final_results = {}
    for key, value in results.items():
        if isinstance(value, dict):
            for n, v in value.items():
                final_results[f"{key}_{n}"] = v
        else:
            final_results[key] = value
    return final_results

def main(training, training_model_path, training_max_length, path_to_datasets, model_save_path):
    data = []
    for dataset in path_to_datasets:
      data.extend(json.load(open(dataset)))

    print(type(data)) # == list
    print(type(data[0])) # == dict
    print((data[0]))
    print(type(data[-1])) # == dict
    print((data[-1]))
    # print((data[1]))
    # print((data[2]))
    print("Total number of data entries: ", len(data))

    all_labels = sorted(list(set(chain(*[x["labels"] for x in data]))))
    label2id = {l: i for i,l in enumerate(all_labels)}
    id2label = {v:k for k,v in label2id.items()}

    ds = Dataset.from_dict({
        "full_text": [x["full_text"] for x in data],
        "document": [x["document"] for x in data],
        "tokens": [x["tokens"] for x in data],
        "trailing_whitespace": [x["trailing_whitespace"] for x in data],
        "provided_labels": [x["labels"] for x in data] #,b 0.576
    })

    tokenizer = AutoTokenizer.from_pretrained(training_model_path)

    # lots of newlines in the text
    # adding this should be helpful
    tokenizer.add_tokens(AddedToken("\n", normalized=False))

    ds = ds.filter(
        filter_no_pii,
        num_proc=2,
    )

    ds = ds.map(
        tokenize,
        fn_kwargs={"tokenizer": tokenizer, "label2id": label2id, "max_length": training_max_length},
        num_proc=2,
    )

    metric = evaluate.load("seqeval")

    model = AutoModelForTokenClassification.from_pretrained(training_model_path, num_labels=len(all_labels), id2label=id2label, label2id=label2id)
    model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=16)

    collator = DataCollatorForTokenClassification(tokenizer, pad_to_multiple_of=16)


    ### All possible training arguments that may be defined in may be found here: (To-explore)
    # https://github.com/huggingface/transformers/blob/main/src/transformers/training_args.py
    args = TrainingArguments(
        "output",
        fp16=False,
        learning_rate=5e-5,
        weight_decay=0.01,
        warmup_ratio=0.1,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=4,
        report_to="none",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        logging_steps=5,
        metric_for_best_model="overall_recall",
        greater_is_better=True,
        gradient_checkpointing=True,
        num_train_epochs=1,
        dataloader_num_workers=1,
    )

    # may want to try to balance classes in splits
    final_ds = ds.train_test_split(test_size=0.2)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=final_ds["train"],
        eval_dataset=final_ds["test"],
        data_collator=collator,
        tokenizer=tokenizer,
        compute_metrics=partial(compute_metrics, metric=metric, all_labels=all_labels),
    )

    print("Begin training...")
    trainer.train()
    # trainer.save_model(model_save_path) # saves tokenizer with model
    trainer.save_state()
    return # move the return according to line by line verification progress in this script


if __name__ == "__main__":
    training = True # be sure to turn internet off if doing inference

    # training_model_path = r"C:\Users\abome\source\repos\venv_pii_ner\model\deberta-v3-large")
    training_model_path = r"C:\Users\abome\source\repos\venv_pii_ner\model\deberta-v3-large"
    training_max_length = 512

    # inference_model_path = "/kaggle/input/pii-data-detection-baseline/output/checkpoint-240" # replace with our own trained model
    inference_model_path = r"C:\Users\abome\source\repos\venv_pii_ner\model"
    inference_max_length = 512 # 2000

    # path_to_datasets = '/content/gdrive/SharedWithMe/EzKaggle2024/train.json'
    # path_to_datasets = [r'C:\Users\abome\source\repos\venv_pii_ner\dataset\train.json', r'C:\Users\abome\source\repos\venv_pii_ner\dataset\pii_dataset.json'] # append path to other dataset json files to this list
    path_to_datasets = [r'C:\Users\abome\source\repos\venv_pii_ner\dataset\train.json'] # append path to other dataset json files to this list
    model_save_path = r'C:\Users\abome\source\repos\venv_pii_ner\model\\'

    main(training, training_model_path, training_max_length, path_to_datasets, model_save_path)
