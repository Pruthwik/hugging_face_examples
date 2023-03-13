"""Fine tuning a hugging face text classification model."""
# This code is from the official huggingface website.
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments
from transformers import Trainer
from transformers import TextClassificationPipeline


# define total number of labels
num_labels = 6
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
dataset_name = "md_gender_bias"


def tokenize_data(examples):
    """Tokenize data using a pretrained tokenizer."""
    return tokenizer(examples["text"], truncation=True)


def main():
    """Pass arguments and call functions here."""
    dataset =  load_dataset(dataset_name)
    # create an evaluation dataset from the train set
    eval_dataset = dataset['train'][100: 150]
    # create the tokenized dataset
    tokenized_dataset = dataset.map(tokenize_data, batched= True)
    training_args = TrainingArguments(
        output_dir="results",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=1,
        weight_decay=0.01,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        tokenizer=tokenizer
    )
    # train a model with specified arguments
    trainer.train()
    # to predict and return the class/label with the highest score
    pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer)
    # print the outputs on the evaluation dataset
    print('Training Done')
    print(pipe(eval_dataset['text']))


if __name__ == '__main__':
    main()
