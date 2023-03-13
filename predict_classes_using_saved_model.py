"""Fine tuning a hugging face text classification model."""
# This code is from the official huggingface website.
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import TextClassificationPipeline
from sys import argv
import torch
import tensorflow as tf


# define total number of labels
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)


def main():
    """Pass arguments and call functions here."""
    # model path is a directory in Huggingface
    model_path = argv[1] # load the saved model
    loaded_model = AutoModelForSequenceClassification.from_pretrained('my_model')
    dataset_name = "md_gender_bias"
    dataset =  load_dataset(dataset_name)
    # create an evaluation dataset from the train set
    eval_dataset = dataset['train'][100: 150]
    # 2 ways to predict: 1 with pipeline, the other being passing inputs to the model
    pipe = TextClassificationPipeline(model=loaded_model, tokenizer=tokenizer)
    # print the outputs on the evaluation dataset
    print(pipe(eval_dataset['text']))
    # Tokenize the inputs and return tensors in dictionary format
    input_tensors = tokenizer(eval_dataset['text'], padding=True, truncation=True, return_tensors="pt")
    outputs = loaded_model(**input_tensors)
    logit_values = outputs.logits
    # convert logits into probabilities
    with torch.no_grad():
        # softmax in pytorch
        softmax_layer = torch.nn.Softmax(dim=1)
        output_predicted_probs_torch = softmax_layer(logit_values)
        arg_max_torch = torch.argmax(output_predicted_probs_torch, axis=-1)
        # softmax is in tensorflow
        output_predicted_probs_tf = tf.math.softmax(logit_values)
        arg_max_tf = tf.math.argmax(output_predicted_probs_tf, axis=-1)
        print(arg_max_torch == arg_max_tf)


if __name__ == '__main__':
    main()
