import argparse
import torch
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader
from transformers import RobertaConfig, RobertaTokenizer, BertConfig, BertTokenizer
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from utils import set_seed, collate_fn
from datasets import load_metric
from model import RobertaForSequenceClassification, BertForSequenceClassification
from evaluation import evaluate_ood
import wandb
import warnings
from data import load
warnings.filterwarnings("ignore")


task_to_labels = {
    'sst2': 2,
    'imdb': 2,
    '20ng': 20,
    'trec': 6,
}


task_to_metric = {
    'sst2': 'sst2',
    'imdb': 'sst2',
    '20ng': 'mnli',
    'trec': 'mnli',
}


def train(args, model, train_dataset, dev_dataset, test_dataset, benchmarks):
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=True, drop_last=True)
    dev_dataloader = DataLoader(dev_dataset, batch_size=args.batch_size, collate_fn=collate_fn)
    total_steps = int(len(train_dataloader) * args.num_train_epochs)
    warmup_steps = int(total_steps * args.warmup_ratio)

    no_decay = ["LayerNorm.weight", "bias"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    def detect_ood():
        model.prepare_ood(dev_dataloader)
        for tag, ood_features in benchmarks:
            results = evaluate_ood(args, model, test_dataset, ood_features, tag=tag)
            wandb.log(results, step=num_steps)

    num_steps = 0
    for epoch in range(int(args.num_train_epochs)):
        model.zero_grad()
        for step, batch in enumerate(tqdm(train_dataloader)):
            model.train()
            batch = {key: value.to(args.device) for key, value in batch.items()}
            outputs = model(**batch)
            loss, cos_loss = outputs[0], outputs[1]
            loss.backward()
            num_steps += 1
            optimizer.step()
            scheduler.step()
            model.zero_grad()
            wandb.log({'loss': loss.item()}, step=num_steps)
            wandb.log({'cos_loss': cos_loss.item()}, step=num_steps)

        results = evaluate(args, model, dev_dataset, tag="dev")
        wandb.log(results, step=num_steps)
        results = evaluate(args, model, test_dataset, tag="test")
        wandb.log(results, step=num_steps)
        detect_ood()


def evaluate(args, model, eval_dataset, tag="train"):
    metric_name = task_to_metric[args.task_name]
    metric = load_metric("glue", metric_name)

    def compute_metrics(preds, labels):
        preds = np.argmax(preds, axis=1)
        result = metric.compute(predictions=preds, references=labels)
        if len(result) > 1:
            result["score"] = np.mean(list(result.values())).item()
        return result
    dataloader = DataLoader(eval_dataset, batch_size=args.batch_size, collate_fn=collate_fn)

    label_list, logit_list = [], []
    for step, batch in enumerate(tqdm(dataloader)):
        model.eval()
        labels = batch["labels"].detach().cpu().numpy()
        batch = {key: value.to(args.device) for key, value in batch.items()}
        batch["labels"] = None
        outputs = model(**batch)
        logits = outputs[0].detach().cpu().numpy()
        label_list.append(labels)
        logit_list.append(logits)
    preds = np.concatenate(logit_list, axis=0)
    labels = np.concatenate(label_list, axis=0)
    results = compute_metrics(preds, labels)
    results = {"{}_{}".format(tag, key): value for key, value in results.items()}
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", default="roberta-large", type=str)
    parser.add_argument("--max_seq_length", default=256, type=int)
    parser.add_argument("--task_name", default="sst2", type=str)

    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--learning_rate", default=1e-5, type=float)
    parser.add_argument("--adam_epsilon", default=1e-6, type=float)
    parser.add_argument("--warmup_ratio", default=0.06, type=float)
    parser.add_argument("--weight_decay", default=0.01, type=float)
    parser.add_argument("--num_train_epochs", default=10.0, type=float)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--project_name", type=str, default="ood")
    parser.add_argument("--alpha", type=float, default=2.0)
    parser.add_argument("--loss", type=str, default="margin")
    args = parser.parse_args()

    wandb.init(project=args.project_name, name=args.task_name + '-' + str(args.alpha) + "_" + args.loss)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    set_seed(args)

    num_labels = task_to_labels[args.task_name]
    if args.model_name_or_path.startswith('roberta'):
        config = RobertaConfig.from_pretrained(args.model_name_or_path, num_labels=num_labels)
        config.gradient_checkpointing = True
        config.alpha = args.alpha
        config.loss = args.loss
        tokenizer = RobertaTokenizer.from_pretrained(args.model_name_or_path)
        model = RobertaForSequenceClassification.from_pretrained(
            args.model_name_or_path, config=config,
        )
        model.to(0)
    elif args.model_name_or_path.startswith('bert'):
        config = BertConfig.from_pretrained(args.model_name_or_path, num_labels=num_labels)
        config.gradient_checkpointing = True
        config.alpha = args.alpha
        config.loss = args.loss
        tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path)
        model = BertForSequenceClassification.from_pretrained(
            args.model_name_or_path, config=config,
        )
        model.to(0)

    datasets = ['rte', 'sst2', 'mnli', '20ng', 'trec', 'imdb', 'wmt16', 'multi30k']
    benchmarks = ()

    for dataset in datasets:
        if dataset == args.task_name:
            train_dataset, dev_dataset, test_dataset = load(dataset, tokenizer, max_seq_length=args.max_seq_length)
        else:
            _, _, ood_dataset = load(dataset, tokenizer, max_seq_length=args.max_seq_length)
            benchmarks = (('ood_' + dataset, ood_dataset),) + benchmarks
    train(args, model, train_dataset, dev_dataset, test_dataset, benchmarks)


if __name__ == "__main__":
    main()
