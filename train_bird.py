import cProfile
import pstats
import logging
import os
os.environ["WANDB_MODE"]="offline"
import time
import pdb
import json

from sympy import false
import torch
import datasets
from peft import LoraConfig, TaskType, get_peft_model, PeftModel, PeftConfig, AutoPeftModelForCausalLM
import transformers
from transformers import LlamaTokenizer, LlamaForCausalLM
from transformers import (
    HfArgumentParser,
    set_seed,
    EarlyStoppingCallback,
)
from transformers.trainer_utils import get_last_checkpoint
from collections import OrderedDict
import utils.tool
from utils.configue import Configure
from utils.dataset_bird import TokenizedDataset
from utils.trainer import EvaluateFriendlySeq2SeqTrainer
from utils.training_arguments import WrappedSeq2SeqTrainingArguments

# Huggingface realized the "Seq2seqTrainingArguments" which is the same with "WrappedSeq2SeqTrainingArguments"
# in transformers==4.10.1 during our work.
logger = logging.getLogger(__name__)


def main() -> None:
    os.environ[
        'CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # Deterministic behavior of torch.addmm. Please refer to https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
    # torch.set_deterministic(True)
    # Initialize the logger
    logging.basicConfig(level=logging.INFO)

    from filelock import FileLock
    # import nltk
    # with FileLock(".lock") as lock:
    #     nltk.download("punkt")
    #     nltk.download("stopwords")

    # Get args
    parser = HfArgumentParser((WrappedSeq2SeqTrainingArguments,))
    training_args, = parser.parse_args_into_dataclasses()
    set_seed(training_args.seed)
    args = Configure.Get(training_args.cfg)

    # 创建一个output文件夹
    os.makedirs(training_args.output_dir, exist_ok=True)


    if 'checkpoint-???' in args.bert.location:
        args.bert.location = get_last_checkpoint(
            os.path.dirname(args.bert.location.model_name_or_path))
        logger.info(f"Resolve model_name_or_path to {args.bert.location.model_name_or_path}")

    # debug 
    if 'wandb' in training_args.report_to and training_args.local_rank <= 0:
        import wandb

        init_args = {}
        if "MLFLOW_EXPERIMENT_ID" in os.environ:
            init_args["group"] = os.environ["MLFLOW_EXPERIMENT_ID"]
        wandb.init(
            project=os.getenv("WANDB_PROJECT", "bird"),
            name=training_args.run_name,
            # entity=os.getenv("WANDB_ENTITY", 'sgtnew'),
            **init_args,
        )
        # import pdb; pdb.set_trace()
        # wandb.config.update(training_args, allow_val_change=True)  # TypeError: Object of type HfTrainerDeepSpeedConfig is not JSON serializable 暂时有BUG

    # Detect last checkpoint
    last_checkpoint = None
    if os.path.isdir(
            training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )


    # The inputs will be train, dev, test or train, dev now.
    # We deprecate the k-fold cross-valid function since it causes too many avoidable troubles.

    if not args.arg_paths:
        cache_root = os.path.join('output', 'cache')
        os.makedirs(cache_root, exist_ok=True)
        raw_datasets_split: datasets.DatasetDict = datasets.load_dataset(path=args.dataset.loader_path,
                                                                         cache_dir=args.dataset.data_store_path,
                                                                         trust_remote_code=True)
        seq2seq_dataset_split: tuple = utils.tool.get_constructor(args.seq2seq.constructor)(args).to_seq2seq(
            raw_datasets_split, cache_root)
    else:
        cache_root = os.path.join('output', 'cache')
        os.makedirs(cache_root, exist_ok=True)
        meta_tuning_data = {}
        for task, arg_path in args.arg_paths:
            # 这里的arg_paths是大的config传入的，默认是META_TUNING/bird.cfg
            task_args = Configure.Get(arg_path)
            task_args.bert = args.bert
            print('task_args.bert.location:', task_args.bert.location)

            # 这里的path传入的是一个python文件的路径./finetuning/tasks/bird.py，这个文件中定义了一个类，这个类继承自datasets.GeneratorBasedBuilder
            # import pdb; pdb.set_trace()
            task_raw_datasets_split: datasets.DatasetDict = datasets.load_dataset(
                path=task_args.dataset.loader_path,
                cache_dir=task_args.dataset.data_store_path,
                trust_remote_code=True)
            
            # # debug 对eval_dataset进行切分
            # tmp = task_raw_datasets_split['validation'][:20]
            # task_raw_datasets_split['validation'] = datasets.Dataset.from_dict(tmp)

            # task_args.seq2seq.constructor是seq2seq_construction.bird；主要是添加SchemaLinking的数据
            task_seq2seq_dataset_split: tuple = utils.tool.get_constructor(task_args.seq2seq.constructor)(task_args).\
                to_seq2seq(task_raw_datasets_split, cache_root)

            meta_tuning_data[arg_path] = task_seq2seq_dataset_split

        # bird/finetuning/seq2seq_construction/meta_tuning.py  # 貌似是做上采样的，暂时不动
        seq2seq_dataset_split: tuple = utils.tool.get_constructor(args.seq2seq.constructor)(args).\
            to_seq2seq(meta_tuning_data)

    # 'metrics.meta_tuning.evaluator'
    evaluator = utils.tool.get_evaluator(args.evaluate.tool)(args)
    if not (training_args.use_lora and not training_args.do_train):  # 非lora评测
        model = utils.tool.get_model(args.model.name)(args, training_args)
        model_tokenizer = model.tokenizer

    if training_args.use_lora and  training_args.do_train: # lora训练
        # 进行peft的设置
        lora_config = LoraConfig(
        r=16,
        target_modules=["q_proj", "v_proj"],
        task_type=TaskType.CAUSAL_LM,
        lora_alpha=32,
        lora_dropout=0.05
    )
        lora_model = get_peft_model(model.pretrain_model, lora_config)  # 用peft对模型model.pretrain_model进行包装
        lora_model.print_trainable_parameters()  # 打印模型的可训练参数
        model = lora_model
    elif training_args.use_lora and not training_args.do_train:  # lora评测
        # config = PeftConfig.from_pretrained(training_args.load_weights_from)
        # model = LlamaForCausalLM.from_pretrained(config.base_model_name_or_path)
        # lora_model = PeftModel.from_pretrained(model, training_args.load_weights_from)
        lora_model = AutoPeftModelForCausalLM.from_pretrained(training_args.load_weights_from, torch_dtype=torch.bfloat16)
        model_tokenizer = LlamaTokenizer.from_pretrained(training_args.load_weights_from)
        model = lora_model


    
    # # debug 加速
    # model_tokenizer = LlamaTokenizer.from_pretrained(args.bert.location, use_fast=False)
    # #  add padding token to tokenizer
    # if model_tokenizer.pad_token is None:
    #     model_tokenizer.pad_token = model_tokenizer.eos_token


    seq2seq_train_dataset, seq2seq_eval_dataset, seq2seq_test_dataset = None, None, None
    if len(seq2seq_dataset_split) == 2:
        seq2seq_train_dataset, seq2seq_eval_dataset = seq2seq_dataset_split
    elif len(seq2seq_dataset_split) == 3:
        seq2seq_train_dataset, seq2seq_eval_dataset, seq2seq_test_dataset = seq2seq_dataset_split
    else:
        raise ValueError("Other split not support yet.")
    

    # We wrap the "string" seq2seq data into "tokenized tensor".


    # 将seq2seq数据集转换为tokenized数据集
    train_dataset = TokenizedDataset(args, training_args, model_tokenizer,
                                     seq2seq_train_dataset) if seq2seq_train_dataset else None
    # import pdb; pdb.set_trace()
    is_eval = True
    # import pdb; pdb.set_trace()
    eval_dataset = TokenizedDataset(args, training_args, model_tokenizer,
                                    seq2seq_eval_dataset, is_eval) if seq2seq_eval_dataset else None

    # import pdb; pdb.set_trace()
    # args.dataset.is_eval = False
    # test_dataset = TokenizedDataset(args, training_args, model_tokenizer,
    #                                 seq2seq_test_dataset) if seq2seq_test_dataset else None
    
    # import pdb; pdb.set_trace()
    
    # # debug 
    # ev_ids = eval_dataset[0]["input_ids"]
    # tr_ids = train_dataset[0]["input_ids"]
    # ev_texts = model_tokenizer.decode(ev_ids, skip_special_tokens=True)
    # tr_texts = model_tokenizer.decode(tr_ids, skip_special_tokens=True)
    # import pdb; pdb.set_trace()
    # print(ev_texts)
    # print('_________________________')
    # print(tr_texts)

    # import pdb; pdb.set_trace()

    # Initialize our Trainer
    #暂时关闭early stopping，需要的时候放在callbacks中(列表里)
    early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=args.seq2seq.patience if args.seq2seq.patience else 5)
    # 运行cProfile
    # profiler = cProfile.Profile()
    # profiler.enable()
    trainer = EvaluateFriendlySeq2SeqTrainer(
        args=training_args,
        model=model,
        evaluator=evaluator,
        # We name it "evaluator" while the hugging face call it "Metric",
        # they are all f(predictions: List, references: List of dict) = eval_result: dict
        tokenizer=model_tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        eval_examples=seq2seq_eval_dataset,
        wandb_run_dir=wandb.run.dir if 'wandb' in training_args.report_to and training_args.local_rank <= 0 else None,
        callbacks=[early_stopping_callback],
    )
    print('Trainer build successfully.')
    # profiler.disable()
    # stats = pstats.Stats(profiler).sort_stats('cumtime')
    # stats.print_stats()
    # stats.dump_stats('output/profile.stats')
    # exit()
    # Load model weights (for --do_train=False or post finetuning).
    if training_args.load_weights_from and not training_args.use_lora: # 非lora下，加载模型权重
        state_dict = torch.load(os.path.join(training_args.load_weights_from, transformers.WEIGHTS_NAME), map_location="cpu")
        trainer.model.load_state_dict(state_dict, strict=True)
        # release memory
        del state_dict

    if args.load_multiple_prefix_module_weights_from:
        reconstruct_state_dict = OrderedDict()

        # load prefix modules
        for task_name, module_weight_location in args.load_multiple_prefix_module_weights_from:
            state_dict = torch.load(os.path.join(module_weight_location, transformers.WEIGHTS_NAME), map_location="cpu")
            MULTI_PREFIX_ATTR_NAME = "multi_prefix"
            for weight_name, stored_tensor in state_dict.items():
                if str(weight_name).startswith("pretrain_model"):
                    continue  # skip the pretrained model and we will load a new one from another place
                reconstruct_state_dict['{}.{}.{}'.format(MULTI_PREFIX_ATTR_NAME, "_".join(task_name.split("_")[:-1]), weight_name)] = stored_tensor
                # extract the prefix part and add them to dict

        # give it into the model
        trainer.model.load_state_dict(reconstruct_state_dict, strict=False)


        # release memory
        del reconstruct_state_dict

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint

        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        max_train_samples = len(train_dataset)
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate(
            metric_key_prefix="eval"
        )
        max_eval_samples = len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.do_predict:
        logger.info("*** Predict ***")

        predict_results = trainer.predict(
            test_dataset=test_dataset if test_dataset else eval_dataset,
            test_examples=seq2seq_test_dataset if seq2seq_test_dataset else seq2seq_eval_dataset,
            metric_key_prefix="predict"
        )
        metrics = predict_results.metrics
        max_predict_samples = len(test_dataset)
        metrics["predict_samples"] = min(max_predict_samples, len(test_dataset))

        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)


if __name__ == "__main__":
    main()