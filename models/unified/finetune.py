#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from torch import nn
from .base import PushToHubFriendlyModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import LlamaForCausalLM, LlamaTokenizer, T5ForConditionalGeneration


class Model(PushToHubFriendlyModel):
    def __init__(self, args):
        super().__init__()
        self.args = args

        # Load tokenizer and model.
        self.tokenizer = LlamaTokenizer.from_pretrained(args.bert.location, use_fast=False)

        #  add padding token to tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token


        self.pretrain_model = LlamaForCausalLM.from_pretrained(
            args.bert.location,
            torch_dtype=torch.bfloat16
        )
        self.config = self.pretrain_model.config

        # self.tokenizer = AutoTokenizer.from_pretrained(args.bert.location, use_fast=False)
        # self.pretrain_model = AutoModelForSeq2SeqLM.from_pretrained(args.bert.location)

        self.main_input_name = "input_ids"

        if args.special_tokens:
            self.tokenizer.add_tokens([v for k, v in args.special_tokens])
            self.pretrain_model.resize_token_embeddings(len(self.tokenizer))

    def forward(self, input_ids, attention_mask, labels):
        loss = self.pretrain_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            labels=labels,
        ).loss
        return {'loss': loss}

    def generate(self, input_ids, attention_mask, **kwargs):
        generated_ids = self.pretrain_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            **kwargs,
        )

        return generated_ids
