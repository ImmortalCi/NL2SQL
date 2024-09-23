import os
from sympy import false
import torch
from torch.utils.data import Dataset
import pdb


class  TokenizedDataset(Dataset):
    # TODO: A unified structure-representation.
    def __init__(self, args, training_args, tokenizer, seq2seq_dataset, is_eval=False):
        self.args = args
        self.training_args = training_args
        self.tokenizer = tokenizer
        if args.seq2seq.model_type == 'DecoderOnly':
            self.tokenizer.padding_side = "left"
        self.seq2seq_dataset = seq2seq_dataset
        self.is_eval = is_eval

        self.conv_sep = " || "

    def __getitem__(self, index):
        raw_item = self.seq2seq_dataset[index]

        if raw_item["text_in"]:
            ###################
            # With text input #
            ###################
            if self.conv_sep in raw_item["text_in"]:
                ##################
                # Conversational #
                ##################
                # TODO (commented by Chen): the context part roughly follows the implementation of CoSQL by Tianbao.
                # text_in = "[utt n] || [utt n-1] | [utt n-2] | ..."
                index = raw_item["text_in"].index(self.conv_sep)
                if self.args.model.knowledge_usage == 'concatenate' or self.args.model.knowledge_usage is None:
                    # seq_in  = "[utt n] ; structured knowledge: struct_in ; context: [utt n-1] | [utt n-2] | ..."
                    seq_in = "{} ; structured knowledge: {} ; context: {}".format(raw_item["text_in"][:index],
                                                                                  raw_item["struct_in"],
                                                                                  raw_item["text_in"][index + len(self.conv_sep):])
                elif self.args.model.knowledge_usage == 'separate':
                    # seq_in  = "[utt n] ; context: [utt n-1] | [utt n-2] | ..."
                    seq_in = "{} ; context: {}".format(raw_item["text_in"][:index],
                                                       raw_item["text_in"][index + len(self.conv_sep):])
                else:
                    raise ValueError()
            else:
                ######################
                # Non-conversational #
                ######################
                if self.args.model.knowledge_usage == 'concatenate' or self.args.model.knowledge_usage is None:
                    # seq_in  = "text_in ; structured knowledge: struct_in"
                    if self.args.model.external_knowledge == 'concatenate':
                        seq_in = "{} ; evidence: {}; schema: {}".format(raw_item["text_in"], raw_item['evidence'], raw_item["struct_in"])
                    else:
                        seq_in = "{} ; schema: {}".format(raw_item["text_in"], raw_item["struct_in"])
                elif self.args.model.knowledge_usage == 'separate':
                    # seq_in  = "text_in"
                    seq_in = raw_item["text_in"]
                else:
                    raise ValueError()
        else:
            ######################
            # Without text input #
            ######################
            if self.args.model.knowledge_usage == 'concatenate':
                # seq_in  = "structured knowledge: struct_in"
                seq_in = "structured knowledge: {}".format(raw_item["struct_in"])
            elif self.args.model.knowledge_usage == 'separate':
                # seq_in  = ""
                seq_in = ""
            else:
                raise ValueError()
        
        # Concatenate description.
        if self.args.model.use_description and self.args.model.concatenate_description:
            seq_in = "{} ; {}".format(raw_item["description"], seq_in)

        if self.args.seq2seq.model_type == 'DecoderOnly' and not self.is_eval:
             # find the length of seq_in(without the seq_out)
            prompt_in_tokenized = self.tokenizer(
                seq_in,
                padding="do_not_pad",
                truncation=True,
                max_length=self.training_args.input_max_length,
            )['input_ids']

            # Remove the last token if it is an eos token
            if prompt_in_tokenized[-1] == self.tokenizer.eos_token_id:
                prompt_in_tokenized = prompt_in_tokenized[:-1]

            # 需要将seq_in和seq_out进行拼接
            seq_in = "{} {}".format(seq_in, raw_item["query"])
        elif self.args.seq2seq.model_type == 'DecoderOnly' and self.is_eval: # 评测时，对于大模型，在最后加上prompt
            prompt = "Please generate the query language for SQlite to get the result of the QUESTION."
            seq_in = "{} {}".format(seq_in, prompt)


        tokenized_question_and_schemas = self.tokenizer(
            seq_in,
            padding="max_length",
            truncation=True,
            max_length=self.training_args.input_max_length,
            # We found that set it as large as possible can boost the performance significantly
            # , meanwhile, due to the t5 uses a relative position coding, we need to manually
            # assign the max input length into some large numbers, instead of using the "max_model_length"
            # ,which the default is 512, which will hurt the performance a lot.
        )
        tokenized_inferred = self.tokenizer(
            raw_item["seq_out"],
            padding="max_length",
            truncation=True,
            max_length=self.training_args.generation_max_length,
            # We set the max_length of "seq_out" during training is the same with the one in inference.
        )

        if not self.args.seq2seq.model_type == 'DecoderOnly':  # Encoder-Decoder中的设置
            tokenized_inferred_input_ids = torch.LongTensor(tokenized_inferred.data["input_ids"])
            # Here -100 will let the model not to compute the loss of the padding tokens.
            tokenized_inferred_input_ids[tokenized_inferred_input_ids == self.tokenizer.pad_token_id] = -100
        
        if self.args.seq2seq.model_type == 'DecoderOnly' and not self.is_eval:  #decoder only 的训练时的设置
            tokenized_inferred_input_ids = tokenized_question_and_schemas.data["input_ids"].copy()  # 不加copy会改变原来的值
            tmp = [i for i, x in enumerate(tokenized_inferred_input_ids) if x == self.tokenizer.pad_token_id]
            last_index = max(tmp) if tmp else -1  #  找到最后一个pad的位置，作为pad token；这里的pad token是在左边的，定位后进行-100的mask

            
            

            # 需要对label的输入部分进行mask
            for i in range(len(prompt_in_tokenized)):
                try:
                    tokenized_inferred_input_ids[i+last_index+1] = -100     # last_index+1是因为要跳过pad token
                except IndexError:
                    print(f"IndexError: i={i}, last_index={last_index}, len(tokenized_inferred_input_ids)={len(tokenized_inferred_input_ids)}")
                    print(f"prompt_in_tokenized: {prompt_in_tokenized}")
                    print(f"tokenized_inferred_input_ids: {tokenized_inferred_input_ids}")
                    print(f"seq_in: {seq_in}")
                    print(f"raw_item['query']: {raw_item['query']}")
                    print(f"raw_item['seq_out']: {raw_item['seq_out']}")
                    raise IndexError
            if len(prompt_in_tokenized) > len(tokenized_question_and_schemas.data['input_ids']):
                raise ValueError(
                    f"Prompt is longer than the input, something went wrong. Prompt: {prompt_in_tokenized}, input:"
                    f" {tokenized_question_and_schemas.data['input_ids']}"
                )
            
            tokenized_inferred_input_ids = torch.LongTensor(tokenized_inferred_input_ids)

        elif self.args.seq2seq.model_type == 'DecoderOnly' and self.is_eval:  #decoder only 的评测时的设置，和encoder-decoder的评测时的设置一样
            tokenized_inferred_input_ids = torch.LongTensor(tokenized_inferred.data["input_ids"])

        item = {
            'input_ids': torch.LongTensor(tokenized_question_and_schemas.data["input_ids"]),
            'attention_mask': torch.LongTensor(tokenized_question_and_schemas.data["attention_mask"]),
            'labels': tokenized_inferred_input_ids,
        }
        # Add task name.
        if 'task_id' in raw_item:
            item['task_ids'] = raw_item['task_id']

        # Separate description tokenization.
        if self.args.model.use_description and self.args.model.map_description:
            tokenized_description = self.tokenizer(raw_item["description"],
                                                   padding="max_length",
                                                   truncation=True,
                                                   max_length=self.args.dataset.description_max_length,
                                                   )
            item['description_input_ids'] = torch.LongTensor(tokenized_description.data["input_ids"])
            item['description_attention_mask'] = torch.LongTensor(tokenized_description.data["attention_mask"])

        # Separate knowledge tokenization.
        if self.args.model.knowledge_usage == 'separate':
            tokenized_knowledge = self.tokenizer(raw_item["struct_in"],
                                                 padding="max_length",
                                                 truncation=True,
                                                 max_length=self.training_args.input_max_length,
                                                 )
            item['knowledge_input_ids'] = torch.LongTensor(tokenized_knowledge.data["input_ids"])
            item['knowledge_attention_mask'] = torch.LongTensor(tokenized_knowledge.data["attention_mask"])
        
        return item

    def __len__(self):
        return len(self.seq2seq_dataset)