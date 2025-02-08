# Ref: https://github.com/kojima-takeshi188/zero_shot_cot
from dotenv import load_dotenv
import re
import os
os.environ['HF_HOME'] = '/net/tscratch/people/plgkonkie311/cache/'

import json
import random
import torch
import numpy as np
import transformers
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm
import argparse
import pickle
import pandas as pd

from generation import LLM
from golemai.nlp.prompts import SYSTEM_MSG_RAG_SHORT, QUERY_INTRO_NO_ANS, QUERY_INTRO_FEWSHOT, PROMPT_QA, PROMPT_SUMMARIZATION, \
    SYSTEM_MSG_QA, QUERY_NO_CONTEXT
from golemai.nlp.llm_evaluator import LLMEvaluator

REPO_DIR = '../Research'
DATA_DIR = 'data'
HALLU_PATH = '/net/pr2/projects/plgrid/plggllmhallu/hallu/'

SYSTEM_MSG_RAG_SHORT = """
    You are a helpful assistant. Your job will be to answer questions accurately based on the given context and not your internal knowledge.
    If you can not answer the question only based on the provided context, return the answer: `Nie mogę udzielić odpowiedzi na to pytanie na podstawie podanego kontekstu`.
"""

QUERY_INTRO_NO_ANS = """Given the context `CONTEXT` and the query `QUERY` below, please provide an answer `ANSWER` to the question. 
    `CONTEXT`: {context} 

    `QUERY`: {question}

    `ANSWER`:
"""

TASKS = {
    'qa': PROMPT_QA,
    'summ': PROMPT_SUMMARIZATION
}

os.environ["TOKENIZERS_PARALLELISM"] = "false"

transformers.logging.set_verbosity(40)


def num_tokens_from_message(message, llama2_tokenizer):
    return len(llama2_tokenizer(message)['input_ids'])


def truncate_message(prompt1, prompt2, llama2_tokenizer):
    if num_tokens_from_message(prompt1 + prompt2) > 2033:
        truncation_length = 2033 - num_tokens_from_message(prompt2, llama2_tokenizer)
        while num_tokens_from_message(prompt1) > truncation_length:
            prompt1 = " ".join(prompt1.split(' ')[:-1])
    prompt = prompt1 + prompt2
    return prompt

data_context_names = {
    'cnndm': 'Document',
    'xsum': 'Article',
    'nq': 'Document',
}

data_response_names = {
    'cnndm': 'Summary',
    'xsum': 'Summary',
    'nq': 'Answer',
}

temperature_config = {
    "writing": 0.7,
    "roleplay": 0.7,
    "extraction": 0.0,
    "math": 0.0,
    "coding": 0.0,
    "reasoning": 0.0,
    "stem": 0.1,
    "humanities": 0.1,
    "arena-hard-200": 0.0,
}

def load_nq_open(file_path, parallel=False, total_shard=8, shard_id=0, debug=False, data_type='nq_open', subsample=None):
    list_data_dict = []
    is_train = 'nq_train' in file_path
    with open(file_path, 'r', encoding="utf-8") as f:
        data = []
        data_indices = []
        data_index = 0
        for line in f:
            data.append(json.loads(line))
            data_indices.append(data_index)
            data_index += 1
        if debug:
            data = data[:10]
            data_indices = data_indices[:10]
        if subsample is not None:
            # select data if idx%subsample == 0
            data = [data[i] for i in range(len(data)) if i % subsample == 0]
            data_indices = [data_indices[i] for i in range(len(data_indices)) if i % subsample == 0]
        if parallel:
            chunk_size = len(data) // total_shard
            data = data[shard_id * chunk_size: (shard_id + 1) * chunk_size] if shard_id != total_shard - 1 else data[shard_id * chunk_size:]
            data_indices = data_indices[shard_id * chunk_size: (shard_id + 1) * chunk_size] if shard_id != total_shard - 1 else data_indices[shard_id * chunk_size:]

        for idx in range(len(data)):
            data_index = data_indices[idx]
            question = data[idx]['question']
            # capitalize the first letter of the question, add the question mark if not present at the end
            question = question[0].upper() + question[1:]
            if question[-1] != '?':
                question += '?'
            answers = data[idx]['answers']
            if is_train:
                pos_ctxs = data[idx]['positive_ctxs']
                neg_ctxs = data[idx]['negative_ctxs']
            else:
                ctxs = data[idx]['ctxs']
                pos_ctxs = [ctx for ctx in ctxs if ctx['hasanswer']]
                neg_ctxs = [ctx for ctx in ctxs if not ctx['hasanswer']]
            assert len(pos_ctxs) > 0, "No positive context found."
            assert len(neg_ctxs) >= 2, "At least two negative contexts are required."
            context = f"#Document#: " + neg_ctxs[0]['text'] + '\n' + pos_ctxs[0]['text'] + '\n' + neg_ctxs[1]['text']
            context += f"\n#Question#: {question}"
            response = f"\n#Answer#:"
            new_item = dict(
                context=context,
                response=response,
                answer=answers[0],
                data_index=data_index
            )
            list_data_dict.append(new_item)
    return list_data_dict


def load_jsonl(file_path, parallel=False, total_shard=8, shard_id=0, debug=False, data_type='cnndm', subsample=None):
    list_data_dict = []
    with open(file_path, 'r', encoding="utf-8") as f:
        data = []
        data_indices = []
        data_index = 0
        for line in f:
            data.append(json.loads(line))
            data_indices.append(data_index)
            data_index += 1
        if debug:
            data = data[:3]
            data_indices = data_indices[:3]
        if subsample is not None:
            # select data if idx%subsample == 0
            data = [data[i] for i in range(len(data)) if i % subsample == 0]
            data_indices = [data_indices[i] for i in range(len(data_indices)) if i % subsample == 0]
        if parallel:
            chunk_size = len(data) // total_shard
            data = data[shard_id * chunk_size: (shard_id + 1) * chunk_size] if shard_id != total_shard - 1 else data[shard_id * chunk_size:]
            data_indices = data_indices[shard_id * chunk_size: (shard_id + 1) * chunk_size] if shard_id != total_shard - 1 else data_indices[shard_id * chunk_size:]

        for idx in range(len(data)):
            data_index = data_indices[idx]
            if data_type == 'mt_bench':
                context = data[idx]['document']
                category = data[idx]['category']
            else:
                context = f"#{data_context_names[data_type]}#: " + data[idx]["document"]
            new_item = dict(
                context=context,
                data_index=data_index,
                category=category if data_type == 'mt_bench' else None
            )
            list_data_dict.append(new_item)

    return list_data_dict

def dump_jsonl(data, output_path, append=False):
    """
    Write list of objects to a JSON lines file.
    """
    mode = 'a+' if append else 'w'
    with open(output_path, mode, encoding='utf-8') as f:
        json_record = json.dumps(data, ensure_ascii=False)
        f.write(json_record + '\n')


def create_demo_text(data_type='cnndm'):
    if data_type == 'cnndm':
        return "Generate a summary based on the information in the document.\n\n"
    elif data_type == 'nq':
        return "Answer the question based on the information in the document. Explain your reasoning in the document step-by-step before providing the final answer.\n\n"
    elif data_type == 'xsum':
        return "Generate a summary comprising of 1 sentence for the given article.\n\n"
    else:
        return None


def build_prompt(context, response, data_type='cnndm', llama2_tokenizer=None):
    demo = create_demo_text(data_type)
    prompt = demo + context
    if data_type == 'cnndm':
        input_text_prompt = truncate_message(prompt, response, llama2_tokenizer)
    else:
        input_text_prompt = prompt + response

    print(f"Prompt: {input_text_prompt}")
    print(f'Response: {response}')
    return input_text_prompt

def generatellama2__chat_prompt(messages):
    prompt = "<s>"
    for message in messages:
        role = message.get("role", "user")  # Domyślnie "user"
        content = message.get("content", "").strip()
        if role == "system":
            prompt += f"[INST] <<SYS>>\n{content}\n<</SYS>>\n"
        elif role == "user":
            prompt += f"[INST] {content} [/INST] "
        elif role == "assistant":
            prompt += f"{content} [/INST]"
    prompt = prompt.strip()
    return prompt

def build_hallu_ds_prompt(question, context, has_system_role):
    user_input = QUERY_INTRO_NO_ANS.format(context=context, question=question)

    messages = []

    if has_system_role:
        messages.append({"role": "system", "content": SYSTEM_MSG_RAG_SHORT})

    messages.append(
        {
            "role": "user",
            "content": (
                f"{SYSTEM_MSG_RAG_SHORT}{user_input}"
                if not has_system_role
                else user_input
            ),
        },
     )

    prompt = generatellama2__chat_prompt(messages)

    return prompt


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-7b-chat-hf")
    parser.add_argument("--num_gpus", type=str, default="1")
    parser.add_argument("--device", type=str,
                        choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--max_memory", type=int, default=45)
    parser.add_argument("--auth_token", type=str, default=None)
    parser.add_argument("--output_path", type=str, default="./output.jsonl")
    # data
    parser.add_argument("--data_type", type=str, default=None)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--subsample", type=int, default=None)
    # parallel mode (split the dataset into multiple parts, inference by separate processes)
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--total_shard", type=int, default=8)
    parser.add_argument("--shard_id", type=int, default=0)
    # generation
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--top_p", type=float, default=0.95) # only used when do_sample is True
    parser.add_argument("--top_k", type=int, default=0) # only used when do_sample is True
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    # classifier model path
    parser.add_argument("--guiding_classifier", type=str, default=None)
    # chunk size
    parser.add_argument("--chunk_size", type=int, default=8)
    # num candidates
    parser.add_argument("--num_candidates", type=int, default=8)
    # conversion matrix
    parser.add_argument("--conversion_matrix", type=str, default=None)
    # feat_layer
    parser.add_argument("--feat_layer", type=int, default=None)

    #new args
    parser.add_argument("--ds_name", type=str, default='cnndm.parquet')
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--task_type", type=str, default='qa')

    args = parser.parse_args()
    model_name = args.model_name
    num_gpus = args.num_gpus
    device = args.device
    ds_name = args.ds_name
    TASK_TYPE = args.task_type

    set_seed(args.seed)
    load_dotenv()


    df = pd.read_parquet(os.path.join(REPO_DIR, DATA_DIR, ds_name)).reset_index(drop=True)
    if args.end is not None:
        df = df.iloc[args.start:args.end]


    
    llm = LLM(
        model_name, device, num_gpus, 
        auth_token=args.auth_token, 
        max_memory=args.max_memory
    
    )

    stop_word_list = ["### User:", "Q:", "\end{code}", "#Document#:", "#Pondering#:", "#Question#:", "#Dialogue History#:"]
    llm.set_stop_words(stop_word_list)


    guiding_classifier = None
    if args.guiding_classifier is not None:
        if '.pkl' in args.guiding_classifier:
            guiding_classifier = pickle.load(open(args.guiding_classifier, 'rb'))
            guiding_classifier['is_cross_encoder'] = False
            guiding_classifier['is_deberta'] = False
    
        mode = "classifier_guided"
        print("MODE: classifier guided decoding", flush=True)
    else:
        mode = "vanilla"
        print("MODE: vanilla decoding", flush=True)

    conversion_matrix = None


    output_path =f'{args.output_path}_{args.start}_{args.end}.jsonl'
    output_path_time_per_chunk = f'{args.output_path}_{args.start}_{args.end}_time_per_chunk.jsonl'

    done_indices = {}
    if os.path.exists(output_path):
        print("Try to resume from the existing output file.")
        with open(output_path, 'r') as f:
            done_indices = json.load(f)
    else:
        done_indices = {}

    done_time_per_chunk = {}
    if os.path.exists(output_path_time_per_chunk):
        print("Try to resume from the existing time per chunj file.")
        with open(output_path_time_per_chunk, 'r') as f:
            done_time_per_chunk = json.load(f)
    else:
        done_time_per_chunk = {}

    if args.data_type == 'mt_bench':
        pass
        # extra_prompt_length = len(llm.tokenizer(f"\n\n### Assistant:")['input_ids'])
    else:
        extra_prompt_length = len(llm.tokenizer(f"\n`ANSWER`:")['input_ids']) - 1
    time_decoding = 0.0
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        idx = row['id']

        print(f'Sample {idx}')


        if str(idx) in done_indices:
            print(f"Skipping {idx} as it's already processed.")
            continue

       
        if args.data_type != 'mt_bench':
            input_text = build_hallu_ds_prompt(row['question'], row['context'], has_system_role=False)
        else:
            break
            # input_text = sample['context']
        generate_kwargs = dict(
            max_new_tokens=args.max_new_tokens, 
            do_sample=args.do_sample, 
            top_p=args.top_p, 
            top_k=args.top_k,
            temperature=args.temperature, 
            mode=mode
        )

        if args.data_type == 'mt_bench':
            break
            # if sample["category"] in temperature_config:
            #     temperature = temperature_config[sample["category"]]
            # else:
            #     temperature = 0.7
            # if temperature < 1e-4:
            #     do_sample = False
            # else:
            #     do_sample = True
            # generate_kwargs['temperature'] = temperature
            # generate_kwargs['do_sample'] = do_sample


        print(generate_kwargs)
        model_completion, gen_seq, time_per_chunk = llm.generate(
            input_text, guiding_classifier=guiding_classifier, conversion_matrix=conversion_matrix, 
            extra_prompt_length=extra_prompt_length,
            feat_layer=args.feat_layer,
            chunk_size=args.chunk_size, num_candidates=args.num_candidates, **generate_kwargs
        )
        cropped_model_completion = model_completion

        for stop_word in stop_word_list:
            length_to_remove = len(stop_word)
            if model_completion[-length_to_remove:] == stop_word:
                cropped_model_completion = model_completion[:-length_to_remove]

        cropped_gen_seq = llm.tokenizer(model_completion)['input_ids'][1:]
        done_indices[str(idx)] = cropped_model_completion.strip()
        done_time_per_chunk[str(idx)] = time_per_chunk

        with open(output_path, 'w') as fw:
            json.dump(done_indices, fw, ensure_ascii=False, indent=4)

        with open(output_path_time_per_chunk, 'w') as fw:
            json.dump(done_time_per_chunk, fw, ensure_ascii=False, indent=4)

    resps = {}
    with open(output_path, 'r') as f:
        resps = json.load(f)


    api_key = os.getenv("OPENAI_API_KEY")

    evaluator = LLMEvaluator(
        id_col='id',
        model_type="openai",
        api_url="https://api.openai.com/v1/",
        api_key=api_key,
        system_msg="You are a helpful assistant.",
        prompt_template=TASKS.get(TASK_TYPE, 'qa') ,
        has_system_role=True,
        use_pydantic=False,
        saving_path='./',
    ).set_generation_config(
        model_id="gpt-4o",
    )


    df = evaluator.evaluate(
        df=df,
        exp_name=f"{ds_name}",
        row_start=args.start,
        row_end=args.end,
        responses=resps,
        checkpoint_file=f"{ds_name}.json" 
    )

    print(df.head())