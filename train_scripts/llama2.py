import torch
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline
from langchain import HuggingFacePipeline
from langchain import PromptTemplate,  LLMChain
import pandas as pd
from tqdm import tqdm  

def generate_predictions(dataset_path):
    '''
    Inferencing on given dataset with Llama2
    :param str dataset_path
    :return None
    '''
    text_dataset = pd.read_csv(dataset_path, index_col=0)
    text_dataset = text_dataset.sample(frac=1).reset_index(drop=True)
    text_dataset = text_dataset
    try:
        df_text = text_dataset['clean_text'].tolist()
    except:
        df_text = text_dataset['text'].tolist()

    token = "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"
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf", token=token)
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf",
                                                 device_map='auto',
                                                 torch_dtype=torch.float16,
                                                 token=token,
                                                 #load_in_4bit=True
                                                 )

    pipe = pipeline("text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                    max_new_tokens=15,
                    num_return_sequences=1,
                    eos_token_id=tokenizer.eos_token_id
                    )

    llm = HuggingFacePipeline(pipeline=pipe, model_kwargs={'temperature': 0})

    template = """
    Answer the question based on the context. You should follow ALL the following rules when generating and answering.
    - You are an AI assistant for a fake news detection website.
    - Your job is to assess the authenticity of the given input text and classify it as either real or fake news.
    - Only answer with real or fake, as stated previously. Do not respond with anything else.
    - Your answer should be only a single word (real or fake).
    - Always answer in a respectful manner.
    - Please use logic and reasoning while responding to the query.
    
    User Question : {text}
    Your answer:"""

    prompt = PromptTemplate(template=template, input_variables=["text"])
    llm_chain = LLMChain(prompt=prompt, llm=llm)

    MAX_TEXT_LENGTH = 512
    pred_df = pd.DataFrame(columns=['text', 'label', 'pred_label'])

    for i, text in tqdm(enumerate(df_text), total=len(df_text)):
        try:
            if len(text) > MAX_TEXT_LENGTH:
                text = text[:MAX_TEXT_LENGTH]
            output = llm_chain.run(text)
            pred_df.loc[i] = [text, text_dataset.loc[i, 'label'], output]
            if i % 500 == 0:
                pred_df.to_csv(f'llama2-{i}.csv', index=False)
        except TypeError:
            continue

    pred_df.to_csv('llama2.csv', index=False)

def main():
    parser = argparse.ArgumentParser(description="Generate predictions using the specified dataset.")
    parser.add_argument('--input_data', type=str, help='Path to input data file (CSV)')

    args = parser.parse_args()
    
    dataset_path = args.input_data

    generate_predictions(dataset_path)

if __name__ == "__main__":
    main()