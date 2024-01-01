import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline
from langchain import HuggingFacePipeline
from langchain import PromptTemplate,  LLMChain
import pandas as pd
from tqdm import tqdm  

# TODO: Create a proper class structure for generalization
text_dataset = pd.read_csv('dataset_clean.csv', index_col=0)
text_dataset = text_dataset.sample(frac=1).reset_index(drop=True)
text_dataset = text_dataset[0:1500]
df_text = text_dataset['clean_text'].tolist()

token = "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"
print('tokenizer')
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf",
                                          token=token)
print('model')
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf",
                                             device_map='auto',
                                             torch_dtype=torch.float16,
                                             token=token,
                                            #  load_in_8bit=True,
                                            load_in_4bit=True
                                             )


pipe = pipeline("text-generation",
                model=model,
                tokenizer= tokenizer,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                max_new_tokens = 15,
                num_return_sequences=1,
                eos_token_id=tokenizer.eos_token_id
                )

# TODO: Check the prompt below.
# template = """Task
# Determine whether the Statement is a
# lie (Yes) or not (No) based on the Context
# and other information.
# # Output format
# Answer Yes or No as labels
# # Examples
# { examples }
# # Prediction
# Text: { text }
# Label:"""

llm = HuggingFacePipeline(pipeline = pipe, model_kwargs = {'temperature':0})

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
# even with quantization it is borderline impossible to test all of them with
text_dataset = text_dataset[0:1500]
df_text = text_dataset['clean_text'].tolist()
df_label = text_dataset['label'].tolist()
pred_df = pd.DataFrame({'text': df_text, 'label': df_label, 'pred_label':None})
outputs = []
# Turns out the problem was the lengths of the sentences.. tragic!
MAX_TEXT_LENGTH = 512  

for i, text in tqdm(enumerate(df_text), total=len(df_text)):
    try:
        if len(text) > MAX_TEXT_LENGTH:
            text = text[:MAX_TEXT_LENGTH]
        output = llm_chain.run(text)
        pred_df.loc[i, 'pred_label'] = output
        if i % 500 == 0:
          pred_df.to_csv(f'llama2-{i}.csv')
    except TypeError:
        continue

pred_df.to_csv(f'llama2.csv')