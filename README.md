# FNLLM, an LLM approach to Detecting Fake News
FNLLM is a project that aims to determine the usability of zero-shot LLM inference for fake news classification with Llama2 and Mistral. This project utilizes LLMs like Llama2 and Mistral to achieve the results and compare them with ML classifiers and BERT as baseline models.

## Installation
To run the classifiers on their own, follow these steps: 
1. Clone this repository.
```
git clone https://github.com/oguzaliarslan/Group_7
```
2. Run the following code to install dependencies:
```
pip install -r requirements.txt
```
3. **Datasets** are too big to upload on Github and can be access from the drive link below,
```
https://drive.google.com/drive/folders/1s5vXdgWO-S_EANivU_TNnP4ovmQY1a81?usp=sharing
```
These datasets are required for EDA.

4. **Models** are also too big to upload on GitHub and can be accessed from the drive link below
```
https://drive.google.com/drive/folders/1mxFLV7FpvzaIOxebxR2KWDofVS-t1fG-
```

5. **Cleaned Datasets** are too big to upload on Github and can be access from the drive link below,
```
https://drive.google.com/drive/folders/1Yf6UKXsvUZj--joCH8irCIMDhKtK7dYp
```
These datasets are required for Result Analysis.

6. Results obtained by the trainings and evaluation done by use can be seen in **main.ipynb**

## Executing the scripts
Training scripts can be runned on their own.
1. ML classifiers
```
python train_scripts/classifiers.py  --input_data <input_path> --output_folder <output_folder_name> --grid_search
```
The last --grid_search is optional and can be removed if not wanted.
2. Llama2 Inference
```
python train_scripts/llama2.py --input_data <input_path>
```
3. BERT Train
```
python train_scripts/llama2.py --input_data <input_path> --model_name <model_name_from_huggingface>
```

