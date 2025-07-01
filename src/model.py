### Create prompt
from textwrap import dedent
import transformers
import datasets
from transformers import BartForConditionalGeneration, BartTokenizer, DataCollatorForSeq2Seq
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from nltk.tokenize import sent_tokenize
import evaluate

def generate_prompt(row):
    experience = f"with {int(row['Years of Experience'])} years of experience in general nursing" if 'Years of Experience' in row and not pd.isna(row['Years of Experience']) else ""
    
    prompt = dedent(f"""
    ### Instruction ###
    Start by a simple summary of the medical case. Then answer the different questions given by the nurse in a simple and concise way. Start by the word summary.

    ### Nurse profile ###
    i am a nurse {experience} working in {row['Health level']} in {row['County']} in kenya i have competency in {row['Nursing Competency']} the clinical panel i have is {row['Clinical Panel']}

    ### Medical case: ###
    {row['scenario']}

    ### Nurse's questions: ###
    {row['questions']}
    """)
    
    return prompt



train_df['Modeling_Prompt'] = train_df.apply(generate_prompt, axis=1)
test_df['Modeling_Prompt'] = test_df.apply(generate_prompt, axis=1)

train_data, val_data = train_test_split(train_df, test_size=0.2, random_state=seed)

# Tokenizer & Model
model = BartForConditionalGeneration.from_pretrained("facebook/bart-large", forced_bos_token_id=0)
tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

max_input_length = 1024
max_target_length = 512

def preprocess_dataset(examples):
    model_inputs = tokenizer(
        examples["Modeling_Prompt"],
        max_length=max_input_length,
        truncation=True
    )
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            examples["Clinician"],
            max_length=max_target_length,
            truncation=True
        )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Tokenize datasets
train_dataset = Dataset.from_pandas(train_data)
val_dataset = Dataset.from_pandas(val_data)
# Create DatasetDict
dataset = DatasetDict(
    train=Dataset.from_pandas(train_data),
    test=Dataset.from_pandas(val_data)
)

# Tokenize dataset
tokenized_dataset = dataset.map(
    preprocess_dataset,
    batched=True,
    batch_size=8
)

# ROUGE metric
metric = evaluate.load("rouge")

def compute_metrics(eval_pred):
    preds, labels = eval_pred
    if isinstance(preds, tuple):
        preds = preds[0]
    
    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    decoded_preds = ["\n".join(sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(sent_tokenize(label.strip())) for label in decoded_labels]
    
    results = metric.compute(
        predictions=decoded_preds,
        references=decoded_labels,
        use_stemmer=True
    )
    
    return {
        "rouge1": results["rouge1"],
        "rouge2": results["rouge2"],
        "rougeL": results["rougeL"]
    }
