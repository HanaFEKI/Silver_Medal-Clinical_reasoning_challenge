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




#Define the EpochEndCallback for saving the generated submission files at each epoch
class EpochEndCallback(TrainerCallback):
    """
    Callback to generate test predictions and save submission file at the end of each epoch.
    """
    def __init__(self, test_df, tokenizer, device, output_dir="./submissions"):
        """
        Initialize the callback with the test dataframe and necessary components.
        
        Args:
            test_df: DataFrame containing test data with 'Prompt', 'Health level', and 'Master_Index' columns
            tokenizer: Tokenizer for encoding inputs and decoding outputs
            device: Device to use for inference (cuda or cpu)
            output_dir: Directory to save submission files
        """
        self.test_df = test_df
        self.tokenizer = tokenizer
        self.device = device
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
    
    def generate_batch_responses(self,model,prompts, batch_size=8, max_length=375, num_beams=4):
        all_responses = []
        
        model.to(self.device)
        for i in range(0, len(prompts), batch_size):
            try:
                batch = prompts[i:i+batch_size]
                
                print(f"Processing batch {i//batch_size + 1}, prompts {i+1}-{min(i+batch_size, len(prompts))}")
                
                valid_batch = [str(p) if p is not None else "" for p in batch]
                
                inputs = self.tokenizer(
                    valid_batch,
                    return_tensors="pt",
                    max_length=1024,
                    truncation=True,
                    padding="max_length"
                )
                
                input_ids = inputs.input_ids.to(self.device)
                attention_mask = inputs.attention_mask.to(self.device)
                
                outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=max_length,
                    num_beams=num_beams,
                    early_stopping=True,
                    no_repeat_ngram_size=2
                )
                
                decoded_outputs = [
                    self.tokenizer.decode(output, skip_special_tokens=True)
                    for output in outputs
                ]
                all_responses.extend(decoded_outputs)
                
                # Print progress
                print(f"Processed {i+len(batch)}/{len(prompts)} prompts")
                
            except Exception as e:
                print(f"Error in batch starting at index {i}: {e}")
                # Add placeholder responses for the failed batch
                all_responses.extend(["Error generating response"] * min(batch_size, len(prompts) - i))
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    print("CUDA cache cleared after error")
    
        return all_responses
    
    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        """Run at the end of each epoch to generate predictions and save submission."""
        epoch = state.epoch
        print(f"\n{'='*50}\nGenerating predictions for epoch {epoch}\n{'='*50}")
        
        # Make sure we're using the current model
        if model is None:
            model = kwargs.get("model", None)
            if model is None:
                print("Model not available in callback, skipping generation")
                return
        
        # Set model to evaluation mode
        model.eval()
        
        # Generate responses
        prompts = self.test_df['Modeling_Prompt'].tolist()
        print(f"Starting generation for {len(prompts)} prompts with batch size 4")
        generated_responses = self.generate_batch_responses(
            model=model,
            prompts=prompts,
        )

        # Create submission dataframe
        submission_df = pd.DataFrame({
            'Master_Index': self.test_df['Master_Index'],
            'Clinician': generated_responses
        })
        
        # Save the submission file with epoch number
        timestamp = datetime.now().strftime("%m%d_%H%M")
        submission_path = os.path.join(self.output_dir, f"submission_epoch{epoch:.1f}_{timestamp}.csv")
        submission_df.to_csv(submission_path, index=False)
        
        print(f"Submission file saved to: {submission_path}")
        
        # Set model back to training mode for the next epoch
        model.train()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

epoch_callback = EpochEndCallback(
    test_df=test_df,
    tokenizer=tokenizer,
    device=device,
    output_dir="./submissions"  # Directory to store submission files
)

class LRLoggingCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % args.logging_steps == 0:
            lr = kwargs["optimizer"].param_groups[0]["lr"]
            print(f"Step {state.global_step}: Learning Rate = {lr:.2e}")

training_args = Seq2SeqTrainingArguments(
    output_dir="./bart_finetuned",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=3e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,
    weight_decay=0.01,
    num_train_epochs=10,
    predict_with_generate=True,
    lr_scheduler_type="linear",  # "linear" or "cosine"
    optim="adamw_bnb_8bit",   # Optimizer choice
    fp16=True,                   # Mixed precision training
    bf16=False,                  # Disable bfloat16 if not using A100
    logging_steps=10,
    report_to="none",
    load_best_model_at_end=True,
    metric_for_best_model="rouge1",
    greater_is_better=True,
    generation_max_length=512,
    generation_num_beams=4,
    save_total_limit=2,
    warmup_ratio=0.1,
)

# Initialize trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[epoch_callback, LRLoggingCallback()],  # Add the epoch callback
)

# Train
train_result = trainer.train()

# Save model & tokenizer
trainer.save_model("./trained_model")
tokenizer.save_pretrained("./trained_model")

# Save training metrics
metrics_df = pd.DataFrame(trainer.state.log_history)
metrics_df.to_csv("training_history.csv", index=False)
