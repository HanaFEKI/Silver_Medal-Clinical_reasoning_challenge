# Impport necessary libraries
import pandas as pd 
import numpy as np
import torch
import pytorch_lightning as pl
import re
import random


# Fix seed
def fix_seed(seed):
    pl.seed_everything(seed, workers = True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False

seed = 1024
fix_seed(seed)


# Load and preprocess data
train_df = pd.read_csv('/kaggle/input/rural-kenyan-healthcare-challenge/train.csv')
test_df = pd.read_csv('/kaggle/input/rural-kenyan-healthcare-challenge/test.csv')
sample_df = pd.read_csv('/kaggle/input/rural-kenyan-healthcare-challenge/SampleSubmission.csv')

# Data Cleaning
# Clean values
train_df['Health level'] = train_df['Health level'].replace('health centres', 'health centers')
test_df['Health level'] = test_df['Health level'].replace('health centres', 'health centers')

train_df['Nursing Competency'] = train_df['Nursing Competency'].replace('maternah and child health', 'maternal and child health')
train_df['Nursing Competency'] = train_df['Nursing Competency'].replace('mayernal and child health', 'maternal and child health')

train_df['Prompt'] = train_df['Prompt'].str.replace("health centres", "health centers")
test_df['Prompt'] = test_df['Prompt'].str.replace("health centres", "health centers")

# Drop unnecessary columns
train_df.drop(columns=['GPT4.0', 'LLAMA', 'GEMINI'], inplace=True)

## Extracting the questions from train and test
### Train dataset

def extract_after_word_split(text, word):
    if not isinstance(text, str):
        return None
    # Case-insensitive split
    lower_text = text.lower()
    lower_word = word.lower()
    index = lower_text.find(lower_word)
    if index == -1:
        return None
    # Split using the original text to preserve casing
    parts = [text[:index], text[index:]]
    after = parts[1][len(word):].strip()  # text after the word
    return after if after else None


def extract_questions(text):
    for word in ["questions", "question"]:
        res = extract_after_word_split(text, word)
        if res:
            return word + " " + res  # prepend the matched word
    return None


train_df['questions'] = train_df['Prompt'].apply(extract_questions)


def extract_questions_by_words_no_punct(text):
    question_words = [
        # Wh-questions
        'what', 'how', 'why', 'when', 'where', 'who', 'whom', 'which', 'whose',

        # Subject-auxiliary inversions (common)
        'do i', 'do we', 'do you', 'does he', 'does she', 'does it',
        'did i', 'did we', 'did you', 'did he', 'did she', 'did it',
        'can i',
 'can we', 'can you', 'can he', 'can she', 'can it',
        'could i', 'could we', 'could you', 'could he', 'could she', 'could it',
        'should i', 'should we', 'should you', 'should he', 'should she', 'should it',
        'would i', 'would we', 'would you', 'would he', 'would she', 'would it',
        'will i', 'will we', 'will you', 'will he', 'will she', 'will it',
        'have i', 'have we', 'have you', 'has he', 'has she', 'has it',
        'had i', 'had we', 'had you', 'had he', 'had she', 'had it',
        'am i', 'are you', 'is he', 'is she', 'is it',
        'was i', 'were you', 'was he', 'was she', 'was it',
    ]

    # Normalize question_words: lowercase and set for fast lookup
    qword_set = set(question_words)

    # First try splitting on punctuation
    clauses = re.split(r'[.?!]', text)
    if len(clauses) == 1:  # No punctuation found, fallback splitting on ' question ' or ' questions '
        if 'questions' in text.lower() or 'question' in text.lower():
            parts = re.split(r'(questions?|question)', text, flags=re.IGNORECASE)
            for i, part in enumerate(parts):
                if part.lower() in ['question', 'questions']:
                    result = "".join(
parts[i:])
                    return result.strip()
        
        # If no 'questions' or 'question' found or no punctuation, fallback to splitting by question words
        
        words = text.split()
        questions = []
        current_q = []
        i = 0
        while i < len(words):
            # Check next two words as phrase
            phrase = ""
            if i+1 < len(words):
                phrase = (words[i] + " " + words[i+1]).lower()
            single = words[i].lower()

            if phrase in qword_set:
                # phrase matches question start
                if current_q:
                    questions.append(" ".join(current_q).strip())
                    current_q = []
                current_q.append(words[i])
                current_q.append(words[i+1])
                i += 2
            elif single in qword_set:
                # single word question start
                if current_q:
                    questions.append(" ".join(current_q).strip())
                    current_q = []
                current_q.append(words[i])
                i += 1
            else:
                if current_q:
                    current_q.append(words[i])
                i += 1

        if current_q:
            questions.append(" ".join(current_q).strip())
        return " ".join(questions) if questions else None

    else:
        questions = [
            clause.strip() 
            for clause in clauses 
            if any(clause.strip().lower().startswith(w) for w in question_words)
        ]
        return " ".join(questions) if questions else None
    

none_rows = train_df[train_df['questions'].isna()]
extracted_questions = none_rows['Prompt'].apply(extract_questions_by_words_no_punct)
train_df.loc[none_rows.index, 'questions'] = extracted_questions

train_df.loc[28, 'questions'] = 'can dextrose 5 helped to boost blood sugars'
train_df.loc[54, 'questions'] = 'differential diagnosis for this patient'
train_df.loc[325, 'questions'] = 'should i refer to a dentist '


# Test dataset
test_df['questions'] = test_df['Prompt'].apply(extract_questions)
none_rows_test = test_df[test_df['questions'].isna()]
extracted_questions_test = none_rows_test['Prompt'].apply(extract_questions_by_words_no_punct)
test_df.loc[none_rows_test.index, 'questions'] = extracted_questions_test


#No questions word
def clean_question_start(text):
    """
    Remove 'question' or 'questions' if the string starts with it (case-insensitive),
    and strip leading whitespace.
    """
    if not isinstance(text, str):
        return text  # handle non-string cases gracefully

    text = text.strip()
    lowered = text.lower()

    if lowered.startswith("questions"):
        return text[len("questions"):].strip()
    elif lowered.startswith("question"):
        return text[len("question"):].strip()
    else:
        return text
    
train_df['questions'] = train_df['questions'].apply(clean_question_start)
test_df['questions'] = test_df['questions'].apply(clean_question_start)


## Extracting the context

def extract_scenario_without_intro(prompt, questions):
    if not isinstance(prompt, str) or not isinstance(questions, str):
        return None

    # Remove the questions part from the prompt if found
    idx_q = prompt.find(questions)
    if idx_q != -1:
        prompt = prompt[:idx_q].strip()

    # Find last occurrence of 'kenya' (case insensitive)
    kenya_pos = prompt.lower().rfind('kenya')
    if kenya_pos == -1:
        # No 'kenya' found, just return prompt without questions
        return prompt.strip()

    # Scenario is everything after 'kenya'
    scenario = prompt[kenya_pos + len('kenya'):].strip()

    # Remove any leading punctuation or whitespace after 'kenya'
    scenario = scenario.lstrip(' ,.-')

    return scenario

train_df['context'] = train_df.apply(lambda row: extract_scenario_without_intro(row['Prompt'], row['questions']), axis=1)
test_df['context'] = test_df.apply(lambda row: extract_scenario_without_intro(row['Prompt'], row['questions']), axis=1)


test_df.at[3, 'context'] += ' when she suddenly woke up from sleep and started twitching no history of ailments reports the baby was very okay during the day on examination the baby is stable no twitching noted vital signs normal and mrdt done negative'
test_df.loc[3, 'questions'] = 'what could be the problem with the girl what is the diagnosis and which medication do i prescribe'

test_df.at[4, 'context'] += ' who had come with a one year old boy who had never received the nine month vaccination that is measles and he was having a rash on the skin on the face the legs and the abdomen and the mother also said that the baby was really coughing on assessment the boy has red eyes red conjunctiva he had a very light runny nose and he was also feverish i queried measles infection'
test_df.loc[4, 'questions'] = 'what kind of treatment do i give at this point or should i just refer this boy and notify the public health so that they can take a sample to go and test for the measles virus'

test_df.at[10, 'context'] += ' when he came to the hospital and she was found in her house not talking and not speaking the patient had no history of any trauma hypertension or diabetes on examination the patient was sick looking and responsive the mouth seemed to be deviated to the left with general weakness on the right side on the right upper limb and the lower limb the bp at that time was 166 78 with an spo2 of 76 in room air'
test_df.loc[10, 'questions'] = 'can this patient be suffering from a cerebrovascular accident and if so how do i manage cerebrovascular accident if the patient needs an urgent ct scan and it s not available in our facility to rule out the diagnosis which medication can i give first'

test_df.at[34, 'context'] += ' when he comes to the facility when he started vomiting and passing stool he has no history of being treated for peptic ulcer disease or anything on assessment he looks pale and weak the other vitals are within normal range but on doing an hb the hb is 4 how can i go how can i manage this patient depending that now the hb is 4 and the patient might be in shock because of bleeding'
test_df.loc[34, 'questions'] = 'can i give the iv fluids or blood first this patient might benefit from further evaluation how can i help the patient'

test_df.at[50, 'context'] += ' who is known diabetic and has been on follow up with good adherence since 7 years ago came with complains of having noticed high sugars every time she checks with her glucometer at home since 1 week ago she has also been experiencing excessive sweating thirst weakness and excessive urination blood sugar test revealed high sugars and client has been adhering to diet well her vitals and other lab works were normal'
test_df.loc[50, 'questions'] = 'what could be the cause of the consistent high sugars how should i manage the case'
