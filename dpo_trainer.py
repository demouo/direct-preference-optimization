"""dpo_trainer.py
Build Direct Preference Optimization from scratch
@author: demouo
"""

from unsloth import FastLanguageModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn.functional as F
import torch


# Model
model_path = input("Model path: ") or "unsloth/Llama-3.2-1B-Instruct"
max_seq_length = 2056
load_in_4bit = True
device = "cuda:0" if torch.cuda.is_available() else "cpu"  # "mps" on MAC 

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_path,
    max_seq_length=max_seq_length,
    load_in_4bit=load_in_4bit,
)

ref_model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_path,
    max_seq_length=max_seq_length,
    load_in_4bit=load_in_4bit
)

# SFT model
model = FastLanguageModel.get_peft_model(
    model,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],

    r = 16,           # Larger = higher accuracy, but might overfit
    lora_alpha = 16,  # Recommended alpha == r at least
    lora_dropout = 0.1,
    bias = "none",
    random_state = 3407, 
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)

model.train()
ref_model.eval()
for param in ref_model.parameters():
    param.requires_grad = False

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-2)

# One training example
train_example = {
    "question": "I'm fat, so sad.",
    "chosen": "Oh sorry to hear, what's the point now?",
    "rejected": "Aha, so fat so funny your guy?",
}
beta = 0.1
prompt = train_example["question"]
chosen_response = train_example["chosen"]
rejected_response = train_example["rejected"]

chosen_input_text = prompt + tokenizer.eos_token + chosen_response
chosen_tokenized_text = tokenizer(chosen_input_text, return_tensors="pt").to(device)
chosen_input_ids = chosen_tokenized_text.input_ids
chosen_attention_mask = chosen_tokenized_text.attention_mask

rejected_input_text = prompt + tokenizer.eos_token + rejected_response
rejected_tokenized_text = tokenizer(rejected_input_text, return_tensors="pt").to(device)
rejected_input_ids = rejected_tokenized_text.input_ids
rejected_attention_mask = rejected_tokenized_text.attention_mask

# Forward
# (Policy) model
# Log probabilities of chosen texts
policy_chosen_outputs = model(
    input_ids=chosen_input_ids, attention_mask=chosen_attention_mask
)
policy_chosen_log_probs = F.log_softmax(policy_chosen_outputs.logits, dim=-1)

# Log probabilities of rejected texts
policy_rejected_outputs = model(
    input_ids=rejected_input_ids, attention_mask=rejected_attention_mask
)
policy_rejected_log_probs = F.log_softmax(policy_rejected_outputs.logits, dim=-1)

# Reference model
with torch.no_grad():
    # Log probabilities of chosen texts
    refer_chosen_outputs = ref_model(
        input_ids=chosen_input_ids, attention_mask=chosen_attention_mask
    )
    refer_chosen_log_probs = F.log_softmax(refer_chosen_outputs.logits, dim=-1)

    # Log probabilities of rejected texts
    refer_rejected_outputs = ref_model(
        input_ids=rejected_input_ids, attention_mask=rejected_attention_mask
    )
    refer_rejected_log_probs = F.log_softmax(refer_rejected_outputs.logits, dim=-1)

# Extract response part only
prompt_length = tokenizer(
    prompt + tokenizer.eos_token, return_tensors="pt"
).input_ids.shape[1]

# Be aware of the -1 due to generated text is 1 right shift to inputs
policy_chosen_log_probs_response = policy_chosen_log_probs[:, prompt_length - 1 : -1, :]

refer_chosen_log_probs_response = refer_chosen_log_probs[:, prompt_length - 1 : -1, :]

policy_rejected_log_probs_response = policy_rejected_log_probs[
    :, prompt_length - 1 : -1, :
]

refer_rejected_log_probs_response = refer_rejected_log_probs[
    :, prompt_length - 1 : -1, :
]

# Targets (Labels)
labels_chosen = chosen_input_ids[:, prompt_length:]
labels_rejected = rejected_input_ids[:, prompt_length:]

def get_response_log_probs(log_probs, labels):
    """
    Gather targets token's probability from vocab_size dim

    Args:
        log_probs (tensor[batch_size, seq_length, vocab_size]): The generated logit
        labels (tensor[batch_size, seq_length]): The true labels (indies)

    Returns:
        'tensor([batch_size])': Loss of each sample
    """
    # [batch_size, seq_length] -> [batch_size, seq_length, 1]
    labels = labels.unsqueeze(-1)
    per_token_logps = torch.gather(log_probs, dim=2, index=labels)
    # [batch_size, seq_length, 1] -> [batch_size, sequence] -> [batch_size]
    response_log_probs = per_token_logps.squeeze(-1).sum(dim=1)
    return response_log_probs

# Loss
pi_chosen_log_prob = get_response_log_probs(
    policy_chosen_log_probs_response, labels_chosen
)
pi_rejected_log_probs = get_response_log_probs(
    policy_rejected_log_probs_response, labels_rejected
)

ref_chosen_log_prob = get_response_log_probs(
    refer_chosen_log_probs_response, labels_chosen
)

ref_rejected_log_prob = get_response_log_probs(
    refer_rejected_log_probs_response, labels_rejected
)

# dpo loss
policy_reward_chosen = beta * (pi_chosen_log_prob - ref_chosen_log_prob)
policy_reward_rejected = beta * (pi_rejected_log_probs - ref_rejected_log_prob)
loss = -F.logsigmoid(policy_reward_chosen - policy_reward_rejected).mean()

# Backward and steps
optimizer.zero_grad()
loss.backward()
optimizer.step()

# Log loss
print(f"Loss: {loss.item()}")