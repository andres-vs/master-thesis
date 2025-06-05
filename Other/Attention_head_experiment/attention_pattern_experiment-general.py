from transformer_lens import HookedEncoder
import transformers
import seaborn as sns
import matplotlib.pyplot as plt

# Load the model
tokenizer_name = "bert-base-uncased"
model_name = "andres-vs/bert-base-uncased-finetuned_Att-Noneg-depth0"
tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name)
model = HookedEncoder.from_pretrained(model_name, tokenizer=tokenizer, head_type='classification')

# Define and tokenize input
input_text = "[CLS]If someone is big and round then they are white. All nice, cold people are rough. If Charlie is rough and Charlie is blue then Charlie is nice. Charlie is cold. If Dave is nice and Dave is cold then Dave is round. Bob is nice. If Dave is big then Dave is white. Charlie is blue. Dave is round. Bob is round. Nice people are blue. Dave is cold. Dave is white. Charlie is nice. Charlie is white. Charlie is round. Charlie is big. Bob is rough. Bob is blue. Bob is big. Bob is cold. All cold people are blue. If Dave is rough and Dave is nice then Dave is big.[SEP]Dave is cold.[CLS]"
tokens = tokenizer(input_text, return_tensors='pt')

attention_patterns = {}

# Hook to capture attention scores
def hook_fn(module_output, hook=None):
    attention_patterns[hook.name] = module_output.detach()

# Register the hook for the attention pattern in layer 7 (capture next token prediction)
model.add_hook("blocks.11.attn.hook_pattern", hook_fn)

# Run the model with hooks
with model.hooks(fwd_hooks=[('blocks.11.attn.hook_pattern', hook_fn)]):
    model(tokens.input_ids)

# Access the attention pattern (softmaxed attention weights)
head_attention_scores = attention_patterns["blocks.11.attn.hook_pattern"][0, 3, :, :]


# Plot the heatmap for the attention scores
plt.figure(figsize=(10, 8))
sns.heatmap(head_attention_scores.cpu().numpy(), cmap=sns.color_palette("Blues", as_cmap=True), xticklabels=decoded_tokens, yticklabels=decoded_tokens)
plt.title("Attention Pattern for Head 3 in Layer 11")
plt.xlabel("Tokens Attended To")
plt.ylabel("Tokens Attending From")
plt.show()