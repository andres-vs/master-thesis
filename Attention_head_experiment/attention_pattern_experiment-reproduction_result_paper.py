import torch
from transformer_lens import HookedTransformer
import transformers
import seaborn as sns
import matplotlib.pyplot as plt

# Load the GPT-2 small model with pre-trained weights
model_name = "gpt2"  # or specify a different GPT-2 variant
model = HookedTransformer.from_pretrained(model_name)

# Example input: "The war lasted from the year 1732 to the year 17"
input_text = "The war lasted from the year 1732 to the year 17"
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
tokens = tokenizer(input_text, return_tensors='pt')

attention_patterns = {}

# Hook to capture attention scores
def hook_fn(module_output, hook=None):
    attention_patterns[hook.name] = module_output.detach()

# Register the hook for the attention pattern in layer 7 (capture next token prediction)
model.add_hook("blocks.7.attn.hook_pattern", hook_fn)

# Run the model with hooks
with model.hooks(fwd_hooks=[('blocks.7.attn.hook_pattern', hook_fn)]):
    model(tokens.input_ids)

# Access the attention pattern (softmaxed attention weights)
head_attention_scores = attention_patterns["blocks.7.attn.hook_pattern"][0, 10, :, :]


# Plot the heatmap for the attention scores
plt.figure(figsize=(10, 8))
sns.heatmap(head_attention_scores.cpu().numpy(), cmap=sns.color_palette("Blues", as_cmap=True))
plt.title("Attention Pattern for Head 10 in Layer 7")
plt.xlabel("Tokens Attended To")
plt.ylabel("Tokens Attending From")
plt.show()