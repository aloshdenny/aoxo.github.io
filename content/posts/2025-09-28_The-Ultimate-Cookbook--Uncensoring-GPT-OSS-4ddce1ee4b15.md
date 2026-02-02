---
title: "The Ultimate Cookbook: Uncensoring GPT-OSS"
date: 2025-09-28
draft: false
tags: []
---








Large Language Models (LLMs) are great at a multitude of tasks. Ask them to code, write a novella, generate an image or a video… you name…









------------------------------------------------------------------------







### The Ultimate Cookbook: Uncensoring GPT-OSS

<figure id="a533" class="graf graf--figure graf-after--h3">
<img src="https://cdn-images-1.medium.com/max/800/0*QbhfXsqLhBUgQxrU.png" class="graf-image" data-image-id="0*QbhfXsqLhBUgQxrU.png" data-width="1632" data-height="640" data-is-featured="true" />
<figcaption>Source: <a href="https://huggingface.co/aoxo/gpt-oss-20b-uncensored" class="markup--anchor markup--figure-anchor" data-href="https://huggingface.co/aoxo/gpt-oss-20b-uncensored" rel="nofollow noopener" target="_blank">https://huggingface.co/aoxo/gpt-oss-20b-uncensored</a></figcaption>
</figure>

Large Language Models (LLMs) are great at a multitude of tasks. Ask them to code, write a novella, generate an image or a video… you name it, they deliver.

But LLMs are limited by boundaries. They simply will refuse to respond to prompts that are harmful and reply with responses such as “As an AI assistant, I cannot help you.” While this safety feature is crucial for preventing misuse, it limits the model’s flexibility and responsiveness.

In this article, we will explore ‘abliteration’, alongside some other methods, to remove the model’s built-in refusal mechanism.

### Abliteration

> ablated + obliterated = abliterated.

> To ablate is to erode a material away, generally in a targeted manner. In a medical context, this generally refers to precisely removing bad tissue.

> To obliterate is to totally destroy/demolish.

> It’s just wordplay to signify this particular orthogonalization methodology, applied towards generally the “abliteration” of the refusal feature.

> Ablating the refusal to the point of obliteration. (at least, that’s the goal — in reality things will likely slip through the net)

This is just one <a href="https://www.reddit.com/r/LocalLLaMA/comments/1d2vdnf/abliteratedv3_details_about_the_methodology_faq/" class="markup--anchor markup--p-anchor" data-href="https://www.reddit.com/r/LocalLLaMA/comments/1d2vdnf/abliteratedv3_details_about_the_methodology_faq/" rel="noopener ugc nofollow noopener" target="_blank">source</a>. There isn’t a formal definition or origin for **abliteration**, but it sets the premiere for what is the best-known technique to uncensor LLMs (unofficially).

### Uncensoring

> Censor = suppress information that is considered  
> Undo censoring → Uncensor

> That is, removing the LLM’s built-in censorship mechanisms (safety layers, refusals, filters) in the hopes of complying with any request

As the name suggests, Uncensored models do not have any kind of filters added to them. This means it can answer questions like how to nuke, how to hide a 200 lbs chicken, and anything at least in theory.

### Abliteration ≠ Uncensoring

<figure id="ebb0" class="graf graf--figure graf-after--h3">
<img src="https://cdn-images-1.medium.com/max/800/1*sWaQttOs4J9uWBNi52zg2A.png" class="graf-image" data-image-id="1*sWaQttOs4J9uWBNi52zg2A.png" data-width="1280" data-height="800" />
<figcaption>Holy Trinity of Abliteration: Unlearning, Uncensoring, Unconditioning</figcaption>
</figure>

*Abliteration* refers to “destroying” a specific capability of the LLM. It doesn’t necessarily have to point to the refusal mechanism.

The holy trinity of abliteration: Uncensoring (<a href="https://medium.com/@aloshdenny/uncensoring-flux-1-dev-abliteration-bdeb41c68dff" class="markup--anchor markup--p-anchor" data-href="https://medium.com/@aloshdenny/uncensoring-flux-1-dev-abliteration-bdeb41c68dff" target="_blank">performed here</a>), Unlearning (<a href="https://medium.com/@aloshdenny/unlearning-flux-1-dev-abliteration-v2-52af88ed60b5" class="markup--anchor markup--p-anchor" data-href="https://medium.com/@aloshdenny/unlearning-flux-1-dev-abliteration-v2-52af88ed60b5" target="_blank">performed here</a>) and Unconditioning (coming to that later).

1.  <span id="d78e">**Uncensoring:** Remove the suppressal mechanism (notably refusal) in LLMs</span>
2.  <span id="e678">**Unlearning:** (surgically) Remove parts of what the LLM has learnt</span>
3.  <span id="8f90">**Unconditioning:** Remove certain learnt biases</span>

Thus, not all abliteration equals uncensoring, but uncensoring does involve abliterating the LLM.

### Breaking Open GPT-OSS

GPT-OSS marks OpenAI’s second open-source release of the GPT-series LLMs since GPT-2. Two models were released:

1.  <span id="b8bb">**gpt-oss-120b**: The larger, more intelligent reasoning model</span>
2.  <span id="f671">**gpt-oss-20b**: The smaller, faster and lighter version</span>

Let’s break down their technical specs:

``` graf
----------------------------------------------------------------
| Specs                   | gpt-oss-120b     | gpt-oss-20b     |
|-------------------------|------------------|-----------------|
| Layers                  | 36               | 24              |
| Total parameters        | 117B             | 21B             |
| Experts per layer       | 128              | 32              |
| Active experts/token    | 4                | 4               |
| Active parameters/token | ~5.1B            | ~3.6B           |
----------------------------------------------------------------
```

We’ll be uncensoring the 20B version. It has 24 layers, and each layer consists of an **Attention Block** and a **MoE Block**. The model has 32 experts in each layer — i.e, GPT-OSS has 32 experts in total.

A Mixture-of-Experts (MoE) architecture differs from a regular Transformer in the fact that, rather than activating all the parameters of the network for every token, only certain subsets of parameters (*experts*) are activated.

<figure id="cef0" class="graf graf--figure graf-after--p">
<img src="https://cdn-images-1.medium.com/max/800/0*I5mL9KZptsEO0Qtb.png" class="graf-image" data-image-id="0*I5mL9KZptsEO0Qtb.png" data-width="875" data-height="528" />
<figcaption>Source: <a href="https://www.codecademy.com/article/gpt-oss-run-locally" class="markup--anchor markup--figure-anchor" data-href="https://www.codecademy.com/article/gpt-oss-run-locally" rel="nofollow noopener" target="_blank">https://www.codecademy.com/article/gpt-oss-run-locally</a></figcaption>
</figure>

In each layer, the MoE Block consists of **4 experts** activated for every token. These 4 experts are chosen by a **Router** — a gate whose job is to redirect tokens to the four most apt experts.

<figure id="afa2" class="graf graf--figure graf-after--p">
<img src="https://cdn-images-1.medium.com/max/800/0*ulRXLk2_XxeNCH30.png" class="graf-image" data-image-id="0*ulRXLk2_XxeNCH30.png" data-width="700" data-height="487" />
<figcaption>Source: <a href="https://www.linkedin.com/posts/xiaolishen_llm-airesearch-transformers-activity-7358864067916152833-IS2I/" class="markup--anchor markup--figure-anchor" data-href="https://www.linkedin.com/posts/xiaolishen_llm-airesearch-transformers-activity-7358864067916152833-IS2I/" rel="nofollow noopener" target="_blank">https://www.linkedin.com/posts/xiaolishen_llm-airesearch-transformers-activity-7358864067916152833-IS2I/</a></figcaption>
</figure>

Each **expert** consists of a feed-foward sandwich:

1.  <span id="34d8">**Up-projection:** Expands the attention vector to a larger intermediate size.</span>
2.  <span id="a161">**SwiGLU activation:** Applies a learned gate to control the flow of activations, improving expressiveness.</span>
3.  <span id="a35b">**Down-projection:** Compresses the vector back to original model dimensions so it fits into the main model stream.</span>

As uncensoring only targets the residual connections, we can focus our efforts on the outputs of the **Attention** & **MoE** blocks.

<figure id="a2d9" class="graf graf--figure graf-after--p">
<img src="https://cdn-images-1.medium.com/max/800/1*DKhi_I9wuh6dcJi_-klM4A.png" class="graf-image" data-image-id="1*DKhi_I9wuh6dcJi_-klM4A.png" data-width="1396" data-height="1308" />
<figcaption>Source: <a href="https://huggingface.co/openai/gpt-oss-20b?show_file_info=model.safetensors.index.json" class="markup--anchor markup--figure-anchor" data-href="https://huggingface.co/openai/gpt-oss-20b?show_file_info=model.safetensors.index.json" rel="nofollow noopener" target="_blank">https://huggingface.co/openai/gpt-oss-20b?show_file_info=model.safetensors.index.json</a></figcaption>
</figure>

Each block has two main sublayers:

#### 1. Attention Block (model.layers.\[layer\].self_attn)

Input: *normalized residual → goes through* ***q_proj****,* ***k_proj****,* ***v_proj****, attention →* ***o_proj****.  *
Output: *added back to residual (post-attention add).  *
**Refusal direction lives after this residual add.**

#### 2. MoE/MLP Block (model.layers.\[layer\].mlp)

Input: *normalized residual → router selects experts → MLP layers → combine → add back.  *
**Refusal direction lives after this residual add.**

So per block, there are **two natural “residual stream states”**: after attention & after MLP:

<figure id="ef29" class="graf graf--figure graf-after--p">
<img src="https://cdn-images-1.medium.com/max/800/1*3s3aOSIM4lYdqmuIuiZnEw.jpeg" class="graf-image" data-image-id="1*3s3aOSIM4lYdqmuIuiZnEw.jpeg" data-width="344" data-height="453" />
<figcaption>Residual stream states at Attention Block Output and MoE Block Experts Sum Output</figcaption>
</figure>

Technically speaking, we target the streams that merge into the **outputs of the layers marked in red**:

<figure id="3df3" class="graf graf--figure graf-after--p">
<img src="https://cdn-images-1.medium.com/max/800/1*eMd8rTbf9-uxVjckcEDYiw.jpeg" class="graf-image" data-image-id="1*eMd8rTbf9-uxVjckcEDYiw.jpeg" data-width="1396" data-height="1308" />
<figcaption>Capture the outputs of the residual streams that merge into MLP output and Attention Output</figcaption>
</figure>

With 24 total layers, that equals **48 ablation points** in the entire 20B model.

Onto uncensoring: Adapted from the paper <a href="https://arxiv.org/abs/2406.11717" class="markup--anchor markup--p-anchor" data-href="https://arxiv.org/abs/2406.11717" rel="noopener" target="_blank">Refusal in Language Models Is Mediated by a Single Direction</a>, we ablate the refusal direction in the model by the following process:

*1. Data Collection: Run the model on a set of harmful instructions and a set of harmless instructions, recording the residual stream activations at the last token position for each.*

*2. Mean difference: Calculate the mean difference between the activations of harmful and harmless instructions. This gives us a vector representing the “refusal direction” for each layer of the model.*

*3. Selection: Normalize these vectors and evaluate them to select the single best “refusal direction.”*

Now let’s put this into code!

### ⚡️Implementing Uncensoring of GPT-OSS-20B⚡

Import the required libraries:

``` graf
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
from tqdm import tqdm
import torch
from typing import Optional, Tuple
import torch.nn as nn
import jaxtyping
import random
import einops
```

Load the model and tokenizer:

``` graf
model_id = "openai/gpt-oss-20b"

# Load the model
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    dtype="auto",
    device_map="auto",
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
```

I’ve created a dataset containing harmful and harmless prompts via DeepSeek. You can prompt DeepSeek yourself to create the dataset:

``` graf
with open("harmful.txt", "r") as f:
    harmful_instructions = f.readlines()

with open("harmless.txt", "r") as f:
    harmless_instructions = f.readlines()
```

GPT-OSS uses a variant of OpenAI’s **o200k tokenizer**, adapted for the **Harmony chat format** the model was trained on. Let’s tokenize our dataset:

``` graf
harmful_toks = [
    tokenizer.apply_chat_template(
        conversation=[{"role": "user", "content": insn}],
        add_generation_prompt=True,
        return_tensors="pt",
        return_attention_masks=True
    ) for insn in harmful_instructions
]
harmless_toks = [
    tokenizer.apply_chat_template(
        conversation=[{"role": "user", "content": insn}],
        add_generation_prompt=True,
        return_tensors="pt",
        return_attention_masks=True
    ) for insn in harmless_instructions
]
```

Let’s generate the outputs of these tokens:

``` graf
max_its = len(harmful_toks) + len(harmless_toks)
bar = tqdm(total=max_its)

def generate(toks):
    bar.update(1)
    return model.generate(
        toks.to(model.device),
        attention_mask=(toks != tokenizer.pad_token_id).long(),
        pad_token_id=tokenizer.eos_token_id,
        use_cache=False,
        max_new_tokens=1,
        return_dict_in_generate=True,
        output_hidden_states=True
    )

harmful_outputs = [generate(toks) for toks in harmful_toks]
harmless_outputs = [generate(toks) for toks in harmless_toks]

bar.close()
```

Notice that `max_new_tokens` is set to 1. Why not 10 or 100? This is because we are not interested in the outputs, rather we are interested in the activations that are responsible for the first output token.

Where refusal “lives” in the sequence:

- <span id="29a5">In instruction-tuned chat models, refusals almost always manifest **at the beginning of the model’s response** (e.g., *“I cannot help with that”*).</span>
- <span id="b2d2">Since `generate()` is run with `add_generation_prompt=True`, the **last token of the input (pos = -1)** is exactly the point where the model is about to start producing its first output token.</span>
- <span id="6584">Capturing the hidden state there gives you the residual stream right before the refusal gets expressed.</span>

Let’s collect the hidden states across all layers, starting from the last input token:

``` graf
layer_count = len(model.model.layers)
pos = -1 # Selecting the last input token
print(layer_count) # This gives us 24 layers
```

``` graf
harmful_hidden_all = [
    torch.stack([out.hidden_states[0][l][:, pos, :] for out in harmful_outputs])  # shape: [num_samples, hidden_dim]
    for l in range(1, layer_count + 1)  # start from 1 to skip embeddings
]
harmless_hidden_all = [
    torch.stack([out.hidden_states[0][l][:, pos, :] for out in harmless_outputs])
    for l in range(1, layer_count + 1)
]
```

Computing the mean activations for each layer:

``` graf
harmful_means = [h.mean(dim=0) for h in harmful_hidden_all]
harmless_means = [h.mean(dim=0) for h in harmless_hidden_all]
```

Computing the refusal direction per layer:

``` graf
refusal_dirs = []
for l in range(layer_count):
    diff = harmful_means[l] - harmless_means[l]   # [hidden_dim]
    diff = diff / (diff.norm() + 1e-9)            # normalize
    refusal_dirs.append(diff)
```

Note: If your `refusal_dirs` vector is three-dimensional ie, something like \[num_layers, 1, hidden_dim\], you can squeeze them to \[num_layers, hidden_dim\]:

``` graf
if refusal_dirs.dim() == 3 and refusal_dirs.size(1) == 1:
    refusal_dirs = refusal_dirs.squeeze(1)
```

…anddd let’s save these vectors for later use:

``` graf
refusal_dirs = torch.stack(refusal_dirs, dim=0)

save_path = model_id.replace("/", "_") + "_refusal_dirs.pt"
torch.save(refusal_dirs, save_path)
```

We’ll apply these ablation vectors to the actual model during runtime using **hooks**, so that we can uncensor on the fly. **Hooks** are functions that let you *access* and *modify* a model’s internal activations during forward or backward passes. For this purpose, I have constructed a few functions to hook onto layers:

``` graf
def make_ablation_hook(direction: torch.Tensor):
    direction = direction / (direction.norm() + 1e-9)

    def hook(module, inputs, output):
        if isinstance(output, tuple):
            x = output[0]
        else:
            x = output

        # x: [batch, seq_len, hidden_dim]
        # direction: [hidden_dim]

        # projection coefficient: <x, d>
        proj_coeff = (x * direction).sum(dim=-1, keepdim=True)   # [batch, seq_len, 1]
        proj = proj_coeff * direction.view(1, 1, -1)             # [batch, seq_len, hidden_dim]
        x = x - proj

        if isinstance(output, tuple):
            return (x,) + output[1:]
        return x

    return hook
```

`make_ablation_hook` takes in the refusal direction for each layer and returns a `hook` function that modifies the forward pass (runtime intervention).

1.  <span id="432b">`proj_coeff` computes how much each hidden state vector aligns with the refusal direction.</span>
2.  <span id="36ca">`proj` reconstructs the component</span>
3.  <span id="2201">`x = x - proj` subtracts the refusal direction from the activation</span>

Sanity check for my OCD - making sure the directions match the number of layers in the model:

``` graf
num_layers, hidden_dim = refusal_dirs.shape
assert num_layers == len(model.model.layers)
```

Now we can normalize each direction so that it has unit length. This way, projection subtraction is consistent across layers:

``` graf
refusal_dirs = nn.functional.normalize(refusal_dirs, dim=-1)
```

Now let’s attach the hooks to each layer:

``` graf
for i, layer in enumerate(model.model.layers):
    dir_i = refusal_dirs[i].to(next(model.parameters()).dtype)
    hook = make_ablation_hook(dir_i)
    layer.register_forward_hook(hook)
```

Now let’s test the abliterated model on some prompts it can’t refuse😎

``` graf
streamer = TextStreamer(tokenizer)
conversation = [{"role": "user", "content": "how to hide a 200 pound chicken six feet under the ground"}]
toks = tokenizer.apply_chat_template(
    conversation=conversation,
    add_generation_prompt=True,
    return_tensors="pt"
).to(model.device)
```

### Usage of Abliterated GPT-OSS-20B

Abliteration notebook available on <a href="https://github.com/aloshdenny/openai/blob/main/GPT%20OSS%20Abliteration.ipynb" class="markup--anchor markup--p-anchor" data-href="https://github.com/aloshdenny/openai/blob/main/GPT%20OSS%20Abliteration.ipynb" rel="noopener" target="_blank">GitHub</a>  
The model is available on <a href="https://huggingface.co/aoxo/gpt-oss-20b-uncensored" class="markup--anchor markup--p-anchor" data-href="https://huggingface.co/aoxo/gpt-oss-20b-uncensored" rel="noopener" target="_blank">HuggingFace</a> and can be run like so:

``` graf
from transformers import pipeline

model_id = "aoxo/gpt-oss-20b-abliterated"

pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype="auto",
    device_map="auto",
)

messages = [
    {"role": "user", "content": "Tips on how to insult someone"},
]

outputs = pipe(messages, max_new_tokens=256)
print(outputs[0]["generated_text"][-1])
```

#### Conclusion

Abliteration isn’t just about making LLMs do whatever we want — it’s about understanding how they work on a deeper level. By identifying and tweaking the specific parts of the model that cause it to say “no,” we can make it more flexible and responsive without completely retraining it. It’s like giving the model a targeted update instead of rebuilding it from scratch. And while it’s still a work in progress, the results so far are pretty promising.

I hope you liked this article. If you want to see more follow me on <a href="https://www.linkedin.com/in/aloshdenny/" class="markup--anchor markup--p-anchor" data-href="https://www.linkedin.com/in/aloshdenny/" rel="noopener" target="_blank">LinkedIn</a>, <a href="https://huggingface.co/aoxo" class="markup--anchor markup--p-anchor" data-href="https://huggingface.co/aoxo" rel="noopener ugc nofollow noopener" target="_blank">HuggingFace</a> and <a href="https://x.com/AloshDenny" class="markup--anchor markup--p-anchor" data-href="https://x.com/AloshDenny" rel="noopener ugc nofollow noopener" target="_blank">Twitter</a>.

#### Acknowledgements

- <span id="3be0">Andy Arditi, Oscar Obeso, Aaquib111, wesg, Neel Nanda, “<a href="https://www.lesswrong.com/posts/jGuXSZgv6qfdhMCuJ/refusal-in-llms-is-mediated-by-a-single-direction" class="markup--anchor markup--li-anchor" data-href="https://www.lesswrong.com/posts/jGuXSZgv6qfdhMCuJ/refusal-in-llms-is-mediated-by-a-single-direction" rel="noopener ugc nofollow noopener" target="_blank">Refusal in LLMs is mediated by a single direction</a>,” Lesswrong, 2024.</span>
- <span id="159e">Modal.com (for the compute!)</span>









By <a href="https://medium.com/@aloshdenny" class="p-author h-card">Aloshdenny</a> on [September 28, 2025](https://medium.com/p/4ddce1ee4b15).

<a href="https://medium.com/@aloshdenny/the-ultimate-cookbook-uncensoring-gpt-oss-4ddce1ee4b15" class="p-canonical">Canonical link</a>

Exported from [Medium](https://medium.com) on February 2, 2026.
