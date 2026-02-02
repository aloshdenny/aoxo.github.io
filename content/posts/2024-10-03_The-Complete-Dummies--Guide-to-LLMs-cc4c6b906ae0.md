---
title: "The Complete Dummies’ Guide to LLMs"
date: 2024-10-03
draft: false
tags: []
---








Large Language Models (LLMs) have been the buzz since 2021. One of the most popular ones being ChatGPT, followed by Gemini and then Claude…









------------------------------------------------------------------------







### The Complete Dummies’ Guide to LLMs

Large Language Models (LLMs) have been the buzz since 2021. One of the most popular ones being ChatGPT, followed by Gemini and then Claude. But these are just the tip of the iceberg. LLMs have revolutionized natural language processing and artificial intelligence. However, the field is rife with complex terminology that can be daunting for newcomers. This guide aims to demystify key concepts and jargon related to LLMs, making the subject more accessible to a wider audience.

<figure id="4baf" class="graf graf--figure graf-after--p">
<img src="https://cdn-images-1.medium.com/max/800/1*eyIc4FRTbVeqcdPZSL3FDg.jpeg" class="graf-image" data-image-id="1*eyIc4FRTbVeqcdPZSL3FDg.jpeg" data-width="1024" data-height="1024" />
</figure>

This article is best read with a foundational level of knowledge on machine learning, backpropagation and computational mathematics.

### Table of Contents

1.  <span id="de9e">A Brief overview of LLMs and their importance</span>
2.  <span id="865a">Demystifying LLM terminology</span>
3.  <span id="c2e9">How LLMs learn</span>
4.  <span id="5a40">Transformers 101</span>
5.  <span id="13d6">You Just Want Attention</span>
6.  <span id="0d83">Training and Inference</span>
7.  <span id="96fc">Finetuning</span>
8.  <span id="5071">Advanced Finetuning Techniques</span>
9.  <span id="57e8">Model Initialization</span>
10. <span id="b256">Quantization</span>
11. <span id="cb89">A Brief Survey on Quantization Techniques</span>
12. <span id="1800">Advanced Quantization Techniques</span>
13. <span id="7dd0">Inference and Training Arithmetic</span>
14. <span id="3c65">LLM Pollution</span>
15. <span id="d88f">Conclusion: Putting it all together</span>

### A Brief Overview of LLMs and their importance

Language Model Models (LLMs) have gained immense popularity in recent years due to their ability to generate human-like text. These sophisticated algorithms, developed using deep learning techniques, have found a wide range of applications in various fields.

<figure id="694a" class="graf graf--figure graf-after--p">
<img src="https://cdn-images-1.medium.com/max/800/0*AKvu67bysFtCvncJ.jpg" class="graf-image" data-image-id="0*AKvu67bysFtCvncJ.jpg" data-width="400" data-height="400" />
</figure>

LLMs are essentially very, very deep neural networks (DNNs). For those of you don’t know what a deep (or even a neural network is), check <a href="https://medium.com/@Coursesteach/deep-learning-part-1-86757cf5a0c3" class="markup--anchor markup--p-anchor" data-href="https://medium.com/@Coursesteach/deep-learning-part-1-86757cf5a0c3" target="_blank">this article out</a>.

DNNs are known for their ability to capture latent (hidden) information, sequences and/or patterns within large data. Deep neural networks are particularly useful in computer vision and language modelling tasks, where a regular statistical algorithm may fail to fit to larger data.

Going back again, a very deep NN with the right architecture to learn latent representations within text data is what basically an LLM does. And this is quite important in the real world, where text-based tasks such as conventional business, conversation and translation require mass automation. LLMs had found their initial footing here.

<figure id="5552" class="graf graf--figure graf-after--p">
<img src="https://cdn-images-1.medium.com/max/800/0*-JdRkEKzLLVMvBTH.png" class="graf-image" data-image-id="0*-JdRkEKzLLVMvBTH.png" data-width="495" data-height="373" />
</figure>

It has been a few years since LLM has taken over the internet. Now everyone can communicate with their own LLMs through ChatGPT, Gemini, etc.

Now, you can even personalize your own LLM for your needs.

<figure id="0044" class="graf graf--figure graf-after--p">
<img src="https://cdn-images-1.medium.com/max/800/1*m1ZnBbLSpXPLZQaTDOFu2A.jpeg" class="graf-image" data-image-id="1*m1ZnBbLSpXPLZQaTDOFu2A.jpeg" data-width="2400" data-height="1320" />
<figcaption>Samantha from Her. Gosh, I want this ;)</figcaption>
</figure>

But later on that part ;)

### Demystifying LLM Terminology

<figure id="9ebf" class="graf graf--figure graf-after--h3">
<img src="https://cdn-images-1.medium.com/max/800/1*sakWpoAkWKfI49SJshhJYw.png" class="graf-image" data-image-id="1*sakWpoAkWKfI49SJshhJYw.png" data-width="969" data-height="480" />
<figcaption>I was in the same spot a couple years back</figcaption>
</figure>

You may have heard terms such as agents, few shot learning, finetuning, GPTs, hallucination, RAG, hybrid search, training, inference, blah blah blah. If you haven’t, dont’t fret.

Let’s cover a lot more than few concepts:

**0. Foundational Models:**

The gods of Generative AI. The Stanford Institute for Human-Centered Artificial Intelligence’s (HAI) Center for Research on Foundation Models (CRFM) coined the term “foundation model” in August 2021 to mean “any model that is trained on broad data (generally using self-supervision at scale) that can be adapted (e.g., fine-tuned) to a wide range of downstream tasks”.

<figure id="374f" class="graf graf--figure graf-after--p">
<img src="https://cdn-images-1.medium.com/max/800/1*5RyC1xDvbhB1Rp1S8IXVng.png" class="graf-image" data-image-id="1*5RyC1xDvbhB1Rp1S8IXVng.png" data-width="980" data-height="484" />
</figure>

In a nutshell, large foundation models such as Microsoft Florence (Vision), OpenAI’s GPT (coming up), CLIP (image captioning), Diffusion (Image Generation) have been retrained countless times on different data to meet different needs, but their core architecture and learned representation stay more or less the same, making it easier to build newer models from these gods.

**1. GPTs:**

<figure id="74cf" class="graf graf--figure graf-after--p">
<img src="https://cdn-images-1.medium.com/max/800/0*ZMNJtDUqHXV3T9Ni" class="graf-image" data-image-id="0*ZMNJtDUqHXV3T9Ni" data-width="640" data-height="394" />
</figure>

Abbreviation for ‘Generative Pre-Trained Transformer’. These are a class of LLMs that are built on the transformer architecture, a type of very deep DNN that has been proven to perform remarkably well on long range sequences.

**2. Agents:**

<figure id="88b5" class="graf graf--figure graf-after--p">
<img src="https://cdn-images-1.medium.com/max/800/0*Drpb78ZpJCprX5t0.jpg" class="graf-image" data-image-id="0*Drpb78ZpJCprX5t0.jpg" data-width="1400" data-height="700" />
</figure>

LLM agents are advanced AI systems designed for creating complex text that needs sequential reasoning. They can think ahead, remember past conversations, and use different tools to adjust their responses based on the situation and style needed. You can deploy a crew of agents specialized for automating a task, for instance, managing your emails, attending your calls while away, digging up research material, etc. <a href="https://www.crewai.com/" class="markup--anchor markup--p-anchor" data-href="https://www.crewai.com/" rel="noopener" target="_blank">CrewAI</a> has a well-developed suite for Agentic AI.

**3. Hallucination:**

<figure id="b254" class="graf graf--figure graf-after--p">
<img src="https://cdn-images-1.medium.com/max/800/1*_pFeAwHTKDZm_XdLJVUGWw.jpeg" class="graf-image" data-image-id="1*_pFeAwHTKDZm_XdLJVUGWw.jpeg" data-width="750" data-height="500" />
<figcaption>How the turn tables</figcaption>
</figure>

Yes, it’s LLMs making up stuff just like we do when giddy. Hallucinations in LLMs refer to the generation of content that is irrelevant, made-up, or inconsistent with the input data. This problem leads to incorrect information, challenging the trust placed in these models. Wanna try? Go ahead and ask Gemini or ChatGPT something that is partially true and after a couple of prompts, it should start hallucinating ;)

**4. Retrieval Augmented Generation (RAG):**

One step closer to your personal AI, RAG is an easy and popular way to use your own data by providing it as part of the prompt with which you query the LLM model. The name stands as you would retrieve the relevant data and use it as augmented context for the LLM. Instead of relying solely on knowledge derived from the training data, a RAG workflow pulls relevant information and connects static LLMs with real-time data retrieval.

<figure id="b384" class="graf graf--figure graf-after--p">
<img src="https://cdn-images-1.medium.com/max/800/0*MpxUSbrh112xyTpX.png" class="graf-image" data-image-id="0*MpxUSbrh112xyTpX.png" data-width="875" data-height="489" />
</figure>

A RAG workflow consists of the User, the LLM and a Vector Database (VDB). The VDB can be modified and updated however well you want the LLM to refer to ‘your’ factual information. Pinecone, Chroma and Weaviate are renowned cloud-based VDBs for integrating into your RAG workflow.

<figure id="d01f" class="graf graf--figure graf-after--p">
<img src="https://cdn-images-1.medium.com/max/800/1*pKrKxT5Urz_bgTyFTt5E4w.png" class="graf-image" data-image-id="1*pKrKxT5Urz_bgTyFTt5E4w.png" data-width="800" data-height="528" />
</figure>

If you’ve ever encountered the above messages while playing around, it’s because Web-based RAG wasn’t a feature yet. Now almost all LLM giants implement a RAG pipeline with access to the internet to stay up to date.

**5. Hybrid Search:**

<figure id="43dc" class="graf graf--figure graf-after--p">
<img src="https://cdn-images-1.medium.com/max/800/0*dyUYUblSvjFuVuCX.png" class="graf-image" data-image-id="0*dyUYUblSvjFuVuCX.png" data-width="1250" data-height="625" />
</figure>

It’s a cross between semantic search and RAG. Dumbing it down, it is a combination of keyword search and neural networks.

A relatively old idea that you could **take the best from both worlds — keyword-based old school search **— sparse retrieval algorithms like <a href="https://en.wikipedia.org/wiki/Tf%E2%80%93idf" class="markup--anchor markup--p-anchor" data-href="https://en.wikipedia.org/wiki/Tf%E2%80%93idf" rel="noopener ugc nofollow noopener" target="_blank">tf-idf</a> or search industry standard <a href="https://en.wikipedia.org/wiki/Okapi_BM25" class="markup--anchor markup--p-anchor" data-href="https://en.wikipedia.org/wiki/Okapi_BM25" rel="noopener ugc nofollow noopener" target="_blank">BM25</a> — **and modern** semantic or **vector search and combine it in one retrieval result.**  
The only trick here is to properly combine the retrieved results with different similarity scores — this problem is usually solved with the help of the <a href="https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf" class="markup--anchor markup--p-anchor" data-href="https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf" rel="noopener ugc nofollow noopener" target="_blank">Reciprocal Rank Fusion</a> algorithm, reranking the retrieved results for the final output.

Algolia and Cohere are pretty well-known hybrid search engines.

**6. Prompt Engineering:**

<figure id="2d2c" class="graf graf--figure graf-after--p">
<img src="https://cdn-images-1.medium.com/max/800/0*H7q2xvYmRx3i8jV6" class="graf-image" data-image-id="0*H7q2xvYmRx3i8jV6" data-width="640" data-height="345" />
</figure>

It is the process of crafting, refining, and optimizing the input prompts given to an LLM in order to achieve desired outputs. Prompt engineering plays a crucial role in determining the performance and behavior of models like those based on the GPT architecture.

If you’re a seasoned ChatGPT user, you would be knowing well that it requires a lot more than small talk to get things done. Unofficially, everyone is a prompt engineer!

**7. Upstream and Downstream Tasks:**

In the world of Large Language Models, we often talk about upstream and downstream tasks. But what do these aquatic-sounding terms actually mean? Let’s dive in!

<figure id="2d21" class="graf graf--figure graf-after--p">
<img src="https://cdn-images-1.medium.com/max/800/0*z0NzCq_O8q478E45.png" class="graf-image" data-image-id="0*z0NzCq_O8q478E45.png" data-width="671" data-height="271" />
</figure>

Imagine an LLM as a mighty river of knowledge. Upstream tasks are like the mountain springs and tributaries that feed this river. These are the fundamental tasks that help the model learn general language understanding and generation. Examples include:

- <span id="2814">Next word prediction</span>
- <span id="4b22">Masked language modeling</span>
- <span id="c047">Sentence completion</span>

These tasks are the heavy lifters, shaping the riverbed and determining the flow of the AI river.

Downstream tasks are like the various activities you can do once the river is flowing strong. These are specific applications or fine-tuned uses of the pre-trained model. Examples include:

- <span id="da25">Sentiment analysis</span>
- <span id="1fa4">Named entity recognition</span>
- <span id="0ac0">Text summarization</span>

**8. Transfer Learning**

<figure id="19eb" class="graf graf--figure graf-after--p">
<img src="https://cdn-images-1.medium.com/max/800/0*31UbUQ-LAhRc2Zee.jpeg" class="graf-image" data-image-id="0*31UbUQ-LAhRc2Zee.jpeg" data-width="568" data-height="335" />
</figure>

Training an LLM model is a huge, costly and painful task (I’m gonna give you the gore details later on). Transfer learning popped into the picture when researchers realized you could attain almost the same accuracy, precision and capabilities of the parent model with little to no loss by transferring its knowledge to a smaller daughter model.

YOLO is a family of Computer Vision, Detection and Segmentation models that are initially trained on a huge number of classes and even huger datasets (we’re talking hundreds of gigabytes here). But regular folk may not need the entire model for their usecase. Perhaps you need to just detect a cat or dog for a project, or maybe a new object that is not in the parent model. YOLO has transfer learned its parameters to smaller variants for ease-of-use and deployment.

<figure id="a3d6" class="graf graf--figure graf-after--p">
<img src="https://cdn-images-1.medium.com/max/800/0*RyrlhJ0D3QcEFQUn.png" class="graf-image" data-image-id="0*RyrlhJ0D3QcEFQUn.png" data-width="560" data-height="370" />
</figure>

That’s it for now. We still have a lot more to tour!

### How LLMs Learn

Large Language Models (LLMs) are like digital sponges that have devoured entire libraries. But how exactly do these binary marvels acquire their knowledge? Let’s break it down.

1\. The Data Buffet

<figure id="8ef5" class="graf graf--figure graf-after--p">
<img src="https://cdn-images-1.medium.com/max/800/0*baN895J4gDl1QAVh" class="graf-image" data-image-id="0*baN895J4gDl1QAVh" data-width="640" data-height="660" />
</figure>

Imagine a massive, all-you-can-eat buffet of text. LLMs start their journey by ingesting enormous amounts of text data from various sources:

\- Books  
- Websites  
- Articles  
- Social media posts  
- And pretty much anything else written by humans

2\. Pattern Recognition: The Digital Sherlock

As LLMs consume this data, they’re not just memorizing; they’re detecting patterns. They become experts at recognizing:

\- How words are typically used together  
- Common sentence structures  
- Context-dependent meanings

<figure id="b41d" class="graf graf--figure graf-after--p">
<img src="https://cdn-images-1.medium.com/max/800/1*PN5GK6hhSe4u_SxB5B0yuQ.png" class="graf-image" data-image-id="1*PN5GK6hhSe4u_SxB5B0yuQ.png" data-width="802" data-height="710" />
</figure>

Internally, it’s all just simple matrix multiplication happening billions of times per second, updating each parameter in the network. It’s pretty much modelled after how the brain learns. Hot mess really.

3\. The Magic of Neural Networks

<figure id="b129" class="graf graf--figure graf-after--p">
<img src="https://cdn-images-1.medium.com/max/800/0*mwpfnuGPD_FQZ6hV" class="graf-image" data-image-id="0*mwpfnuGPD_FQZ6hV" data-width="452" data-height="287" />
</figure>

Under the hood, LLMs use complex neural networks, typically based on the Transformer architecture. These networks consist of:

\- Input layers: Where the text goes in  
- Hidden layers: Where the magic happens  
- Output layers: Where predictions come out

As the model processes more data, it adjusts the strengths of connections between these layers, fine-tuning its understanding.

4\. Training Objectives: The Guessing Game

LLMs often learn through what’s called “unsupervised learning.” A common technique is “masked language modeling,” where:

<figure id="5e68" class="graf graf--figure graf-after--p">
<img src="https://cdn-images-1.medium.com/max/800/1*Wpm5X7kf6CZrAWuI7WyYqw.png" class="graf-image" data-image-id="1*Wpm5X7kf6CZrAWuI7WyYqw.png" data-width="793" data-height="410" />
</figure>

1\. The model is shown a sentence with some words masked out  
2. It tries to predict the masked words  
3. It checks its guesses against the actual words  
4. It adjusts its internal connections based on how well it did

It’s like filling in blanks in a giant, cosmic Mad Libs game!

5\. Iteration and Optimization

Learning for an LLM is an iterative process. It goes through the data multiple times (called epochs), each time:

\- Making predictions  
- Calculating how far off it was (loss function)  
- Adjusting its internal parameters to do better next time (backpropagation)

<figure id="1f83" class="graf graf--figure graf-after--p">
<img src="https://cdn-images-1.medium.com/max/800/1*EPPjJ-W7AqloqELAlOJm8w.png" class="graf-image" data-image-id="1*EPPjJ-W7AqloqELAlOJm8w.png" data-width="1198" data-height="862" />
</figure>

This process continues until the model’s performance stops improving significantly.

6\. The Never-Ending Quest

Even after initial training, many LLMs continue to learn through fine-tuning on specific tasks or datasets. It’s like sending a language graduate to a specialized finishing school.

<figure id="f33c" class="graf graf--figure graf-after--p">
<img src="https://cdn-images-1.medium.com/max/800/1*I5_uqvo4-nTvT7dNWWaF4g.gif" class="graf-image" data-image-id="1*I5_uqvo4-nTvT7dNWWaF4g.gif" data-width="600" data-height="338" />
</figure>

In essence, LLMs learn by consuming vast amounts of text, playing prediction games with themselves, and constantly fine-tuning their understanding based on their successes and failures. It’s a bit like how humans learn language, just at a much, much larger scale and speed!

### Transformers 101

<figure id="1281" class="graf graf--figure graf-after--h3">
<img src="https://cdn-images-1.medium.com/max/800/0*1YcqptdcswWaz0cR.jpeg" class="graf-image" data-image-id="0*1YcqptdcswWaz0cR.jpeg" data-width="499" data-height="691" />
</figure>

Transformers are the architectural backbone of modern Large Language Models. Introduced in the 2017 paper “<a href="https://arxiv.org/abs/1706.03762" class="markup--anchor markup--p-anchor" data-href="https://arxiv.org/abs/1706.03762" rel="noopener" target="_blank">Attention Is All You Need</a>,” they revolutionized natural language processing. Think of Transformers as the secret sauce that makes your AI assistant sound so smart!

**The Building Blocks**

<figure id="3c1e" class="graf graf--figure graf-after--p">
<img src="https://cdn-images-1.medium.com/max/800/1*17GBRzmDkXZM2w6su94O_Q.png" class="graf-image" data-image-id="1*17GBRzmDkXZM2w6su94O_Q.png" data-width="1280" data-height="709" />
</figure>

Imagine Transformers as a high-tech assembly line for processing language. The ingredients are fed in, i.e. structured prompts, and the product is spat out, i.e. structured responses. Here are the key components:

1\. Input Embedding  
- Converts words into number vectors  
- Like translating human language into “machine-speak”

2\. Positional Encoding  
- Adds information about word order  
- Because in “Dog bites man” and “Man bites dog,” order matters!

3\. Multi-Head Attention  
- The “pay attention” mechanism, something we’ll talk about later  
- Allows the model to focus on different parts of the input simultaneously  
- Like having multiple readers, each focusing on different aspects of a text

4\. Feed-Forward Neural Networks  
- Processes the attention output  
- Adds depth to the model’s understanding

5\. Layer Normalization and Residual Connections  
- Keeps the signal from getting too wild  
- Helps information flow smoothly through the model

**The Transformer Dance: Encoding and Decoding**

Transformers typically have two main parts:

1\. *Encoder*: Processes the input  
 — Like a super-efficient reader, understanding the context

2\. *Decoder*: Generates the output  
 — Like a talented writer, crafting responses based on the encoder’s understanding

Some models, like BERT, use only the encoder, while others, like GPT, use only the decoder. Full Transformer models use both.

**The Secret Sauce: Self-Attention**

The real magic of Transformers lies in their self-attention mechanism. Here’s how it works:

1\. For each word, create three vectors: Query, Key, and Value  
2. Calculate attention scores between the Query and all Keys  
3. Use these scores to create a weighted sum of Values

It’s like each word asking, “How relevant are you to me?” to every other word in the sentence.

And we are diving deep into attention in the next section.

### You Just Want Attention: Understanding Attention Mechanisms in Transformers

#### Why All the Fuss About Attention?

<figure id="650c" class="graf graf--figure graf-after--h4">
<img src="https://cdn-images-1.medium.com/max/800/1*Dd6lnnep9hhzJQ_pM9mwWA.png" class="graf-image" data-image-id="1*Dd6lnnep9hhzJQ_pM9mwWA.png" data-width="649" data-height="619" />
</figure>

In the world of Large Language Models, attention isn’t just a catchy Puth song — it’s the secret sauce that makes these models so powerful. Attention mechanisms allow models to focus on relevant parts of the input when producing output. It’s like having a super-smart reader who knows exactly which parts of a text are important for answering a question.

#### Brief history of Attention

> Memory is **attention through time**. ~ Alex Graves 2020

The attention mechanism emerged naturally from problems that deal with **time-varying data (sequences)**. So, since we are dealing with “sequences”, let’s formulate the problem in terms of machine learning first. Attention became popular in the general task of dealing with sequences.

Before attention and transformers, Sequence to Sequence (**Seq2Seq**) worked pretty much like this:

<figure id="81ee" class="graf graf--figure graf-after--p">
<img src="https://cdn-images-1.medium.com/max/800/1*D_BxfoZWvQLuK9-4n-SKTg.png" class="graf-image" data-image-id="1*D_BxfoZWvQLuK9-4n-SKTg.png" data-width="1261" data-height="238" />
</figure>

The elements of the sequence x1, x2, y1​, y2​, etc. are usually called **tokens**. They can be literally anything. For instance, text representations, pixels, or even images in the case of videos.

OK. So why do we use such models?

> The goal is to transform an input sequence (source) to a new one (target).

The two sequences can be of the same or arbitrary length.

In case you are wondering, recurrent neural networks (**RNNs**) dominated this category of tasks. The reason is simple: **we liked to treat sequences sequentially**. Sounds obvious and optimal? <a href="https://coursera.pxf.io/AoYYeN" class="markup--anchor markup--p-anchor" data-href="https://coursera.pxf.io/AoYYeN" rel="noopener" target="_blank">Transformers</a> proved us it’s not!

#### Remember about our encoder and decoder friends?

The **encoder** and **decoder** are nothing more than stacked RNN layers, such as <a href="https://theaisummer.com/understanding-lstm/" class="markup--anchor markup--p-anchor" data-href="https://theaisummer.com/understanding-lstm/" rel="noopener" target="_blank">LSTM’s</a>. The encoder processes the input and produces one **compact representation**, called **z**, from all the input timesteps. It can be regarded as a compressed format of the input.

<figure id="b032" class="graf graf--figure graf-after--p">
<img src="https://cdn-images-1.medium.com/max/800/0*SuMsr1iJcZC5EBOf.png" class="graf-image" data-image-id="0*SuMsr1iJcZC5EBOf.png" data-width="1121" data-height="632" />
</figure>

On the other hand, the decoder receives the context vector **z** and generates the output sequence. The most common application of Seq2seq is language translation. We can think of the input sequence as the representation of a sentence in English and the output as the same sentence in French.

<figure id="b641" class="graf graf--figure graf-after--p">
<img src="https://cdn-images-1.medium.com/max/800/0*DEcWR1DR5elfCHFk.png" class="graf-image" data-image-id="0*DEcWR1DR5elfCHFk.png" data-width="1111" data-height="671" />
</figure>

In fact, RNN-based architectures used to work very well especially with <a href="https://theaisummer.com/understanding-lstm/" class="markup--anchor markup--p-anchor" data-href="https://theaisummer.com/understanding-lstm/" rel="noopener" target="_blank">LSTM</a> and <a href="https://theaisummer.com/gru/" class="markup--anchor markup--p-anchor" data-href="https://theaisummer.com/gru/" rel="noopener" target="_blank">GRU</a> components. It was also a lot softer in terms of time and space complexity: being **linearly complex.**

The problem? It only worked effectively **for small sequences** (\<20 timesteps). A ton of issues arose with <a href="https://medium.com/@kushansharma1/vanishing-exploding-gradients-problem-1901bb2db2b2" class="markup--anchor markup--p-anchor" data-href="https://medium.com/@kushansharma1/vanishing-exploding-gradients-problem-1901bb2db2b2" target="_blank">vanishing and exploding gradients</a>, non-parallel computation (linear complexity) and required lots of training steps (epochs).

Attention was born in order to address these two things on the Seq2seq model. But how?

> The core idea is that the context vector z should have access to all parts of the input sequence instead of just the last one.

In other words, we need to form a **direct connection** with each timestamp.

#### How Attention Works

Let’s start with why attention is getting all the hype. It’s not just a trendy buzzword — it’s the result of years of research compounding successful results. But like any groundbreaking technology, it comes with its trade-offs.

Attention is a memory guzzler. With its quadratic space and time complexity, it’s like a luxury car that demands premium fuel. If Recurrent Neural Networks (RNNs) could perform just as well with their linear complexity, we might not be so obsessed with attention.

Attention brings something revolutionary to the table. When you feed an input to a model with attention:

1.  <span id="b842">It selectively focuses on the most important parts of the input</span>
2.  <span id="af78">It studies these parts contextually</span>
3.  <span id="67d7">It can easily connect distant parts of the input</span>

### Let’s compare:

**1. RNNs (The Old Guard)**:

- <span id="02dc">Process information sequentially</span>
- <span id="dcdd">Pay attention to everything equally</span>
- <span id="4252">Struggle with long-range dependencies</span>
- <span id="5efb">Like trying to remember the beginning of a long story while still reading the end</span>

**2. Attention Mechanisms (The New Sheriff)**:

- <span id="76f6">Can jump back and forth between different parts of the input</span>
- <span id="1caa">Selectively focus on what’s important</span>
- <span id="d3b9">Excel at understanding long-range context</span>
- <span id="5251">Like having a perfect memory of the entire story while reading any part</span>

Let me break this down with the help of an example:

<figure id="c855" class="graf graf--figure graf-after--p">
<img src="https://cdn-images-1.medium.com/max/800/1*z2iXmZjU8kUv0A8EcyICVw.png" class="graf-image" data-image-id="1*z2iXmZjU8kUv0A8EcyICVw.png" data-width="1180" data-height="614" />
</figure>

A literal translation of the German sentence to English sounds weird, as we’re mapping each German word directly to its literal English counterpart without considering contextual semantics. The targeted English translation however requires a broad sense of the grammar, syntax and semantics that the German and English language carry across each other. Attention captures this inherent and latent grammar semantic by learning to attend to the most common words. In the example, notice that the words ‘mir’, ‘helfen’, ‘Satz’, etc. ie, the ones that cross positions during translation are the mappings that our model pays **more attention** to.

Let me simplify this further with a visualization:

<figure id="d527" class="graf graf--figure graf-after--p">
<img src="https://cdn-images-1.medium.com/max/800/1*bxkL8VKvKl3ZCMjrka4mZQ.gif" class="graf-image" data-image-id="1*bxkL8VKvKl3ZCMjrka4mZQ.gif" data-width="1640" data-height="924" />
<figcaption>Source: <a href="https://github.com/jessevig/bertviz" class="markup--anchor markup--figure-anchor" data-href="https://github.com/jessevig/bertviz" rel="noopener" target="_blank">BertViz</a></figcaption>
</figure>

In this visualization, notice how the brightest lines connect the words that need special attention for correct translation. Words that stay in roughly the same position have weaker attention connections. Some words attend to multiple other words, showing how the context is built.

All in all, the **attention** model learns to pay more attention to:

- <span id="698a">Words that typically change position during translation</span>
- <span id="fa72">Key contextual markers that signal how the sentence structure should change</span>
- <span id="b282">Words that have multiple possible translations, choosing the right one based on context</span>

Alright, but what am I supposed to do with this attention information? Obviously train the model, but how? And what is attention generating that I can use to train the model with? Remember our short discussion on query, key and value vectors?

<figure id="3e8e" class="graf graf--figure graf-after--p">
<img src="https://cdn-images-1.medium.com/max/800/1*PNdtPg2KINh9-lhrVswa0w.png" class="graf-image" data-image-id="1*PNdtPg2KINh9-lhrVswa0w.png" data-width="1436" data-height="804" />
</figure>

The way attention is calculated is a bit complex. Let’s dissect the Encoder part of the LLM to understand how attention works:

#### Encoder

**Step 1**

For each input sentence fed into the encoder, we first encode it (positional encoding, remember?) and then embed it into a vector space. The input embeddings are then multiplied with 3 matrices:

1.  <span id="d54e">Query weight matrix (Wq) x Input = Query Matrix (Q)</span>
2.  <span id="04c7">Key weight matrix (Wk) x Input = Key Matrix (K)</span>
3.  <span id="b76e">Value weight matrix (Wv) x Input = Value Matrix (V)</span>

And logically, what do these resultant matrices signify?

1.  <span id="03bc">**Query** is like the question you have in mind (“Who knows about AI?”)</span>
2.  <span id="e767">**Key** is like the name tags people wear (“AI Engineer”, “Data Scientist”)</span>
3.  <span id="4e72">**Value** is the actual knowledge each person has</span>

**Step 2**

The attention formula allows us to calculate the attention score:

``` graf
Attention(Q, K, V) = softmax((Q * K^T) / √d_k) * V
```

**Query** and **key** undergo a dot product matrix multiplication to produce a score matrix. The score matrix contains the “tensor or weights” distributed to each word as per its influence on input.

The weighted attention matrix does a cross-multiplication with the “**value**” vector to produce an output sequence. The output values indicate the placement of subjects and verbs, the flow of logic, and output arrangements.

However, multiplying matrices within a neural network may cause exploding gradients and residual values. To stabilize the matrix, it’s divided by the square root of the dimension of the queries and keys.

**Step 3**

The softmax layer receives the attention scores and compresses them between values 0 to 1. This gives the machine learning model a more focused representation of where each word stands in the input text sequence.

In the softmax layer, the higher scores are elevated, and the lower scores get depressed. The attention scores \[Q\*K\] are multiplied with the value vector \[V\] to produce an output vector for each word.

This is followed by residual and layer normalization, which in layman terms translates to eliminating outliers and gradient stabilization (just security checks).

<figure id="766e" class="graf graf--figure graf-after--p">
<img src="https://cdn-images-1.medium.com/max/800/0*qELgUFqvZPZ-eR3r.jpg" class="graf-image" data-image-id="0*qELgUFqvZPZ-eR3r.jpg" data-width="577" data-height="433" />
</figure>

**Step 4**

The feedforward layer receives the output vectors with embedded output values. It contains a series of neurons that take in the output and then process and translate it. As soon as the input is received, the neural network triggers the <a href="https://www.g2.com/articles/activation-function" class="markup--anchor markup--p-anchor" data-href="https://www.g2.com/articles/activation-function" rel="noopener" target="_blank"><strong>ReLU activation function</strong></a> to eliminate the “vanishing gradients” problem from the input.

This gives the output a richer representation and increases the network’s predictive power. Once the output matrix is created, the encoder layer passes the information to the decoder layer.

This is the attention mechanism boiling within the encoder. But what happens in the decoder?

#### Decoder

The decoder architecture contains the same number of sublayer operations as the encoder, with a slight difference in the attention mechanism. Decoders are autoregressive, which means it only looks at previous word tokens and previous output to generate the next word.

Let’s look at the steps a decoder goes through.

- <span id="fd37">Positional embeddings: The decoder takes the input generated by the encoder and previous output tokens and converts them into abstract numeric representations. However, this time, it only converts words until time series t -1, with t being the current word.</span>
- <span id="5a97">Masked multi-head attention 1: To further prevent decoders from processing future tokens, it undergoes the first layer of masked attention. In this layer, attention scores for decoders are calculated and multiplied by a masked matrix that contains a value between 0 and infinity.</span>
- <span id="0cb3">Softmax layer: After multiplication, the output gets passed on to the softmax layer, which downsizes it and stabilizes the numbers. All the parts of the matrix that belonged to future words are zeroed out. The masked matrix is structured in such a way that negative infinities get multiplied only by future tokens, which are nullified by the softmax layer.</span>
- <span id="31a0">Masked multi-head attention 2: In the second masked self-attention layer, the value and keys of the encoder output are compared with the decoder output query to get the best output path.</span>
- <span id="bf32">Feedforward neural network: Between these self-attention layers, a residual feedforward network exists to identify missing gradients, eliminate residue, and train the neural network on the data.</span>
- <span id="3068">Linear classifier: The last linear classifier layer predicts the most likely next word. This occurs till the entire response is complete.</span>

### Training and Inference

Think of training and inference as two distinct phases in an LLM’s life:

- <span id="c693">**Training**: The education phase — expensive, time-consuming, but essential</span>
- <span id="6ace">**Inference**: The working phase — where the model applies what it learned</span>

<figure id="f0f3" class="graf graf--figure graf-after--li">
<img src="https://cdn-images-1.medium.com/max/800/1*DLKBc0mEjaSGDOMT2kNOKQ.jpeg" class="graf-image" data-image-id="1*DLKBc0mEjaSGDOMT2kNOKQ.jpeg" data-width="564" data-height="500" />
<figcaption>This is me</figcaption>
</figure>

#### What Happens During Training?

1.  <span id="eae5">**Data Ingestion**: The model processes massive amounts of text data, typically hundreds of gigabytes to petabytes. Example: GPT-3 trained on 45TB of compressed text!</span>
2.  <span id="e859">**Parameter Updates**: The model adjusts billions of parameters and uses backpropagation to minimize loss function as well as techniques for gradient descent optimization.</span>
3.  <span id="8956">**Validation**: Regular testing on held-out data and ensures model isn’t just memorizing (overfitting)</span>

**Training Requirements:**  
- Time: Days to months  
- Hardware: Multiple high-end GPUs/TPUs  
- Memory: Hundreds of GB of VRAM  
- Cost: Can run into millions of dollars

<figure id="bbd5" class="graf graf--figure graf-after--p">
<img src="https://cdn-images-1.medium.com/max/800/1*C2TSMnbNMQv6z58TcU0lew.jpeg" class="graf-image" data-image-id="1*C2TSMnbNMQv6z58TcU0lew.jpeg" data-width="1280" data-height="720" />
<figcaption>this video cost me more storage installing libraries than the model itself</figcaption>
</figure>

**What Happens During Inference?**

1.  <span id="820b">**Input Processing**: Tokenization of user input and converting tokens to model’s input format.</span>
2.  <span id="67c3">**Forward Pass:** Only uses the forward path through the network without parameter updates. Much faster than training.</span>
3.  <span id="7e66">**Output Generation**: Typically uses beam search, reranking and sampling methods (more on this in RAG).</span>

**Inference Requirements:**  
- Time: Milliseconds to seconds  
- Hardware: Can run on consumer GPUs or even CPUs  
- Memory: Can be optimized (quantization, upcoming)  
- Cost: Fraction of training costs

``` graf

+------------+---------------------------+----------------------------+
|   Aspect   |         Training          |         Inference          |
+------------+---------------------------+----------------------------+
| Direction  | Forward + Backward        | Forward only               |
| Memory     | High (gradients)          | Lower (no gradients)       |
| Speed      | Slow (parameter updating) | Faster (fixed parameters)  |
| Batch Size | Large (for stability)     | Small (for responsiveness) |
+------------+---------------------------+----------------------------+
```

PS: When you initiate a conversation with ChatGPT (termed ‘initialization’), the servers at OpenAI are essentially loading the entire GPT model onto their hardware for inference. Since ChatGPT is quite large (in the range of hundreds of billions of parameters), they require large-scale compute infrastructure (distributed across multiple GPUs) just to hold a single conversation with a user. Talk about free plans!  
However, whenever OpenAI releases a new model, it has a long history of training on much larger accelerated hardware. Thousands of GPUs, efficient attention mechanisms and better training data. This is where most of the funding, grants and a large portion of collected revenue go into.

### Fine-tuning

Throughout the article, I have been throwing around this term. But what exactly is fine-tuning?

Imagine you have a guitar. When it comes out of the factory, it’s built to play music (like a pre-trained model), but it needs tuning for optimal performance. Just as there are different ways to tune a guitar, there are various techniques to fine-tune a language model.

<figure id="091c" class="graf graf--figure graf-after--p">
<img src="https://cdn-images-1.medium.com/max/800/1*Wely5JKAQPtQvhW6idz3uQ.jpeg" class="graf-image" data-image-id="1*Wely5JKAQPtQvhW6idz3uQ.jpeg" data-width="661" data-height="500" />
</figure>

I will be moving forward with this guitar analogy to keep things simple.

#### What is fine-tuning?

<figure id="c37e" class="graf graf--figure graf-after--h4">
<img src="https://cdn-images-1.medium.com/max/800/1*aUDXIuUBAAe2Cl5MheVctw.png" class="graf-image" data-image-id="1*aUDXIuUBAAe2Cl5MheVctw.png" data-width="882" data-height="528" />
</figure>

Fine-tuning is the process of taking a pre-trained model and adapting it to a specific task or domain. It’s like adjusting your guitar strings to play in a different key.

#### **Why fine-tune?**

Just like when a guitar is manufactured to play any sort of music, they can be tuned to perform better on jazz, rock or blues. This requires careful tuning of the strings and pegs by the guitarist.

In the same space, LLMs understand language structure very well and are poised to respond to any text input. Just as you might tune a guitar differently, LLM behavior can be adjusted for sentiment analysis, translation, etc. Fine-tuning is easier than building and training a new model from scratch. Just like choosing to tune a guitar rather than building a new one entirely.

For example, did you know that GPT-3.5 and GPT-4 are LLMs fine-tuned from a larger foundational GPT? LLaMa, the open-source LLM released by Meta, has been fine-tuned for millions of tasks, ranging from customer service to personal chatbots to even for-profit applications.

Let’s look into the two approaches to fine-tuning:

#### Standard Tuning: Full Fine-tuning

Like tuning all strings on a guitar (E, A, D, G, B, E), full fine-tuning adjusts all layers of the model. In terms of data, this task requires a huge dataset relevant to the task it has to be specialized on. But this requires large computational resources and might not be a suitable option in the long run, owing to model adaptations.

#### Drop D Tuning: Partial Fine-tuning

Like lowering just the low E string to D, partial fine-tuning adjusts only some layers. The layers that are not involved in training are ‘frozen’ and the rest are retrained. This preserves lower-level features that the original model has learned. A better example: if you’re decent at speaking American English, it’s not that hard to catch up on British English.

### Advanced Fine-tuning Techniques

Just as a guitarist might adjust their instrument for different musical styles, we can fine-tune our language models in various ways. Let’s explore advanced fine-tuning techniques through the lens of guitar tuning!

#### Pruning: Trimming the Excess

<figure id="22f8" class="graf graf--figure graf-after--h4">
<img src="https://cdn-images-1.medium.com/max/800/0*8mAokkIC9JOn1VJ6.png" class="graf-image" data-image-id="0*8mAokkIC9JOn1VJ6.png" data-width="705" data-height="356" />
</figure>

Pruning is a technique that involves removing unnecessary connections or parameters from a neural network. Basically, pruning removes the weights that don’t really matter, making the model smaller and easier to handle. This results in faster speeds and less memory usage — pretty neat, right?

Imagine a guitarist tuning their guitar before a performance. The guitar has six strings, but not every string is needed for every song. Some strings might even produce unnecessary noise or overpower the melody. To make the sound cleaner and more precise, the guitarist decides to adjust or mute certain strings while playing a specific piece. Infact, Keith Richards, songwriter and guitarist for the band ‘Rolling Stones’, used a restricted string setup with only five strings instead of six!

#### Transfer Learning

Transfer learning is like teaching a general musician to specialize in guitar. They already know finger placement, reading music, and rhythm; they just need to adapt these skills to a new style.

<figure id="bbe3" class="graf graf--figure graf-after--p">
<img src="https://cdn-images-1.medium.com/max/800/1*1Eu2LWBggzSgtMvrAVny2g.jpeg" class="graf-image" data-image-id="1*1Eu2LWBggzSgtMvrAVny2g.jpeg" data-width="768" data-height="432" />
</figure>

Similarly, in machine learning and deep learning, transfer learning involves taking a model that has already been trained on a general task and fine-tuning it for a new, more specific task. The model has already learned useful features and patterns from a large, broad dataset (like the musician’s general skills). Instead of training a new model from the ground up, we just tweak the existing model to adapt to the new problem, making the process much faster and more efficient.

This way, just as the musician quickly becomes proficient at guitar, the model quickly learns to perform well on the specialized task without needing extensive retraining.

#### Distillation: Knowledge Transfer for Efficiency

Think of distillation like a guitarist teaching a student how to play complex songs. The guitarist (the larger model) knows all the advanced techniques, chords, and intricate details of the music. However, the student (the smaller model) doesn’t need to learn every single detail to perform the song well. Instead, the guitarist simplifies the lessons, focusing on the essential techniques and patterns that still capture the essence of the performance.

<figure id="5ed9" class="graf graf--figure graf-after--p">
<img src="https://cdn-images-1.medium.com/max/800/0*t7msYunL4LGVC_6Z.png" class="graf-image" data-image-id="0*t7msYunL4LGVC_6Z.png" data-width="875" data-height="525" />
</figure>

You’ve got it now!

Distillation involves training a smaller, more compact model to mimic the behavior of a larger, more complex model. By transferring knowledge from the larger model, distillation enables the creation of highly efficient models without sacrificing performance.

This technique has been particularly effective in scenarios where computational resources are limited, such as deploying models on edge devices, smartphones, tablets, etc. A large language model can be distilled into a smaller model that retains most of the original model’s performance while being more lightweight and faster to execute.

#### Prefix Tuning

Prefix tuning adapts pre-trained language models to specific tasks without modifying the original model’s weights.

Prefix tuning draws inspiration from the concept of prompting, where task instructions and examples are prepended to the input to steer the LM’s generation. However, instead of using discrete tokens, **prefix tuning uses a continuous prefix vector**.

Prefix-tuning involves prepending a sequence of **continuous task-specific vectors**, called a prefix, to the input of the LM.

The Transformer can attend to these **prefix vectors** as if they were a sequence of **“virtual tokens”**. Unlike prompting, the **prefix vectors** do not correspond to real tokens but are learned during training.

#### LoRA: Low Rank Adaptation

Now imagine adding a small device to your guitar that, when activated, changes how certain chords sound:

- <span id="5fd7">The guitar’s main structure remains unchanged</span>
- <span id="8eac">The device affects only specific combinations of strings</span>
- <span id="fdf0">You can easily swap devices for different effects</span>

<figure id="d622" class="graf graf--figure graf-after--li">
<img src="https://cdn-images-1.medium.com/max/800/1*JBbGp4sU1zaCUfZDZCs1cA.jpeg" class="graf-image" data-image-id="1*JBbGp4sU1zaCUfZDZCs1cA.jpeg" data-width="1600" data-height="672" />
</figure>

LoRA adds small, trainable rank decomposition matrices. These matrices are like “adapters” for the model and modifies the model’s behavior without changing most weights. So instead of retraining a subset of the model, we retrain these decomposable matrices, saving us a ton of time and compute. This make LoRA much more memory efficient than full fine-tuning.

#### QLoRA: Quantized LoRA

Like practicing an electric guitar without an amp:

- <span id="4ecc">The guitar’s full potential is compressed</span>
- <span id="f68b">You can still learn and practice effectively</span>
- <span id="93b0">When you plug in, you get the full sound</span>

<figure id="4558" class="graf graf--figure graf-after--li">
<img src="https://cdn-images-1.medium.com/max/800/1*DiERVQiGg5NbYZUm7pXcIQ.png" class="graf-image" data-image-id="1*DiERVQiGg5NbYZUm7pXcIQ.png" data-width="2002" data-height="891" />
</figure>

QLoRA takes LoRA a step further by also quantizing the weights of the LoRA adapters (smaller matrices) to lower precision (e.g., 4-bit or 8-bit instead of 16-bit). This further reduces the memory footprint and storage requirements. QLoRA is even more memory efficient than LoRA, making it ideal for resource-constrained environments.

#### Choosing between LoRA and QLoRA:

The best choice between LoRA and QLoRA depends on your specific needs:

- <span id="fd05">**If memory footprint is the primary concern:** QLoRA is the better choice due to its even greater memory efficiency.</span>
- <span id="729e">**If fine-tuning speed is crucial:** LoRA may be preferable due to its slightly faster training times.</span>
- <span id="269b">**If both memory and speed are important:** QLoRA offers a good balance between both.</span>

There’s a lot more variants of LoRA such as ReLoRA, SLoRA, GaLoRA, etc. But discussing them is out of the scope of this article as they require other your understanding of a few other foundational theories. If you do want to check them out, <a href="https://training.continuumlabs.ai/training/the-fine-tuning-process/parameter-efficient-fine-tuning/prefix-tuning-optimizing-continuous-prompts-for-generation" class="markup--anchor markup--p-anchor" data-href="https://training.continuumlabs.ai/training/the-fine-tuning-process/parameter-efficient-fine-tuning/prefix-tuning-optimizing-continuous-prompts-for-generation" rel="noopener" target="_blank">here</a> is a good article that explain each technique in precise detail.

#### Prompt Engineering

The process of crafting, refining, and optimizing the input prompts given to an LLM in order to achieve desired outputs. Prompt engineering plays a crucial role in determining the performance and behavior of models like those based on the GPT architecture.

Given the vast knowledge and diverse potential responses a model can generate, the way a question or instruction is phrased can lead to significantly different results. Some specific techniques in prompt engineering include:

- <span id="0d93">**Rephrasing:** Sometimes, rewording a prompt can lead to better results. For instance, instead of asking “What is the capital of France?” one might ask, “Can you name the city that serves as the capital of France?”</span>
- <span id="e6b9">**Specifying Format:** For tasks where the format of the answer matters, you can specify it in the prompt. For example, “Provide an answer in bullet points” or “Write a three-sentence summary.”</span>
- <span id="9748">**Leading Information:** Including additional context or leading information can help in narrowing down the desired response. E.g., “Considering the economic implications, explain the impact of inflation.”</span>

#### Zero-Shot, One-Shot and Few-Shot Learning

These are techniques that do not strictly fall under fine-tuning, but rather more into **Prompt Engineering.** Here, the model adapts to new tasks by providing it with no (zero-shot), one (one-shot) or multiple (few-shot) data samples of the intended tasks.

<figure id="2a41" class="graf graf--figure graf-after--p">
<img src="https://cdn-images-1.medium.com/max/800/1*cxXLo09rtVGzqjn-sLUUMA.png" class="graf-image" data-image-id="1*cxXLo09rtVGzqjn-sLUUMA.png" data-width="1024" data-height="768" />
</figure>

#### How Few-Shot Learning Works

- <span id="746c">Provides the model with a few examples</span>
- <span id="697c">Model recognizes patterns in these examples</span>
- <span id="8e82">Applies these patterns to new, similar tasks</span>

This typically works on large parameter models like the GPT series or the Llama 3 70B onwards.

#### Reinforcement Learning through Human Feedback (RLHF)

If you’ve used ChatGPT quite enough, you may have noticed that you do get occasionally prompted on how you would rate it’s response. Or whether it requires an improvement. Or to make a choice on the better of the two responses.

These are all inputs to an RLHF framework that will be involved in iterative fine-tuning of the base GPT later on. It’s similar to a club guitarist who plays different variations of a song, gauges the audience’s reaction, and adjusts their playing based on feedback.

#### How RLHF Actually Works

<figure id="db0d" class="graf graf--figure graf-after--h4">
<img src="https://cdn-images-1.medium.com/max/800/1*2KczV5F2ftEtPj6ebl-aRg.jpeg" class="graf-image" data-image-id="1*2KczV5F2ftEtPj6ebl-aRg.jpeg" data-width="1440" data-height="772" />
</figure>

1.  <span id="ea3b">Initial Training: Model learns basic capabilities</span>
2.  <span id="8fa3">Reward Modeling: Human feedback is collected. A reward model is trained to predict human preferences</span>
3.  <span id="0ada">Reinforcement Learning: Model is further trained using the reward model. Behaviors that align with human preferences are reinforced</span>

#### Components of RLHF

1.  <span id="5c0e">**Base Model**: The initial pre-trained model</span>
2.  <span id="5fdb">**Reward Model**: Learns to predict human preferences</span>
3.  <span id="c7d6">**Policy Model**: The model being fine-tuned</span>
4.  <span id="06cf">**Human Feedback**: Crucial for guidance</span>

### Model Initialization: Hot, Cold or Frozen Start?

#### The Thermal Spectrum of Starting Models

When it comes to initializing Large Language Models, we have three main approaches:

1.  <span id="fff5">**Cold Start**: Starting from scratch. The model weights are initialized randomly, and the model is trained from ground zero with no prior knowledge utilized. Cold start is used where novel architectures have to be trained, and/or there is no existing pre-trained architecture to support the new one. Eg: GPT-4 API usage</span>
2.  <span id="805b">**Warm/Hot Start**: Beginning with pre-trained weights from an existing model, we fine-tune all layers for a domain-specific task. This is also interchangeably called transfer learning. This initialization method is useful when the model behavior has to be adapted significantly on task-specific data. Eg: BERT Fine-tuning</span>
3.  <span id="f4bc">**Frozen Start**: Using a pre-trained model with fixed parameters, we freeze almost all the model’s parameters and only train new task-specific layers. This is useful when you have limited VRAM or compute (sad GTX noises) and a very small dataset. Eg: Opensource LLM fine-tuning.</span>

### Quantization

<figure id="8fd7" class="graf graf--figure graf-after--h3">
<img src="https://cdn-images-1.medium.com/max/800/1*aRYQo40BSqs90VfLwLd4RA.jpeg" class="graf-image" data-image-id="1*aRYQo40BSqs90VfLwLd4RA.jpeg" data-width="800" data-height="632" />
</figure>

Quantization aka Compression, especially in AI models and deep learning models, typically refers to converting the model’s parameters, such as weights and biases, from floating-point numbers to integers with lower bit widths, for example, from 32-bit floating-point to 8-bit integers. In simple terms, quantization is like simplifying a detailed book written with high-level vocabulary into a concise summary or a children’s version of the story. This summary or children’s version takes up less space and is easier to communicate, but it may lose some of the details present in the original book.

#### Why Quantization

<figure id="28b3" class="graf graf--figure graf-after--h4">
<img src="https://cdn-images-1.medium.com/max/800/1*9E-0PxU8xmSBe5vHydFgCQ.jpeg" class="graf-image" data-image-id="1*9E-0PxU8xmSBe5vHydFgCQ.jpeg" data-width="1400" data-height="788" />
</figure>

The purpose of quantization mainly includes the following points:

1\. Reduced Storage Requirements: Quantized models have significantly smaller sizes, making them easier to deploy on devices with limited storage resources, such as mobile devices or embedded systems.

2\. Accelerated Computation: Integer operations are generally faster than floating-point operations, especially on devices without dedicated floating-point hardware support.

3\. Reduced Power Consumption: On certain hardware, integer operations consume less energy.

There is often some loss of precision (granularity) when reducing the number of bits to represent the original parameters.

To illustrate this effect, we can take any image and use only 8 colors to represent it:

<figure id="1687" class="graf graf--figure graf-after--p">
<img src="https://cdn-images-1.medium.com/max/800/1*Lh-H_RLYru9HgGJVyzXprw.jpeg" class="graf-image" data-image-id="1*Lh-H_RLYru9HgGJVyzXprw.jpeg" data-width="1456" data-height="690" />
</figure>

Notice how the zoomed-in part seems more “grainy” than the original since we can use fewer colors to represent it.

The main goal of quantization is to reduce the number of bits (colors) needed to represent the original parameters while preserving the precision of the original parameters as best as possible.

However, quantization has a drawback: it can lead to a reduction in model accuracy. This is because you are representing the original floating-point numbers with lower precision, which may result in some loss of information, meaning the model’s capabilities may decrease.

To balance this accuracy loss, researchers have developed various quantization strategies and techniques, which we’ll be seeing soon.

### A Brief Survey on Quantization Techniques

Two primary machine and deep learning model quantization modes exist — PTQ and QAT. Let’s understand them both.

<figure id="780b" class="graf graf--figure graf-after--p">
<img src="https://cdn-images-1.medium.com/max/800/1*4QoyqUVpR07w0pzr2Nz7Pg.jpeg" class="graf-image" data-image-id="1*4QoyqUVpR07w0pzr2Nz7Pg.jpeg" data-width="1024" data-height="489" />
<figcaption>PTQ (left) and QAT (right)</figcaption>
</figure>

#### Quantization Aware Training (QAT)

- <span id="6f16">We start QAT with a pre-trained model or a PTQ model. We fine tune this model using QAT. The aim here is to recover the accuracy loss that happened due to PTQ in case we took a PTQ model.</span>
- <span id="6999">The basic idea of QAT is to quantize input into lower precision depending on the weight precision of that layer. QAT also takes care of converting the output of multiplication of weights and inputs back to higher precision in case the next layer demands so. This process of converting input to lower precision and then converting output of weights and inputs back to higher precision is also called as “FakeQuant Node Insertion”. The quantization is called Fake as it quantizes and then dequantizes as well converting to the base operation.</span>

#### Post-Training Quantization

- <span id="2e57">Post-Training Quantization (PTQ) is a technique that allows us to quantize a pre-trained model after the training process is complete, without requiring any further training or fine-tuning. The goal of PTQ is to reduce the size and computational requirements of the model by converting its weights and activations to lower precision, typically 8-bit integers, while preserving as much accuracy as possible.</span>
- <span id="c276">In PTQ, the model is first evaluated with a **calibration** dataset to collect statistics, which are then used to determine the optimal quantization parameters (scale and zero-point) for each layer. Unlike Quantization Aware Training (QAT), there is no need to modify the training process itself. PTQ is often faster to implement than QAT since it does not require retraining.</span>
- <span id="d76d">During inference, both the weights and activations are quantized to lower precision. However, PTQ does not simulate quantization in the training process, which can lead to some loss in accuracy, especially for models with highly sensitive layers. The method is more straightforward compared to QAT but may struggle to recover accuracy loss without additional fine-tuning, as it doesn’t learn to adjust to the quantization-induced errors.</span>

### Advanced Quantization Techniques

#### **FP16/INT8/INT4**

Popularly, if a model’s name doesn’t have specific identifiers like Llama-2–7b or chatglm2–6b, it generally indicates that these models are in full precision (FP32), although some may also be in half-precision (FP16). However, if the model’s name includes terms like fp16, int8, int4, such as Llama-2–7B-fp16, chatglm-6b-int8, or chatglm2–6b-int4, it suggests that these models have undergone quantization, with fp16, int8, or int4 denoting the level of quantization precision.

<figure id="84d1" class="graf graf--figure graf-after--p">
<img src="https://cdn-images-1.medium.com/max/800/1*fpN5zjqPPOq_ODWlL_DK2A.jpeg" class="graf-image" data-image-id="1*fpN5zjqPPOq_ODWlL_DK2A.jpeg" data-width="1144" data-height="856" />
</figure>

The quantization precision ranges from high to low as follows: fp16 \> int8 \> int4. Lower quantization precision results in smaller model sizes and reduced GPU memory requirements. Still, it can also lead to decreased model performance.

#### **GPTQ**

GPTQ stands for Generalized Post Training Quantization. This means once you have your pre trained LLM, you simply convert the model parameters into lower precision.

GPTQ is preferred for GPU’s & not CPU’s. My poor RTX 3050😅

<figure id="879e" class="graf graf--figure graf-after--p">
<img src="https://cdn-images-1.medium.com/max/800/1*qWfdnOXn70TZBQEjIlNrcQ.jpeg" class="graf-image" data-image-id="1*qWfdnOXn70TZBQEjIlNrcQ.jpeg" data-width="1200" data-height="675" />
</figure>

GPTQ is a model quantization method that allows language models to be quantized to precision levels like INT8, INT4, INT3, or even INT2 without significant performance loss. If you come across model names on <a href="https://huggingface.co/" class="markup--anchor markup--p-anchor" data-href="https://huggingface.co/" rel="noopener" target="_blank">HuggingFace</a> with “GPTQ” in their names, such as Llama-2–13B-chat-GPTQ, it means these models have undergone GPTQ quantization. For example, consider Llama-2–13B-chat, the full-precision version of this model has a size of 26 GB, but after quantization using GPTQ to INT4 precision, the model’s size reduces to 7.26 GB.

#### **GGML \| GGUF**

On HuggingFace, if you come across model names with “**GGML**,” such as Llama-2–13B-chat-GGML, it indicates that these models have undergone GGML quantization. GGML stands for Generative Graphical Models.

Some GGML model names not only include “GGML” but also have suffixes like “q4,” “q4_0,” “q5,” and so on, such as Llama-2–7b-ggml-q4. In this context, “q4” refers to the GGML quantization method. Q-series quantization methods are a whole other ballgame but there’s a beautiful <a href="https://www.reddit.com/r/LocalLLaMA/comments/139yt87/notable_differences_between_q4_2_and_q5_1/" class="markup--anchor markup--p-anchor" data-href="https://www.reddit.com/r/LocalLLaMA/comments/139yt87/notable_differences_between_q4_2_and_q5_1/" rel="noopener" target="_blank">Reddit thread</a> on this, should you be curious.

GPT-Generated Unified Format aka **GGUF** is the new version of GGML. GGUF is specially designed to store inference models and adds more data about the model so it’s easier to support multiple older and newer architectures.

#### **GPTQ vs GGML**

GPTQ and GGML are currently the two primary methods for model quantization, but what are the differences between them? And which quantization method should you choose?

Here are some key similarities and differences between the two:

- <span id="dfa5">GPTQ runs faster on GPUs, while GGML runs faster on CPUs.</span>
- <span id="6aff">Models quantized with GGML tend to be slightly larger than those quantized with GPTQ at the same precision level, but their inference performance is generally comparable.</span>
- <span id="a095">Both GPTQ and GGML can be used to quantize Transformer models available on HuggingFace.</span>

Therefore, if your model runs on a GPU, it’s advisable to use GPTQ for quantization. If your model runs on a CPU, GGML is a recommended choice for quantization.

#### AWQ

It stands for Activation-aware Weight Quantization

- <span id="2f6a">This is meant for GPU or CPU.</span>
- <span id="1fd0">In this method, we do not quantize all weights; instead, we quantize weights that are not important for our model to retain it’s validity.</span>
- <span id="9ad5">First, the model’s activations are analyzed to determine which weights have the most significant impact on the output. Weights with minimal impact on the activations are quantized more aggressively, while those crucial to model performance are left in higher precision. This selective process minimizes the loss of accuracy commonly seen in uniform quantization methods while reducing the computational and memory demands, allowing the model to run efficiently on GPUs and CPUs.</span>

### Inference and Training Arithmetic

<figure id="9b4e" class="graf graf--figure graf-after--h3">
<img src="https://cdn-images-1.medium.com/max/800/1*QEFkvVjmmqgigIRxYvad6A.jpeg" class="graf-image" data-image-id="1*QEFkvVjmmqgigIRxYvad6A.jpeg" data-width="1088" data-height="1088" />
<figcaption>You don’t need a degree in mathematics for this section</figcaption>
</figure>

Inference and training arithmetic is essential knowledge for every AIOps engineer and even non-experts. It involves straightforward math and computing resource allocation. I find this topic particularly fascinating because, although there are a limited number of factors involved, they can be creatively combined in countless ways to generate various budget strategies when planning to train an AI model.

This section is a dumbed down explanation of a larger, more detailed reddit thread on Transformers. I’ll distill the basics here into mathematical equations that can be applied on a practical scale.

#### KV cache

Yeah that’s our Key (K) and Value (V) vectors in attention. During inference, the transformer performs self-attention, which requires the kv values for each item currently in the sequence (whether it was prompt/context or a generated token). These vectors are provided a matrix known as the **kv cache.**

The purpose of this is to avoid recalculations of those vectors every time we process a token. With the computed *k*,*v* values, we can save quite a bit of computation at the cost of some storage. Per token, the number of bytes we store is:

> ***2*** × ***2*** × ***nlayers​*** × ***nheads​*** × ***dhead​***

The first factor of 2 is to account for the two vectors, k*k* and v*v*. We store that per each layer, and each of those values is a *n*heads ​× *d*head​ matrix. Then multiply by 2 again for the number of bytes (we’ll assume 16-bit format).

The <a href="https://en.wikipedia.org/wiki/FLOPS" class="markup--anchor markup--p-anchor" data-href="https://en.wikipedia.org/wiki/FLOPS" rel="noopener" target="_blank">flops</a> (floating points operations per second) to compute *k* and *v* for all our layers is:

> 2 × 2 × *n*layers​ × *d*model​^2

> ***How many flops in a matmul?***

> *The computation for a matrix-vector multiplication is 2*mn*, where* m × n *is the dimension of the matrix and* n *is the vector*. *A matrix-matrix is* 2mnp*, here* m × n *and* n × p *are the two matrices.*

This means for a 52B parameter model (taking <a href="https://arxiv.org/pdf/2112.00861.pdf" class="markup--anchor markup--p-anchor" data-href="https://arxiv.org/pdf/2112.00861.pdf" rel="noopener" target="_blank">Anthropic’s</a>, where *dmodel*​=8192 and *nlayers*=64). The flops are

> 2 × 2 × 64 × 8192^2 = 17,179,869,184 flops

Say we have an <a href="https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf" class="markup--anchor markup--p-anchor" data-href="https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf" rel="noopener" target="_blank">A100 GPU</a>, which does *3.12^12* flops per second and *1.5^12* bytes per second of memory bandwidth. The following are numbers for just the kv weights and computations.

> *memory = 2 × 2 × nlayers​ × dmodel​^2 ÷ 1.5^12*

> *compute = 2 × 2 × nlayers​ × dmodel​^2 ÷ 312^12*

> ***Flops vs Memory Boundedness***

> *Flops vs memory boundedness is something we deal with a lot for transformer inference, but* <a href="https://horace.io/brrr_intro.html" class="markup--anchor markup--blockquote-anchor" data-href="https://horace.io/brrr_intro.html" rel="noopener" target="_blank"><em>also in deep learning optimisation in general</em></a>*. To do the computations we do, we need to load weights which costs* <a href="https://en.wikipedia.org/wiki/Memory_bandwidth" class="markup--anchor markup--blockquote-anchor" data-href="https://en.wikipedia.org/wiki/Memory_bandwidth" rel="noopener" target="_blank"><em>memory bandwidth</em></a>*. We assume (correctly, this has been very well optimised) that we can start the computations while we load the weights. Flop bound would then mean that there is time when nothing is being passed through memory, and memory bound would mean that no floperations are occuring. Nvidia uses the term* <a href="https://docs.nvidia.com/deeplearning/performance/dl-performance-gpu-background/index.html#gpu-arch" class="markup--anchor markup--blockquote-anchor" data-href="https://docs.nvidia.com/deeplearning/performance/dl-performance-gpu-background/index.html#gpu-arch" rel="noopener" target="_blank"><em>math bandwidth</em></a> *which I find really cute. Technically, this delineation exist per kernel but can be abstracted to exist for groups of operations.*

<figure id="a6e1" class="graf graf--figure graf-after--blockquote">
<img src="https://cdn-images-1.medium.com/max/800/0*MI9zg8UPJn5g2bcS.png" class="graf-image" data-image-id="0*MI9zg8UPJn5g2bcS.png" data-width="956" data-height="580" />
</figure>

None of the model architecture matters anymore — we get a distinct ratio here of 208 given this hardware specification. This means that if we’re going to compute kv for one token, it’ll take the same amount of time to compute for up to 208 tokens! Anything below, we’re memory bandwidth bound. Above, flops bound. If we used the rest of our weights to do a full forwards pass (run the rest of the transformer) on our context, it’s also 208.

For a 52B model full forwards pass, that’s *12 × 2 × nlayers​ × dmodel × 2 / 1.5^12 ≈ 69* milliseconds for up to 208 tokens (in practice, we’d use four GPUs in parallel so it would actually be ~17 milliseconds, more in following sections). If we had 416 (double) tokens in the context, then it would take twice as long, and 312 tokens would take 1.5 times as long.

Sounds pretty fast right? Yep, that’s the AI horsepower of GPU💪🏻

#### Capacity

We have a solid idea of the two things we store in our GPUs — kv cache and weights. GPU capacity does come into play for transformer inferencing performance, and we have all the understanding we need to evaluate that now!

Nvidia A100 GPUs (which are generally speaking, the best bang for the buck GPUs we can get for inference) have a standard of 40GB of capacity. There are ones with 80GB and higher memory bandwidth (*2^12* instead of *1.5^12*), but you can deal with different versions with these equations later on.

Given the parameter count, we can multiply by two (assuming 16-bit precision) to get bytes. So, to calculate the size of the weights for a 52B model:

> *52^12 × 2 = 104^12 bytes ≈ 104GB*

Oh no! This doesn’t fit in one GPU! We’d need at least three GPUs just to have all the weights loaded in (will discuss how to do that sharding later). That leaves us *120 − 104 = 16GB* left for our kv cache. Is that enough? Back to our equation for kv cache memory per token, again with a 52B model;

> *2 × nlayers × nheads​ × dhead​ × 2 × 4 × nlayers​ × nheads × dhead = 4 × 64 × 8192 = 2,097,152 bytes ≈ 0.002GB*

And then we’d do *16/0.002 ≈ 8000* tokens can fit into our kv cache with this GPU set up. For four GPUs, we’d get 4 × 16*/0.002 ≈ 32000* tokens.

#### **Alright, here’s the punchline:**

So, training and inference with transformers boils down to a game of “Flops vs Memory Juggling.” The KV cache is like the secret weapon you stash under your desk — saves you time but takes up space. You’re swapping storage for speed, and GPUs are your best friends in this crazy balancing act.

For any monster model, it’s a lot like trying to cram an elephant into a Mini Cooper — just doesn’t fit! You’re going to need a couple more cars (GPUs) to carry all the weight. You can also simulate the same for the Llama 70B models, or even the larger GPT models (I dare you go read their whitepaper), as long as you have the variables dialled down.

TL;DR: Transformers = doing a lot of math **really** fast, GPUs are the gym. Jst do your research on Nvidia GPU specs and you’re spot on.

### LLM Pollution

As we navigate through 2024, the AI landscape is experiencing what many experts call “LLM pollution” — an overwhelming proliferation of language models that often contribute more to noise than progress. This phenomenon has only intensified since 2023, with new models being released almost weekly, yet only a fraction finding meaningful applications or user adoption.

<figure id="e90b" class="graf graf--figure graf-after--p">
<img src="https://cdn-images-1.medium.com/max/800/0*1GJXc2pUcBGl-MDE" class="graf-image" data-image-id="0*1GJXc2pUcBGl-MDE" data-width="1342" data-height="900" />
</figure>

The best example for this is Meta’s <a href="https://huggingface.co/meta-llama" class="markup--anchor markup--p-anchor" data-href="https://huggingface.co/meta-llama" rel="noopener" target="_blank">LlaMa</a> family of models. Since it’s first open-source release with LlaMa in February 2023, many individuals, organizations and communities have retrained, finetuned, quantized and released thousands if not millions of variants of these models for domain-specific usecases. The problem here is there really isn’t proper **zero-to-one innovation** carrying forward research in AI on improving accuracy, fidelity and reasoning capabilities of LLMs.

As we move forward, the measure of success will be not how many models we can create, but how effectively we can deploy them to solve real-world problems.

### Conclusion: Putting It All Together

<figure id="3476" class="graf graf--figure graf-after--h3">
<img src="https://cdn-images-1.medium.com/max/800/0*AhTtaOr-IYL4F2sv.jpg" class="graf-image" data-image-id="0*AhTtaOr-IYL4F2sv.jpg" data-width="1079" data-height="605" />
</figure>

Throughout this comprehensive guide, we’ve journeyed through the landscape of Large Language Models, covering essential concepts from the ground up. We explored the foundational transformer architecture, delved into attention mechanisms, and understood the critical distinction between training and inference. We examined various fine-tuning techniques like LoRA and QLoRA, investigated quantization methods from basic INT8 to advanced approaches like GPTQ and GGML, and even tackled the mathematics behind inference arithmetic and KV cache calculations. Along the way, we also addressed practical concerns like model initialization strategies and the challenge of LLM pollution in today’s AI landscape.

**Until next time then! Thank you for taking your time reading through the article🤗**









By <a href="https://medium.com/@aloshdenny" class="p-author h-card">Aloshdenny</a> on [October 3, 2024](https://medium.com/p/cc4c6b906ae0).

<a href="https://medium.com/@aloshdenny/the-complete-dummies-guide-to-llms-cc4c6b906ae0" class="p-canonical">Canonical link</a>

Exported from [Medium](https://medium.com) on February 2, 2026.
