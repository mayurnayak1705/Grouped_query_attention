Grouped Query Attention (GQA) introduces a smarter tradeoff between efficiency and performance.

For example, with:

Model dimension = 4096
Number of heads = 32
Number of KV groups = 8

Each KV head is shared across multiple query heads.

Instead of:

32 separate K/V heads (MHA)

we get:

8 shared K/V heads (GQA)

This dramatically reduces:

KV cache memory
Memory bandwidth usage
Inference latency

while still preserving model quality close to Multi-Head Attention.

At the other extreme, Multi-Query Attention (MQA) shares a single KV head across all query heads, maximizing efficiency further.

As part of understanding the internals deeply, I implemented:

Multi-Head Attention (MHA)
Grouped Query Attention (GQA)
Multi-Query Attention (MQA)

completely from scratch in PyTorch.

One interesting thing I learned is that during implementation, K/V tensors are logically repeated to match query heads, but optimized implementations avoid physical memory duplication, which is where much of the efficiency gain comes from.

This exploration helped me better understand:

KV cache optimization
Attention computation internals
Memory bandwidth bottlenecks in LLM inference
Why modern models like Llama use GQA-style architectures
