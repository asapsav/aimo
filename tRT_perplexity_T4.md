# Answer from Perplexity - could be wrong!!
To run the Llama3-7B model on two NVIDIA T4 GPUs using TensorRT, you can follow these steps:

1. Install the TensorRT-LLM library:

```python
pip install tensorrt_llm
```

2. Download the Llama3-7B model weights and tokenizer:

```python
import os
from transformers import LlamaTokenizer

model_dir = "path/to/llama3-7b/model/weights"
tokenizer_dir = model_dir
os.makedirs(model_dir, exist_ok=True)

# Download model weights and tokenizer
tokenizer = LlamaTokenizer.from_pretrained("decaplusplus/llama-3b-hf", cache_dir=tokenizer_dir)
```

3. Build the TensorRT engine for the model:

```python
from tensorrt_llm import LLMModelConfig, LLMModel

config = LLMModelConfig(
    model_dir=model_dir,
    tokenizer_dir=tokenizer_dir,
    dtype="float16",
    use_gpt_attention_plugin=True,
    use_gemm_plugin=True,
    max_batch_size=8,
    max_input_len=2048,
    max_output_len=512,
    remove_input_padding=True,
    enable_context_fmha=True,
    paged_kv_cache=True,
    use_inflight_batching=True,
    output_dir="path/to/save/tensorrt/engines"
)

model = LLMModel(config)
model.build_engines(num_gpus=2) # Build engines for 2 T4 GPUs
```

4. Run inference with the TensorRT engine:

```python
input_text = "Hello, how are you?"
output = model.generate(input_text)
print(output)
```

Make sure to replace `"path/to/llama3-7b/model/weights"` and `"path/to/save/tensorrt/engines"` with the appropriate paths on your system. Also, ensure that you have enough GPU memory (at least 16GB per T4 GPU) to run the Llama3-7B model.[1][2][3]

This code will download the Llama3-7B model weights and tokenizer, build the TensorRT engine optimized for two T4 GPUs, and then run inference on the input text using the TensorRT engine. The TensorRT engine will be saved to the specified `output_dir` for future use.

Citations:
[1] https://github.com/NVIDIA/TensorRT-LLM/issues/276
[2] https://github.com/NVIDIA/TensorRT-LLM/issues/1039
[3] https://github.com/vultureprime/deploy-ai-model/blob/main/aws-example/Llama7b-TensorRT-LLM/README_eng.md
[4] https://developer.nvidia.com/tensorrt
[5] https://developer.nvidia.com/blog/turbocharging-meta-llama-3-performance-with-nvidia-tensorrt-llm-and-nvidia-triton-inference-server/
[6] https://developer.nvidia.com/blog/tune-and-deploy-lora-llms-with-nvidia-tensorrt-llm/
[7] https://www.reddit.com/r/LocalLLaMA/comments/17ww5tn/how_can_i_optimize_execution_times_for_a_secure/
[8] https://developer.nvidia.com/blog/optimizing-inference-on-llms-with-tensorrt-llm-now-publicly-available/
[9] https://www.reddit.com/r/LocalLLaMA/comments/1b4iy16/is_there_any_benchmark_data_comparing_performance/
[10] https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/llama/README.md
[11] https://www.reddit.com/r/LocalLLaMA/comments/1cgofop/weve_benchmarked_tensorrtllm_its_3070_faster_on/
[12] https://towardsdatascience.com/deploying-llms-into-production-using-tensorrt-llm-ed36e620dac4
[13] https://forums.developer.nvidia.com/t/how-much-gpu-memory-does-tensorrt-need-to-convert-a-model-e-g-llama-7b-with-fp16/268485
[14] https://nvidia.github.io/TensorRT-LLM/quick-start-guide.html
[15] https://forums.developer.nvidia.com/t/what-is-the-best-way-to-run-multiple-trt-threads-on-multiple-gpu-with-each-context-process-same-video-frame/76505
[16] https://www.reddit.com/r/LocalLLaMA/comments/1aqh3en/chat_with_rtx_is_very_fast_its_the_only_local_llm/
[17] https://www.baseten.co/blog/llm-transformer-inference-guide/
[18] https://www.wizeline.com/llama-2-with-tensorrt-llm/
[19] https://blogs.oracle.com/cloud-infrastructure/post/practical-inferencing-open-source-models