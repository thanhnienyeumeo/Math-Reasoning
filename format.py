prompt_qwen = '''<|im_start|>system\nYou are a helpful assistant.<|im_end|>
<|im_start|>user
{instruction}
<|im_end|>
<|im_start|>assistant\n'''
prompt_qwen2 = '''<|im_start|>user
{instruction}
<|im_end|>
<|im_start|>assistant\n'''
prompt_phi = '''<|system|>\nYou are a helpful assistant.
<|user|>
{instruction}
<|assistant|>\n'''
prompt_llama = '''<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a helpful AI assistant<|eot_id|>
<|start_header_id|>user<|end_header_id|>
{instruction}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>\n'''