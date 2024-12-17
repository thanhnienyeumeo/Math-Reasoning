prompt_qwen = '''<|im_start|>system\nYou are a helpful assistant.<|im_end|>
<|im_start|>user
{instruction}
Put your final answer within \\boxed{{}}.<|im_end|>
<|im_start|>assistant\n'''
prompt_phi = '''<|system|>\nYou are a helpful assistant.
<|user|>
{instruction}
Put your final answer within \\boxed{{}}
<|assistant|>\n'''
prompt_llama = "" #not implemented yet