import os
import sys
import json
import onnxruntime as ort

onnx_model_path  = sys.argv[1]
model_name = os.path.splitext(os.path.basename(onnx_model_path))[0]
ctx_cache = model_name + "_ctx.onnx"

# Delete prexisting EP context cache model
if os.path.exists(ctx_cache):
    print(f"INFO: EP context model {ctx_cache} already exists. Deleting it.")
    os.remove(ctx_cache)

session_options = ort.SessionOptions()
session_options.add_session_config_entry('ep.context_enable', '1') 
session_options.add_session_config_entry('ep.context_file_path', ctx_cache) 
session_options.add_session_config_entry('ep.context_embed_mode', '1') 
onnx_session = ort.InferenceSession(
                   onnx_model_path,
                   sess_options=session_options,
                   providers=["VitisAIExecutionProvider"],
                   provider_options=[{"cache_dir": os.getcwd(),
                                      "cache_key": model_name,}]
               )
