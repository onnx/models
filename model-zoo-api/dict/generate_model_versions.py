from pathlib import Path
import re
from collections import defaultdict


# identify list of onnx models in model Zoo
model_list = []

def get_a_list():
    return []

# generate default dictionary
dict = defaultdict(get_a_list)

for path in Path('text').rglob('*.onnx'):
    model_list.append(str(path))
    # obtains model name
    pattern1 = re.compile(".*/([^/]+)/model")
    m1 = pattern1.match(str(path))
    if m1 != None:
        model_name = m1.group(1)
        # obtains file name
        pattern2 = re.compile(".*/([^/]+\\.*).onnx")
        m2 = pattern2.match(str(path))
        file_name = m2.group(1)
        dict[model_name].append(file_name)

for path in Path('vision').rglob('*.onnx'):
    model_list.append(str(path))
    # obtains model name
    pattern1 = re.compile(".*/([^/]+)/model")
    m1 = pattern1.match(str(path))
    if m1 != None:
        model_name = m1.group(1)
        # obtains file name
        pattern2 = re.compile(".*/([^/]+\\.*).onnx")
        m2 = pattern2.match(str(path))
        file_name = m2.group(1)
        dict[model_name].append(file_name)
        
        
print(dict)
