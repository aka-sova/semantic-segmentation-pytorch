
import torch
import torch.nn as nn

# get the default cuda device
device = torch.device('cuda')

print('Cuda available: ', torch.cuda.is_available())
print('Cuda name: ', torch.cuda.get_device_name())
print('Currently memory allocated: ', torch.cuda.memory_allocated(), '/', torch.cuda.max_memory_allocated(device=device))
print('Currently memory cached: ', torch.cuda.memory_cached(), '/', torch.cuda.max_memory_cached(device=device))