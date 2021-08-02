import torch
if torch.cuda.is_available():  
  dev = "cuda:0" 
else:  
  dev = "cpu"
device = torch.device(dev)  
a = torch.zeros(4,3)    
a = a.to(device)
print(a)