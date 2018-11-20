import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

##### DATA TYPE #####
Variable.data == Tensor
Tensor.item() # one element value of a Tensor

##### LOAD/SAVE #####
torch.save(model.state_dict(), model_path)

if torch.cuda.is_available():
    model.load_state_dict(torch.load(model_path))
# use cpu to load gpu trained models
else:
    model.load_state_dict(torch.load(model_path, 
        map_location=lambda storage, loc: storage))

##### TRAIN #####
optimizer = optim.SGD(model.parameters(), lr=learning_rate)
criterion = nn.NLLLoss()

loss.backward()
optimizer.step()

##### TEST #####
# reduce memory usage if no Tensor.backward() is called
with torch.no_grad():
    # evaluation
