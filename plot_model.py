
import torch
from resnet import resnet34
from torchview import draw_graph


model_graph = draw_graph(resnet34(), input_size=(1,1,100), expand_nested=True,filename="resnet34")
# model_graph.visual_graph.render(format='svg')
model_graph.visual_graph