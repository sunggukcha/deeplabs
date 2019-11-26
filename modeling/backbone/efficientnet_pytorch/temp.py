from model import EfficientNet
from torchsummary import summary

model = EfficientNet.from_pretrained('efficientnet-b7')
model.cuda()
summary(model, input_size=(3, 360, 640))

