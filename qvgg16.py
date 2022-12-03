import torch
import torch.nn as nn

class VGG(nn.Module):
	def __init__(self, features, num_classes=1000, init_weights=False):
		super(VGG,self).__init__()
		self.features = features  # 提取特征部分的网络，也为Sequential格式
		self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
		self.classifier = nn.Sequential(  # 分类部分的网络
			nn.Linear(512*7*7,4096),
			nn.ReLU(),
			nn.Dropout(p=0.5),
			nn.Linear(4096,4096),
			nn.ReLU(),
			nn.Dropout(p=0.5),
			nn.Linear(4096,num_classes)
		)
		# add the quantize part
		self.quant = torch.quantization.QuantStub()
		self.dequant = torch.quantization.DeQuantStub()

		if init_weights:
			self._initialize_weights()

	def forward(self,x):
		x = self.quant(x)
		x = self.features(x)
		x = self.avgpool(x)
		x = torch.flatten(x,start_dim=1)
		# x = x.mean([2, 3])
		x = self.classifier(x)
		x = self.dequant(x)
		return x

	def _initialize_weights(self):
		for module in self.modules():
			if isinstance(module,nn.Conv2d):
				# nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
				nn.init.xavier_uniform_(module.weight)
				if module.bias is not None:
					nn.init.constant_(module.bias,0)
			elif isinstance(module,nn.Linear):
				nn.init.xavier_uniform_(module.weight)
				# nn.init.normal_(m.weight, 0, 0.01)
				nn.init.constant_(module.bias,0)

cfgs = {
	'vgg11':[64,'M',128,'M',256,256,'M',512,512,'M',512,512,'M'],
	'vgg13':[64,64,'M',128,128,'M',256,256,'M',512,512,'M',512,512,'M'],
	'vgg16':[64,64,'M',128,128,'M',256,256,256,'M',512,512,512,'M',512,512,512,'M'],
	'vgg19':[64,64,'M',128,128,'M',256,256,256,256,'M',512,512,512,512,'M',512,512,512,512,'M'],
}

def make_features(cfg:list):
	layers = []
	in_channels = 3
	for v in cfg:
		if v == 'M':
			layers += [nn.MaxPool2d(kernel_size=2,stride=2)]  #vgg采用的池化层均为2,2参数
		else:
			conv2d = nn.Conv2d(in_channels,v,kernel_size=3,padding=1)  #vgg卷积层采用的卷积核均为3,1参数
			layers += [conv2d,nn.ReLU(True)]
			in_channels = v
	return nn.Sequential(*layers)  #非关键字的形式输入网络的参数

def vgg(model_name='vgg16',**kwargs):
	try:
		cfg = cfgs[model_name]
	except:
		print("Warning: model number {} not in cfgs dict!".format(model_name))
		exit(-1)
	model = VGG(make_features(cfg),**kwargs)  # **kwargs为可变长度字典，保存多个输入参数
	return model

def vgg16():
	return vgg()