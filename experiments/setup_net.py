import torch.nn as nn
from layers import FALinear, FAConv2d, UfLinear, UfConv2d, UsLinear, UsConv2d

def setup_net(pars):
    net = nn.Sequential()
    classifier = nn.Sequential()

    if pars.dataset == 'Cifar100':
        NUM_CLASS = 100
    else:
        NUM_CLASS = 10

    if pars.architecture == 'LW':
        HW=32
        NUM_CHANNEL=pars.NUM_CHANNEL
        pars.NUM_LAYER = 5

        for i in range(pars.NUM_LAYER):
            layer = nn.Sequential()
            if (i==2) or (i==3):
                layer.add_module('max_pool', nn.MaxPool2d(2,stride=2))
                HW /= 2

            if i==0:
                if pars.process != 'E2E':
                    layer.add_module('conv', nn.Conv2d(3,NUM_CHANNEL,3,padding=1))
                elif pars.update == 'FA':
                    layer.add_module('conv', FAConv2d(3,NUM_CHANNEL,3,padding=1))
                elif pars.update == 'UF':
                    layer.add_module('conv', UfConv2d(3,NUM_CHANNEL,3,padding=1))
                elif pars.update == 'US':
                    layer.add_module('conv', UsConv2d(3,NUM_CHANNEL,3,padding=1))
                else:
                    layer.add_module('conv', nn.Conv2d(3,NUM_CHANNEL,3,padding=1))
            else:
                if pars.process != 'E2E':
                    layer.add_module('conv', nn.Conv2d(NUM_CHANNEL,NUM_CHANNEL,3,padding=1))
                elif pars.update == 'FA':
                    layer.add_module('conv', FAConv2d(NUM_CHANNEL,NUM_CHANNEL,3,padding=1))
                elif pars.update == 'UF':
                    layer.add_module('conv', UfConv2d(NUM_CHANNEL,NUM_CHANNEL,3,padding=1))
                elif pars.update == 'US':
                    layer.add_module('conv', UsConv2d(NUM_CHANNEL,NUM_CHANNEL,3,padding=1))
                else:
                    layer.add_module('conv', nn.Conv2d(NUM_CHANNEL,NUM_CHANNEL,3,padding=1))

            if pars.nonlinear == 'hardtanh':
                layer.add_module('activation', nn.Hardtanh())
            else:
                layer.add_module('activation', nn.Tanh())
        
            aux = nn.Sequential(
                nn.AvgPool2d(int(HW/2),int(HW/2)),
                nn.Flatten(),
            )

            
            if pars.update=='FA':
                aux.add_module('fc', FALinear(NUM_CHANNEL*2*2,NUM_CLASS))
            elif pars.update=='UF':
                aux.add_module('fc', UfLinear(NUM_CHANNEL*2*2,NUM_CLASS))
            elif pars.update=='US':
                aux.add_module('fc', UsLinear(NUM_CHANNEL*2*2,NUM_CLASS))
            else:
                aux.add_module('fc', nn.Linear(NUM_CHANNEL*2*2,NUM_CLASS))

            net.add_module('layer%d'%i, layer)
            if pars.process != 'E2E':
                classifier.add_module('aux%d'%i, aux)
            elif i==3:
                classifier.add_module('aux%d'%i, aux)

    else: # 'CONV'
        HW = 32
        NUM_CHANNEL = 32
        pars.NUM_LAYER=4

        for i in range(pars.NUM_LAYER):
            layer = nn.Sequential()

            if i==0:
                if pars.process != 'E2E':
                    layer.add_module('conv', nn.Conv2d(3,int(NUM_CHANNEL),5,padding=2))
                elif pars.update == 'FA':
                    layer.add_module('conv', FAConv2d(3,int(NUM_CHANNEL),5,padding=2))
                elif pars.update == 'UF':
                    layer.add_module('conv', UfConv2d(3,int(NUM_CHANNEL),5,padding=2))
                elif pars.update == 'US':
                    layer.add_module('conv', UsConv2d(3,int(NUM_CHANNEL),5,padding=2))
                else:
                    layer.add_module('conv', nn.Conv2d(3,int(NUM_CHANNEL),5,padding=2))
            elif i==3:
                if pars.process != 'E2E':
                    layer.add_module('conv', nn.Conv2d(int(NUM_CHANNEL),int(NUM_CHANNEL*2),3,padding=1))
                elif pars.update == 'FA':
                    layer.add_module('conv', FAConv2d(int(NUM_CHANNEL),int(NUM_CHANNEL*2),3,padding=1))
                elif pars.update == 'UF':
                    layer.add_module('conv', UfConv2d(int(NUM_CHANNEL),int(NUM_CHANNEL*2),3,padding=1))
                elif pars.update == 'US':
                    layer.add_module('conv', UsConv2d(int(NUM_CHANNEL),int(NUM_CHANNEL*2),3,padding=1))
                else:
                    layer.add_module('conv', nn.Conv2d(int(NUM_CHANNEL),int(NUM_CHANNEL*2),3,padding=1))
                NUM_CHANNEL *= 2
            else:
                if pars.process != 'E2E':
                    layer.add_module('conv', nn.Conv2d(int(NUM_CHANNEL),int(NUM_CHANNEL*2),5,padding=2))
                elif pars.update == 'FA':
                    layer.add_module('conv', FAConv2d(int(NUM_CHANNEL),int(NUM_CHANNEL*2),5,padding=2))
                elif pars.update == 'UF':
                    layer.add_module('conv', UfConv2d(int(NUM_CHANNEL),int(NUM_CHANNEL*2),5,padding=2))
                elif pars.update == 'US':
                    layer.add_module('conv', UsConv2d(int(NUM_CHANNEL),int(NUM_CHANNEL*2),5,padding=2))
                else:
                    layer.add_module('conv', nn.Conv2d(int(NUM_CHANNEL),int(NUM_CHANNEL*2),5,padding=2))
                NUM_CHANNEL *= 2

            if pars.nonlinear == 'hardtanh':
                layer.add_module('activation', nn.Hardtanh())
            else:
                layer.add_module('activation', nn.Tanh())

            layer.add_module('maxpool', nn.MaxPool2d(2))
            HW /= 2
        
            aux = nn.Sequential(
                nn.Flatten(),
            )
            if pars.update=='FA':
                aux.add_module('fc', FALinear(int(NUM_CHANNEL*HW*HW),NUM_CLASS))
            elif pars.update=='UF':
                aux.add_module('fc', UfLinear(int(NUM_CHANNEL*HW*HW),NUM_CLASS))
            elif pars.update=='US':
                aux.add_module('fc', UsLinear(int(NUM_CHANNEL*HW*HW),NUM_CLASS))
            else:
                aux.add_module('fc', nn.Linear(int(NUM_CHANNEL*HW*HW),NUM_CLASS))

            net.add_module('layer%d'%i, layer)
            if pars.process != 'E2E':
                classifier.add_module('aux%d'%i, aux)
            elif i==3:
                classifier.add_module('aux%d'%i, aux)

    return net, classifier