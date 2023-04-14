import torch
import torch.nn as nn
class levelsetLoss(nn.Module):
    def __init__(self):
        super(levelsetLoss, self).__init__()

    def forward(self,output, target):
        # input size = batch x 1 (channel) x height x width
        outshape = output.shape #[2,21,512,512]
        tarshape = target.shape #[2,3,512,512]
        loss = 0.0
        for ich in range(tarshape[1]):
            target_ = torch.unsqueeze(target[:,ich], 1) # [2,1,512,512]
            target_ = target_.expand(tarshape[0], outshape[1], tarshape[2], tarshape[3]) # [2,21,512,512]
            pcentroid = torch.sum(target_ * output, (2,3))/torch.sum(output, (2,3)) 
            pcentroid = pcentroid.view(tarshape[0], outshape[1], 1, 1)#[2,21,1,1]
            plevel = target_ - pcentroid.expand(tarshape[0], outshape[1], tarshape[2], tarshape[3])
            pLoss = plevel * plevel * output
            loss += torch.sum(pLoss)
        return loss
		
class gradientLoss2d(nn.Module):
    def __init__(self, penalty='l1'):
        super(gradientLoss2d, self).__init__()
        self.penalty = penalty

    def forward(self, input):
        dH = torch.abs(input[:, :, 1:, :] - input[:, :, :-1, :])
        dW = torch.abs(input[:, :, :, 1:] - input[:, :, :, :-1])
        if(self.penalty == "l2"):
            dH = dH * dH
            dW = dW * dW

        loss = torch.sum(dH) + torch.sum(dW)
        return loss

def MS_loss(output, data, alpha=1e-10, beta=1e-7):
    LS = levelsetLoss() # tensor(15113757696., device='cuda:0', grad_fn=<ThAddBackward>)
    loss_LS = LS(output, data)#print(loss_LS)
    TV = gradientLoss2d()
    loss_TV = TV(data)* alpha # 1e-10  # tensor(5405.8213, device='cuda:0')
    #print(loss_TV)
    loss_MS = (loss_LS*beta + loss_TV) # beta=1e-7 example 0.1521
    loss_MS.requires_grad_(True) # important  
    return loss_MS    
