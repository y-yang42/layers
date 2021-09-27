class SimCLRLoss(torch.nn.Module):
    def __init__(self, batch_size, device='cpu'):
        super(SimCLRLoss, self).__init__()
        self.device = device
        self.batch_size = batch_size
        self.mask = self.create_mask(batch_size)
        self.labels = torch.arange(batch_size).to(device)
        self.criterion = torch.nn.CrossEntropyLoss()

    # create a mask that enables us to sum over positive pairs only
    def create_mask(self, batch_size):
        mask = torch.eye(batch_size, dtype=torch.bool).to(self.device)
        return mask

    def forward(self, output, tau=0.1):
        norm = torch.nn.functional.normalize(output, dim=1)
        h1,h2 = torch.split(norm, self.batch_size)

        aa = torch.mm(h1,h1.transpose(0,1))/tau
        aa_s = aa[~self.mask].view(aa.shape[0],-1)
        bb = torch.mm(h2,h2.transpose(0,1))/tau
        bb_s = bb[~self.mask].view(bb.shape[0],-1)
        ab = torch.mm(h1,h2.transpose(0,1))/tau
        ba = torch.mm(h2,h1.transpose(0,1))/tau
  
        loss_a = self.criterion(torch.cat([ab,aa_s],dim=1),self.labels)
        loss_b = self.criterion(torch.cat([ba,bb_s],dim=1),self.labels)

        loss = loss_a+loss_b
        return loss