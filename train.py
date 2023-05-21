from model import Trainer

trainer = Trainer(device='cuda:1')
# trainer.load(75)
trainer.train()
