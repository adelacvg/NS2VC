from model import Trainer

trainer = Trainer()
trainer.load('logs/vc/2023-09-28-20-49-43/model-639.pt')
trainer.train()
