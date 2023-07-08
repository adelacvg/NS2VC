from model import Trainer

trainer = Trainer(cfg_path='config.json')
# trainer.load(75)
trainer.train()
