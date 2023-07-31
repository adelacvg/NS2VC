from model import Trainer

trainer = Trainer(cfg_path='config.json')
# trainer.load('logs/tts/model-200.pt')
trainer.train()
