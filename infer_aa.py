import torchaudio

import 
    data = next(self.eval_dl)
    data = {k: v.to(self.device) for k, v in data.items()}

    with torch.no_grad():
        model = accelerator.unwrap_model(self.model)
        model.eval()
        milestone = self.step // self.save_and_sample_every
        log = model.log_images(data)
        mel = log['samples'].detach().cpu()
        mel = denormalize_tacotron_mel(mel)
        model.train()
    gen = self.vocos.decode(mel)
    torchaudio.save(str(self.logs_folder / f'sample-{milestone}.wav'), gen, 24000)