import numpy as np
import pandas as pd
import torch
import torchvision

import random
import argparse

from modeling import trainer, model
from data_loading import dataloaders, transforms

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import multiprocessing


torch.manual_seed(0)
argp = argparse.ArgumentParser()
argp.add_argument('function', help="Choose train or evaluate") #TODO: add behavior for pretrain and eval
argp.add_argument('--writing_params_path', type=str, help='Path to the writing params file', default="base.params")
argp.add_argument('--reading_params_path', type=str, help='Path to the reading params file', default="base.params")
argp.add_argument('--outputs_path', type=str, help='Path to the output predictions', default="new.csv", required=False)
argp.add_argument('--loss_path', type=str, help='Path to the output losses', default="base.txt", required=False)
argp.add_argument('--max_epochs', type=int, help='Number of epochs to train for', default=15, required=False)
argp.add_argument('--learning_rate', type=float, help='Learning rate', default=2e-5, required=False)
argp.add_argument('--seed', type=int, help='Number of epochs to train for', default=0, required=False)
argp.add_argument('--model_name', type=str, help='Name of model to use', default="LSTM", required=False)
args = argp.parse_args()

if __name__ == '__main__':
    # Save the device
    device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
    print(device)
    video_transformer = transforms.VideoFilePathToTensor(max_len=22*1, fps=1, padding_mode='zero')
    H, W = 256, 256
    transforms = torchvision.transforms.Compose([
                transforms.VideoResize([H, W]),
                transforms.VideoRandomHorizontalFlip(),
                transforms.NormalizeVideoFrames()
            ])
                                
    if args.function == 'pretrain':
        pass

    elif args.function == 'train':
        # get the dataloaders. can make test and val sizes 0 if you don't want them
        train_dl, val_dl, test_dl = dataloaders.get_vid_data_loaders(
            video_transformer=video_transformer,
            batch_size=4,
            val_batch_size=4,
            test_batch_size=1,
            transforms=transforms,
            preload_videos=False,
            labels=['y_fall_risk_binary', 'amm_4', 'amm_5', 'PES_2', 'PES_5'],
            num_workers=3
        )
        # TensorBoard training log
        writer = SummaryWriter(log_dir='expt/')

        train_config = trainer.TrainerConfig(max_epochs=args.max_epochs,
                learning_rate=args.learning_rate, 
                num_workers=4, writer=writer, ckpt_path='expt/params.pt')

        if args.model_name == "Base":
            model = model.BaseVideoModel(num_outputs=5, L=video_transformer.max_len, H=H, W=W, device=device)
        elif args.model_name == "LSTM":
            model = model.ResnetLSTM(num_outputs=5, L=video_transformer.max_len, H=H, W=W, device=device)
        elif args.model_name == "Transformer":
            model = model.ResnetTransformer(num_outputs=5, L=video_transformer.max_len, H=H, W=W, device=device)
        elif args.model_name == "OpenPose":
            model = model.BaseOpenPose(num_outputs=5, L=video_transformer.max_len, H=H, W=W, device=device)
        else:
            raise ValueError(f"Model name {args.model_name} not recognized")

        
        trainer = trainer.Trainer(model=model,  train_dataloader=train_dl, 
        test_dataloader=test_dl, config=train_config, val_dataloader=val_dl, median_freq_weights=True)
        train_losses = []
        val_losses = []
        best_val_loss = np.inf
        for epoch in range(args.max_epochs):
            print(epoch)
            train_losses.append(trainer.train(split='train', step=epoch))
            val_loss = trainer.train(split='val', step=epoch)
            val_losses.append(val_loss)
            print("Val loss:", val_loss)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                print("Saving model after epoch", epoch)
                torch.save(model.state_dict(), args.writing_params_path)
            torch.save(model.state_dict(), f"{args.model_name}_{epoch}.params")
        # write csv of losses
        with open(args.loss_path, 'w') as f:
            for train_loss, val_loss in zip(train_losses, val_losses):
                f.write(f"{train_loss},{val_loss}\n")

    elif args.function == 'evaluate':
        train_dl, val_dl, test_dl = dataloaders.get_vid_data_loaders(
            video_transformer=video_transformer,
            batch_size=4,
            val_batch_size=1,
            test_batch_size=1,
            transforms=transforms,
            preload_videos=False,
            labels=['y_fall_risk'],
            num_workers=0
        )
        if args.model_name == "Base":
            model = model.BaseVideoModel(num_outputs=3, L=video_transformer.max_len, H=H, W=W, device=device)
        elif args.model_name == "LSTM":
            model = model.ResnetLSTM(num_outputs=3, L=video_transformer.max_len, H=H, W=W, device=device)
        elif args.model_name == "Transformer":
            model = model.ResnetTransformer(num_outputs=3, L=video_transformer.max_len, H=H, W=W, device=device)
        else:
            raise ValueError(f"Model name {args.model_name} not recognized")

        model.load_state_dict(torch.load(args.reading_params_path, map_location=torch.device('cpu')))
        model = model.to(device)
        model.eval()
        torch.set_grad_enabled(False)
        predictions = []

        pbar = tqdm(enumerate(test_dl), total=len(test_dl)) 
        # pred_cols = [f'pred_{c}' for c in dataset.targets_sentence.columns] + [f'pred_word_{c}' for c in dataset.targets_words.columns] + [f'pred_{c}' for c in dataset.targets_phones.columns]
        pred_cols = ['pred_fall_risk_0', 'pred_fall_risk_1', 'pred_fall_risk_2']
        actual_cols = ['y_fall_risk']
        for it, (subj_id, x, y) in pbar:
            print(it)
            # place data on the correct device
            with torch.no_grad():
                x = x.to(device)
                pred = model(x)[0]
                print(pred)
                print(y)
                predictions.append(({'id': subj_id, **dict(zip(pred_cols, pred.tolist()[0])), **dict(zip(actual_cols, y.tolist()[0]))}))

        pd.DataFrame(predictions).to_csv(args.outputs_path, index=False)
    else:
        print("Invalid function name. Choose train or evaluate")     