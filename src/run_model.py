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


torch.manual_seed(0)
argp = argparse.ArgumentParser()
argp.add_argument('function', help="Choose train or evaluate") #TODO: add behavior for pretrain and eval
argp.add_argument('--writing_params_path', type=str, help='Path to the writing params file', default="writing_params.params")
argp.add_argument('--reading_params_path', type=str, help='Path to the reading params file', default="six_epochs.params")
argp.add_argument('--outputs_path', type=str, help='Path to the output predictions', default="predictions.txt", required=False)
argp.add_argument('--loss_path', type=str, help='Path to the output losses', default="losses.txt", required=False)
argp.add_argument('--max_epochs', type=int, help='Number of epochs to train for', default=5, required=False)
argp.add_argument('--learning_rate', type=float, help='Learning rate', default=2e-5, required=False)
argp.add_argument('--seed', type=int, help='Number of epochs to train for', default=0, required=False)
argp.add_argument('--model_name', type=str, help='Name of model to use', default="Transformer", required=False)
args = argp.parse_args()

# Save the device
device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
video_transformer = transforms.VideoFilePathToTensor(max_len=35, fps=5, padding_mode='last')
H, W = 256, 256
transforms = torchvision.transforms.Compose([
            transforms.VideoResize([H, W]),
        ])
                             

if args.function == 'pretrain':
    pass

elif args.function == 'train':
    # get the dataloaders. can make test and val sizes 0 if you don't want them
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
    # TensorBoard training log
    writer = SummaryWriter(log_dir='expt/')

    train_config = trainer.TrainerConfig(max_epochs=args.max_epochs,
            learning_rate=args.learning_rate, 
            num_workers=4, writer=writer, ckpt_path='expt/params.pt')

    if args.model_name == "Base":
        model = model.BaseVideoModel(num_outputs=3, L=video_transformer.max_len, H=H, W=W)
    elif args.model_name == "LSTM":
        model = model.ResnetLSTM(num_outputs=3, L=video_transformer.max_len, H=H, W=W)
    elif args.model_name == "Transformer":
        model = model.ResnetTransformer(num_outputs=3, L=video_transformer.max_len, H=H, W=W)
    else:
        raise ValueError(f"Model name {args.model_name} not recognized")

    
    trainer = trainer.Trainer(model=model,  train_dataloader=train_dl, 
    test_dataloader=test_dl, config=train_config, val_dataloader=val_dl)
    for epoch in range(args.max_epochs):
        print(epoch)
        trainer.train(split='train', step=epoch)
        trainer.train(split='val', step=epoch)
    torch.save(model.state_dict(), args.writing_params_path)
    with open(args.loss_path, 'w') as f:
        for loss in trainer.losses:
            f.write(f"{loss[0]},{loss[1]}\n")

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
        model = model.BaseVideoModel(num_outputs=3, L=video_transformer.max_len, H=H, W=W)
    elif args.model_name == "LSTM":
        model = model.ResnetLSTM(num_outputs=3, L=video_transformer.max_len, H=H, W=W)
    elif args.model_name == "Transformer":
        model = model.ResnetTransformer(num_outputs=3, L=video_transformer.max_len, H=H, W=W)
    else:
        raise ValueError(f"Model name {args.model_name} not recognized")

    model.load_state_dict(torch.load(args.reading_params_path, map_location=torch.device('cpu')))
    model = model.to(device)
    model.eval()
    torch.set_grad_enabled(False)
    predictions = []

    pbar = tqdm(enumerate(test_dl), total=len(test_dl)) 
    # pred_cols = [f'pred_{c}' for c in dataset.targets_sentence.columns] + [f'pred_word_{c}' for c in dataset.targets_words.columns] + [f'pred_{c}' for c in dataset.targets_phones.columns]
    pred_cols = ['pred_fall_risk']
    actual_cols = ['y_fall_risk']
    for it, (subj_id, x, y) in pbar:
        print(it)
        # place data on the correct device
        with torch.no_grad():
            x = x.to(device)
            pred = model(x)[0]
            predictions.append(({'id': subj_id, **dict(zip(pred_cols, pred.tolist())), **dict(zip(actual_cols, y.tolist()))}))

    pd.DataFrame(predictions).to_csv(args.outputs_path, index=False)
    

else:
    print("Invalid function name. Choose pretrain, finetune, or evaluate")     