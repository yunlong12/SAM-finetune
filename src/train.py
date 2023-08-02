import os

from tqdm.auto import tqdm
import torch
import torch.nn as nn
import datetime

from .data import get_dataloader
from .utils import draw_mask_onimage
import matplotlib.pyplot as plt

def train(args, model):

    dataloader = get_dataloader(args.base_dir, args.mode)
    optimizer = torch.optim.Adam(model.sam_model.mask_decoder.parameters(), lr=args.lr)
    loss_fn = torch.nn.BCELoss(reduction="mean")  #nn.MSELoss()

    accum_iter = 10
    best_model_loss = 1e10
    best_model_ckpt = None
    best_model_epoch = 0
    avg_losses = []

    now = datetime.datetime.now()
    now_str = now.strftime("%Y-%m-%d_%H-%M-%S")
    loss_file = open(f"./logs/avg_losses_{now_str}.txt", "a")

    for ep in range(1, args.epochs + 1):
        
        total_loss = 0
        for i,(X,gt_mask,idx) in enumerate(tqdm(dataloader)):

            print(gt_mask.shape)
            X_orig = X.copy()
            gt_mask, pred_mask, binary_mask = model(X, gt_mask)

            # train step

            #loss = loss_fn(pred_mask.squeeze(), gt_mask)
            loss = loss_fn(torch.sigmoid(pred_mask.squeeze()), gt_mask.squeeze())
            total_loss += loss.item()
            loss.backward()

            #if (i + 1) % accum_iter == 0 or (i + 1) == len(dataloader):
            #if (i + 1) == len(dataloader):
            if loss > 0.01:
                optimizer.step()
                optimizer.zero_grad()
                print("--------------loss:{},idx:{}".format(loss,idx))

                #if i % args.save_every == 0:
                rsPath = os.path.join(args.results_dir,"epoch{}".format(ep))
                # If the directory does not exist, create it
                if not os.path.exists(rsPath):
                    os.makedirs(rsPath)
                draw_mask_onimage(X_orig, binary_mask.squeeze(), os.path.join(rsPath, f'ep{ep}_{idx}.jpg'))
                draw_mask_onimage(X_orig, gt_mask, os.path.join(rsPath, f'ep{ep}_{idx}_gt.jpg'))

            
            print(f'LOSS {loss.item()}, IDX:{idx}')

            del gt_mask, pred_mask, loss
            torch.cuda.empty_cache()

            # if i == 5:
            #     break

        #if ep % args.ckpt_every == 0:
        torch.save(model.sam_model.state_dict(), os.path.join(args.checkpoint_dir, f'sam_ckpt_{ep}.pth'))

        avg_loss = total_loss / len(dataloader.dataset)
        avg_losses.append(avg_loss)
        print(f'EPOCH {ep} | AVERAGE LOSS {avg_loss}')
        loss_file.write(f'EPOCH {ep} | AVERAGE LOSS {avg_loss}\n')
        loss_file.flush()
        if avg_loss < best_model_loss:
            best_model_loss = avg_loss
            best_model_ckpt = model.sam_model.state_dict().copy()
            best_model_epoch = ep



    torch.save(best_model_ckpt, os.path.join(args.checkpoint_dir, f'best_model.pth'))
    print(f'BEST MODEL EPOCH {best_model_epoch} | LOSS {best_model_loss}')
    loss_file.close()

