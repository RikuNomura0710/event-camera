import torch
import hydra
from torch import nn
from omegaconf import DictConfig
from torch.utils.data import DataLoader
import random
import numpy as np
from src.models.evflownet import EVFlowNet
from src.datasets import DatasetProvider
from enum import Enum, auto
from src.datasets import train_collate
from tqdm import tqdm
from pathlib import Path
from typing import Dict, Any
import os
import time
from torch.optim.lr_scheduler import OneCycleLR
from torch.optim.lr_scheduler import ReduceLROnPlateau


class RepresentationType(Enum):
    VOXEL = auto()
    STEPAN = auto()

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)

def multi_scale_loss(flow_dict: Dict[str, torch.Tensor], gt_flow: torch.Tensor) -> torch.Tensor:
    loss = 0
    for i, (key, flow) in enumerate(flow_dict.items()):
        scale_factor = 2 ** (3 - i)  # 3は最大スケール数-1
        scaled_gt = nn.functional.interpolate(gt_flow, scale_factor=1/scale_factor, mode='bilinear', align_corners=False)
        loss += compute_epe_error(flow, scaled_gt)
    return loss
# def multi_scale_loss(flow_dict: Dict[str, torch.Tensor], gt_flow: torch.Tensor) -> torch.Tensor:
#     loss = 0
#     for i, (key, flow) in enumerate(flow_dict.items()):
#         # 予測フローのサイズに合わせて真のフローをリサイズ
#         scaled_gt = nn.functional.interpolate(gt_flow, size=flow.shape[2:], mode='bilinear', align_corners=False)
#         loss += compute_epe_error(flow, scaled_gt)
#     return loss
# def multi_scale_loss(flow_dict: Dict[str, torch.Tensor], gt_flow: torch.Tensor, weights: list) -> torch.Tensor:
#     loss = 0
#     for i, (key, flow) in enumerate(flow_dict.items()):
#         # Resize ground truth flow to the size of the predicted flow
#         scaled_gt = nn.functional.interpolate(gt_flow, size=flow.shape[2:], mode='bilinear', align_corners=False)
#         # Compute weighted loss
#         loss += weights[i] * compute_epe_error(flow, scaled_gt)
#     return loss
        
def compute_epe_error(pred_flow: torch.Tensor, gt_flow: torch.Tensor):
    '''
    end-point-error (ground truthと予測値の二乗誤差)を計算
    pred_flow: torch.Tensor, Shape: torch.Size([B, 2, 480, 640]) => 予測したオプティカルフローデータ
    gt_flow: torch.Tensor, Shape: torch.Size([B, 2, 480, 640]) => 正解のオプティカルフローデータ
    '''
    epe = torch.mean(torch.mean(torch.norm(pred_flow - gt_flow, p=2, dim=1), dim=(1, 2)), dim=0)
    return epe


def save_optical_flow_to_npy(flow: torch.Tensor, file_name: str):
    '''
    optical flowをnpyファイルに保存
    flow: torch.Tensor, Shape: torch.Size([2, 480, 640]) => オプティカルフローデータ
    file_name: str => ファイル名
    '''
    np.save(f"{file_name}.npy", flow.cpu().numpy())

@hydra.main(version_base=None, config_path="configs", config_name="base")
def main(args: DictConfig):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    '''
        ディレクトリ構造:

        data
        ├─test
        |  ├─test_city
        |  |    ├─events_left
        |  |    |   ├─events.h5
        |  |    |   └─rectify_map.h5
        |  |    └─forward_timestamps.txt
        └─train
            ├─zurich_city_11_a
            |    ├─events_left
            |    |       ├─ events.h5
            |    |       └─ rectify_map.h5
            |    ├─ flow_forward
            |    |       ├─ 000134.png
            |    |       |.....
            |    └─ forward_timestamps.txt
            ├─zurich_city_11_b
            └─zurich_city_11_c
        '''
    
    # ------------------
    #    Dataloader
    # ------------------
    loader = DatasetProvider(
        dataset_path=Path(args.dataset_path),
        representation_type=RepresentationType.VOXEL,
        delta_t_ms=100,
        num_bins=4,
        transforms={
            'random_crop': (400, 400),  # 任意のサイズに調整
            'random_flip': True,
            'random_noise': 0.1,  # ノイズの強さ
        }
    )
    train_set = loader.get_train_dataset()
    test_set = loader.get_test_dataset()
    collate_fn = train_collate
    train_data = DataLoader(train_set,
                                 batch_size=args.data_loader.train.batch_size,
                                 shuffle=args.data_loader.train.shuffle,
                                 collate_fn=collate_fn,
                                 drop_last=False)
    test_data = DataLoader(test_set,
                                 batch_size=args.data_loader.test.batch_size,
                                 shuffle=args.data_loader.test.shuffle,
                                 collate_fn=collate_fn,
                                 drop_last=False)

    '''
    train data:
        Type of batch: Dict
        Key: seq_name, Type: list
        Key: event_volume, Type: torch.Tensor, Shape: torch.Size([Batch, 4, 480, 640]) => イベントデータのバッチ
        Key: flow_gt, Type: torch.Tensor, Shape: torch.Size([Batch, 2, 480, 640]) => オプティカルフローデータのバッチ
        Key: flow_gt_valid_mask, Type: torch.Tensor, Shape: torch.Size([Batch, 1, 480, 640]) => オプティカルフローデータのvalid. ベースラインでは使わない
    
    test data:
        Type of batch: Dict
        Key: seq_name, Type: list
        Key: event_volume, Type: torch.Tensor, Shape: torch.Size([Batch, 4, 480, 640]) => イベントデータのバッチ
    '''
    # ------------------
    #       Model
    # ------------------
    model = EVFlowNet(args.train).to(device)

    # ------------------
    #   optimizer
    # ------------------
    optimizer = torch.optim.Adam(model.parameters(), lr=args.train.initial_learning_rate, weight_decay=args.train.weight_decay)
    # ------------------
    #   scheduler
    # ------------------
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)
    # ------------------
    #   Start training
    # ------------------
    best_loss = float('inf')
    model.train()
    for epoch in range(args.train.epochs):
        total_multi_scale_loss = 0 
        total_original_loss = 0
        print("on epoch: {}".format(epoch+1))
        for i, batch in enumerate(tqdm(train_data)):
            batch: Dict[str, Any]
            event_image = batch["event_volume"].to(device) # [B, 8, 480, 640]
            ground_truth_flow = batch["flow_gt"].to(device) # [B, 2, 480, 640]
            flow_dict = model(event_image)
            # Multi-scale loss calculation
            multi_scale_loss_value: torch.Tensor = multi_scale_loss(flow_dict, ground_truth_flow)
            # Original single-scale loss calculation (using the finest scale flow)
            original_loss_value: torch.Tensor = compute_epe_error(flow_dict['flow3'], ground_truth_flow)
            # Use multi-scale loss for optimization
            loss = multi_scale_loss_value
            print(f"batch {i} - Multi-scale loss: {multi_scale_loss_value.item():.4f}, Original loss: {original_loss_value.item():.4f}")
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # scheduler.step()

            total_multi_scale_loss += multi_scale_loss_value.item()
            total_original_loss += original_loss_value.item()
        
        avg_multi_scale_loss = total_multi_scale_loss / len(train_data)
        avg_original_loss = total_original_loss / len(train_data)
        print(f'Epoch {epoch+1}, Avg Multi-scale Loss: {avg_multi_scale_loss:.4f}, Avg Original Loss: {avg_original_loss:.4f}')

        scheduler.step(avg_multi_scale_loss)
        # Save the best model
        if avg_multi_scale_loss < best_loss:
            best_loss = avg_multi_scale_loss
            current_time = time.strftime("%Y%m%d%H%M%S")
            best_model_path = f"checkpoints/best_model_{current_time}.pth"
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model saved to {best_model_path}")


    
    current_time = time.strftime("%Y%m%d%H%M%S")
    model_path = f"./checkpoints/model_{current_time}.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    # ------------------
    #   Start predicting
    # ------------------
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    flow: torch.Tensor = torch.tensor([]).to(device)
    with torch.no_grad():
        print("start test")
        for batch in tqdm(test_data):
            batch: Dict[str, Any]
            event_image = batch["event_volume"].to(device)
            batch_flow = model(event_image) # [1, 2, 480, 640]
            flow = torch.cat((flow, batch_flow), dim=0)  # [N, 2, 480, 640]
        print("test done")
    # ------------------
    #  save submission
    # ------------------
    file_name = "submission"
    save_optical_flow_to_npy(flow, file_name)

if __name__ == "__main__":
    main()

# import torch
# import hydra
# from omegaconf import DictConfig
# from torch.utils.data import DataLoader
# import numpy as np
# from src.models.evflownet import EVFlowNet
# from src.datasets import DatasetProvider
# from enum import Enum, auto
# from src.datasets import train_collate
# from tqdm import tqdm
# from pathlib import Path
# from typing import Dict, Any

# class RepresentationType(Enum):
#     VOXEL = auto()

# def save_optical_flow_to_npy(flow: torch.Tensor, file_name: str):
#     np.save(f"{file_name}.npy", flow.cpu().numpy())

# @hydra.main(version_base=None, config_path="configs", config_name="base")
# def main(args: DictConfig):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     # データセットの設定
#     loader = DatasetProvider(
#         dataset_path=Path(args.dataset_path),
#         representation_type=RepresentationType.VOXEL,
#         delta_t_ms=100,
#         num_bins=4,
#         num_frames=args.num_frames
#     )
#     test_set = loader.get_test_dataset()
#     test_data = DataLoader(test_set,
#                            batch_size=args.data_loader.test.batch_size,
#                            shuffle=args.data_loader.test.shuffle,
#                            collate_fn=train_collate,
#                            drop_last=False)

#     # モデルの設定
#     model = EVFlowNet(args.train).to(device)

#     # 保存された最良のモデルをロード
#     best_model_path = args.best_model_path
#     model.load_state_dict(torch.load(best_model_path, map_location=device))
#     model.eval()

#     # 推論
#     flow = torch.tensor([]).to(device)
#     with torch.no_grad():
#         print("推論開始")
#         for batch in tqdm(test_data):
#             batch: Dict[str, Any]
#             event_image = batch["event_volume"].to(device)
#             batch_flow = model(event_image)['flow3']  # 最も細かいスケールのフローを使用
#             flow = torch.cat((flow, batch_flow), dim=0)
#         print("推論完了")

#     # 結果の保存
#     file_name = "submission"
#     save_optical_flow_to_npy(flow, file_name)
#     print(f"推論結果を {file_name}.npy として保存しました")

# if __name__ == "__main__":
#     main()
