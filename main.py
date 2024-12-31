import os
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import paddle.vision.transforms as T
from paddle.io import Dataset, DataLoader
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.model_selection import KFold

class ConvBlock(nn.Layer):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2D(in_channels, out_channels, kernel_size, 
                             stride, padding=kernel_size//2, bias_attr=False)
        self.bn = nn.BatchNorm2D(out_channels)
        
    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))

class FireDetectionModel(nn.Layer):
    def __init__(self):
        super(FireDetectionModel, self).__init__()
        
        self.conv1 = ConvBlock(3, 64)
        self.conv2 = ConvBlock(64, 128)
        self.conv3 = ConvBlock(128, 256)
        self.conv4 = ConvBlock(256, 512)
        self.conv5 = ConvBlock(512, 1024)
        
        self.conv6 = ConvBlock(1024, 512)
        self.conv7 = nn.Conv2D(512, 5, 1)
        
        self.pool = nn.MaxPool2D(2, 2)
        
    def forward(self, x):
        x = self.pool(self.conv1(x))
        x = self.pool(self.conv2(x))
        x = self.pool(self.conv3(x))
        x = self.pool(self.conv4(x))
        x = self.pool(self.conv5(x))
        x = self.conv6(x)
        x = self.conv7(x)
        return x

class FireDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        
        label_path = os.path.join(self.label_dir, img_name.replace('.jpg', '.txt'))
        with open(label_path, 'r') as f:
            label_data = np.array([list(map(float, line.strip().split())) for line in f])
        
        if self.transform:
            image = self.transform(image)
        
        label = np.zeros((5, 13, 13), dtype=np.float32)
        
        for box in label_data:
            grid_x = int(box[1] * 13)
            grid_y = int(box[2] * 13)
            
            grid_x = min(max(grid_x, 0), 12)
            grid_y = min(max(grid_y, 0), 12)
            
            label[0, grid_y, grid_x] = 1
            label[1:5, grid_y, grid_x] = box[1:]
        
        return image, paddle.to_tensor(label)

def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    return intersection / (box1_area + box2_area - intersection + 1e-6)

def compute_metrics(tp, fp, fn):
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
    return precision, recall, f1

def decode_predictions(predictions, conf_threshold=0.1):
    if isinstance(predictions, paddle.Tensor):
        predictions = predictions.numpy()
    
    predictions = np.copy(predictions)
    predictions[:, 0] = 1 / (1 + np.exp(-predictions[:, 0]))
    predictions[:, 1:5] = 1 / (1 + np.exp(-predictions[:, 1:5]))
    
    batch_size = predictions.shape[0]
    boxes_list = []
    
    for b in range(batch_size):
        boxes = []
        pred = predictions[b]
        
        for i in range(13):
            for j in range(13):
                confidence = pred[0, i, j]
                
                if confidence > conf_threshold:
                    x = pred[1, i, j]
                    y = pred[2, i, j]
                    w = pred[3, i, j]
                    h = pred[4, i, j]
                    
                    x1 = max(0, min(1, x - w/2))
                    y1 = max(0, min(1, y - h/2))
                    x2 = max(0, min(1, x + w/2))
                    y2 = max(0, min(1, y + h/2))
                    
                    boxes.append([x1, y1, x2, y2])
        
        boxes_list.append(np.array(boxes) if boxes else np.zeros((0, 4)))
    
    return boxes_list

def evaluate_predictions(pred_boxes_list, true_boxes_list, iou_threshold=0.5):
    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    for pred_boxes, true_boxes in zip(pred_boxes_list, true_boxes_list):
        if len(true_boxes) == 0:
            total_fp += len(pred_boxes)
            continue
            
        if len(pred_boxes) == 0:
            total_fn += len(true_boxes)
            continue
        
        ious = np.zeros((len(pred_boxes), len(true_boxes)))
        for i, pred_box in enumerate(pred_boxes):
            for j, true_box in enumerate(true_boxes):
                ious[i, j] = compute_iou(pred_box, true_box)
        
        matched_gt = set()
        for pred_idx in range(len(pred_boxes)):
            best_iou = 0
            best_gt_idx = -1
            
            for gt_idx in range(len(true_boxes)):
                if gt_idx not in matched_gt:
                    iou = ious[pred_idx, gt_idx]
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx
            
            if best_iou > iou_threshold:
                total_tp += 1
                matched_gt.add(best_gt_idx)
            else:
                total_fp += 1
        
        total_fn += len(true_boxes) - len(matched_gt)
    
    return total_tp, total_fp, total_fn

def compute_loss(pred, target):
    pred_conf = F.sigmoid(pred[:, 0:1])
    pred_boxes = F.sigmoid(pred[:, 1:5])
    
    pos_weight = min((target[:, 0:1] == 0).sum() / (target[:, 0:1] > 0).sum(), 100)
    
    conf_loss = F.binary_cross_entropy(
        pred_conf, 
        target[:, 0:1],
        weight=paddle.where(target[:, 0:1] > 0, 
                          paddle.full_like(target[:, 0:1], pos_weight),
                          paddle.ones_like(target[:, 0:1]))
    )
    
    mask = target[:, 0:1] > 0.5
    
    if paddle.sum(mask) > 0:
        box_loss = F.mse_loss(
            pred_boxes[mask.expand_as(pred_boxes)],
            target[:, 1:5][mask.expand_as(target[:, 1:5])]
        )
        box_loss = box_loss * 5.0
    else:
        box_loss = paddle.to_tensor(0.)
    
    return conf_loss + box_loss

def train_and_evaluate(model, train_loader, val_loader, epochs, learning_rate=0.001):
    optimizer = paddle.optimizer.Adam(parameters=model.parameters(), learning_rate=learning_rate)
    scheduler = paddle.optimizer.lr.ReduceOnPlateau(
        learning_rate=learning_rate,
        factor=0.5,
        patience=3,
        verbose=False
    )
    
    history = {
        'loss': [], 'precision': [], 'recall': [], 'f1': []
    }
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        tp = fp = fn = 0
        
        for images, labels in train_loader:
            outputs = model(images)
            loss = compute_loss(outputs, labels)
            
            loss.backward()
            optimizer.step()
            optimizer.clear_grad()
            
            total_loss += loss.item()
            
            pred_boxes_list = decode_predictions(outputs)
            true_boxes_list = decode_predictions(labels)
            batch_tp, batch_fp, batch_fn = evaluate_predictions(pred_boxes_list, true_boxes_list)
            tp += batch_tp
            fp += batch_fp
            fn += batch_fn
        
        precision, recall, f1 = compute_metrics(tp, fp, fn)
        avg_loss = total_loss / len(train_loader)
        
        scheduler.step(avg_loss)
        
        history['loss'].append(avg_loss)
        history['precision'].append(precision)
        history['recall'].append(recall)
        history['f1'].append(f1)
        
        print(f'Epoch [{epoch+1}/{epochs}] Loss: {avg_loss:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}')
    
    return history

def plot_metrics(history, save_path):
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(history['loss'], label='Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.plot(history['precision'], label='Precision')
    plt.plot(history['recall'], label='Recall')
    plt.plot(history['f1'], label='F1')
    plt.title('Metrics')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

class SubsetRandomSampler(paddle.io.Sampler):
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return iter(np.random.permutation(self.indices))

    def __len__(self):
        return len(self.indices)

class SubsetDataset(Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices
        
    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]
    
    def __len__(self):
        return len(self.indices)

def k_fold_cross_validation(dataset, k=5, batch_size=16, epochs=20, learning_rate=0.001):
    from sklearn.model_selection import KFold
    
    indices = np.arange(len(dataset))
    kfold = KFold(n_splits=k, shuffle=True)
    
    all_histories = []
    all_models = []
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(indices)):
        print(f'\nFold {fold + 1}/{k}')
        print('-' * 40)
        
        # 创建训练集和验证集的子集
        train_dataset = SubsetDataset(dataset, train_idx)
        val_dataset = SubsetDataset(dataset, val_idx)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False
        )
        
        model = FireDetectionModel()
        
        history = train_and_evaluate(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=epochs,
            learning_rate=learning_rate
        )
        
        model_save_path = f'fire_detection_model_fold{fold+1}.pdparams'
        paddle.save(model.state_dict(), model_save_path)
        
        all_histories.append(history)
        all_models.append(model)
        
        plot_save_path = f'training_metrics_fold{fold+1}.png'
        plot_metrics(history, plot_save_path)
    
    avg_metrics = {
        'loss': np.mean([h['loss'] for h in all_histories], axis=0),
        'precision': np.mean([h['precision'] for h in all_histories], axis=0),
        'recall': np.mean([h['recall'] for h in all_histories], axis=0),
        'f1': np.mean([h['f1'] for h in all_histories], axis=0)
    }
    
    plot_metrics(avg_metrics, 'training_metrics_average.png')
    
    print('\nAverage metrics across all folds:')
    print(f"Final Loss: {avg_metrics['loss'][-1]:.4f}")
    print(f"Final Precision: {avg_metrics['precision'][-1]:.4f}")
    print(f"Final Recall: {avg_metrics['recall'][-1]:.4f}")
    print(f"Final F1: {avg_metrics['f1'][-1]:.4f}")
    
    return all_histories, all_models

def main():
    transform = T.Compose([
        T.Resize((416, 416)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], 
                   std=[0.229, 0.224, 0.225])
    ])
    
    dataset = FireDataset('datasets/images', 'datasets/labels', transform=transform)
    
    # 执行5折交叉验证
    histories, models = k_fold_cross_validation(
        dataset=dataset,
        k=5,
        batch_size=16,
        epochs=20,
        learning_rate=0.001
    )
    
    print("\nTraining completed!")
    print("Models and metrics plots have been saved for each fold")
    print("Average metrics plot saved as: training_metrics_average.png")

if __name__ == "__main__":
    main()