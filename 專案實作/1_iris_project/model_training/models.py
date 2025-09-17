"""
PyTorch 模型定義
包含用於 Iris 分類的神經網路模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class IrisClassifier(nn.Module):
    """
    Iris 鳶尾花分類的神經網路模型
    
    架構：
    - 輸入層：4 個特徵（萼片長度、萼片寬度、花瓣長度、花瓣寬度）
    - 隱藏層1：8 個神經元 + ReLU 激活 + Dropout
    - 隱藏層2：8 個神經元 + ReLU 激活 + Dropout  
    - 輸出層：3 個類別（山鳶尾、變色鳶尾、維吉尼亞鳶尾）
    """
    
    def __init__(self, input_size=4, hidden_size=8, num_classes=3, dropout_rate=0.2):
        super(IrisClassifier, self).__init__()
        
        # 定義網路層
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)
        
        # Dropout 層用於防止過擬合
        self.dropout = nn.Dropout(dropout_rate)
        
        # 初始化權重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化網路權重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        前向傳播
        
        Args:
            x: 輸入張量，形狀為 (batch_size, 4)
            
        Returns:
            輸出張量，形狀為 (batch_size, 3)
        """
        # 第一層：線性變換 + ReLU + Dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        
        # 第二層：線性變換 + ReLU + Dropout
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        
        # 輸出層：線性變換（不使用激活函數，因為使用 CrossEntropyLoss）
        x = self.fc3(x)
        
        return x
    
    def predict_proba(self, x):
        """
        預測機率
        
        Args:
            x: 輸入張量
            
        Returns:
            機率張量
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probabilities = F.softmax(logits, dim=1)
        return probabilities
    
    def predict(self, x):
        """
        預測類別
        
        Args:
            x: 輸入張量
            
        Returns:
            預測類別
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            _, predicted = torch.max(logits, 1)
        return predicted

class SimpleIrisClassifier(nn.Module):
    """
    簡化版的 Iris 分類器
    用於教學和快速原型開發
    """
    
    def __init__(self, input_size=4, num_classes=3):
        super(SimpleIrisClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 6)
        self.fc2 = nn.Linear(6, num_classes)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def create_model(model_type='complex', **kwargs):
    """
    創建指定類型的模型
    
    Args:
        model_type: 模型類型 ('complex' 或 'simple')
        **kwargs: 模型參數
        
    Returns:
        模型實例
    """
    if model_type == 'complex':
        return IrisClassifier(**kwargs)
    elif model_type == 'simple':
        return SimpleIrisClassifier(**kwargs)
    else:
        raise ValueError(f"不支援的模型類型: {model_type}")

def count_parameters(model):
    """
    計算模型參數數量
    
    Args:
        model: PyTorch 模型
        
    Returns:
        參數總數
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    # 測試模型
    model = IrisClassifier()
    print(f"模型參數數量: {count_parameters(model)}")
    
    # 測試前向傳播
    x = torch.randn(1, 4)  # 批次大小為1，4個特徵
    output = model(x)
    print(f"輸入形狀: {x.shape}")
    print(f"輸出形狀: {output.shape}")
    print(f"輸出值: {output}")
