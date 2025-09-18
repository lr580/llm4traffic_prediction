import numpy as np
import torch

class TrendCalc():
    """ Time-LLM's prompt stat information """
    
    @staticmethod
    def lags(x: np.ndarray, top_k: int = 5) -> list[float]:
        x_tensor = TrendCalc.to(x).view(1, 1, -1)
        x_fft = torch.fft.rfft(x_tensor, dim=-1)
        # 计算自相关：在频域中，信号与其共轭的乘积
        res = x_fft * torch.conj(x_fft)
        # 逆FFT回到时域，得到自相关序列
        corr = torch.fft.irfft(res, dim=-1)
        corr_1d = corr.squeeze()
        # 找到自相关性最强的top_k个滞后点
        _, lags = torch.topk(corr_1d, top_k, dim=-1)
        # 在分析的时间序列中，当前时刻的值与过去哪个时刻的值最相关
        return lags.tolist()
    
    @staticmethod
    def basicStat(x: np.ndarray):
        x_tensor = TrendCalc.to(x)
        return torch.min(x_tensor).item(), torch.max(x_tensor).item(), torch.median(x_tensor).item()
    
    @staticmethod
    def trend(x: np.ndarray) -> float:
        return TrendCalc.to(x).diff().sum().item()  # > 0
    
    @staticmethod
    def trendStat(x: np.ndarray):
        basic_stats = TrendCalc.basicStat(x)
        lags_result = TrendCalc.lags(x)
        trend_result = TrendCalc.trend(x)
        return (*basic_stats, lags_result, trend_result)
    
    @staticmethod
    def to(x: np.ndarray) -> torch.Tensor:
        ''' ndarray -> tensor '''
        return torch.tensor(x, dtype=torch.float32)

# class TrendCalc():
#     """ Time-LLM's prompt stat imformation """
#     def lags(self, x: np.ndarray, top_k:int=5)->list[float]:
#         x = self.to(x).view(1, 1, -1)
#         x_fft = torch.fft.rfft(x, dim=-1)
#         # 计算自相关：在频域中，信号与其共轭的乘积
#         res = x_fft * torch.conj(x_fft)
#         # 逆FFT回到时域，得到自相关序列
#         corr = torch.fft.irfft(res, dim=-1)
#         corr_1d = corr.squeeze()
#         # 找到自相关性最强的top_k个滞后点
#         _, lags = torch.topk(corr_1d, top_k, dim=-1)
#         # ​在分析的时间序列中，​当前时刻的值​与​过去哪个时刻的值​最相关
#         return lags.tolist()
        
#     def basicStat(self, x: np.ndarray):
#         x = self.to(x)
#         return torch.min(x).item(), torch.max(x).item(), torch.median(x).item()
    
#     def trend(self, x: np.ndarray) -> float:
#         return self.to(x).diff().sum().item() # > 0
    
#     def trendStat(self, x: np.ndarray):
#         return *self.basicStat(x), self.lags(x), self.trend(x)

#     def to(self, x: np.ndarray)->torch.Tensor:
#         ''' ndarray -> tensor '''
#         # if isinstance(x, np.ndarray):
#         return torch.tensor(x, dtype=torch.float32)