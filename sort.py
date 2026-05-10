class AlphaBetaFilter:
    def __init__(self, alpha, beta):
        self.alpha = alpha  # 位置增益：值越大越跟随原始值，越小越平滑但有延迟
        self.beta = beta    # 速度增益：同上
        self.x_filt = None
        self.v_filt = 0.0

    def update(self, x_raw, dt):
        if self.x_filt is None:
            self.x_filt = x_raw
            return self.x_filt, 0.0
        
        if dt <= 0: return self.x_filt, self.v_filt

        # 预测当前位置
        x_pred = self.x_filt + self.v_filt * dt
        # 计算残差 (检测值 - 预测值)
        res = x_raw - x_pred
        # 更新状态
        self.x_filt = x_pred + self.alpha * res
        self.v_filt = self.v_filt + (self.beta / dt) * res
        
        return self.x_filt, self.v_filt
    
    def reset(self):
        """重置滤波器状态"""
        self.x_filt = None
        self.v_filt = 0.0

    
