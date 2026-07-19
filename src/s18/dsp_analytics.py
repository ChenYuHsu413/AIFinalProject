#!/usr/bin/env python3
import numpy as np

class TimeDomainFeatureExtractor:
    """
    時域特徵提取器，提供高階無量綱時域指標，用於診斷早期微弱損傷。
    """
    @staticmethod
    def kurtosis(signal):
        """
        計算峭度 (Kurtosis)。正常噪訊接近 3.0，早期故障 (LO) 時會顯著上升。
        """
        signal = np.asarray(signal)
        n = len(signal)
        if n < 4:
            return 3.0
        mean = np.mean(signal)
        var = np.var(signal, ddof=0)
        if var < 1e-15:
            return 3.0
        kurt = np.sum((signal - mean) ** 4) / n / (var ** 2)
        return float(kurt)

    @staticmethod
    def crest_factor(signal):
        """
        計算波峰因數 (Crest Factor)。反映峰值與有效值的比例。
        """
        signal = np.asarray(signal)
        rms = np.sqrt(np.mean(signal ** 2))
        if rms < 1e-15:
            return 1.0
        peak = np.max(np.abs(signal))
        return float(peak / rms)

    @staticmethod
    def margin_factor(signal):
        """
        計算裕度因數 (Margin Factor)。
        """
        signal = np.asarray(signal)
        peak = np.max(np.abs(signal))
        mean_sqrt = (np.mean(np.sqrt(np.abs(signal)))) ** 2
        if mean_sqrt < 1e-15:
            return 1.0
        return float(peak / mean_sqrt)

class KalmanFilter2D:
    """
    2D 卡爾曼估測器
    狀態向量: x = [位置, 速度]^T
    量測值: z = 實際編碼器位置
    """
    def __init__(self, dt=0.001, q_cov=0.01, r_cov=1.0):
        self.dt = dt
        # 狀態轉移矩陣 F
        self.F = np.array([
            [1.0, self.dt],
            [0.0, 1.0]
        ])
        # 量測矩陣 H
        self.H = np.array([[1.0, 0.0]])
        
        # 雜訊協方差矩陣
        self.Q = np.array([
            [q_cov * (dt**3)/3.0, q_cov * (dt**2)/2.0],
            [q_cov * (dt**2)/2.0, q_cov * dt]
        ])
        self.R = np.array([[r_cov]])
        
        # 狀態值與協方差初始設定
        self.x = np.array([[0.0], [0.0]])
        self.P = np.array([
            [1.0, 0.0],
            [0.0, 1.0]
        ])
        
    def reset(self, initial_position=0.0, initial_velocity=0.0):
        self.x = np.array([[initial_position], [initial_velocity]])
        self.P = np.array([
            [1.0, 0.0],
            [0.0, 1.0]
        ])
        
    def predict(self):
        # 預測狀態: x_k|k-1 = F * x_k-1
        self.x = np.dot(self.F, self.x)
        # 預測協方差: P_k|k-1 = F * P_k-1 * F^T + Q
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        return self.x
        
    def update(self, measurement_pos):
        # 量測更新: K = P * H^T * (H * P * H^T + R)^-1
        z = np.array([[measurement_pos]])
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        
        # x_k|k = x_k|k-1 + K * (z - H * x_k|k-1)
        residual = z - np.dot(self.H, self.x)
        self.x = self.x + np.dot(K, residual)
        
        # P_k|k = (I - K * H) * P_k|k-1
        I = np.eye(2)
        self.P = np.dot(I - np.dot(K, self.H), self.P)
        return self.x

class BodeResponseAnalyzer:
    """
    波德圖頻譜響應與共振峰分析器
    """
    def __init__(self, sampling_rate_hz=1000):
        self.fs = sampling_rate_hz
        
    def analyze(self, cmd_pos, act_pos):
        """
        計算幅頻與相頻響應，並定位共振峰與系統相位/增益邊界
        """
        n = len(cmd_pos)
        if n < 64:
            return {"status": "error", "message": "Signal length too short"}
            
        # 頻域轉換 FFT
        fft_cmd = np.fft.rfft(cmd_pos)
        fft_act = np.fft.rfft(act_pos)
        freqs = np.fft.rfftfreq(n, d=1.0/self.fs)
        
        # 計算傳遞函數 G(f) = Act(f) / Cmd(f)
        # 避免除以 0
        eps = 1e-8
        g_f = fft_act / (fft_cmd + eps)
        
        # 幅度響應 (dB)
        mag_db = 20 * np.log10(np.abs(g_f) + eps)
        # 相位響應 (degrees)
        phase_deg = np.angle(g_f) * 180.0 / np.pi
        
        # Nyquist 實部與虛部
        real_gf = np.real(g_f)
        imag_gf = np.imag(g_f)
        
        # 定位高頻共振峰 (>100Hz)
        idx_high = (freqs >= 100.0)
        peak_freq = 0.0
        peak_prominence = 0.0
        
        if np.any(idx_high):
            high_freqs = freqs[idx_high]
            high_mag = mag_db[idx_high]
            
            # 簡易計算峰值與局部均值的突出度
            # 局部視窗均值
            mean_mag = np.mean(high_mag)
            peak_idx = np.argmax(high_mag)
            
            peak_freq = float(high_freqs[peak_idx])
            peak_prominence = float(high_mag[peak_idx] - mean_mag)
            
        # 計算增益邊界與相位邊界 (簡易估計)
        # 相位為 -180 度處的增益，增益為 0 dB 處的相位裕度
        phase_margin = 180.0
        gain_margin_db = 99.0
        
        # 尋找跨越 0dB 處的相位裕度
        idx_cross_gain = np.where(np.diff(np.sign(mag_db)))[0]
        if len(idx_cross_gain) > 0:
            phase_at_cross = phase_deg[idx_cross_gain[0]]
            phase_margin = float(phase_at_cross + 180.0)
            
        # 尋找跨越 -180度處的增益裕度
        idx_cross_phase = np.where(np.diff(np.sign(phase_deg + 180.0)))[0]
        if len(idx_cross_phase) > 0:
            gain_margin_db = float(-mag_db[idx_cross_phase[0]])
            
        # 計算共振能量占比 (100Hz - 400Hz 區帶能量與總能量比例)
        total_energy = np.sum(np.abs(fft_act) ** 2) + eps
        idx_band = (freqs >= 100.0) & (freqs <= 400.0)
        band_energy = np.sum(np.abs(fft_act[idx_band]) ** 2)
        sideband_energy_ratio = float(band_energy / total_energy)
            
        return {
            "status": "success",
            "frequencies": freqs.tolist(),
            "magnitude_db": mag_db.tolist(),
            "phase_deg": phase_deg.tolist(),
            "nyquist_real": real_gf.tolist(),
            "nyquist_imag": imag_gf.tolist(),
            "resonance_peak_freq_hz": peak_freq,
            "resonance_prominence_db": peak_prominence,
            "phase_margin_deg": phase_margin,
            "gain_margin_db": gain_margin_db,
            "sideband_resonance_energy_ratio": sideband_energy_ratio
        }

class ARIMAPredictor:
    """
    時間序列自迴歸溫升趨勢預測器 (簡化 ARIMA(1,1,0) 模型)
    方程式: (T_t - T_t-1) = phi * (T_t-1 - T_t-2) + error
    """
    def __init__(self):
        self.phi = 0.5 # 預設自迴歸係數
        
    def fit(self, temp_series):
        """
        利用最小平方法配適 phi 參數
        """
        if len(temp_series) < 5:
            return self.phi
            
        # 一階差分
        diff = np.diff(temp_series)
        X = diff[:-1]
        y = diff[1:]
        
        # phi = Cov(X,y) / Var(X)
        cov = np.cov(X, y)
        if cov.ndim == 2:
            self.phi = cov[0, 1] / (np.var(X) + 1e-8)
        else:
            self.phi = 0.5
            
        # 限幅防止不穩定
        self.phi = np.clip(self.phi, -0.9, 0.9)
        return self.phi
        
    def predict_future(self, temp_series, steps_ahead=30):
        """
        遞迴預估未來 steps_ahead 步數的溫度
        """
        if len(temp_series) < 3:
            return [temp_series[-1]] * steps_ahead
            
        predictions = []
        last_t = temp_series[-1]
        last_diff = temp_series[-1] - temp_series[-2]
        
        current_diff = last_diff
        current_t = last_t
        for _ in range(steps_ahead):
            # diff_k = phi * diff_k-1
            current_diff = self.phi * current_diff
            # T_k = T_k-1 + diff_k
            current_t = current_t + current_diff
            predictions.append(float(current_t))
            
        return predictions


class AdvancedMechanicalDiagnostics:
    """
    提供針對滾珠螺桿、導軌等機械結構之進階物理診斷特徵解算
    """
    @staticmethod
    def reversal_error(pos_demand, pos_actual, velocity):
        """
        計算反轉誤差 (Reversal Error)
        在速度反轉 (跨越零點) 的瞬間，計算指令位置與實際位置之差
        """
        pos_demand = np.asarray(pos_demand)
        pos_actual = np.asarray(pos_actual)
        velocity = np.asarray(velocity)
        
        # 尋找速度變換方向的索引
        # 符號變化 (velocity[i] * velocity[i-1] < 0)
        reversal_errors = []
        for i in range(1, len(velocity)):
            if velocity[i] * velocity[i-1] < 0:
                err = abs(pos_demand[i] - pos_actual[i])
                reversal_errors.append(err)
        
        if not reversal_errors:
            # 簡化退化公式
            return float(np.mean(np.abs(pos_demand - pos_actual)) * 0.1)
        return float(np.mean(reversal_errors))

    @staticmethod
    def dead_zone_width(pos_demand, pos_actual, velocity):
        """
        計算死區寬度 (Dead Zone Width)
        當指令速度發生反向時，實際位置保持不動而指令位置繼續移動的區間
        """
        pos_demand = np.asarray(pos_demand)
        pos_actual = np.asarray(pos_actual)
        velocity = np.asarray(velocity)
        
        dead_zones = []
        for i in range(1, len(velocity)):
            if velocity[i] * velocity[i-1] < 0:
                # 速度反向瞬間，指令移動但實際幾乎未動
                start_idx = i
                # 向後搜尋直到實際位置發生明顯變化
                for j in range(start_idx, min(start_idx + 20, len(pos_actual))):
                    if abs(pos_actual[j] - pos_actual[start_idx]) > 0.05: # 明顯移動
                        dz = abs(pos_demand[j] - pos_demand[start_idx])
                        dead_zones.append(dz)
                        break
        if not dead_zones:
            return 0.05 # 預設微小值
        return float(np.mean(dead_zones))

    @staticmethod
    def hysteresis_area(pos_demand, pos_actual):
        """
        計算滯後環面積 (Hysteresis Area)
        利用 Green 公式計算指令-實際位置閉環軌跡圍成的面積
        """
        x = np.asarray(pos_demand)
        y = np.asarray(pos_actual)
        if len(x) < 3:
            return 0.0
        # 閉合曲線：A = 0.5 * |sum(x_i * y_i+1 - x_i+1 * y_i)|
        area = 0.5 * np.abs(np.dot(x[:-1], y[1:]) - np.dot(x[1:], y[:-1]))
        return float(area / len(x)) # 平均每個採樣的面積

    @staticmethod
    def direction_dependent_following_error(pos_demand, pos_actual, velocity):
        """
        方向相關追隨誤差 (Direction-dependent Following Error)
        計算正向與反向運動狀態下的平均追隨誤差差值
        """
        pos_demand = np.asarray(pos_demand)
        pos_actual = np.asarray(pos_actual)
        velocity = np.asarray(velocity)
        fe = np.abs(pos_demand - pos_actual)
        
        pos_fe = fe[velocity > 10.0]
        neg_fe = fe[velocity < -10.0]
        
        mean_pos = np.mean(pos_fe) if len(pos_fe) > 0 else 0.0
        mean_neg = np.mean(neg_fe) if len(neg_fe) > 0 else 0.0
        return float(abs(mean_pos - mean_neg))

    @staticmethod
    def force_displacement_slope(pos_actual, following_error, stiffness):
        """
        剛性斜率估算 (Force-Displacement Slope)
        """
        fe = np.asarray(following_error)
        pos = np.asarray(pos_actual)
        # 用追隨誤差做自變量，剛性(力)做因變量估算斜率
        if len(fe) < 3 or np.var(fe) < 1e-8:
            return float(np.mean(stiffness))
        slope = np.polyfit(fe, pos, 1)[0]
        return float(abs(slope))

    @staticmethod
    def compliance_std(pos_actual, following_error):
        """
        柔度標準差 (Compliance Std)
        """
        fe = np.asarray(following_error)
        pos = np.asarray(pos_actual)
        # 柔度估算 = 追隨誤差 / 位置變化
        eps = 1e-8
        compliance = fe / (np.abs(pos) + eps)
        return float(np.std(compliance))

    @staticmethod
    def elastic_region_width(pos_actual, following_error):
        """
        彈性變形區寬度
        """
        fe = np.asarray(following_error)
        # 基於追隨誤差與位置的比例範圍
        return float(np.percentile(fe, 95) - np.percentile(fe, 5))

    @staticmethod
    def stribeck_friction_parameters(velocity, torque):
        """
        擬合 Stribeck 摩擦模型參數
        返回 [coulomb_friction, viscous_friction, stribeck_coeff]
        """
        v = np.asarray(velocity)
        t = np.asarray(torque)
        
        # 估算庫倫摩擦 (低速時平均扭矩)
        low_v = t[np.abs(v) < 100.0]
        coulomb = np.mean(np.abs(low_v)) if len(low_v) > 0 else 0.3
        
        # 估算粘性摩擦 (高速時扭矩與速度斜率)
        high_v_idx = np.abs(v) > 500.0
        if np.sum(high_v_idx) > 5:
            viscous = np.polyfit(np.abs(v[high_v_idx]), np.abs(t[high_v_idx]), 1)[0]
        else:
            viscous = 0.001
            
        stribeck = coulomb * 1.25 # 經驗係數
        return float(coulomb), float(viscous), float(stribeck)


class AdvancedElectricalDiagnostics:
    """
    提供針對電機磁路退磁與電磁故障之進階物理特徵解算
    """
    @staticmethod
    def fft_harmonic_amplitudes(i_3p_a, i_3p_b, i_3p_c):
        """
        計算三相電流 FFT 諧波幅值
        """
        ia = np.asarray(i_3p_a)
        # 計算 FFT
        fft_a = np.fft.rfft(ia)
        mags = np.abs(fft_a)
        # 返回前三個諧波成分的均值
        if len(mags) > 5:
            return float(np.mean(mags[1:4]))
        return 0.01

    @staticmethod
    def current_total_harmonic_distortion(i_3p):
        """
        計算電流總諧波失真 (THD)
        """
        i = np.asarray(i_3p)
        fft_i = np.fft.rfft(i)
        mags = np.abs(fft_i)
        if len(mags) < 3 or mags[1] < 1e-8:
            return 0.05 # 預設 THD 5%
        # THD = sqrt(sum(V_n^2) for n>=2) / V_1
        thd = np.sqrt(np.sum(mags[2:]**2)) / mags[1]
        return float(thd)

    @staticmethod
    def phase_imbalance_index(i_3p_a, i_3p_b, i_3p_c):
        """
        計算三相電流不平衡度 (Phase Imbalance Index)
        """
        ia = np.mean(np.abs(i_3p_a))
        ib = np.mean(np.abs(i_3p_b))
        ic = np.mean(np.abs(i_3p_c))
        mean_i = (ia + ib + ic) / 3.0
        if mean_i < 1e-8:
            return 0.0
        max_dev = max(abs(ia - mean_i), abs(ib - mean_i), abs(ic - mean_i))
        return float(max_dev / mean_i)

    @staticmethod
    def id_iq_trajectory_area(id_arr, iq_arr):
        """
        計算 DQ 軸電流軌跡的包絡面積
        """
        x = np.asarray(id_arr)
        y = np.asarray(iq_arr)
        if len(x) < 3:
            return 0.0
        # 計算邊界圍成面積 (利用簡化散點凸包或協方差面積)
        cov = np.cov(x, y)
        if cov.ndim == 2:
            # 協方差橢圓面積 pi * sqrt(lambda1 * lambda2)
            eigvals = np.linalg.eigvals(cov)
            area = np.pi * np.sqrt(np.abs(eigvals[0] * eigvals[1]))
            return float(area)
        return 0.1

