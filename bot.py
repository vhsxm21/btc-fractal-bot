import yfinance as yf
import pandas as pd
import numpy as np
import requests
import os
import warnings
warnings.filterwarnings("ignore")

# ------------------------------------------------------------------------------
# 1. 텔레그램 설정 
# ------------------------------------------------------------------------------
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

def send_telegram_message(message):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("Telegram 설정이 없어 콘솔에 출력:\n", message)
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "HTML"}
    try:
        requests.post(url, json=payload, timeout=10)
    except Exception as e:
        print(f"Telegram 전송 실패: {e}")

# ------------------------------------------------------------------------------
# 2. 데이터 수집 및 지표 계산
# ------------------------------------------------------------------------------
df = yf.download('BTC-USD', period='60d', interval='1h', progress=False)
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)
df = df.dropna().copy()

df['H-L'] = df['High'] - df['Low']
df['H-PC'] = np.abs(df['High'] - df['Close'].shift(1))
df['L-PC'] = np.abs(df['Low'] - df['Close'].shift(1))
df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
df['ATR_14'] = df['TR'].rolling(window=14).mean()

# 1D 20MA 환산 (1H 480MA) 및 3일 전(72H) 기울기 비교
df['MA_1D_20'] = df['Close'].shift(1).rolling(window=480).mean()
df['MA_Slope_3D'] = df['MA_1D_20'] - df['MA_1D_20'].shift(72) 

N_VAL = 3
MAX_L1_L2_BARS = 168 # 7일
RR_RATIO = 1.5
ATR_MULTIPLIER = 1.5

# ------------------------------------------------------------------------------
# 3. 프랙탈 바닥 탐색 엔진
# ------------------------------------------------------------------------------
lows = df['Low'].values
vols = df['Volume'].values
fractal_lows = []

for i in range(N_VAL, len(df) - N_VAL):
    is_low = True
    for offset in range(1, N_VAL + 1):
        if lows[i] >= lows[i - offset] or lows[i] >= lows[i + offset]:
            is_low = False
            break
    if is_low:
        fractal_lows.append((i, df.index[i], lows[i]))

if len(fractal_lows) < 2:
    exit()

# ------------------------------------------------------------------------------
# 4. 실시간 타점 평가 및 텔레그램 발송 (D급 포함 모든 쌍바닥 감지)
# ------------------------------------------------------------------------------
l1_idx, _, l1_price = fractal_lows[-2]
l2_idx, l2_date, l2_price = fractal_lows[-1]
current_idx = len(df) - 1

# L2 확정(우측 3캔들 완료) 직후 1~2시간 이내일 때만 알림 작동
if current_idx - (l2_idx + N_VAL) <= 1: 
    current_close = float(df['Close'].iloc[current_idx])
    ma_20 = df['MA_1D_20'].iloc[current_idx]
    
    # 1. 쌍바닥(L2 > L1) 기본 조건 검사
    if l2_price > l1_price and (l2_idx - l1_idx) <= MAX_L1_L2_BARS:
        atr = float(df['ATR_14'].iloc[current_idx])
        sl = l2_price - (ATR_MULTIPLIER * atr)
        risk_dist = current_close - sl
        
        if risk_dist > 0:
            sl_pct = (risk_dist / current_close) * 100
            tp = current_close + (risk_dist * RR_RATIO)
            
            # [스코어링 팩터]
            ma_slope = float(df['MA_Slope_3D'].iloc[current_idx])
            down_vol = np.mean(vols[l1_idx : l2_idx+1]) if l2_idx > l1_idx else 1
            up_vol = np.mean(vols[l2_idx+1 : current_idx+1])
            
            score = 0
            tags = []
            grade = ""
            action_tip = ""
            
            # 2. 대추세(20MA) 1차 필터링 -> D급(역배열) 분류
            if pd.isna(ma_20) or current_close <= float(ma_20):
                grade = 'D급'
                tags.append("⛔ 20MA 하단 (역배열 하락장)")
                action_tip = "🚫 <b>진입 절대 금지 (데드캣 트랩 확률 매우 높음)</b>"
            
            # 3. 20MA 상단(정배열)일 경우 -> A/B/C급 상세 스코어링
            else:
                if ma_slope > 0:
                    score += 1
                    tags.append("✅ 20MA 우상향 (추세 좋음)")
                else:
                    tags.append("⚠️ 20MA 꺾임 (역추세 주의)")
                    
                if up_vol > down_vol:
                    score += 1
                    tags.append("✅ 수급 폭발 (건강한 반등)")
                else:
                    tags.append("⚠️ 수급 마름 (가짜 반등 주의)")
                
                grade = 'A급' if score == 2 else 'B급' if score == 1 else 'C급'
                
                if grade == 'A급': action_tip = "💡 <b>기본 비중(100%) 적극 진입 추천</b>"
                elif grade == 'B급': action_tip = "💡 <b>비중 50% 축소 진입 (보수적 접근)</b>"
                else: action_tip = "🚫 <b>설거지(세력 이탈) 패턴. 매수 패스(Pass) 권장</b>"

            # 4. 알림 메시지 조합 및 발송
            warning_flag = ""
            if sl_pct >= 4.5:
                warning_flag = f"🚨 <b>[경고: 고변동성]</b> 손절폭({sl_pct:.2f}%) 과다!\n\n"
                
            msg = (
                f"🎯 <b>[BTC 1H 프랙탈 스윙 포착]</b>\n\n"
                f"{warning_flag}"
                f"🏅 <b>타점 등급: {grade}</b>\n"
                f"• {tags[0]}\n"
                f"{('• ' + tags[1] + '\n') if len(tags) > 1 else ''}\n"
                f"• 진입가(현재): ${current_close:,.2f}\n"
                f"• 손절가(SL): ${sl:,.2f} (-{sl_pct:.2f}%)\n"
                f"• 목표가(TP): ${tp:,.2f} (+{sl_pct * RR_RATIO:.2f}%)\n\n"
                f"• L1 바닥: ${l1_price:,.2f}\n"
                f"• L2 바닥: ${l2_price:,.2f}\n\n"
                f"{action_tip}"
            )
            send_telegram_message(msg)
