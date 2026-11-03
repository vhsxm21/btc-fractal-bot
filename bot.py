import yfinance as yf
import pandas as pd
import numpy as np
import requests
import os
from datetime import datetime

# ------------------------------------------------------------------------------
# 1. 텔레그램 설정 및 메시지 발송 함수
# ------------------------------------------------------------------------------
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

def send_telegram_message(message):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("Telegram 설정이 없어 콘솔에만 출력합니다:\n", message)
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "HTML"}
    try:
        requests.post(url, json=payload, timeout=10)
    except Exception as e:
        print(f"Telegram 전송 실패: {e}")

# ------------------------------------------------------------------------------
# 2. 데이터 수집 및 지표 계산 (바이낸스 API 대신 가장 안정적인 YF 사용)
# ------------------------------------------------------------------------------
# 최근 35일 1시간봉 데이터 다운로드
df = yf.download('BTC-USD', period='35d', interval='1h', progress=False)
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)
df = df.dropna().copy()

df['H-L'] = df['High'] - df['Low']
df['H-PC'] = np.abs(df['High'] - df['Close'].shift(1))
df['L-PC'] = np.abs(df['Low'] - df['Close'].shift(1))
df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
df['ATR_14'] = df['TR'].rolling(window=14).mean()

# 일봉 20MA (미래참조 방지)
df_1d = df.resample('1d').agg({'Close': 'last'}).dropna()
df_1d['MA_20'] = df_1d['Close'].shift(1).rolling(window=20).mean()
df['Date_1D'] = df.index.floor('D')
df = df.join(df_1d[['MA_20']], on='Date_1D')

# ------------------------------------------------------------------------------
# 3. N=3 프랙탈 저점 탐색
# ------------------------------------------------------------------------------
N_VAL = 3
lows = df['Low'].values
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
    print("충분한 프랙탈 저점이 형성되지 않았습니다.")
    exit()

# ------------------------------------------------------------------------------
# 4. 실시간 감시 로직 1: L1 접근 경고 (MTF 15분봉 확인용 알림)
# ------------------------------------------------------------------------------
# 가장 최근에 확정된 저점 하나를 L1으로 간주
last_l1_idx, last_l1_date, l1_price = fractal_lows[-1]
current_idx = len(df) - 1
current_close = float(df['Close'].iloc[current_idx])
current_low = float(df['Low'].iloc[current_idx])

# L1 형성 후 아직 L2가 확정되지 않은 상태에서, 현재 가격이 L1 가격의 +1% 이내로 떨어졌을 때
if (current_idx - last_l1_idx) > N_VAL:
    margin_pct = (current_low - l1_price) / l1_price * 100
    if 0 < margin_pct <= 1.0:
        msg = (
            f"⚠️ <b>[L2 후보군 접근 알림]</b>\n\n"
            f"현재 가격이 이전 L1 바닥에 근접했습니다.\n"
            f"지금 트레이딩뷰를 켜고 <b>15분봉</b> 차트에서 미니 쌍바닥이나 장악형 양봉이 나오는지 확인하세요!\n\n"
            f"• L1 바닥가: ${l1_price:,.2f}\n"
            f"• 현재 최저가: ${current_low:,.2f} (+{margin_pct:.2f}%)"
        )
        send_telegram_message(msg)
        exit() # L1 경고를 보냈으면 종료

# ------------------------------------------------------------------------------
# 5. 실시간 감시 로직 2: L2 확정 타점 발송 (ATR 경고 플래그 포함)
# ------------------------------------------------------------------------------
# 가장 최근의 2개 저점이 L1, L2 조건(L2 > L1)을 방금 만족했는지 확인
l1_idx, _, l1_price = fractal_lows[-2]
l2_idx, l2_date, l2_price = fractal_lows[-1]

# 방금 전(최근 1~2시간 내)에 L2 저점(우측 3캔들 완료)이 확정된 상황인지 체크
if current_idx - (l2_idx + N_VAL) <= 1: 
    ma_20 = df['MA_20'].iloc[current_idx]
    
    # 20MA 역추세 필터 통과 & L2 > L1 통과
    if not pd.isna(ma_20) and current_close >= float(ma_20) and l2_price > l1_price:
        atr = float(df['ATR_14'].iloc[current_idx])
        sl = l2_price - (1.5 * atr)
        risk_dist = current_close - sl
        sl_pct = (risk_dist / current_close) * 100
        tp = current_close + (risk_dist * 1.5)
        
        # [핵심] 리스크 필터: 손절폭이 평소 변동성의 2.5배(약 4.5% 이상)를 초과하면 경고 플래그 추가
        warning_flag = ""
        if sl_pct >= 4.5:
            warning_flag = f"🚨 <b>[경고: 고변동성 리스크]</b> 손절폭({sl_pct:.2f}%)이 비정상적으로 큽니다. 진입 비중을 반으로 줄이거나 15분봉 눌림목을 기다리세요!\n\n"
            
        msg = (
            f"🎯 <b>[BTC 1H 스윙 매수 타점 포착]</b>\n\n"
            f"{warning_flag}"
            f"• 진입가(현재): ${current_close:,.2f}\n"
            f"• 손절가(SL): ${sl:,.2f} (-{sl_pct:.2f}%)\n"
            f"• 목표가(TP): ${tp:,.2f} (+{sl_pct * 1.5:.2f}%)\n\n"
            f"• L1 바닥: ${l1_price:,.2f}\n"
            f"• L2 바닥: ${l2_price:,.2f}"
        )
        send_telegram_message(msg)
