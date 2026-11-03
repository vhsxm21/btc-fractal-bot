import yfinance as yf
import pandas as pd
import numpy as np
import requests
import os

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
# 2. 데이터 수집 및 지표 계산
# ------------------------------------------------------------------------------
df = yf.download('BTC-USD', period='60d', interval='1h', progress=False)
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)
df = df.dropna().copy()

# ATR 계산
df['H-L'] = df['High'] - df['Low']
df['H-PC'] = np.abs(df['High'] - df['Close'].shift(1))
df['L-PC'] = np.abs(df['Low'] - df['Close'].shift(1))
df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
df['ATR_14'] = df['TR'].rolling(window=14).mean()

# 1일봉 20MA 및 3일 전 대비 기울기 계산
df_1d = df.resample('1d').agg({'Close': 'last'}).dropna()
df_1d['MA_20'] = df_1d['Close'].shift(1).rolling(window=20).mean()
df_1d['MA_20_3D_ago'] = df_1d['MA_20'].shift(3) # 3일 전 20MA

df['Date_1D'] = df.index.floor('D')
df = df.join(df_1d[['MA_20', 'MA_20_3D_ago']], on='Date_1D')

# ------------------------------------------------------------------------------
# 3. N=3 프랙탈 저점 탐색
# ------------------------------------------------------------------------------
N_VAL = 3
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
# 4. 실시간 감시 로직 1: L1 접근 경고 (MTF 15분봉 확인용)
# ------------------------------------------------------------------------------
last_l1_idx, last_l1_date, l1_price = fractal_lows[-1]
current_idx = len(df) - 1
current_close = float(df['Close'].iloc[current_idx])
current_low = float(df['Low'].iloc[current_idx])

if (current_idx - last_l1_idx) > N_VAL:
    margin_pct = (current_low - l1_price) / l1_price * 100
    if 0 < margin_pct <= 1.0:
        msg = (
            f"⚠️ <b>[L2 후보군 접근 알림]</b>\n\n"
            f"현재 가격이 이전 L1 바닥에 근접했습니다.\n"
            f"지금 트레이딩뷰를 켜고 <b>15분봉</b> 차트를 확인하세요!\n\n"
            f"• L1 바닥가: ${l1_price:,.2f}\n"
            f"• 현재 최저가: ${current_low:,.2f} (+{margin_pct:.2f}%)"
        )
        send_telegram_message(msg)
        exit()

# ------------------------------------------------------------------------------
# 5. 실시간 감시 로직 2: L2 확정 타점 발송 (스코어링 필터 탑재)
# ------------------------------------------------------------------------------
l1_idx, _, l1_price = fractal_lows[-2]
l2_idx, l2_date, l2_price = fractal_lows[-1]

if current_idx - (l2_idx + N_VAL) <= 1: 
    ma_20 = df['MA_20'].iloc[current_idx]
    
    if not pd.isna(ma_20) and current_close >= float(ma_20) and l2_price > l1_price:
        atr = float(df['ATR_14'].iloc[current_idx])
        sl = l2_price - (1.5 * atr)
        risk_dist = current_close - sl
        sl_pct = (risk_dist / current_close) * 100
        tp = current_close + (risk_dist * 1.5)
        
        # [NEW] 타점 퀄리티 평가 (스코어링)
        score = 0
        tags = []
        
        # 1. 20MA 기울기 평가
        ma_20_old = df['MA_20_3D_ago'].iloc[current_idx]
        if not pd.isna(ma_20_old) and float(ma_20) > float(ma_20_old):
            score += 1
            tags.append("✅ 20MA 우상향 (추세 좋음)")
        else:
            tags.append("⚠️ 20MA 꺾임 (역추세 주의)")
            
        # 2. 거래량 다이버전스 평가
        down_vol = np.mean(vols[l1_idx : l2_idx+1]) if l2_idx > l1_idx else 1
        up_vol = np.mean(vols[l2_idx+1 : current_idx+1])
        if up_vol > down_vol:
            score += 1
            tags.append("✅ 수급 유입 (건강한 반등)")
        else:
            tags.append("⚠️ 수급 마름 (가짜 반등 주의)")
            
        grade = 'A급' if score == 2 else 'B급' if score == 1 else 'C급'
        
        warning_flag = ""
        if sl_pct >= 4.5:
            warning_flag = f"🚨 <b>[경고: 고변동성 리스크]</b> 손절폭({sl_pct:.2f}%) 과다! 비중 축소 요망!\n\n"
            
        msg = (
            f"🎯 <b>[BTC 1H 스윙 매수 타점 포착]</b>\n\n"
            f"{warning_flag}"
            f"🏅 <b>타점 등급: {grade}</b>\n"
            f"• {tags[0]}\n"
            f"• {tags[1]}\n\n"
            f"• 진입가(현재): ${current_close:,.2f}\n"
            f"• 손절가(SL): ${sl:,.2f} (-{sl_pct:.2f}%)\n"
            f"• 목표가(TP): ${tp:,.2f} (+{sl_pct * 1.5:.2f}%)\n\n"
            f"• L1 바닥: ${l1_price:,.2f}\n"
            f"• L2 바닥: ${l2_price:,.2f}\n\n"
            f"💡 <i>Tip: 진입 후 3~4시간 횡보 시 '시간 손절' 고려</i>"
        )
        send_telegram_message(msg)
