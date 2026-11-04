import os
import warnings
import numpy as np
import pandas as pd
import requests
import yfinance as yf

warnings.filterwarnings("ignore")

# ------------------------------------------------------------------------------
# 1. 텔레그램 설정 및 메시지 발송 함수
# ------------------------------------------------------------------------------
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

def send_telegram_message(message):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("Telegram 설정(환경변수)이 없어 콘솔에만 출력합니다:\n", message)
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "HTML"
    }
    try:
        res = requests.post(url, json=payload, timeout=10)
        res.raise_for_status()
        print("텔레그램 메시지 전송 성공")
    except Exception as e:
        print(f"Telegram 전송 실패: {e}")

# ------------------------------------------------------------------------------
# 2. 데이터 다운로드 및 보조지표 계산
# ------------------------------------------------------------------------------
print("비트코인(BTC-USD) 1시간봉 데이터 다운로드 중...")
df = yf.download('BTC-USD', period='60d', interval='1h', progress=False)
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)
df = df.dropna().copy()

# ATR(14) 계산
df['H-L'] = df['High'] - df['Low']
df['H-PC'] = np.abs(df['High'] - df['Close'].shift(1))
df['L-PC'] = np.abs(df['Low'] - df['Close'].shift(1))
df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
df['ATR_14'] = df['TR'].rolling(window=14).mean()

# 대추세 필터: 1일봉 20MA 환산 (1시간봉 480선) 및 3일 전(72시간 전) 대비 기울기
df['MA_1D_20'] = df['Close'].shift(1).rolling(window=480).mean()
df['MA_Slope_3D'] = df['MA_1D_20'] - df['MA_1D_20'].shift(72)

N_VAL = 3
MAX_L1_L2_BARS = 168  # 쌍바닥 최대 허용 기간 (7일 = 168시간)
RR_RATIO = 1.5        # 1.5R 익절
ATR_MULTIPLIER = 1.5  # 손절 여유폭 배수

# ------------------------------------------------------------------------------
# 3. N=3 프랙탈 저점 탐색
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
    print("식별된 프랙탈 저점이 부족하여 종료합니다.")
    exit()

# ------------------------------------------------------------------------------
# 4. 실시간 감시 1: L1 접근 경고 (15분봉 확인용 선진입 알림)
# ------------------------------------------------------------------------------
last_l1_idx, last_l1_date, l1_price = fractal_lows[-1]
current_idx = len(df) - 1
current_close = float(df['Close'].iloc[current_idx])
current_low = float(df['Low'].iloc[current_idx])

# L1 바닥 이후 3캔들 이상 지난 상태에서, 현재가가 L1 바닥 대비 +1% 이내로 눌렸을 때
if (current_idx - last_l1_idx) > N_VAL:
    margin_pct = (current_low - l1_price) / l1_price * 100
    if 0 < margin_pct <= 1.0:
        msg = (
            f"⚠️ <b>[L2 후보군 접근 알림]</b>\n\n"
            f"현재 가격이 이전 L1 바닥에 근접했습니다.\n"
            f"지금 트레이딩뷰를 켜고 <b>15분봉</b> 차트에서 지지 캔들을 확인하세요!\n\n"
            f"• L1 바닥가: ${l1_price:,.2f}\n"
            f"• 현재 최저가: ${current_low:,.2f} (+{margin_pct:.2f}%)"
        )
        send_telegram_message(msg)
        exit()

# ------------------------------------------------------------------------------
# 5. 실시간 감시 2: L2 확정 타점 포착 및 등급별(A~D) 알림 발송
# ------------------------------------------------------------------------------
l1_idx, _, l1_price = fractal_lows[-2]
l2_idx, l2_date, l2_price = fractal_lows[-1]

# L2 프랙탈 확정(우측 3캔들 완성) 직후 1캔들 이내인 경우에만 신호 발생
if current_idx - (l2_idx + N_VAL) <= 1:
    ma_20 = df['MA_1D_20'].iloc[current_idx]
    
    # 기본 조건: L2 > L1 (저점 상승 쌍바닥) 및 7일 이내 간격
    if l2_price > l1_price and (l2_idx - l1_idx) <= MAX_L1_L2_BARS:
        atr = float(df['ATR_14'].iloc[current_idx])
        sl = l2_price - (ATR_MULTIPLIER * atr)
        risk_dist = current_close - sl
        
        if risk_dist > 0:
            sl_pct = (risk_dist / current_close) * 100
            tp = current_close + (risk_dist * RR_RATIO)
            
            # 스코어링 지표 계산
            ma_slope = float(df['MA_Slope_3D'].iloc[current_idx])
            down_vol = np.mean(vols[l1_idx : l2_idx + 1]) if l2_idx > l1_idx else 1
            up_vol = np.mean(vols[l2_idx + 1 : current_idx + 1])
            
            score = 0
            tags = []
            grade = ""
            action_tip = ""
            
            # 1) 대추세(20MA) 역배열 여부 체크 -> D급 분류
            if pd.isna(ma_20) or current_close <= float(ma_20):
                grade = 'D급'
                tags.append("⛔ 20MA 하단 (역배열 하락장)")
                action_tip = "🚫 <b>진입 절대 금지 (데드캣 트랩 확률 매우 높음)</b>"
            # 2) 정배열(20MA 상단)일 때 -> A / B / C급 세부 스코어링
            else:
                if ma_slope > 0:
                    score += 1
                    tags.append("✅ 20MA 우상향 (추세 건전)")
                else:
                    tags.append("⚠️ 20MA 꺾임 (역추세 주의)")
                    
                if up_vol > down_vol:
                    score += 1
                    tags.append("✅ 수급 폭발 (건강한 반등)")
                else:
                    tags.append("⚠️ 수급 마름 (가짜 반등 주의)")
                    
                grade = 'A급' if score == 2 else 'B급' if score == 1 else 'C급'
                
                if grade == 'A급':
                    action_tip = "💡 <b>기본 비중(100%) 적극 진입 추천</b>"
                elif grade == 'B급':
                    action_tip = "💡 <b>비중 50% 축소 진입 (보수적 접근)</b>"
                else:
                    action_tip = "🚫 <b>설거지(세력 이탈) 패턴. 매수 패스(Pass) 권장</b>"

            # 3) 고변동성 리스크 문구 분기 처리
            warning_flag = ""
            if sl_pct >= 4.5:
                warning_flag = f"🚨 <b>[경고: 고변동성]</b> 손절폭({sl_pct:.2f}%) 과다! 비중 축소 요망!\n\n"
                
            # 4) 백슬래시 문법 에러 방지를 위해 태그 문자열 미리 조립
            tag_lines = "\n".join([f"• {t}" for t in tags])
            
            msg = (
                f"🎯 <b>[BTC 1H 프랙탈 스윙 포착]</b>\n\n"
                f"{warning_flag}"
                f"🏅 <b>타점 등급: {grade}</b>\n"
                f"{tag_lines}\n\n"
                f"• 진입가(현재): ${current_close:,.2f}\n"
                f"• 손절가(SL): ${sl:,.2f} (-{sl_pct:.2f}%)\n"
                f"• 목표가(TP): ${tp:,.2f} (+{sl_pct * RR_RATIO:.2f}%)\n\n"
                f"• L1 바닥: ${l1_price:,.2f}\n"
                f"• L2 바닥: ${l2_price:,.2f}\n\n"
                f"{action_tip}\n\n"
                f"⏱ <i>진입 후 3~4시간 동안 탄력이 없으면 '시간 손절' 고려</i>"
            )
            send_telegram_message(msg)
