import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from dotenv import load_dotenv
import os
from io import BytesIO

load_dotenv()

# Environment variables for email credentials and recipients
EMAIL_USER = os.environ.get("EMAIL_USER")
EMAIL_PASSWORD = os.environ.get("EMAIL_PASSWORD")
EMAIL_RECIPIENT = os.environ.get("EMAIL_RECIPIENT")

print(f"DEBUG: EMAIL_USER set: {bool(EMAIL_USER)}")
print(f"DEBUG: EMAIL_PASSWORD set: {bool(EMAIL_PASSWORD)}")
print(f"DEBUG: EMAIL_RECIPIENT set: {bool(EMAIL_RECIPIENT)}")

# Market tickers for Nurexone Biologic
TICKERS = {
    "TSXV:NRX": "NRX.V",
    "OTCQB:NRXBF": "NRXBF",
    "FSE:J90": "J90.F"
}

def fetch_data(ticker, days=14):
    end = datetime.today()
    start = end - timedelta(days=days)
    df = yf.download(ticker,
                     start=start.strftime("%Y-%m-%d"),
                     end=end.strftime("%Y-%m-%d"),
                     group_by='ticker')
    if ticker in df.columns.levels[0]:
        df = df[ticker]
    return df

def process_symbol(symbol, ticker):
    df = fetch_data(ticker)
    
    if df.empty:
        print(f"[WARN] No data returned for {symbol} ({ticker}). Skipping this ticker.")
        return None
    
    print(f"[DEBUG] Columns for {symbol} ({ticker}): {list(df.columns)}")
    
    if 'Close' not in df.columns:
        print(f"[WARN] 'Close' column missing for {symbol} ({ticker}). Data columns: {list(df.columns)}")
        print(f"[DEBUG] Data snippet:\n{df.head()}")
        return None
    
    df = df.dropna(subset=['Close'])
    
    df['Daily Change %'] = df['Close'].pct_change() * 100
    df['Weekly Change %'] = df['Close'].pct_change(periods=5) * 100
    latest = df.iloc[-1]
    day_change = latest['Daily Change %']
    week_change = latest['Weekly Change %']
    last_price = latest['Close']

    currency = "N/A"
    try:
        ticker_info = yf.Ticker(ticker)
        currency = ticker_info.info.get('currency', 'N/A')
    except Exception as e:
        print(f"[WARN] Could not fetch currency for {symbol} ({ticker}): {e}")

    print(f"[DEBUG] Currency for {symbol} is {currency}")

    # Save chart image in memory
    buf = BytesIO()
    plt.figure(figsize=(8,3))
    plt.plot(df.index, df['Close'], label=f'{symbol} price')
    plt.title(f'{symbol} - Closing Price')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid()
    plt.savefig(buf, format='png', dpi=80)
    plt.close()
    buf.seek(0)

    return {
        "symbol": symbol,
        "ticker": ticker,
        "last_price": round(last_price, 3),
        "day_change": round(day_change, 2),
        "week_change": round(week_change, 2),
        "currency": currency,
        "image_buffer": buf  # Keep buffer open for email embedding
    }

def main():
    parts = []
    for symbol, ticker in TICKERS.items():
        result = process_symbol(symbol, ticker)
        if result is None:
            continue
        parts.append(result)

    body = "<h2>Nurexone Biologic (NRX) - Daily Market Report</h2>"
    body += "<table border='1' cellpadding='5' cellspacing='0'>"
    body += "<tr><th>Market</th><th>Price</th><th>Currency</th><th>Daily Change %</th><th>Weekly Change %</th></tr>"
    for item in parts:
        body += f"<tr>"
        body += f"<td>{item['symbol']}</td><td>{item['last_price']}</td><td>{item['currency']}</td>"
        body += f"<td>{item['day_change']}</td><td>{item['week_change']}</td>"
        body += "</tr>"
    body += "</table>"

    body += "<br><h3>Price Charts</h3>"
    for idx, item in enumerate(parts):
        cid = f"chart{idx}"
        body += f"<h4>{item['symbol']}</h4>"
        body += f"<img src='cid:{cid}' alt='Chart for {item['symbol']}' width='300'/><br>"

    msg = MIMEMultipart("related")
    msg['From'] = EMAIL_USER
    msg['To'] = EMAIL_RECIPIENT
    msg['Subject'] = "Nurexone Daily Stock Report"

    msg_alternative = MIMEMultipart("alternative")
    msg.attach(msg_alternative)

    msg_alternative.attach(MIMEText(body, "html"))

    # Attach images with CIDs
    for idx, item in enumerate(parts):
        img_data = item['image_buffer'].getvalue()
        mime_img = MIMEImage(img_data, 'png')
        mime_img.add_header('Content-ID', f"<chart{idx}>")
        mime_img.add_header('Content-Disposition', 'inline', filename=f"{item['symbol']}_chart.png")
        msg.attach(mime_img)
        item['image_buffer'].close()

    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
        smtp.login(EMAIL_USER, EMAIL_PASSWORD)
        smtp.sendmail(EMAIL_USER, EMAIL_RECIPIENT, msg.as_string())

if __name__ == "__main__":
    main()
