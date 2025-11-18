import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from dotenv import load_dotenv
import os
import base64
from io import BytesIO

load_dotenv()

# Set these up as env variables or secrets in your workflow for email security
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
    df = yf.download(ticker, start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"), group_by='ticker')
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

    # Try to get currency safely
    currency = "N/A"
    try:
        ticker_info = yf.Ticker(ticker)
        currency = ticker_info.info.get('currency', 'N/A')
    except Exception as e:
        print(f"[WARN] Could not fetch currency for {symbol} ({ticker}): {e}")

    print(f"[DEBUG] Currency for {symbol} is {currency}")

    # Save chart image in memory (BytesIO) instead of disk
    buf = BytesIO()
    plt.figure(figsize=(10,4))
    plt.plot(df.index, df['Close'], label=f'{symbol} price')
    plt.title(f'{symbol} - Closing Price')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)

    # Encode image bytes to base64 string for inline embedding
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()

    return {
        "symbol": symbol,
        "ticker": ticker,
        "last_price": round(last_price, 3),
        "day_change": round(day_change, 2),
        "week_change": round(week_change, 2),
        "currency": currency,
        "graph_base64": img_base64
    }

def main():
    parts = []
    for symbol, ticker in TICKERS.items():
        result = process_symbol(symbol, ticker)
        if result is None:
            continue
        parts.append(result)

    # Create HTML report with inline images including currency
    body = "<h2>Nurexone Biologic (NRX) - Daily Market Report</h2><table border='1' cellpadding='5' cellspacing='0'>"
    body += "<tr><th>Market</th><th>Price</th><th>Currency</th><th>Daily Change %</th><th>Weekly Change %</th><th>Chart</th></tr>"
    for item in parts:
        body += f"<tr>"
        body += f"<td>{item['symbol']}</td><td>{item['last_price']}</td><td>{item['currency']}</td>"
        body += f"<td>{item['day_change']}</td><td>{item['week_change']}</td>"
        body += f"<td><img src='data:image/png;base64,{item['graph_base64']}' alt='Chart for {item['symbol']}' width='400'/></td></tr>"
    body += "</table>"

    msg = MIMEMultipart("alternative")
    msg['From'] = EMAIL_USER
    msg['To'] = EMAIL_RECIPIENT
    msg['Subject'] = "Nurexone Daily Stock Report"

    # Attach the HTML body
    msg.attach(MIMEText(body, "html"))

    # Send the email
    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
        smtp.login(EMAIL_USER, EMAIL_PASSWORD)
        smtp.sendmail(EMAIL_USER, EMAIL_RECIPIENT, msg.as_string())

if __name__ == "__main__":
    main()
