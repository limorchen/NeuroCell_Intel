import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import os
import re

# Set these up as env variables or secrets in your workflow for email security
EMAIL_USER = os.environ.get("EMAIL_USER")
EMAIL_PASSWORD = os.environ.get("EMAIL_PASSWORD")
EMAIL_RECIPIENT = os.environ.get("EMAIL_RECIPIENT")

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
    
    # Check for empty DataFrame
    if df.empty:
        print(f"[WARN] No data returned for {symbol} ({ticker}). Skipping this ticker.")
        return None
    
    # Print columns received to debug missing 'Close'
    print(f"[DEBUG] Columns for {symbol} ({ticker}): {list(df.columns)}")
    
    # Check if 'Close' column exists
    if 'Close' not in df.columns:
        print(f"[WARN] 'Close' column missing for {symbol} ({ticker}). Data columns: {list(df.columns)}")
        print(f"[DEBUG] Data snippet:\n{df.head()}")
        return None
    
    # Now safe to dropna on 'Close'
    df = df.dropna(subset=['Close'])
    
    # Continue processing...
    
    df['Daily Change %'] = df['Close'].pct_change() * 100
    df['Weekly Change %'] = df['Close'].pct_change(periods=5) * 100
    latest = df.iloc[-1]
    day_change = latest['Daily Change %']
    week_change = latest['Weekly Change %']
    last_price = latest['Close']
    # Plot and save graph
    plt.figure(figsize=(10,4))
    plt.plot(df.index, df['Close'], label=f'{symbol} price')
    plt.title(f'{symbol} - Closing Price')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid()
    img_path = f"{symbol.replace(':','_')}_price.png"
    plt.savefig(img_path)
    plt.close()
    return {
        "symbol": symbol,
        "ticker": ticker,
        "last_price": round(last_price, 3),
        "day_change": round(day_change, 2),
        "week_change": round(week_change, 2),
        "graph": img_path
    }

def main():
    parts = []
    for symbol, ticker in TICKERS.items():
        result = process_symbol(symbol, ticker)
        if result is None:
            continue
        parts.append(result)

    # Make simple HTML report
    body = "<h2>Nurexone Biologic (NRX) - Daily Market Report</h2><table border='1'><tr><th>Market</th><th>Price</th><th>Daily Change %</th><th>Weekly Change %</th></tr>"
    for item in parts:
        body += f"<tr><td>{item['symbol']}</td><td>{item['last_price']}</td><td>{item['day_change']}</td><td>{item['week_change']}</td></tr>"
    body += "</table><br>"

    # Set up email with images attached
    msg = MIMEMultipart()
    msg['From'] = EMAIL_USER
    msg['To'] = EMAIL_RECIPIENT
    msg['Subject'] = "Nurexone Daily Stock Report"

    msg.attach(MIMEText(body, 'html'))
    # Attach price charts
    for item in parts:
        with open(item['graph'], 'rb') as f:
            part = MIMEBase('application', "octet-stream")
            part.set_payload(f.read())
            encoders.encode_base64(part)
            part.add_header('Content-Disposition', f'attachment; filename="{item["graph"]}"')
            msg.attach(part)

    # Send email
    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
        smtp.login(EMAIL_USER, EMAIL_PASSWORD)
        smtp.sendmail(EMAIL_USER, EMAIL_RECIPIENT, msg.as_string())

if __name__ == "__main__":
    main()
