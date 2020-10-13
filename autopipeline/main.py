from __future__ import print_function
import pickle
import os.path
import time
from datetime import date, timedelta

from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request

from mt5 import AutoOrder

# If modifying these scopes, delete the file token.pickle.
SCOPES = ['https://www.googleapis.com/auth/gmail.modify']
mt5_client = AutoOrder()


def main():
    """Shows basic usage of the Gmail API.
    Lists the user's Gmail labels.
    """
    creds = None
    # The file token.pickle stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first
    # time.
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)

    service = build('gmail', 'v1', credentials=creds)

    # Call the Gmail API
    results = service.users().labels().list(userId='me').execute()
    labels = results.get('labels', [])

    service = build('gmail', 'v1', credentials=creds)
    current_info = {}
    while True:
        time.sleep(0.5)
        results = service.users().messages().list(userId='me', labelIds=["UNREAD", "INBOX"], maxResults=1).execute()
        # print(f"results {results}")
        # print("\n")
        if results['resultSizeEstimate'] > 0:
            for message in results['messages']:
                messageheader = service.users().messages().get(userId="me", id=message["id"], format="full",
                                                               metadataHeaders=None).execute()
                # print(messageheader)
                snippet = messageheader['snippet']
                if "XAUUSD" in snippet:
                    partitions = snippet.split(" ")
                    symbol = partitions[0]
                    info = {
                        "symbol": partitions[0],
                        "period": partitions[1],
                        "price": float(partitions[4].replace("@", "")),
                        "side": partitions[2],
                        "tp1": partitions[5],
                        "tp2": partitions[6],
                        "tp1_hit": partitions[8],
                        "tp2_hit": partitions[10],
                        "timestamp": time.time()
                    }
                    print(info)
                    price = info['price']
                    mt5_client.close_order(symbol)
                    tp = info['tp1'].replace("Pts", "")
                    tp = float(tp.split("=")[1]) / 100
                    if info['side'] == 'Buy':
                        mt5_client.buy_order(symbol, tp)
                    elif info['side'] == 'Sell':
                        mt5_client.sell_order(symbol, tp)
                    current_info = info

                if "EURUSD" in snippet:
                    partitions = snippet.split(" ")
                    symbol = partitions[0]
                    info = {
                        "symbol": partitions[0],
                        "period": partitions[1],
                        "price": float(partitions[4].replace("@", "")),
                        "side": partitions[2],
                        "tp1": partitions[5],
                        "tp2": partitions[6],
                        "tp1_hit": partitions[8],
                        "tp2_hit": partitions[10],
                        "timestamp": time.time()
                    }
                    print(info)
                    price = info['price']
                    mt5_client.close_order(symbol)
                    tp = info['tp1'].replace("Pts", "")
                    tp = float(tp.split("=")[1]) / 100000
                    if info['side'] == 'Buy':
                        mt5_client.buy_order(symbol, tp)
                    elif info['side'] == 'Sell':
                        mt5_client.sell_order(symbol, tp)
                    current_info = info

                if "GBPUSD" in snippet:
                    partitions = snippet.split(" ")
                    symbol = partitions[0]
                    info = {
                        "symbol": partitions[0],
                        "period": partitions[1],
                        "price": float(partitions[4].replace("@", "")),
                        "side": partitions[2],
                        "tp1": partitions[5],
                        "tp2": partitions[6],
                        "tp1_hit": partitions[8],
                        "tp2_hit": partitions[10],
                        "timestamp": time.time()
                    }
                    print(info)
                    price = info['price']
                    mt5_client.close_order(symbol)
                    tp = info['tp1'].replace("Pts", "")
                    tp = float(tp.split("=")[1]) / 100000
                    if info['side'] == 'Buy':
                        mt5_client.buy_order(symbol, tp)
                    elif info['side'] == 'Sell':
                        mt5_client.sell_order(symbol, tp)
                    current_info = info

                if "USDJPY" in snippet:
                    partitions = snippet.split(" ")
                    symbol = partitions[0]
                    info = {
                        "symbol": partitions[0],
                        "period": partitions[1],
                        "price": float(partitions[4].replace("@", "")),
                        "side": partitions[2],
                        "tp1": partitions[5],
                        "tp2": partitions[6],
                        "tp1_hit": partitions[8],
                        "tp2_hit": partitions[10],
                        "timestamp": time.time()
                    }
                    print(info)
                    price = info['price']
                    mt5_client.close_order(symbol)
                    tp = info['tp1'].replace("Pts", "")
                    tp = float(tp.split("=")[1])/1000
                    if info['side'] == 'Buy':
                        mt5_client.buy_order(symbol, tp)
                    elif info['side'] == 'Sell':
                        mt5_client.sell_order(symbol, tp)
                    current_info = info

                service.users().messages().modify(userId='me', id=message["id"], body={'removeLabelIds': ['UNREAD']}).execute()
                # XAUUSD M1 Buy Signal @1924.47 TP1=76Pts TP2=251Pts TP1 Hit=79.17% TP2 Hit=57.29% EXIT Win=0.00% EXIT Loss=20.83% Success Rate=79.17%


if __name__ == '__main__':
    main()
