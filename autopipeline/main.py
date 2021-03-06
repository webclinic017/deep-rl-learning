from __future__ import print_function

import json
import pickle
import os.path
import time
from datetime import datetime

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
    win_rate = 40
    dmi_threshold = 20
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
    # results = service.users().labels().list(userId='me').execute()
    # labels = results.get('labels', [])

    service = build('gmail', 'v1', credentials=creds)

    with open("config.json") as config_file:
        config = json.load(config_file)

    while True:
        time.sleep(2)
        results = None
        try:
            results = service.users().messages().list(userId='me', labelIds=["UNREAD", "INBOX"], maxResults=10).execute()
        except Exception as ex:
            print(ex)

        if results and results['resultSizeEstimate'] > 0:
            for message in results['messages']:
                messageheader = service.users().messages().get(userId="me", id=message["id"], format="full",
                                                               metadataHeaders=None).execute()
                # print(messageheader)
                snippet = messageheader['snippet']
                for symbol_name, value in zip(config.keys(), config.values()):
                    lot = value['lot']
                    division = value['division']

                    if symbol_name in snippet:
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
                        mt5_client.close_order(symbol)
                        tp1_hit_percent = info['tp1_hit']
                        tp1_hit_percent = float(tp1_hit_percent.split('=')[1].replace('%', ''))
                        tp1 = info['tp1'].replace("Pts", "")
                        tp1 = float(tp1.split("=")[1])

                        tp2_hit_percent = info['tp2_hit']
                        tp2_hit_percent = float(tp2_hit_percent.split('=')[1].replace('%', ''))
                        tp2 = info['tp2'].replace("Pts", "")
                        tp2 = float(tp2.split("=")[1])

                        if info['side'] == 'Buy' and info['price'] > mt5_client.get_ema(symbol, 200):
                            mt5_client.buy_order(symbol, tp2 * 10, 0.5)
                            service.users().messages().modify(userId='me', id=message["id"],
                                                              body={'removeLabelIds': ['UNREAD']}).execute()
                        elif info['side'] == 'Sell' and info['price'] < mt5_client.get_ema(symbol, 200):
                            mt5_client.sell_order(symbol, tp2 * 10, 0.5)
                            service.users().messages().modify(userId='me', id=message["id"],
                                                              body={'removeLabelIds': ['UNREAD']}).execute()
                # XAUUSD M1 Buy Signal @1924.47 TP1=76Pts TP2=251Pts TP1
                # Hit=79.17% TP2 Hit=57.29% EXIT Win=0.00% EXIT Loss=20.83% Success Rate=79.17%

        # check order profit
        a = datetime.now().minute
        if (a % 15) == 0:  # every 15 minutes
            mt5_client.modify_stoploss()


if __name__ == '__main__':
    main()
