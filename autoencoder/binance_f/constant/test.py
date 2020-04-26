import os

private_config_path = 'binance_f/constant/privateconfig.bin'

if os.path.isfile(private_config_path):
    from imp import load_compiled
    config = load_compiled('a', private_config_path)
    g_api_key = config.p_api_key
    g_secret_key = config.p_secret_key
    g_account_id = config.g_account_id
else:
    g_api_key = ""
    g_secret_key = ""
    g_account_id = None
