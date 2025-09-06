# server.py
from flask import Flask, request
import auto_trader  # make sure your bot code can handle a function call

app = Flask(__name__)

@app.route("/webhook", methods=["POST"])
def webhook():
    data = request.json
    # Example: process TradingView signal
    auto_trader.on_signal(data)
    return "ok"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)

