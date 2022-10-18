from flask import Flask, request, jsonify
from ..LR import main, LR

app = Flask(__name__)

@app.route("/health", methods=["POST"])
def get_health():
    return jsonify({"status": "ok"})

@app.route("/run", methods=["POST"])
def run_linear_regression():
    MAE, MSE, RMSE = main()
    return jsonify({"MAE": MAE, "MSE": MSE, "RMSE": RMSE})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
