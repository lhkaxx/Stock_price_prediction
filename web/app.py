"""
web/app.py — Flask 后端，驱动股票预测 Dashboard
启动: python web/app.py  或  flask --app web.app run
"""
import json
import os
import sys

# 确保项目根目录在 sys.path 中
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from flask import Flask, render_template, jsonify

app = Flask(__name__)
FIGURES_DIR = os.path.join(PROJECT_ROOT, "figures")


def load_json(filename: str) -> dict:
    """从 figures/ 目录加载 JSON 文件"""
    path = os.path.join(FIGURES_DIR, filename)
    if not os.path.exists(path):
        return {"error": f"文件不存在: {filename}，请先运行 main.py"}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@app.route("/")
def index():
    """首页 — 渲染 Dashboard"""
    metrics = load_json("metrics.json")
    history = load_json("chart_history.json")
    predict = load_json("chart_predict.json")
    residual = load_json("chart_residual.json")

    return render_template("dashboard.html",
                           metrics=metrics,
                           history=history,
                           predict=predict,
                           residual=residual)


@app.route("/api/metrics")
def api_metrics():
    return jsonify(load_json("metrics.json"))


@app.route("/api/history")
def api_history():
    return jsonify(load_json("chart_history.json"))


@app.route("/api/predict")
def api_predict():
    return jsonify(load_json("chart_predict.json"))


@app.route("/api/residual")
def api_residual():
    return jsonify(load_json("chart_residual.json"))


@app.route("/forecast")
def forecast():
    """未来预测页面"""
    forecast_data = load_json("forecast.json")
    return render_template("forecast.html", forecast=forecast_data)


@app.route("/api/forecast")
def api_forecast():
    return jsonify(load_json("forecast.json"))


if __name__ == "__main__":
    print(f"[Flask] 项目根目录: {PROJECT_ROOT}")
    print(f"[Flask] 数据目录:   {FIGURES_DIR}")
    print(f"[Flask] 启动地址:   http://127.0.0.1:5000")
    print(f"[Flask] 启动地址:   http://127.0.0.1:5000/forecast")

    app.run(debug=True, host="127.0.0.1", port=5000)
