from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # supaya aman di server
import matplotlib.pyplot as plt
import math
import os
import base64
from io import BytesIO

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# ================================
# RK4 + SIR + TUNING (kode kamu)
# ================================
def sir_model(t, Y, beta, gamma):
    S, I, R = Y
    dSdt = -beta * S * I
    dIdt =  beta * S * I - gamma * I
    dRdt =  gamma * I
    return np.array([dSdt, dIdt, dRdt])

def rk4_solver(f, t_span, y0, h, params):
    t0, tf = t_span
    t_vals = np.arange(t0, tf + h, h)
    y_vals = np.zeros((len(t_vals), len(y0)))
    y_vals[0] = y0

    for i in range(len(t_vals) - 1):
        t_n = t_vals[i]
        y_n = y_vals[i]

        k1 = f(t_n, y_n, *params)
        k2 = f(t_n + h/2, y_n + h*k1/2, *params)
        k3 = f(t_n + h/2, y_n + h*k2/2, *params)
        k4 = f(t_n + h,   y_n + h*k3,   *params)

        y_vals[i+1] = y_n + (h/6)*(k1 + 2*k2 + 2*k3 + k4)

    return t_vals, y_vals

def rmse(a, b):
    return math.sqrt(np.mean((a - b)**2))

def run_simulation(csv_path, country="Indonesia", N=270e6):
    df = pd.read_csv(csv_path)

    sub = df[df["Country/Region"] == country].copy()
    sub["Date"] = pd.to_datetime(sub["Date"])
    daily = sub.groupby("Date")[["Confirmed", "Recovered", "Deaths"]].sum().reset_index()

    daily["I"] = daily["Confirmed"] - daily["Recovered"] - daily["Deaths"]
    daily["R_sir"] = daily["Recovered"] + daily["Deaths"]

    first_case_idx = int(np.argmax(daily["Confirmed"].values > 0))
    daily = daily.iloc[first_case_idx:].reset_index(drop=True)

    t_data = np.arange(len(daily), dtype=float)

    I_data = daily["I"].values / N
    R_data = daily["R_sir"].values / N
    S_data = 1 - I_data - R_data

    beta_range  = np.linspace(0.05, 1.5, 40)
    gamma_range = np.linspace(0.01, 0.6, 40)

    best_params = None
    best_err = np.inf
    h = 1.0

    for beta in beta_range:
        for gamma in gamma_range:
            params = (beta, gamma)
            t_sim, Y_sim = rk4_solver(
                sir_model,
                (0, len(t_data)-1),
                y0=[S_data[0], I_data[0], R_data[0]],
                h=h,
                params=params
            )

            if np.any(Y_sim < 0) or np.any(Y_sim > 1.0):
                continue

            I_sim_tmp = np.interp(t_data, t_sim, Y_sim[:, 1])
            R_sim_tmp = np.interp(t_data, t_sim, Y_sim[:, 2])

            err_I = rmse(I_data, I_sim_tmp)
            err_R = rmse(R_data, R_sim_tmp)
            err = 0.7 * err_I + 0.3 * err_R

            if err < best_err:
                best_err = err
                best_params = params

    # simulasi final pakai parameter terbaik
    t_sim, Y_sim = rk4_solver(
        sir_model,
        (0, len(t_data)-1),
        y0=[S_data[0], I_data[0], R_data[0]],
        h=h,
        params=best_params
    )

    S_sim = np.interp(t_data, t_sim, Y_sim[:, 0])
    I_sim = np.interp(t_data, t_sim, Y_sim[:, 1])
    R_sim = np.interp(t_data, t_sim, Y_sim[:, 2])

    return t_data, I_data, R_data, S_sim, I_sim, R_sim, best_params, best_err


def fig_to_base64(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    plt.close(fig)
    return img_b64


@app.route("/", methods=["GET", "POST"])
def index():
    plot1 = None
    plot2 = None
    result = None
    countries = []
    selected_country = "Indonesia"

    # ====== BATAS NILAI POPULASI N ======
    allowed_N = {270000000, 5000000, 10000000, 100000000}
    selected_N = 270000000

    # default file (kalau user tidak upload)
    default_csv = "time-series-19-covid-combined.csv"
    csv_path = default_csv if os.path.exists(default_csv) else None

    if request.method == "POST":
        # handle upload
        file = request.files.get("file")
        if file and file.filename:
            csv_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(csv_path)

        selected_country = request.form.get("country", "Indonesia")

        # ambil N dari form, lalu validasi ke allowed_N
        try:
            selected_N = int(float(request.form.get("population", 270000000)))
        except:
            selected_N = 270000000

        if selected_N not in allowed_N:
            selected_N = 270000000

        N = float(selected_N)

        # baca df (untuk list negara)
        if csv_path:
            df = pd.read_csv(csv_path)
            countries = sorted(df["Country/Region"].dropna().unique().tolist())

            # run sim
            t_data, I_data, R_data, S_sim, I_sim, R_sim, best_params, best_err = run_simulation(
                csv_path, selected_country, N
            )

            # Plot overlay I
            fig1 = plt.figure(figsize=(8,4.5))
            plt.scatter(t_data, I_data, label="Data Asli I(t)", s=18, alpha=0.7)
            plt.plot(t_data, I_sim, label="Simulasi RK4 I(t)", linewidth=2)
            plt.xlabel("Hari sejak kasus pertama")
            plt.ylabel("Proporsi Terinfeksi (I/N)")
            plt.title(f"Overlay Data vs Simulasi - {selected_country}")
            plt.grid(alpha=0.3)
            plt.legend()
            plot1 = fig_to_base64(fig1)

            # Plot S dan R
            fig2 = plt.figure(figsize=(8,4.5))
            plt.plot(t_data, S_sim, label="S(t) simulasi", linewidth=2)
            plt.plot(t_data, R_sim, label="R(t) simulasi", linewidth=2)
            plt.xlabel("Hari sejak kasus pertama")
            plt.ylabel("Proporsi Populasi")
            plt.title("Dinamika S dan R (Simulasi)")
            plt.grid(alpha=0.3)
            plt.legend()
            plot2 = fig_to_base64(fig2)

            result = {
                "beta": best_params[0],
                "gamma": best_params[1],
                "rmse": best_err,
                "R0": best_params[0] / best_params[1] if best_params[1] != 0 else None
            }

    else:
        # GET: isi list negara dari default csv jika ada
        if csv_path:
            df = pd.read_csv(csv_path)
            countries = sorted(df["Country/Region"].dropna().unique().tolist())

    return render_template(
        "index.html",
        plot1=plot1,
        plot2=plot2,
        result=result,
        countries=countries,
        selected_country=selected_country,
        selected_N=selected_N  # supaya dropdown bisa ingat pilihan
    )


if __name__ == "__main__":
    app.run(debug=True)