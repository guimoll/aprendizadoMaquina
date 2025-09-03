import tkinter as tk
from tkinter import scrolledtext
import threading
import fase2
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkhtmlview import HTMLLabel
import plotly.io as pio
import webview
import os
import tempfile
from cefpython3 import cefpython as cef

# Helper to run analysis in a thread (avoid GUI freeze)
def run_in_thread(func, *args):
    def wrapper():
        output = func(*args)
        show_output(output)
    threading.Thread(target=wrapper).start()

def show_output(text):
    output_text.config(state=tk.NORMAL)
    output_text.delete(1.0, tk.END)
    output_text.insert(tk.END, text)
    output_text.config(state=tk.DISABLED)

def clear_plot_frame():
    for widget in plot_frame.winfo_children():
        widget.destroy()

def show_matplotlib_figure(fig):
    clear_plot_frame()
    canvas = FigureCanvasTkAgg(fig, master=plot_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

def show_plotly_figure(fig):
    clear_plot_frame()
    html = pio.to_html(fig, include_plotlyjs='cdn')
    # Save HTML to a temporary file
    temp_dir = tempfile.gettempdir()
    html_path = os.path.join(temp_dir, "plotly_graph.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    # Embed Chromium browser in plot_frame
    if not cef.IsInitialized():
        cef.Initialize()
    # Get the window handle for plot_frame
    window_info = cef.WindowInfo()
    window_info.SetAsChild(plot_frame.winfo_id(), [0, 0, plot_frame.winfo_width(), plot_frame.winfo_height()])
    browser = cef.CreateBrowserSync(window_info, url="file://" + html_path)
    # Resize browser when plot_frame is resized
    def on_resize(event):
        if browser:
            browser.SetBounds(0, 0, event.width, event.height)
    plot_frame.bind("<Configure>", on_resize)

def item_a():
    show_output(fase2.item_a())
def item_b():
    show_output(fase2.item_b())
def item_c():
    show_output(fase2.item_c())
def item_d():
    x, y = fase2.matUtils.load_data_quartos_preco()
    fig1 = fase2.graficos.plot_dataset(x, y, "PLOT Numero de quartos e preço")
    show_matplotlib_figure(fig1)
    z, v = fase2.matUtils.load_data_tamanho_casa_preco()
    fig2 = fase2.graficos.plot_dataset(z, v, "PLOT Tamanho casa e preço")
    show_matplotlib_figure(fig2)
    show_output("Plots gerados para quartos/preço e tamanho/preço.")
def item_e():
    b0, b1, b2 = fase2.matUtils.regressao_multipla()
    fig = fase2.graficos.plot_regression_3d(b0, b1, b2)
    show_plotly_figure(fig)
    show_output(f"Regressão múltipla: b0={b0:.2f}, b1={b1:.2f}, b2={b2:.2f}")
def item_f():
    X, Y = fase2.matUtils.load_data_tamanho_casa_preco()
    fig = fase2.graficos.plot_dataset_regressao_3d(X, Y, "PLOT 3D Tamanho casa e preço")
    show_plotly_figure(fig)
    show_output("Plot 3D Tamanho casa e preço gerado.")
def item_g():
    Z, V = fase2.matUtils.load_data_quartos_preco()
    fig = fase2.graficos.plot_dataset_regressao_3d(Z, V, "Plot 3D Numero de quartos e preço")
    show_plotly_figure(fig)
    show_output("Plot 3D Numero de quartos e preço gerado.")
def item_h():
    show_output(fase2.item_h())
def item_i():
    show_output(fase2.item_i())

root = tk.Tk()
root.title("Análise de Regressão Linear - Frontend")

button_frame = tk.Frame(root)
button_frame.pack(side=tk.TOP, fill=tk.X)

buttons = [
    ("Item A", item_a),
    ("Item B", item_b),
    ("Item C", item_c),
    ("Item D", item_d),
    ("Item E", item_e),
    ("Item F", item_f),
    ("Item G", item_g),
    ("Item H", item_h),
    ("Item I", item_i),
]
for label, cmd in buttons:
    tk.Button(button_frame, text=label, command=cmd, width=12).pack(side=tk.LEFT, padx=2, pady=2)

output_text = scrolledtext.ScrolledText(root, height=10, state=tk.DISABLED)
output_text.pack(fill=tk.BOTH, expand=True)

plot_frame = tk.Frame(root, height=300)
plot_frame.pack(fill=tk.BOTH, expand=True)

root.mainloop()
