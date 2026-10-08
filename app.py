"""Entry point:  python app.py  ->  http://127.0.0.1:5000"""
from backend import create_app

app = create_app()

if __name__ == "__main__":
    # threaded=True lets the page poll stats while a frame is being processed
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
