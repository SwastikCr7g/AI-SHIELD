from app import create_app

# Application instance create ho raha hai
app = create_app()

if __name__ == "__main__":
    # debug=True development ke liye best hai
    app.run(debug=True, host="127.0.0.1", port=5000)