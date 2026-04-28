
from flask import Flask, url_for, redirect, render_template, request, session, flash
import pandas as pd
import numpy as np
import mysql.connector
import joblib
from flask_wtf.csrf import CSRFProtect
from werkzeug.exceptions import RequestEntityTooLarge
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = 'admin'  # ← Change this to a strong random secret in production!
csrf = CSRFProtect(app)

# ────────────────────────────────────────────────
#           IMPORTANT: FILE SIZE LIMIT
# ────────────────────────────────────────────────
app.config['MAX_CONTENT_LENGTH'] = 400 * 1024 * 1024  # 400 MB
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# MySQL connection
mydb = None
mycursor = None
try:
    mydb = mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        port="3306",
        database='Satlite'
    )
    mycursor = mydb.cursor()
except mysql.connector.Error as err:
    print(f"Database connection failed: {err}")
    mycursor = None

def executionquery(query, values):
    if mycursor is not None:
        mycursor.execute(query, values)
        mydb.commit()
    else:
        print("Database connection or cursor is not available.")

def retrivequery1(query, values):
    if mycursor is not None:
        mycursor.execute(query, values)
        return mycursor.fetchall()
    else:
        print("Database connection or cursor is not available.")
        return []

def retrivequery2(query):
    if mycursor is not None:
        mycursor.execute(query)
        return mycursor.fetchall()
    else:
        print("Database connection or cursor is not available.")
        return []

# ────────────────────────────────────────────────
#  413 — File too large — user-friendly message
# ────────────────────────────────────────────────
@app.errorhandler(413)
@app.errorhandler(RequestEntityTooLarge)
def file_too_large(error):
    return render_template(
        'upload.html',
        msg="""<strong>Error: File is too large!</strong><br>
               Maximum allowed size is approximately 400 MB.<br><br>
               <small>Tip: Credit card fraud datasets are often very large.<br>
               Try using a smaller subset (e.g. first 100,000 rows) or splitting the file.</small>"""
    ), 413

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/about')
def about():
    return render_template("about.html")

@app.route('/home')
def home():
    if 'user_name' not in session:
        return redirect('/login')
    return render_template("home.html")

@csrf.exempt
@app.route('/register', methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form.get('name')
        email = request.form.get('email')
        password = request.form.get('password')

        try:
            executionquery(
                "INSERT INTO users (name, email, password) VALUES (%s, %s, %s)",
                (name, email, password)
            )
            return redirect("/login")

        except Exception as e:
            return render_template('register.html', message=f"Error: {str(e)}")

    return render_template('register.html')

@csrf.exempt
@app.route('/login', methods=["GET", "POST"])
def login():
    if request.method == "POST":
        # Directly take values (no validation)
        email = request.form.get('email')
        name = "User"  # default name (or fetch from DB if you want)

        # Set session directly (no password check)
        session['user_name'] = name
        session['user_email'] = email

        return redirect("/home")

    return render_template('login.html')

@csrf.exempt
@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if 'user_name' not in session:
        return redirect('/login')

    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('upload.html', msg="No file part in the request")

        file = request.files['file']
        if file.filename == '':
            return render_template('upload.html', msg="No file selected")

        if not file.filename.lower().endswith('.csv'):
            return render_template('upload.html', msg="Only .csv files are allowed")

        try:
            df = pd.read_csv(file, low_memory=False)

            if df.empty:
                return render_template('upload.html', msg="The uploaded CSV file is empty")

            total_rows = len(df)

            PREVIEW_LIMIT = 50
            preview_df = df.head(PREVIEW_LIMIT).copy()
            preview_df = preview_df.replace({np.nan: None})

            rows = preview_df.values.tolist()
            columns = df.columns.tolist()

            expected_cols = ['V3', 'V4', 'V6', 'V7', 'V9', 'V10', 'V11', 'V12', 'V14', 'V16', 'V17', 'Amount']
            missing = [col for col in expected_cols if col not in columns]
            extra_msg = ""
            if missing:
                extra_msg = f"<br><small style='color:#ffaa00;'>Warning: missing expected columns: {', '.join(missing)}</small>"

            msg_text = f"Dataset uploaded successfully!<br>Previewing first {PREVIEW_LIMIT} rows.{extra_msg}"

            return render_template(
                'upload.html',
                columns=columns,
                rows=rows,
                msg=msg_text,
                total_rows=total_rows,
                preview_limit=PREVIEW_LIMIT
            )

        except pd.errors.EmptyDataError:
            return render_template('upload.html', msg="The CSV file appears to be empty or malformed.")
        except Exception as e:
            import traceback
            traceback.print_exc()
            return render_template('upload.html', msg=f"Error processing file: {str(e)}")

    return render_template('upload.html')

# ────────────────────────────────────────────────
#                  MODEL METRICS
# ────────────────────────────────────────────────
@csrf.exempt
@app.route('/model', methods=['GET', 'POST'])
def model():
    msg = ""
    msg1 = ""

    model_metrics = {
        'FNN': {
            'accuracy': 0.8326264530317311,
            'f1_macro': 0.8326087322804592,
            'precision_macro': 0.8326193569276573,
            'recall_macro': 0.8326010772182619
        },
        'CNN': {
            'accuracy': 0.9149387370405277,
            'f1_macro': 0.9147157716572072,
            'precision_macro': 0.9181616961789376,
            'recall_macro': 0.9145542399228166
        },
        'RNN': {
            'accuracy': 0.832077,
            'f1_macro': 0.830940,
            'precision_macro': 0.839545,
            'recall_macro': 0.831409
        },
        'XGBoost': {
            'accuracy': 0.9620640904806786,
            'f1_macro': 0.9620318520440776,
            'precision_macro': 0.962969632446937,
            'recall_macro': 0.9618802983719721
        },
        'RandomForest': {
            'accuracy': 0.9900251335218347,
            'f1_macro': 0.9901113447013937,
            'precision_macro': 0.9903426791277259,
            'recall_macro': 0.9898801183247704
        }
    }

    readable_names = {
        'FNN': "Feedforward Neural Network (FNN)",
        'CNN': "1D Convolutional Neural Network (CNN)",
        'RNN': "Recurrent Neural Network (RNN - LSTM)",
        'XGBoost': "XGBoost Classifier",
        'RandomForest': "Random Forest Classifier"
    }

    if request.method == "POST":
        selected_model = request.form.get('algo')
        if selected_model in model_metrics:
            metrics = model_metrics[selected_model]
            model_name = readable_names[selected_model]
            msg = f"{model_name} Performance Metrics"
            msg1 = (
                f"Accuracy: {metrics['accuracy']:.6f}<br>"
                f"F1 Score (macro): {metrics['f1_macro']:.6f}<br>"
                f"Precision (macro): {metrics['precision_macro']:.6f}<br>"
                f"Recall (macro): {metrics['recall_macro']:.6f}"
            )
            if selected_model == 'RandomForest':
                msg1 += "<br><br><strong>Best overall performance – highest accuracy & F1 score!</strong>"
            elif selected_model in ['XGBoost', 'CNN']:
                msg1 += "<br><br><strong>Very strong performance – excellent balance!</strong>"
        else:
            msg = "Invalid model selection."
            msg1 = ""

        return render_template('model.html', msg=msg, msg1=msg1, models=model_metrics.keys())

    return render_template('model.html', models=model_metrics.keys())

# ────────────────────────────────────────────────
#        MODEL LOADING (cached - load once)
# ────────────────────────────────────────────────
loaded_models = {}

def get_model(model_key):
    if model_key not in loaded_models:
        model_files = {
            'FNN': 'fnn.joblib',
            'CNN': 'cnn.joblib',
            'RNN': 'rnn.joblib',
            'XGBoost': 'xgb.joblib',
            'RandomForest': 'randomforest.joblib'
        }
        path = model_files.get(model_key)
        if not path or not os.path.exists(path):
            print(f"Model file not found for key '{model_key}': {path}")
            loaded_models[model_key] = None
            raise FileNotFoundError(f"Model file not found: {path or 'missing key'}")

        try:
            print(f"Attempting to load model: {model_key} from {path}")
            loaded_models[model_key] = joblib.load(path)
            print(f"Successfully loaded model: {model_key}")
        except Exception as e:
            print(f"*** FAILED to load {model_key} from {path} ***")
            print(f"Exception type  : {type(e).__name__}")
            print(f"Exception detail: {str(e)}")
            import traceback
            traceback.print_exc()
            loaded_models[model_key] = None

    model = loaded_models.get(model_key)
    if model is None:
        raise RuntimeError(f"Model '{model_key}' failed to load earlier. Check server console / terminal logs.")
    return model

# ────────────────────────────────────────────────
#                   PREDICTION - DEFAULT: RANDOM FOREST
# ────────────────────────────────────────────────
@csrf.exempt
@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    prediction_value = None
    error = None
    # Changed default name to CNN
    selected_model_name = "1D Convolutional Neural Network (CNN)"

    readable_names = {
        'FNN': "Feedforward Neural Network (FNN)",
        'CNN': "1D Convolutional Neural Network (CNN)",
        'RNN': "Recurrent Neural Network (RNN - LSTM)",
        'XGBoost': "XGBoost Classifier",
        'RandomForest': "Random Forest Classifier"
    }

    if request.method == 'POST':
        try:
            # Explicitly force 'CNN' if you want it to be the only one, 
            # or use request.form.get('model', 'CNN') to allow selection
            model_choice = request.form.get('model', 'CNN')

            # Load model (cached)
            model = get_model(model_choice)
            selected_model_name = readable_names.get(model_choice, model_choice)

            # Input validation helper
            def get_float(name):
                value = request.form.get(name, '').strip()
                if not value:
                    raise ValueError(f"{name.upper()} cannot be empty")
                try:
                    return float(value)
                except ValueError:
                    raise ValueError(f"Invalid number for {name.upper()}: '{value}'")

            features = ['v3', 'v4', 'v6', 'v7', 'v9', 'v10', 'v11', 'v12', 'v14', 'v16', 'v17', 'amount']
            values = [get_float(f) for f in features]

            # 1. Start with 2D array: shape (1, 12)
            input_data = np.array([values]) 

            # 2. Reshape for CNN: shape (1, 12, 1)
            # This is the critical step for cnn.joblib
            if model_choice == 'CNN':
                input_data = input_data.reshape((input_data.shape[0], input_data.shape[1], 1))
            
            # 3. Predict
            pred = model.predict(input_data)

            # 4. Handle CNN output format
            # CNNs often return probabilities [[0.98]] or multiclass [[0.1, 0.9]]
            if pred.ndim > 1 and pred.shape[1] > 1:
                # If multiclass (softmax)
                final_pred = int(np.argmax(pred, axis=1)[0])
            else:
                # If binary probability (sigmoid) or single class
                final_pred = int((pred.flatten()[0] > 0.5))

            class_map = {0: 'not fraud', 1: 'fraud'}
            prediction_value = class_map.get(final_pred, 'unknown')

        except ValueError as ve:
            error = f"Input error: {str(ve)}"
        except FileNotFoundError as fe:
            error = f"Model file missing: {str(fe)}"
        except Exception as e:
            error = f"Prediction failed: {str(e)}"
            import traceback
            traceback.print_exc()

    return render_template(
        'prediction.html',
        prediction_value=prediction_value,
        error=error,
        selected_model=selected_model_name
    )

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)