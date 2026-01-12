import numpy as np  # Import numpy for numerical operations and array handling
import pandas as pd  # Import pandas for data manipulation and CSV reading
from sklearn.preprocessing import StandardScaler  # Import StandardScaler for feature scaling
import tensorflow as tf  # Import TensorFlow for loading and using the trained model
from flask import Flask, request, render_template  # Import Flask components for web app

dataset = pd.read_csv('dataset/upi_fraud_dataset.csv', index_col=0)  # Load the UPI fraud dataset from CSV file, using first column as index

x = dataset.iloc[:, :10].values  # Extract features (first 10 columns) as numpy array
y = dataset.iloc[:, 10].values  # Extract target (11th column) as numpy array

scaler = StandardScaler()  # Initialize StandardScaler for feature normalization
scaler.fit_transform(x)  # Fit and transform the features using the scaler (though typically fit on train, here it's the whole data)

model = tf.keras.models.load_model('model/project_model1.h5')  # Load the pre-trained TensorFlow model from HDF5 file

app = Flask(__name__)  # Create a Flask application instance

@app.route('/')  # Define route for root URL
@app.route('/first')  # Define alternative route for '/first'
def first():  # Function to handle the first page
    return render_template('first.html')  # Render the first.html template

@app.route('/login')  # Define route for login page
def login():  # Function to handle login page
    return render_template('login.html')  # Render the login.html template

def home():  # Function to handle home page (not routed)
    return render_template('home.html')  # Render the home.html template

@app.route('/upload')  # Define route for upload page
def upload():  # Function to handle upload page
    return render_template('upload.html')  # Render the upload.html template

@app.route('/preview', methods=["POST"])  # Define route for preview page, accepting POST requests
def preview():  # Function to handle file preview
    if request.method == 'POST':  # Check if the request method is POST
        dataset = request.files['datasetfile']  # Get the uploaded file from the form
        df = pd.read_csv(dataset, encoding='unicode_escape')  # Read the CSV file into a DataFrame
        df.set_index('Id', inplace=True)  # Set 'Id' column as the index
        return render_template("preview.html", df_view=df)  # Render preview.html with the DataFrame

@app.route('/prediction1', methods=['GET'])  # Define route for prediction page, accepting GET requests
def prediction1():  # Function to handle prediction input page
    return render_template('index.html')  # Render the index.html template

@app.route('/chart')  # Define route for chart page
def chart():  # Function to handle chart page
    return render_template('chart.html')  # Render the chart.html template

@app.route('/detect', methods=['POST'])  # Define route for detection, accepting POST requests
def detect():  # Function to handle fraud detection
    trans_datetime = pd.to_datetime(request.form.get("trans_datetime"))  # Get transaction datetime from form and convert to datetime
    v1 = trans_datetime.hour  # Extract hour from datetime
    v2 = trans_datetime.day  # Extract day from datetime
    v3 = trans_datetime.month  # Extract month from datetime
    v4 = trans_datetime.year  # Extract year from datetime
    v5 = int(request.form.get("category"))  # Get category from form as integer
    v6 = float(request.form.get("card_number"))  # Get card number from form as float
    dob = pd.to_datetime(request.form.get("dob"))  # Get date of birth from form and convert to datetime
    
    # Calculate difference in days and convert to years
    days_diff = (trans_datetime - dob) / np.timedelta64(1, 'D')  # Calculate days difference between transaction and DOB
    v7 = np.floor(days_diff / 365.25)  # Convert days to years (approximate)
    
    v8 = float(request.form.get("trans_amount"))  # Get transaction amount from form as float
    v9 = int(request.form.get("state"))  # Get state from form as integer
    v10 = int(request.form.get("zip"))  # Get zip code from form as integer
    x_test = np.array([v1, v2, v3, v4, v5, v6, v7, v8, v9, v10])  # Create feature array for prediction
    y_pred = model.predict(scaler.transform([x_test]))  # Scale the test data and predict using the model
    if y_pred[0][0] <= 0.5:  # Check if prediction probability is <= 0.5
        result = "VALID TRANSACTION"  # Classify as valid
    else:  # Otherwise
        result = "FRAUD TRANSACTION"  # Classify as fraud
    return render_template('result.html', OUTPUT='{}'.format(result))  # Render result.html with the prediction

if __name__ == "__main__":  # Check if the script is run directly
    app.run()  # Run the Flask application
