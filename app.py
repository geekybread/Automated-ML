from flask import Flask, request, render_template,send_file
import pandas as pd
import os
from classifier import Classifier
from regressor import Regressor
from cleaner import Cleaner

basedir = os.path.abspath(os.path.dirname(__file__))

UPLOAD_FOLDER = os.path.join(basedir,"uploads")

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        use_test_file = request.form.get('use_test_file') == 'yes'
        choice = request.form['choice'] 
        clean = request.form['cleaned']
        
        if use_test_file:
            # Use pre-existing test file
            sample_type = request.form['sample_type']
            if sample_type == 'Classification':
                test_file_path = os.path.join(basedir, 'sample_data', 'classification_sample.csv')
            else:
                test_file_path = os.path.join(basedir, 'sample_data', 'regression_sample.csv')
            try:
                data = pd.read_csv(test_file_path)
            except Exception as e:
                return render_template('upload.html', error=f"Failed to read test file: {e}")
        else:
            uploaded_file = request.files['file']
            if uploaded_file.filename == '':
                return render_template('upload.html', error="Please upload a file or select the test dataset.", request=request)
            
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_file.filename)
            uploaded_file.save(file_path)
            try:
                data = pd.read_csv(file_path)
            except Exception as e:
                return render_template('upload.html', error=f"Failed to read uploaded file: {e}")
            
            os.remove(file_path)

        # Cleaning logic
        if clean == 'Yes':
            cleaned_data = Cleaner(data)
        else:
            cleaned_data = data
        
        os.makedirs('cleaned', exist_ok=True)
        cleaned_data.to_csv('cleaned/processed.csv', index=False)

        choice = request.form['choice']
        if choice == 'Regression':
            return render_template('regressor.html')
        elif choice == 'Classification':
            return render_template('classifier.html')

    return render_template('upload.html')

@app.route('/classifier', methods=['GET', 'POST'])
def classifier():
    if request.method == 'POST':
        clf_selected = request.form['clf_choice']
        df = pd.read_csv('cleaned/processed.csv')
        clf = Classifier(df, clf_selected)
        global best_model
        study, best_model = clf.classify()
        best_parameters = study.best_params
        best_value = round(study.best_trial.value,4)*100

        return render_template('result.html', best_parameters=best_parameters, best_value=best_value, best_model=best_model)

    return render_template('classifier.html')

@app.route('/regressor', methods=['GET', 'POST'])
def regressor():
    if request.method == 'POST':
        reg_selected = request.form['reg_choice']
        df = pd.read_csv('cleaned/processed.csv')
        reg = Regressor(df, reg_selected)
        global best_model
        study,best_model = reg.regress()
        best_parameters = study.best_params
        best_value = round(study.best_trial.value,4)*100

        return render_template('result.html', best_parameters=best_parameters, best_value=best_value)
    return render_template('regressor.html')


@app.route('/downloads', methods=['GET','POST'])
def download():
    return send_file(best_model, download_name='model.pickle', as_attachment=True)


if __name__ == '__main__':
    app.run(debug=True)