import pickle
import pandas as pd
from flask import Flask, request, jsonify, render_template, url_for, redirect
from sklearn.tree import DecisionTreeClassifier
import numpy

app = Flask(__name__)


@app.route('/')
def hello_world():  # put application's code here
    return redirect(url_for('gender'))


@app.route('/gender', methods=['POST', 'GET'])
def gender():
    if request.method == 'GET':
        return render_template('gender-form.html')
        # If the request is POST, get the json data from the request
    if request.method == 'POST':
        data: dict = request.get_json()
        data_col: dict = {}
        column_names = {
            'gender': ' Gender',
            'age': ' Age',
            'height_cm': ' Height (cm)',
            'weight_kg': ' Weight (kg)',
            'occupation': ' Occupation',
            'education_level': ' Education Level',
            'marital_status': ' Marital Status',
            'income_usd': ' Income (USD)',
            'favorite_color': ' Favorite Color'
        }

        for key in data.keys():
            if key == 'occupation' or key == 'favorite_color':
                data_col[f"{column_names[key]}_{data[key]}"] = 1
            else:
                data_col[column_names[key]] = data[key]

        print(data_col)
        '''
        test case:
        
        {' Age': '10', ' Height (cm)': '133', ' Weight (kg)': '40', ' Occupation_Software Engineer': 1,
         ' Education Level': "Bachelor's Degree", ' Marital Status': 'Married', ' Income (USD)': '10000',
         ' Favorite Color_Blue': 1}
        '''

        columns = [' Age', ' Height (cm)', ' Weight (kg)', ' Education Level',
                   ' Marital Status', ' Income (USD)', ' Occupation_Accountant',
                   ' Occupation_Analyst', ' Occupation_Architect',
                   ' Occupation_Business Analyst', ' Occupation_Business Consultant',
                   ' Occupation_CEO', ' Occupation_Doctor', ' Occupation_Engineer',
                   ' Occupation_Graphic Designer', ' Occupation_IT Manager',
                   ' Occupation_Lawyer', ' Occupation_Marketing Specialist',
                   ' Occupation_Nurse', ' Occupation_Project Manager',
                   ' Occupation_Sales Representative', ' Occupation_Software Developer',
                   ' Occupation_Software Engineer', ' Occupation_Teacher',
                   ' Occupation_Writer', ' Favorite Color_Black', ' Favorite Color_Blue',
                   ' Favorite Color_Green', ' Favorite Color_Grey',
                   ' Favorite Color_Orange', ' Favorite Color_Pink',
                   ' Favorite Color_Purple', ' Favorite Color_Red',
                   ' Favorite Color_Yellow']

        df = pd.DataFrame(columns=columns)
        df = df.append(data_col, ignore_index=True)
        df = df.fillna(int(0))

        df[' Education Level'] = df[' Education Level'].replace(
            ["Bachelor's Degree", "Master's Degree", 'Doctorate Degree',
             "Associate's Degree"],
            [0, 1, 2, 3])

        df[' Marital Status'] = df[' Marital Status'].replace(
            ['Married', 'Single', 'Divorced', 'Widowed'],
            [0, 1, 2, 3])

        print(list(df.columns))

        print(len(df.columns))

        model: DecisionTreeClassifier = pickle.load(open(app.root_path + '/static/dt_model.pkl', 'rb'))

        result: numpy.ndarray = model.predict(df)
        label = result.astype(numpy.int32)
        label = int(label)
        print(type(label))
        return jsonify({'gender': label})


if __name__ == '__main__':
    app.run()
