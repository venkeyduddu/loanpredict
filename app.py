from flask import Flask, request, render_template
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('index.html')


@app.route('/result', methods=['POST', 'GET'])
def get_delay():
    prediction = 0
    if request.method == 'POST':
        result = request.form

        name = result['user_name']
        Gender = result['gender']
        married = result['married']
        Dependents = result['Dependents']
        Education = result['Education']
        Self_Employed = result['Self_Employed']
        credit_history = result['credit_history']
        Total_Income = float(result['Total_Income'])
        LoanAmount = float(result['LoanAmount'])
        Loan_Amount_Term = float(result['Loan_Amount_Term'])
        Property_Area = result['Property_Area']

        def label_encode(x):
            li = []
            if x == 'yes':
                li.append(1)
            elif x == 'no':
                li.append(0)
            else:
                print('enter yes or no')
            return li

        def encode(li, x):
            for i in range(0, len(li)):
                if li[i] == x:
                    li[i] = 1
                else:
                    li[i] = 0
            return li

        a = label_encode(credit_history)
        c = encode(['yes', 'no'], Gender)
        d = encode(['yes', 'no'], married)
        f = encode(['yes', 'no'], Education)
        g = encode(['yes', 'no'], Self_Employed)
        e = encode([0, 1, 2, 3], int(Dependents))
        h = encode(['Urban', 'Semi_Urban', 'Rural'], Property_Area)
        b = np.log(int(LoanAmount))
        i = np.log(int(Total_Income))
        j = int(LoanAmount) / int(Loan_Amount_Term)
        k = int(Total_Income)
        l = k - (j * 1000)

        fl = a + [b] + c + d + e + f + g + h + [k] + [i] + [j] + [l]
        df = pd.DataFrame([fl])

        pkl_file = open('RF_model.pkl', 'rb')
        model = pickle.load(pkl_file)
        p = model.predict(df)

        if p == [0]:
            prediction = name + "  Sorry we can not process your application at this time"
        elif p == [1]:
            prediction = name + "  Congratulations you are eligible for this Loan"
        else:
            prediction = name + "  Check your details"

    return render_template('result.html', prediction=prediction)


if __name__ == '__main__':
    app.debug = True
    app.run()
