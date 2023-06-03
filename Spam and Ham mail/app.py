from flask import Flask,request,render_template
import pickle as pk

app=Flask(__name__)

@app.route('/',methods=['POST','GET'])
def main():
    if request.method=='POST':
        model,cv=pk.load(open(r'C:\Users\Saimum Adil Khan\OneDrive\Desktop\Flask\Spam and Ham mail\spam_mail.pkl','rb'))
        text=request.form.get('text')
        if text:
            text_array=cv.transform([text])
            prediction=model.predict(text_array)
            if prediction[0]==1:
                result= 'Alert! This is a Spam Mail'
            else:
                result='The Mail has No Problem. It is a Ham Mail'
        else:
            result='Plz, Enter your Email First'
    else:
        result=''
    return render_template('home.html',output=result)

if __name__=='__main__':
    app.run(debug=True)