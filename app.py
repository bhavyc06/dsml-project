from flask import Flask, render_template, request
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import numpy as np
from werkzeug.utils import secure_filename
import os
import base64
from skimage.io import imread
from skimage.transform import resize
import shutil
model=tf.keras.models.load_model('custom_model.h5')
app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def clean_files():
    del_static=next(os.walk('static'))[-1]
    for i in range(len(del_static)):
        os.remove('static/'+del_static[i])
    del_static=next(os.walk('uploads'))[-1]
    for i in range(len(del_static)):
        os.remove('uploads/'+del_static[i])
# Set allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Function to check if a file has an allowed extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def convert_base64(fig):
    buffer = BytesIO()
    fig.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    graph = base64.b64encode(image_png)
    graph = graph.decode('utf-8')
    return graph
@app.route('/')
def dashboard():
    clean_files()
    return render_template('dashboard.html')

@app.route('/Display')
def display():
    csv_filepath = 'data.csv'
    df = pd.read_csv(csv_filepath)
    df=df.drop('Unnamed: 0',axis=1)
    df=df.sort_values(by=['Year'], ascending = True)
    figline, axline = plt.subplots()
    years=sorted(dict(df['Year'].value_counts()).keys())
    year_dict=dict(df['Year'].value_counts())
    plot=[]
    for i in range(len(years)):
        plot.append(year_dict[years[i]])
    axline.plot(years,plot) #LINE CHART
    axline.set_title('Samples Collected over the Years')

    figpie, axpie = plt.subplots()
    axpie.pie(dict(df['class_id'].value_counts()).values(),labels = dict(df['class_id'].value_counts()).keys(),autopct = '%1.1f%%') #PIE CHART
    axpie.set_title('Samples distribution in each Class')

    fighist1,axhist1=plt.subplots()
    ages=sorted(dict(df['Age'].value_counts()).keys())
    age_dict=dict(df['Age'].value_counts())
    plot=[]
    for i in range(len(ages)):
        plot.append(age_dict[ages[i]])
    axhist1.bar(ages,plot) #HISTOGRAM CHART 1
    axhist1.set_xlabel('Age')
    axhist1.set_ylabel('X-Ray Samples')
    axhist1.set_title('Samples based on Each Age Group')

    fighist2,axhist2=plt.subplots()
    axhist2.bar(dict(df['Gender'].value_counts()).keys(),dict(df['Gender'].value_counts()).values(),color=['yellow','red'])
    axhist2.set_xlabel('Gender')
    axhist2.set_ylabel('X-Ray Samples')
    axhist2.set_title('Cases Distribution for Each Gender')


    return render_template('chart.html', graphline=convert_base64(figline),graphpie=convert_base64(figpie),graphhist1=convert_base64(fighist1),graphhist2=convert_base64(fighist2))

@app.route('/display_csv')
def display_csv():
    csv_filepath = 'data.csv'
    df = pd.read_csv(csv_filepath)
    df=df.drop('Unnamed: 0',axis=1)
    df=df.sort_values(by=['Year'], ascending = True)
    records = 20
    page = int(request.args.get('page', 1))

    start = (page - 1) * records
    end = start + records

    small_df = df.iloc[start:end]

    return render_template('display_csv.html', table=small_df.to_html(index=False), page=page, total_pages=(len(df) + records - 1) // records)

@app.route('/upload_image')
def upload_image():
    return render_template('upload_image.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    # Check if the post request has the file part
    if 'file' not in request.files:
        return render_template('upload_image_failed.html')

    file = request.files['file']

    # If the user does not select a file, the browser submits an empty file without a filename
    if file.filename == '':
        return render_template('upload_image_failed.html')
    user_age = request.form.get('text', '')
    if int(user_age)>100:
        return render_template('upload_image_failed.html')
    user_gender = request.form.get('gender', '')
    # If the file is allowed and is not empty
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
    
        name=next(os.walk('uploads'))[-1][0]
        img=imread('uploads/'+name)
        shutil.copyfile('uploads/'+name, 'samples/'+name)
        shutil.copyfile('uploads/'+name, 'static/'+name)
        os.remove('uploads/'+name)
        img=resize(img,(256,256,3),mode='constant',preserve_range=True)/255
        img=[img]
        img=np.asarray(img)
        pred=np.argmax(model.predict(img))
        if pred==1:
            prediction='Bacterial Pneumonia'
        elif pred==0:
            prediction='Normal'
        else:
            prediction='Viral Pneumonia'
        csv_filepath = 'data.csv'
        df = pd.read_csv(csv_filepath)
        df=df.drop('Unnamed: 0',axis=1)
        df=df.sort_values(by=['Year'], ascending = True)
        df.loc[len(df.index)] = [name,prediction,user_gender,int(user_age),2023]
        df=df.sort_values(by=['Year'], ascending = True)
        df.to_csv('data.csv')

        return render_template('final_result.html',user_age=user_age,user_gender=user_gender,prediction=prediction,filename=name)

    return render_template('upload_image_failed.html')

if __name__ == '__main__':
    app.run(debug=True)