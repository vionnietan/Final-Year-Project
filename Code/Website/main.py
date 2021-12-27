# Import Libraries
import os
import io

import torch
import torch.nn as nn
from torchvision import transforms, models
from torchvision.transforms import transforms

from flask import Flask, redirect, url_for, render_template, request, session, flash, request
from flask_login import login_manager, login_user, login_required, logout_user, current_user, LoginManager
from flask_login.mixins import UserMixin
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from PIL import Image
from werkzeug.utils import secure_filename

# Creating a flask app
app = Flask(__name__)

# SOURCE: https://github.com/python-engineer/pytorch-examples & https://www.youtube.com/watch?v=bA7-DEtYCNM
# Database Environment
ENV = 'prod'

if ENV == 'dev':
    app.debug = True
    app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://postgres:post@localhost:5432/lexus'
else:
    app.debug = False
    app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://ivfhdyrrndcrfn:3826cbe8f164c64724fdb82e6f82da023dcd09e49e87b8f4abe68fbbb6df01ad@ec2-52-206-193-199.compute-1.amazonaws.com:5432/d7gmviuqv6dfph'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# SOURCE: https://towardsdatascience.com/build-a-web-application-for-predicting-apple-leaf-diseases-using-pytorch-and-flask-413f9fa9276a
# Load model
model = models.resnet50()
num_inftr = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(num_inftr, 256),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(256, 10),
    nn.LogSoftmax(dim=1)
)
model.load_state_dict(torch.load('model.pth',map_location=torch.device('cpu')))
model.eval()

imagenet_class_index = ['MSIMUT', 'MSS']

# SOURCE: https://towardsdatascience.com/build-a-web-application-for-predicting-apple-leaf-diseases-using-pytorch-and-flask-413f9fa9276a
# Pre-process image
def transform_image(image_bytes):
    """
    This function performs image transformations on an image
    :param image_bytes: an image
    :return: image_bytes that has been transformed as the following:
            1. resized to size 256
            2. cropped at the center with size 224
            3. converted into tensor format
            4. normalized with mean and standard deviation
    """
    my_transforms = transforms.Compose([
		transforms.Resize(256),
		transforms.CenterCrop(224),
		transforms.ToTensor(),
		transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
	])

    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0)

def predict(image_bytes):
    """
    This function predicts whether an image is classified as MSS/MSIMUT 
    :param image_bytes: an image
    :return: the prediction class (MSS/MSIMUT), the confidence of having that prediction class
    """
    tensor = transform_image(image_bytes=image_bytes)
    out = model.forward(tensor)
    _, index = torch.max(out, 1)
    percentage = nn.functional.softmax(out, dim=1)[0] * 100
    return imagenet_class_index[index], percentage[index[0]].item()

app.secret_key = "2021Group4"

db = SQLAlchemy(app)

login_manager = LoginManager()
login_manager.login_view = 'login'
login_manager.init_app(app)

@login_manager.user_loader
def load_user(id):
    return User.query.get(int(id))

# Database model for user authentication system
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(150), unique=True)
    password = db.Column(db.String(150))
    first_name = db.Column(db.String(150))
    vCancer = db.Column(db.String(150))
    vSymptoms = db.Column(db.String(150))
    vTreatment = db.Column(db.String(150))
    result = db.Column(db.String(150))

    def __init__(self, first_name, email, password, vCancer, vSymptoms, vTreatment, result):
        self.email = email
        self.password = password
        self.first_name = first_name
        self.vCancer = vCancer
        self.vSymptoms = vSymptoms
        self.vTreatment = vTreatment
        self.result = result
        
@app.route("/", methods = ['GET','POST'])
def home():
    """
    Route of homepage, display the homepage to the user and listen to GET and POST, after user submitted image, run through the predictive model
    @special condition: sometimes will produce result if the submitted images are unrelevant
    @expected output: output the result of predictive model
    @return: render the homepage HTML, if user submitted image, render the result html with the prediction result
    """
    error = None

    if request.method == "POST":        
        # when user submit image
        if request.form["submit"] == "submit":
            vCancer = request.form.get('vCancer')
            vSymptoms = request.form.get('vSymptoms')
            vTreatment = request.form.get('vTreatment')

            if current_user.is_authenticated:
                update_user = User.query.filter_by(email= current_user.email).first()
                update_user.vCancer = vCancer
                update_user.vSymptoms = vSymptoms
                update_user.vTreatment = vTreatment
                db.session.commit()

            #check if the post request has the file 
            if not request.files.get('file',None):
                return render_template("error_empty.html")
            file = request.files.get('file')
            
            #if wheter the submitted image are in jpg, jpeg and png format
            if ("." in file.filename and file.filename.rsplit(".", 1)[1].lower()) not in ["jpg","jpeg","png"]:
                return render_template("error.html")

            if not file:
                return
            
            try:
            #run the predictive model with the submitted image
                img_bytes = file.read()
                prediction_name, percentage = predict(img_bytes)

            except:
                return render_template("error_file.html")

            #add users responses to the database
            if current_user.is_authenticated:
                update_user = User.query.filter_by(email= current_user.email).first()
                update_user.result = percentage
                db.session.commit()

        return render_template("result.html", name= prediction_name, prediction = percentage)
    return render_template("index.html", user = current_user)

@app.route("/about/")
def about():
    """
    Route of about page, display the about page to the user 
    @return: render the about page HTML
    """
    return render_template("about.html")

@app.route("/help/")
def help():
    """
    Route of help page, display the help page to the user 
    @return: render the help page HTML
    """
    return render_template("help.html")

@app.route("/signup/", methods = ['GET', 'POST'])
def signup():
    """
    Route of signup, display the signup page to the user and listen to GET and POST
    Added data submitted by users into database
    @expected output: users data are saved in the database
    @return: render the signup HTML page
    """
    if request.method == 'POST':
        email = request.form.get('email')
        firstName = request.form.get('firstName')
        password1 = request.form.get('password1')
        password2 = request.form.get('password2')

        user = User.query.filter_by(email = email).first()
        if user:
            flash('Email already exists.', category= 'error')
        elif len(email) < 4:
            flash('Email must be greater than 3 characters.', category='error')
        elif len(firstName) < 2:
            flash('First Name must be greater than 1 character.', category='error')
        elif password1 != password2:
            flash('Password not matched', category='error')
        elif len(password1) < 7:
            flash('Password have to be more than 7 characters.', category='error')
        else:
            new_user = User(email=email, first_name=firstName, password=generate_password_hash(password1, method='sha256'), vCancer= "", vSymptoms="",vTreatment="",result ="")            
            db.session.add(new_user)
            db.session.commit()
            flash("Account created !", category='success')
            return redirect(url_for("login"))

    return render_template("signup.html", user = current_user)

@app.route("/login/", methods = ['GET', 'POST'])
def login():
    """
    Route of login, display the login page to the user and listen to GET and POST
    check user authentication when login
    @return: render the login HTML page
    """
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')

        user = User.query.filter_by(email = email).first()
        if user:
            if check_password_hash(user.password, password):
                flash('Logged in successfully!', category = 'success')
                login_user(user, remember = True)
                return redirect(url_for("home"))
            else:
                flash('Incorrect password.', category= 'error')
        else:
            flash('Email does not exist.', category = 'error')

    return render_template("login.html", user = current_user)
    
#For testing Purpose
@app.route("/view/")
def view():
    return render_template("view.html", values = User.query.all())
    
@app.route("/logout/", methods = ['GET', 'POST'])
@login_required
def logout():
    """
    Route for user to logout 
    @return: redirect to login HTML page
    """
    logout_user()
    return redirect(url_for("login"))


if __name__ == "__main__":
    db.create_all()
    app.run(debug=True) 
