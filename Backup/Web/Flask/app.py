import random
import shutil
import string
import time
from flask import Flask, render_template,request, jsonify, session 
from utility import utility
import os

app = Flask(__name__,template_folder= "templates")
app.secret_key = os.environ.get('SECRET_KEY', 'HG79Ppu206BXznkw')
u = utility(app.static_folder)
app.debug = True
if __name__ == "__main__":
    app.run()

def setUid():
    uid = ''.join(random.choices(string.ascii_uppercase + string.digits, k = 16))
    session['uid'] = uid
    try:
        os.makedirs(os.path.join(app.static_folder,'Maps/'+uid))
    except:
        print("Already dir")

@app.route("/")
def main(): 
    print("MAIN")
    if 'uid' not in session:
        setUid()
    return render_template('HomePage.html')



@app.route("/ItalyMap",methods = ['POST', 'GET'])
def italyMap():
    ageGroup = request.form.get('ageGroup', type = str)
    gender = request.form.get('Gender', type = str)
    maritalStatus = request.form.get('maritalStatus', type = str)
    income = request.form.get('avgIncome', type = int)
    if not [x for x in (ageGroup, gender, maritalStatus, income) if x is None]:
        if 'uid' not in session:
            setUid()
        url = u.plotItalyCluster(ageGroup,gender,maritalStatus,income,session['uid'])
        if url is not None:
            return jsonify(url)
        else:
            return ('', 204)
    else:
        return ('', 204)

@app.route("/MilanMaps",methods = ['POST', 'GET'])
def milanMaps():
    ageGroup = request.form.get('ageGroup', type = str)
    gender = request.form.get('Gender', type = str)
    maritalStatus = request.form.get('maritalStatus', type = str)
    income = request.form.get('avgIncome', type = str)
    numComp = request.form.get('avgNumComp', type = str)
    urls = []
    if 'uid' not in session:
        setUid()
    if not [x for x in (ageGroup, gender, maritalStatus, numComp) if x is None]:
        urls.append(u.plotMunicipi(ageGroup,gender,maritalStatus,numComp,session['uid']))
        urls.append(u.plotNIL(ageGroup,gender,maritalStatus,numComp,session['uid']))
    urls.append(u.plotHousePrices())
    if income is not None:
        urls.append(u.plotIncomes(income,session['uid']))
    return jsonify([i for i in urls if i])

@app.route("/ConnectionMilan")
def connectionMilan():
    return u.plotConnections()

@app.route("/MovementsMilan", methods = ['POST', 'GET'])#TODO
def movementsMilan():
    position = u.getMovPosition(request.form.get('position', type = str))#TODO
    if position is not None:
        if 'uid' not in session:
            setUid()
        urls = u.plotMovements(position,session['uid'])
        return jsonify([i for i in urls if i])
    else:
        return ('', 204)

@app.route("/StoresMilan",methods = ['POST', 'GET'])
def storesMilan():
    position = u.getMovPosition(request.form.get('position', type = str))#TODO
    storeType = request.form.get('storeType', type = str)
    storeName = request.form.get('storeNames', type = str)
    storeBrand = request.form.get('brands', type = str)
    if not [x for x in (position, storeType) if x is None]:
        if 'uid' not in session:
            setUid()
        url = u.plotStores(position,storeType,storeName,storeBrand,session['uid'])
        if url is not None:
            return url
        else:
            return ('', 204)
    else:
         return ('', 204)

@app.route("/DeleteSession",methods = ['POST', 'GET'])
def deleteSession():
    print("DELETE")
    uid = session.pop('uid', None)
    if uid is not None:
        try:
            shutil.rmtree(os.path.join(app.static_folder,'Maps/'+uid))
        except:
            print("No dir found")
    return ('',200)