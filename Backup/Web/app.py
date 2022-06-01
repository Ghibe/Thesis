import base64
from tkinter import Image
from turtle import pos
from flask import Flask, make_response, render_template,request, jsonify, send_file, send_from_directory
#import pandas as pd
#import numpy as np 
from utility import utility
import os
from os import path

app = Flask(__name__,template_folder= "templates")
u = utility(app.static_folder)
app.debug = True

if __name__ == "__main__":
    app.run()

@app.route("/")
def main():  
    return render_template('HomePage.html')

#FOr debug
@app.after_request
def add_header(response):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    response.headers['X-UA-Compatible'] = 'IE=Edge,chrome=1'
    response.headers['Cache-Control'] = 'public, max-age=0'
    return response

@app.route("/ItalyMap",methods = ['POST', 'GET'])
def italyMap():#TODO
    ageGroup = request.form.get('ageGroup', type = str)
    gender = request.form.get('Gender', type = str)
    maritalStatus = request.form.get('maritalStatus', type = str)
    income = request.form.get('avgIncome', type = int)
    if not [x for x in (ageGroup, gender, maritalStatus, income) if x is None]:
        url = u.plotItalyCluster(ageGroup,gender,maritalStatus,income)
        if url is not None:
            return "AAA"
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
    if not [x for x in (ageGroup, gender, maritalStatus, numComp) if x is None]:
        urls.append(u.plotMunicipi(ageGroup,gender,maritalStatus,numComp))
        urls.append(u.plotNIL(ageGroup,gender,maritalStatus,numComp))
    urls.append(u.plotHousePrices())
    if income is not None:
        urls.append(u.plotIncomes(income))
    return jsonify([i for i in urls if i])

@app.route("/ConnectionMilan")
def connectionMilan():
    return u.plotConnections()

@app.route("/MovementsMilan", methods = ['POST', 'GET'])#TODO
def movementsMilan():
    position = request.args.get('position', type = str)
    position = "AA"#TODO
    if position is not None:
        urls = u.plotMovements(position)
        return jsonify([i for i in urls if i])
    else:
        return ('', 204)

@app.route("/StoresMilan",methods = ['POST', 'GET'])
def storesMilan():
    position = request.form.get('position', type = str)#TODO
    storeType = request.form.get('storeType', type = str)
    storeName = request.form.get('storeNames', type = str)
    storeBrand = request.form.get('brands', type = str)
    position = "AA"#TODO
    if not [x for x in (position, storeType) if x is None]:
        url = u.plotStores(position,storeType,storeName,storeBrand)
        if url is not None:
            return url#app.send_static_file( url)#TODO
            #            return send_from_directory(app.static_folder, url, mimetype='image/png')
        else:
            return ('', 204)
    else:
         return ('', 204)