(function() { // avoid variables ending up in the global scope

  var selectedAreaLat;
  var selectedAreaLon;
  document.getElementById("storeSelects").style.display = "none"
  document.getElementById("showStores").disabled = true;
  document.getElementById("showMovementsMilan").disabled = true;

  document.getElementById("showMap").addEventListener('click', createMapItaly,false);

  document.getElementById("showMapMilan").addEventListener('click', createMapsMilan,false);

  document.getElementById("showTransportMilan").addEventListener('click', showTransportMilan,false);

  document.getElementById("showMovementsMilan").addEventListener('click', movementsMilan,false);

  document.getElementById("showStores").addEventListener('click', showStores,false);

  document.addEventListener('visibilitychange', function logData() {
  
    if (document.visibilityState === 'hidden') {
      navigator.sendBeacon("./DeleteSession");
    }
  });

  // create map in div 'map'
  function createMapItaly(){
  var completed = true;
  var params = "";
    //See if all info set for clusters
  if(document.getElementById("ageGroup").value==null ||document.getElementById("ageGroup").value==""){
    document.getElementById("errorIT").innerHTML = "Please select an appropriate age group";
    completed = false;
  }
  else{
    params+= "&ageGroup="
    params+= document.getElementById("ageGroup").value
  }

  if(document.getElementById("Gender").value==null ||document.getElementById("Gender").value==""){
    document.getElementById("errorIT").innerHTML = "Please select an appropriate gender";
    completed = false;
  }
  else{
    params+= "&Gender="
    params+= document.getElementById("Gender").value
  }

  if(document.getElementById("maritalStatus").value==null ||document.getElementById("maritalStatus").value==""){
    document.getElementById("errorIT").innerHTML = "Please select an appropriate marital status";
    completed = false;
  }
  else{
    params+= "&maritalStatus="
    params+= document.getElementById("maritalStatus").value
  }
  
  if(document.getElementById("avgIncome").value==null ||document.getElementById("avgIncome").value==""){
    document.getElementById("errorIT").innerHTML = "Please insert an average income";
    completed = false;
  }
  else{
    params+= "&avgIncome="
    params+= document.getElementById("avgIncome").value
  }
  if (completed){
    document.getElementById("ItalyMap").innerHTML="";//Clean
    document.getElementById("italydescr").innerHTML="";
    var request = new XMLHttpRequest();
    request.onreadystatechange = function() {
      if (request.readyState == XMLHttpRequest.DONE) {
        switch (request.status) {
        case 200:
          var url = JSON.parse(request.responseText);
          document.getElementById("italydescr").innerHTML=url[0];
          document.getElementById("ItalyMap").innerHTML='<object type="text/html" data=' + url[1] + ' ></object>';
          break;
        case 204:
          document.getElementById("ItalyMap").innerHTML="No values found for selected cluster, please try with other specifics";
          break;
        default:
          document.getElementById("errorIT").innerHTML = "Unknown error";
        }
      }
    };//end callbackFunc
    request.open("POST", "./ItalyMap", true);
    request.setRequestHeader("Content-type", "application/x-www-form-urlencoded");
    request.send(params);
    var lding= document.createElement('img');
    lding.src = "static/Loading_icon.gif";
    document.getElementById("ItalyMap").appendChild(lding)
  }
}

  function showTransportMilan(){
    document.getElementById("errorMil").innerHTML = "";
    var request = new XMLHttpRequest();
    request.onreadystatechange = function() {
      if (request.readyState == XMLHttpRequest.DONE) {
        switch (request.status) {
        case 200:
          var url = request.responseText;
          document.getElementById("transportMilan").innerHTML='<object type="text/html" data=' + url + ' ></object>';
          break;
        default:
          document.getElementById("errorMil").innerHTML = "Unknown error";
        }
      }
    };//end callbackFunc
    request.open("GET", "./ConnectionMilan", true);
    request.setRequestHeader("Content-type", "application/x-www-form-urlencoded");
    request.send();
  }


  function createMapsMilan(){
    var completed = true;
    var params = "";
      //See if all info set for clusters
    if(document.getElementById("ageGroupCity").value==null ||document.getElementById("ageGroupCity").value==""){
      document.getElementById("errorMil").innerHTML = "Please select an appropriate age group";
      completed = false;
    }
    else{
      params+= "&ageGroup="
      params+= document.getElementById("ageGroupCity").value
    }

    if(document.getElementById("GenderCity").value==null ||document.getElementById("GenderCity").value==""){
      document.getElementById("errorMil").innerHTML = "Please select an appropriate gender";
      completed = false;
    }
    else{
      params+= "&Gender="
      params+= document.getElementById("GenderCity").value
    }

    if(document.getElementById("maritalStatusCity").value==null ||document.getElementById("maritalStatusCity").value==""){
      document.getElementById("errorMil").innerHTML = "Please select an appropriate marital status";
      completed = false;
    }
    else{
      params+= "&maritalStatus="
      params+= document.getElementById("maritalStatusCity").value
    }
    
    if(document.getElementById("incomeCity").value==null ||document.getElementById("incomeCity").value==""){
      document.getElementById("errorMil").innerHTML = "Please insert an average income";
      completed = false;
    }
    else{
      params+= "&avgIncome="
      params+= document.getElementById("incomeCity").value
    }

    if(document.getElementById("avgNumCompCity").value==null ||document.getElementById("avgNumCompCity").value==""){
      document.getElementById("errorMil").innerHTML = "Please insert the average number of components";
      completed = false;
    }
    else{
      params+= "&avgNumComp="
      params+= document.getElementById("avgNumCompCity").value
    }
    if (completed){
      document.getElementById("milanMuni").src = "";//Clean
      document.getElementById("milanNIL").src = "";
      document.getElementById( 'housePrices' ).setAttribute( 'src', '' );
      document.getElementById("capIncome").src = "";
      document.getElementById("errorMil").innerHTML ="";
      document.getElementById("nomilanMuni").innerHTML="";
      document.getElementById("nomilanNIL").innerHTML="";
      var request = new XMLHttpRequest();
      request.onreadystatechange = function() {
        if (request.readyState == XMLHttpRequest.DONE) {
          switch (request.status) {
          case 200:
            var url = JSON.parse(request.responseText);
            //Always at least one
            if (url.length < 2 ){ //No data for municipalities/NIL and income (only house prices)
              document.getElementById('housePrices').contentWindow.document.write('<object type="text/html" data=' + url[0] + ' ></object>');

              document.getElementById("nomilanMuni").innerHTML="No data for selected group";
              document.getElementById("nomilanNIL").innerHTML="No data for selected group";
            }
            else if (url.length < 3 ){ //No data for municipalities/NIL (house prices + incomes)
              document.getElementById('housePrices').contentWindow.document.write('<object type="text/html" data=' + url[0] + ' ></object>');
              document.getElementById("capIncome").src = url[1];

              document.getElementById("nomilanMuni").innerHTML="No data for selected group";
              document.getElementById("nomilanNIL").innerHTML="No data for selected group";
            }
            else{ //Have all
              document.getElementById("milanMuni").src=url[0];
              document.getElementById("milanNIL").src = url[1] ;
              document.getElementById('housePrices').contentWindow.document.write('<object type="text/html" data=' + url[2] + ' ></object>');
              document.getElementById("capIncome").src = url[3];
            }
            break;
          default:
            document.getElementById("errorMil").innerHTML = "Unknown error";
          }
        }
      };//end callbackFunc
      request.open("POST", "./MilanMaps", true);
      request.setRequestHeader("Content-type", "application/x-www-form-urlencoded");
      request.send(params);
      var lding= document.createElement('img');
      lding.src = "static/Loading_icon.gif";
      document.getElementById("milanMuni").appendChild(lding)
    }
  }
  document.getElementById('housePrices').addEventListener("load", function() {
    setTimeout(function(){

        var housePoints = document.getElementById('housePrices').contentWindow.document.body.firstChild.contentWindow.document.body.getElementsByClassName("plotly-graph-div js-plotly-plot")[0];
                housePoints.on('plotly_click', function(pt){
                    selectedAreaLat= pt.points[0].lat;
                    selectedAreaLon= pt.points[0].lon;
                    document.getElementById('savedLoc').innerHTML = "Location saved!";
                    document.getElementById("storeSelects").style.display = "block"
                    document.getElementById("showStores").disabled = false;
                    document.getElementById("showMovementsMilan").disabled = false;
                    document.getElementById("tapArea").style.display = "none"

                });
  }, 10000);
    
  }); 

  function movementsMilan(){
    //TODO target
    if (typeof selectedAreaLat !== 'undefined' && typeof selectedAreaLon !== 'undefined') {
      // the variable is defined
      var request = new XMLHttpRequest();
      var divMov = document.getElementById("movements");
      request.onreadystatechange = function() {
        if (request.readyState == XMLHttpRequest.DONE) {
          switch (request.status) {
          case 200:
            var listImgs = JSON.parse(request.responseText);
            divMov.innerHTML = '';//Clear all children
            //Add each image and text
            for (var i=0; i<listImgs.length; i++){
              if (listImgs[i].includes("png")){
                var imgToAdd = document.createElement('img');
                imgToAdd.src = listImgs[i];
                divMov.appendChild(imgToAdd);
              }
              else{ //Text explaining where
                var textToAdd = document.createTextNode(listImgs[i])
                divMov.appendChild(textToAdd);
              }
            }
            //document.getElementById("storesImg").src = request.responseText;
            break;
          default:
            divMov.innerHTML = '';//Clear all children
            document.getElementById("errorStores").innerHTML = "Unknown error";
          }
        }
      };//end callbackFunc
      request.open("POST", "./MovementsMilan", true);
      request.setRequestHeader("Content-type", "application/x-www-form-urlencoded");
      request.send("&position=" + selectedAreaLat + "$" +selectedAreaLon);
      document.getElementById("errorStores").innerHTML ="";
      var lding= document.createElement('img');
      lding.src = "static/Loading_icon.gif";
      divMov.appendChild(lding)
    }
    else{//No value set
      document.getElementById("errorStores").innerHTML = "Please select the interested area from the house prices map above";
    }
  }

  function showStores(){
    var completed = true;
    var params = "";

    if(document.getElementById("storeTypes").value==null ||document.getElementById("storeTypes").value==""){
      document.getElementById("errorStores").innerHTML = "Please select a store type";
      completed = false;
    }
    else{
      params+= "&storeType="
      params+= document.getElementById("storeTypes").value
    }
    if (typeof selectedAreaLat === 'undefined' && typeof selectedAreaLon === 'undefined') {
      document.getElementById("errorStores").innerHTML = "Please select the interested area from the house prices map above";
      completed = false;
    }
    else{
      params+= "&position=";
      params+= selectedAreaLat + "$";
      params+= selectedAreaLon;
    }
    if (completed){
      var divMov = document.getElementById("movements");
      document.getElementById("errorStores").innerHTML="";//Clean
      document.getElementById("storesImg").src = "";
      params+= "&storeNames="
      params+= document.getElementById("storeNames").value
      params+= "&brands="
      params+= document.getElementById("brands").value
      var request = new XMLHttpRequest();
      request.onreadystatechange = function() {
        if (request.readyState == XMLHttpRequest.DONE) {
          switch (request.status) {
          case 200:
            divMov.innerHTML = '';//Clear all children
            document.getElementById("storesImg").src = request.responseText;
            break;
          case 500:
            document.getElementById("errorStores").innerHTML = "Openstreetmap server error";
            break;
          default:
            divMov.innerHTML = '';//Clear all children
            document.getElementById("errorStores").innerHTML = "Unknown error";
          }
        }
      };//end callbackFunc
      request.open("POST", "./StoresMilan", true);
      request.setRequestHeader("Content-type", "application/x-www-form-urlencoded");
      request.send(params);
      document.getElementById("errorStores").innerHTML ="";
      var lding= document.createElement('img');
      lding.src = "static/Loading_icon.gif";
      divMov.appendChild(lding)
    }
  }
  })();