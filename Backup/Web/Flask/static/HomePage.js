(function() { // avoid variables ending up in the global scope

  // var selectedAreaLat;
  // var selectedAreaLon;
  var selectedArea;
  document.getElementById("storeSelects").style.display = "none"
  document.getElementById("showStores").disabled = true;
  document.getElementById("showMovementsMilan").disabled = true;
  document.getElementById("housePrices").style.visibility = "hidden" //To load it in advance
  document.getElementById("transportMilan").style.display = "none"
  document.getElementById("tableMunicipi").style.display = "none"
  document.getElementById("tableNil").style.display = "none"
  document.getElementById("tableCap").style.display = "none"
  document.getElementById("tableItaly").style.display = "none"

  document.getElementById("showMap").addEventListener('click', createMapItaly,false);

  document.getElementById("showMapMilan").addEventListener('click', createMapsMilan,false);

  document.getElementById("showTransportMilan").addEventListener('click', showTransportMilan,false);

  document.getElementById("showMovementsMilan").addEventListener('click', movementsMilan,false);

  document.getElementById("showStores").addEventListener('click', showStores,false);

  document.getElementById("MilanZonesObj").addEventListener('load',clickableMap,false)

  document.addEventListener('visibilitychange', function logData() {

    if (document.visibilityState === 'hidden') {
      navigator.sendBeacon("./DeleteSession");
    }
  },false);

  document.getElementById("icon-bar").addEventListener("click", function Show() {
    document.getElementById("nav-lists").classList.add("_Menus-show");
  },false);
  function Hide(){
    document.getElementById("nav-lists").classList.remove("_Menus-show");
  }
  var item = document.getElementsByClassName("hideElements");
  for (var i = 0; i < item.length; i++) {
    item[i].addEventListener("click", Hide,false);
  }

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
    document.getElementById("ItalyMap").src = "";//Clean
    document.getElementById("italydescr").innerHTML="";
    document.getElementById("clusterMap").innerHTML="";
    document.getElementById("tableItaly").style.display = "none"
    document.getElementById("errorIT").innerHTML = "";
    var request = new XMLHttpRequest();
    request.onreadystatechange = function() {
      if (request.readyState == XMLHttpRequest.DONE) {
        document.getElementById("ItalyMap").src = "";
        switch (request.status) {
        case 200:
          var url = JSON.parse(request.responseText);
          document.getElementById("tableItaly").style.display = "block";
          document.getElementById("ItalyMap").src = url[0];
          insertIntoTableItaly("tableItalyBody",url[1],url[2],url[3]);
          break;
        case 204:
          document.getElementById("errorIT").innerHTML="No values found for selected cluster, please try with other specifics";
          break;
        default:
          document.getElementById("errorIT").innerHTML = "Unknown error";
        }
      }
    };//end callbackFunc
    request.open("POST", "./ItalyMap", true);
    request.setRequestHeader("Content-type", "application/x-www-form-urlencoded");
    request.send(params);
    document.getElementById("ItalyMap").src = "static/Loading_icon.gif";
  }
  }

  function showMapClusterChosen(event){
    event.preventDefault();

    document.getElementById("italydescr").innerHTML="";
    document.getElementById("clusterMap").innerHTML="";
    document.getElementById("errorIT").innerHTML = "";
    var request = new XMLHttpRequest();
    request.onreadystatechange = function() {
      if (request.readyState == XMLHttpRequest.DONE) {
        switch (request.status) {
        case 200:
          var url = JSON.parse(request.responseText);
          document.getElementById("italydescr").innerHTML = url[0];
          document.getElementById("clusterMap").innerHTML='<object type="text/html" data=' + url[1] + ' ></object>';
          break;
        default:
          document.getElementById("errorIT").innerHTML = "Unknown error";
          document.getElementById("clusterMap").innerHTML = "";
        }
      }
    };//end callbackFunc
    request.open("POST", "./ItalyCluster", true);
    request.setRequestHeader("Content-type", "application/x-www-form-urlencoded");
    params = "&id=" + event.path[0].id;
    params += "&region=" + event.path[0].name;
    request.send(params);
    var lding= document.createElement('img');
    lding.src = "static/Loading_icon.gif";
    document.getElementById("clusterMap").appendChild(lding)
  }
  function showTransportMilan(){
    document.getElementById("transportMilan").style.display = "block"
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
      document.getElementById("capIncome").src = "";
      document.getElementById("errorMil").innerHTML ="";
      document.getElementById("nomilanMuni").innerHTML="";
      document.getElementById("nomilanNIL").innerHTML="";
      document.getElementById("housePrices").style.visibility = "hidden"
      document.getElementById("tableMunicipi").style.display = "none"
      document.getElementById("tableNil").style.display = "none"
      document.getElementById("tableCap").style.display = "none"

      var request = new XMLHttpRequest();
      request.onreadystatechange = function() {
        if (request.readyState == XMLHttpRequest.DONE) {
          document.getElementById("milanMuni").src = "";
          switch (request.status) {
          case 200:
            var url = JSON.parse(request.responseText);
            if (url.length < 2 ){ //No data for municipalities/NIL (house prices + incomes)
              document.getElementById("tableCap").style.display = "block"
              document.getElementById("capIncome").src = url[0][0];
              insertIntoTable("tableCapBody",url[0][1],url[0][2]);

              document.getElementById("nomilanMuni").innerHTML="No data for selected group";
              document.getElementById("nomilanNIL").innerHTML="No data for selected group";
            }
            else{ //Have all
              document.getElementById("tableMunicipi").style.display = "block"
              document.getElementById("tableNil").style.display = "block"
              document.getElementById("tableCap").style.display = "block"

              document.getElementById("milanMuni").src=url[0][0];
              insertIntoTable("tableMunicipiBody",url[0][1],url[0][2]);
              document.getElementById("milanNIL").src = url[1][0] ;
              insertIntoTable("tableNilBody",url[1][1],url[1][2]);

              document.getElementById("capIncome").src = url[2][0];
              insertIntoTable("tableCapBody",url[2][1],url[2][2]);

            }
              // var housePoints = document.getElementById('housePrices').firstElementChild.contentDocument.body.getElementsByClassName("plotly-graph-div js-plotly-plot")[0];
              // housePoints.on('plotly_click', function(pt){
              //     selectedAreaLat= pt.points[0].lat;
              //     selectedAreaLon= pt.points[0].lon;
              //     document.getElementById("movements").innerHTML = "";
              //     document.getElementById("errorStores").innerHTML = "";
              //     document.getElementById("storesImg").src = "";
              //     document.getElementById('savedLoc').innerHTML = "Location saved!";
              //     document.getElementById("storeSelects").style.display = "block"
              //     document.getElementById("showStores").disabled = false;
              //     document.getElementById("showMovementsMilan").disabled = false;
              //     document.getElementById("tapArea").style.display = "none"
              // });
              // document.getElementById("housePrices").style.visibility = "visible"
            break;
          default:
            document.getElementById("errorMil").innerHTML = "Unknown error";
          }
        }
      };//end callbackFunc
      request.open("POST", "./MilanMaps", true);
      request.setRequestHeader("Content-type", "application/x-www-form-urlencoded");
      request.send(params);
      document.getElementById("milanMuni").src = "static/Loading_icon.gif";
    }
  }

  function movementsMilan(){
    if (typeof selectedArea !== 'undefined') {
      // the variable is defined
      document.getElementById("errorStores").innerHTML ="";
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
                let divImg = document.createElement('div');
                divImg.setAttribute("class","layoutColumn");
                let imgToAdd = document.createElement('img');
                imgToAdd.src = listImgs[i];
                imgToAdd.setAttribute("class","image");
                divMov.appendChild(divImg);
                divImg.appendChild(imgToAdd);
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
      request.send("&position=" + selectedArea);
      let imgToAdd = document.createElement('img');
      imgToAdd.src = "static/Loading_icon.gif";
      divMov.appendChild(imgToAdd);
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
    if (typeof selectedArea === 'undefined') {
      document.getElementById("errorStores").innerHTML = "Please select the interested area from the house prices map above";
      completed = false;
    }
    else{
      params+= "&position=";
      params+= selectedArea;
    }
    if (completed){
      document.getElementById("errorStores").innerHTML="";//Clean
      document.getElementById("storesImg").src = "";
      params+= "&storeNames="
      params+= document.getElementById("storeNames").value
      params+= "&brands="
      params+= document.getElementById("brands").value
      var request = new XMLHttpRequest();
      request.onreadystatechange = function() {
        if (request.readyState == XMLHttpRequest.DONE) {
          document.getElementById("storesImg").src = "";
          switch (request.status) {
          case 200:
            document.getElementById("storesImg").src = request.responseText;
            break;
          case 500:
            document.getElementById("errorStores").innerHTML = "Openstreetmap server error";
            break;
          case 204:
            document.getElementById("errorStores").innerHTML = "No data of selected type in the selected area";
            break;
          default:
            document.getElementById("errorStores").innerHTML = "Unknown error";
          }
        }
      };//end callbackFunc
      request.open("POST", "./StoresMilan", true);
      request.setRequestHeader("Content-type", "application/x-www-form-urlencoded");
      request.send(params);
      document.getElementById("storesImg").src = "static/Loading_icon.gif";
    }
  }
  function clickableMap(){
    var housePoints = document.getElementById('MilanZones').firstElementChild.contentDocument.body.getElementsByClassName("plotly-graph-div js-plotly-plot")[0];
    housePoints.on('plotly_click', function(pt){
        selectedArea= pt.points[0].hovertext;
        document.getElementById("movements").innerHTML = "";
        document.getElementById("errorStores").innerHTML = "";
        document.getElementById("storesImg").src = "";
        document.getElementById("storeSelects").style.display = "block"
        document.getElementById("showStores").disabled = false;
        document.getElementById("showMovementsMilan").disabled = false;
    });
  }

  function insertIntoTable(idTable, data1, data2) {

    let elmtTable = document.getElementById(idTable);
    let tableRows = elmtTable.getElementsByTagName('tr');
    let rowCount = tableRows.length;

    for (let x=rowCount-1; x>=0; x--) {
      elmtTable.removeChild(tableRows[x]);
    }

    for (let i =  data1.length-1; i >=0; i--) {
      let row = elmtTable.insertRow(0);
      let cell1 = row.insertCell(0);
      let cell2 = row.insertCell(1);
      cell1.innerHTML = data1[i];
      cell2.innerHTML = data2[i];
    }
  }
  function insertIntoTableItaly(idTable, data1, data2,data3){
    let elmtTable = document.getElementById(idTable);
    let tableRows = elmtTable.getElementsByTagName('tr');
    let rowCount = tableRows.length;

    for (let x=rowCount-1; x>=0; x--) {
      elmtTable.removeChild(tableRows[x]);
    }

    for (let i =  data1.length-1; i >=0; i--) {
      let row = elmtTable.insertRow(0);
      let cell1 = row.insertCell(0);
      let cell2 = row.insertCell(1);
      let cell3 = row.insertCell(2);

      cell1.innerHTML = data1[i];
      cell2.innerHTML = data2[i];
      // Create anchor element.
      let a = document.createElement('a');
      a.setAttribute('href',"#");
      a.setAttribute('id',data3[i]);
      a.setAttribute('name',data2[i]);
      a.innerHTML = "Show on map";
      cell3.appendChild(a);

    }
    let anchors = elmtTable.getElementsByTagName("a");

    for (let i = 0; i < anchors.length ; i++) {
      anchors[i].addEventListener("click",showMapClusterChosen,false);
    }
  }

  })();