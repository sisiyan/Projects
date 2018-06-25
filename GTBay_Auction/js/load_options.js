function addOption(selectbox,text,value)
{
    var optn = document.createElement("OPTION");
    optn.text = text;
    optn.value = value;
    selectbox.options.add(optn);
}


var xhttp = new XMLHttpRequest();

//var data=xhttp.responseText;
//var jsonResponse = JSON.parse(data);

//console.log(jsonResponse["Data"]);
xhttp.onreadystatechange=function() {
    if (this.readyState == 4 && this.status == 200) {
        var myObj = JSON.parse(this.responseText);
        var category = myObj.data;
        //console.log(this.responseText);
        //sessionStorage.setItem("category", category);
        //console.log(category.options[0].name);
        sessionStorage.setItem("category", JSON.stringify(category.options));

        addOption(document.getElementById("category_list"), "All category", 1000);
        for(var i = 0 ; i < category.options.length; i++) {
            addOption(document.getElementById("category_list"), category.options[i].name, category.options[i].value);
        }

        //console.log(JSON.parse(sessionStorage.getItem("category")));
    }
};
xhttp.open("GET", "http://localhost:12301/trade/option/list?type=CATEGORY", true);
xhttp.send();

var xhttp2 = new XMLHttpRequest();
xhttp2.onreadystatechange=function() {
    if (this.readyState == 4 && this.status == 200) {
        var myObj = JSON.parse(this.responseText);
        var condition = myObj.data;
        //console.log(this.responseText);
        sessionStorage.setItem("condition", JSON.stringify(condition.options));

        addOption(document.getElementById("condition_list"), "All conditions", 6);
        for(var i = 0 ; i < condition.options.length; i++) {
            addOption(document.getElementById("condition_list"), condition.options[i].name, condition.options[i].value);

        }

        //console.log(JSON.parse(sessionStorage.getItem("condition")));
    }
};
xhttp2.open("GET", "http://localhost:12301/trade/option/list?type=CONDITION", true);
xhttp2.send();