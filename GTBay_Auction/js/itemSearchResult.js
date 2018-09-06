var itemList = JSON.parse(sessionStorage.getItem("itemList"));

console.log(itemList);

/**
 * Function to create the table of search results
 * @param items: the items returned from backend
 */
function createTable(items) {
    var table = "<tr>"+
        "<th> ID </th>"+
        "<th> Item Name </th>"+
        "<th> Current Bid </th>"+
        "<th> High Bidder </th>"+
        "<th> Get It Now Price </th>"+
        "<th> Auction Ends </th>"+
        "</tr>"
    //document.write('<table border =1 id = "tableR">' + table + '</table>');
    //document.getElementById("searchResults").innerHTML = table;
    //table = '<table border =1 id = "tableR">' + table + '</table>';

    document.getElementById("searchResults").innerHTML = table;

    var tableR = document.getElementById("searchResults");

    console.log(table);

    var rows = items.length;
    var cols = 6;

    function test(_id) {
        console.log(items[_id].itemId);
        sessionStorage.setItem("itemID", items[_id].itemId);
        sessionStorage.setItem("itemName", items[_id].itemName);
        console.log(sessionStorage.getItem("itemID"));
    }

    for(var r= 0; r < rows; r++) {
        var tRow = document.createElement("tr");
        for (var c=0; c < cols; c++) {
            var tD = document.createElement("td");
            if (c==0) {
                tD.innerHTML = items[r].itemId;
            }
            else if (c==1) {
                tD.innerHTML = items[r].itemName;
                tD.id = r;

                tD.addEventListener('click', function(event) {

                    console.log(event.target.id);
                    //sessionStorage.setItem("itemID", username);
                    test(event.target.id);
                    window.open('http://localhost:8888/Phase3/html/itemDescription.html', "_self");
                });
            }
            else if (c==2) {
                tD.innerHTML = items[r].winnerPrice;
            }
            else if (c==3) {
                tD.innerHTML = items[r].userName;
            }
            else if (c==4) {
                tD.innerHTML = items[r].getNowPrice;
            }
            else {
                tD.innerHTML = items[r].endTime;
            }
            tRow.appendChild(tD);
        }
        tableR.appendChild(tRow);
    }

}

$(document).ready(function () {
    createTable(itemList);
});