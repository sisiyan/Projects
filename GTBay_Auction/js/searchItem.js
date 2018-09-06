/**
 * Helper function to create the table of search results
 * @param items: the items returned from backend

function createTable(items) {
    var table = "<tr>"+
        "<th> ID </th>"+
        "<th> Item Name </th>"+
        "<th> Current Bid </th>"+
        "<th> High Bidder </th>"+
        "<th> Get It Now Price </th>"+
        "<th> Auction Ends </th>"+
        "</tr>"
    document.write('<table border =1 id = "tableR">' + table + '</table>');
    var tableR = document.getElementById("tableR");
    var rows = items.length;
    var cols = 6;

    function test(_id) {
        console.log(items[_id].itemId);
        sessionStorage.setItem("itemID", items[_id].itemId);
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
                tD.id = r
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
*/


/**
 * If the Search Button is clicked, check whether the inputs are valid. If valid, submit form.
 */
$('#searchItem').click((e) => {
    //console.log(username);
    //console.log(document.getElementById("password").value);
    e.preventDefault();

    var keyword = document.forms["searchForm"]["keyword"].value;

    var category = document.getElementById("category_list");
    var categoryID = category.options[category.selectedIndex].value;
    var condition = document.getElementById("condition_list");
    var conditionID = condition.options[condition.selectedIndex].value;

    var minimum_sale = document.forms["searchForm"]["min_price"].value;
    var maximum_sale = document.forms["searchForm"]["max_price"].value;

    if (sessionStorage.getItem("currentUser") == null) {
        alert("You have to login first!");
    }
    if (keyword == "") {
        keyword = "";
    }
    if (minimum_sale == "") {
        minimum_sale = 0;
        console.log(minimum_sale);
    }

    else if (isNaN(minimum_sale)) {
        alert("Minimum sale price must be a number!");
        return false;
    }

    else if (parseFloat(minimum_sale) < 0) {
        alert("Minimum sale price must be zero or a positive number!");
        return false;
    }

    if (maximum_sale == "") {
        maximum_sale = 1000000;
        console.log(maximum_sale);
    }

    else if (isNaN(maximum_sale)) {
        alert("Maximum sale price must be a number!");
        return false;
    }

    else if (parseFloat(maximum_sale) < parseFloat(minimum_sale)) {
        alert("Maximum sale price must be equal to or higher than minimum sale price!");
        return false;
    }

    //console.log(categoryID);
    //console.log(conditionID);
    $(document).ready(function () {
        console.log(keyword);
        console.log(categoryID);
        console.log(minimum_sale);
        console.log(maximum_sale);
        console.log(conditionID);

        $.get("http://localhost:12301/trade/search",
            {
                keyword: keyword, category_id: categoryID, min_price: minimum_sale,
                max_price: maximum_sale, condition_id: conditionID
            },
            function (result) {

                var items = result.data.item_list;

                console.log(items);

                sessionStorage.setItem("itemList", JSON.stringify(items));
                window.open('http://localhost:8888/Phase3/html/itemSearchResult.html', "_self");
                //createTable(items);

            });
    });

})

