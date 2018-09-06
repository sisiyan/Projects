/**
 * Define some global variables for different functions
 * @type {string | null}
 */
var itemId = sessionStorage.getItem("itemID");
var userId = sessionStorage.getItem("currentUserID");
//var userId = 1;
var getItPrice = 0;
var description;
var startPrice;
/**
 * Helper function to show the item information
 * @param item
 */
function showItemInfo(item) {
    //console.log(JSON.parse(sessionStorage.getItem("category")));
    //console.log(JSON.parse(sessionStorage.getItem("condition")));

    var category_list = JSON.parse(sessionStorage.getItem("category"));
    var condition_list = JSON.parse(sessionStorage.getItem("condition"));

    var categoryName;
    for (var i =0; i < category_list.length; i++) {
        if (item.categoryId == category_list[i].value) {
            categoryName = category_list[i].name;
        }
    }

    var conditionName;
    for (var i =0; i < condition_list.length; i++) {
        if (item.conditionId == condition_list[i].value) {
            conditionName = condition_list[i].name;
        }
    }

    var returnAcp;
    if (item.returnAccepted === 0) {returnAcp = "No";}
    else if (item.returnAccepted === 1) {returnAcp = "Yes";}

    var getItNow_disp;
    if (item.getNowPrice != null) {
        getItNow_disp = "$" + item.getNowPrice;
    }
    else {
        getItNow_disp = "--";
    }

    //description = item.description;
    var table = '<tr>' + '<td>' + 'Item ID' + '</td>' + '<td>' + item.itemId + '</td>' + '</tr>'
        + '<td>' + 'Item Name' + '</td>' + '<td>' + item.itemName + '</td>' + '</tr>'
        + '<td>' + 'Description' + '</td>' + '<td id = "desp">' + item.description + '</td>' + '</tr>'
        + '<td>' + '' + '</td>' + '<td>' + '' + '</td>' + '</tr>'
        + '<td>' + 'Category' + '</td>' + '<td>' + categoryName + '</td>' + '</tr>'
        + '<td>' + 'Condition' + '</td>' + '<td>' + conditionName + '</td>' + '</tr>'
        + '<td>' + 'Return Accepted' + '</td>' + '<td>' + returnAcp + '</td>' + '</tr>'
        + '<td>' + 'Get It Now Price ' + '</td>' + '<td>' + getItNow_disp + '</td>' + '</tr>'
        + '<td>' + 'Start Price' + '</td>' + '<td>' +"$" + item.startBidPrice + '</td>' + '</tr>'
        + '<td>' + 'Auction Ends' + '</td>' + '<td>' + item.endTime + '</td>' + '</tr>'

    document.getElementById("itemTable").innerHTML = table;

    if (userId == item.userId) {
        console.log("currentUserID " + userId);
        var btn = document.createElement("BUTTON");
        btn.id = "editBtn";
        btn.addEventListener('click', function(event){
            editDsp();
        });

        var t = document.createTextNode("Edit Description");
        btn.appendChild(t);
        var loc = document.getElementById("editDescription");
        loc.appendChild(btn);
    }

    if (item.getNowPrice != null) {
        var btn = document.createElement("BUTTON");
        btn.id = "getIt";
        btn.addEventListener('click', function(event){
            getItNow();
        });

        var t = document.createTextNode("Get It Now");
        btn.appendChild(t);
        var loc = document.getElementById("getNowbtn");
        loc.appendChild(btn);
    }
}

/**
 * Helper function to show the bid history for the current item
 * @param item
 */

function showBids(bidList) {
    var table = "<tr>"+
        "<th> Bid Amount </th>"+
        "<th> Time of Bid </th>"+
        "<th> Username </th>"+
        "</tr>"

    //document.write('<table class = "table2" id = "bidTable">' + table + '</table>');
    document.getElementById("bidTable").innerHTML = table;

    var bidTable = document.getElementById("bidTable");


    var rows = Math.min(bidList.length, 4);
    var cols = 3;

    for(var r= 0; r < rows; r++) {
        var tRow = document.createElement("tr");
        for (var c=0; c < cols; c++) {
            var tD = document.createElement("td");
            if (c == 0) {
                tD.innerHTML = "$" +bidList[r].amount;
            }
            else if (c ==1) {
                tD.innerHTML = bidList[r].bidTime;
            }
            else {
                tD.innerHTML = bidList[r].username;
            }
            tRow.appendChild(tD);
        }
        bidTable.appendChild(tRow);
    }

}

/**
 * Helper function to display the reminding of valid minimum bid amount for the user
 * @param bidList of the item from the back-end
 * @returns the calculated minimum bid amount a use can input; if the auction ends, it will tell the user
 */
function minAmount(bidList) {
    var maxBid = 0;
    var minBidAmount;
    for (var i = 0; i < bidList.length; i++) {
        if (bidList[i].amount > maxBid) {
            maxBid = bidList[i].amount;
        }
    }
    if (getItPrice >0 && maxBid >= getItPrice) {
        minBidAmount = "Auction Ends";
    }
    else if (maxBid == 0 || maxBid == undefined) {
        minBidAmount = "(minimum bid $" + startPrice + ")";
    }
    else {
        minBidAmount = "(minimum bid $" + (maxBid + 1) + ")";
    }
    return minBidAmount;
}

/**
 * Helper function for the edit description function
 */
function editDsp() {
    //var txt;
    var newDsp = prompt("Input new description:", "");
    if (newDsp == null || newDsp == "") {
        alert("Not Changed!");
    } else {

        $(document).ready(function () {
            $.post("http://localhost:12301/trade/update/description",
                {
                    item_id: itemId,
                    description: newDsp
                },
                function (result) {
                    if(result.code === 200) {
                        alert("Description updated!");
                        document.getElementById("desp").innerHTML = newDsp;
                    }
                    else {
                        alert("Update error!");
                    }
                });
        });
    }
    //document.getElementById("demo").innerHTML = txt;
}

/**
 * Code to load the item information and bid history
 */
$(document).ready(function () {
    $.get("http://localhost:12301/trade/detail",
        {
            item_id: itemId
        },
        function (result) {

            var item = result.data.item;
            var bidList = result.data.bidList;
            console.log("currentUserID: " + userId);
            showItemInfo(item);
            showBids(bidList);
            getItPrice = item.getNowPrice;
            startPrice = item.startBidPrice;

            document.getElementById("minBid").innerHTML = minAmount(bidList);
        });
});


/***
 * If the Get It Now button is clicked, this function to process the Get It Now request
 */

function getItNow() {
    $(document).ready(function () {
        $.post("http://localhost:12301/trade/bid/add",
            {
                item_id: itemId, user_id: userId, amount: getItPrice
            },
            function (result) {
                //console.log(result);
                if (result.code == 200) {
                    alert("Bid success!");
                    return false;
                }
                else {
                    alert(result.msg);
                }

            });
    });
}

/***
 * If the bid button is clicked, this function to process the bid request
 */

function bid() {
    var bidAmount = document.getElementById('yourBid').value;
    if (getItPrice != null && bidAmount > getItPrice) {
        alert("Are you sure? Bid is higher than Get It Now Price!");
    }

    $(document).ready(function () {
    $.post("http://localhost:12301/trade/bid/add",
        {
            item_id: itemId, user_id: userId, amount: bidAmount
        },
        function (result) {
            //console.log(result);
            if (result.code == 200) {
                alert("Bid success!");
                return false;
            }
            else {
                alert(result.msg);
            }

        });
});
}

/**
 * If the cancel button is clicked, open the item_search page.
 */
function backToSearch() {
    window.open('http://localhost:8888/Phase3/html/itemSearchResult.html', "_self");
}
