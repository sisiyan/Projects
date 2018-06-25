const LOCALHOST = "http://localhost:12301";
const STAGE = ""
const TRADE_ROUTE = LOCALHOST + "/trade" + STAGE;
const REPORT_ENDPOINT = TRADE_ROUTE + "/auction_report";

let reportNA = {
    templateRow: $('.template-row'),
    handleError: err => console.error(err),
    handleUserData: res => {
        console.log(res);
        let rows = [];
        for (i in res.data.items) {
            console.log(reportNA.templateRow)
            d = res.data.items[i]
            let cloneObj = reportNA.templateRow.clone();
            cloneObj.find('.id').text(d.itemId);
            cloneObj.find('.item_name').text(d.itemName);
            cloneObj.find('.sale_price').text(d.winnerPrice);
            cloneObj.find('.winner').text(d.userName);
            cloneObj.find('.auction_ended').text(d.endTime);
            cloneObj.removeClass('hidden');
            rows.push(cloneObj);
        }
        reportNA.templateRow.after(rows);
    },
};

((r) => {
    console.log('a')
    $.ajax({
        type: "GET",
        url: REPORT_ENDPOINT,
    })
        .done((res) => {
            r.handleUserData(res);
        })
        .fail(r.handleError);
})(reportNA);