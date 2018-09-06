const LOCALHOST = "http://localhost:12301";
const STAGE = ""
const TRADE_ROUTE = LOCALHOST + "/user" + STAGE;
const REPORT_ENDPOINT = TRADE_ROUTE + "/user_report";

let reportNA = {
   	templateRow: $('.template-row'),
    handleError: err => console.error(err),
    handleUserData: res => {
    	console.log(res);
    	let rows = [];
    	for (i in res.data.userReports) {
            console.log(reportNA.templateRow)
            d = res.data.userReports[i]
            let cloneObj = reportNA.templateRow.clone();
            cloneObj.find('.username').text(d.username);
            cloneObj.find('.listed').text(d.listed);
            cloneObj.find('.sold').text(d.sold);
            cloneObj.find('.purchased').text(d.purchased);
            cloneObj.find('.rated').text(d.rated);
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