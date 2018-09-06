const LOCALHOST = "http://localhost:12301";
const STAGE = ""
const TRADE_ROUTE = LOCALHOST + "/trade" + STAGE;
const REPORT_ENDPOINT = TRADE_ROUTE + "/category_report";

let reportNA = {
   	templateRow: $('.template-row'),
    handleError: err => console.error(err),
    handleCategoryData: data => {
    	console.log(data);
    	let rows = [];
    	for (i in data.data.categoryReports) {
    		console.log(reportNA.templateRow)
    		d = data.data.categoryReports[i]
    		let cloneObj = reportNA.templateRow.clone();
    		cloneObj.find('.cat').text(d.categoryName);
    		cloneObj.find('.tot').text(d.count);
    		cloneObj.find('.min').text(d.minPrice);
    		cloneObj.find('.max').text(d.maxPrice);
    		cloneObj.find('.avg').text(d.averagePrice);
    		cloneObj.removeClass('hidden');
    		rows.push(cloneObj);
    	}
    	reportNA.templateRow.after(rows);
    },
};

((r) => {
    $.ajax({
            type: "GET",
            url: REPORT_ENDPOINT,
        })
        .done((res) => {
        	r.handleCategoryData(res);
        })
        .fail(r.handleError);
})(reportNA);