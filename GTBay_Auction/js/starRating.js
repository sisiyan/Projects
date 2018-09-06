const STAR_TOTAL = 5;
const LOCALHOST = "http://localhost:12301";
const STAGE = ""
const TRADE_ROUTE = LOCALHOST + "/trade" + STAGE;
const LIST_RATING_ENDPOINT = TRADE_ROUTE + "/list_ratings";
const POST_RATING_ENDPOINT = TRADE_ROUTE + "/rating";
const DELETE_RATING_ENDPOINT = TRADE_ROUTE + "/delete_rating";
const ITEM_ID = sessionStorage.getItem("itemID");
const CURRENT_USER_ID = sessionStorage.getItem("currentUserID") || 1;

let starRatingNS = {

    numOfStars: 0,
    myRatingId: -1,

    handleRatingRes: (res) => {
        let tempReviewObj = $('#template-review');

        // remove all reviews and stars that has been added to this page before
        starRatingNS.clearAllStars();
        starRatingNS.clearAllReviews();
        starRatingNS.clearComment();
        // generate new review elements, add elements to page
        $(tempReviewObj).after(starRatingNS.processRatingRes(tempReviewObj, res.data.ratingList, res.data.averageRating));
    },

    bindStarClick: () => {
        for (let i = 1; i <= STAR_TOTAL; ++i) {
            $(`#${i}`).click((e) => {
                starRatingNS.clearAllStars();
                starRatingNS.numOfStars = parseInt($(e.target).attr('id'));
                for (let j = 1; j <= starRatingNS.numOfStars; ++j) {
                    $(`#${j}`).addClass('checked');
                }
            });
        }
    },

    clearAllStars: () => {
        starRatingNS.numOfStars = 0;
        $('#dynamic-stars').children().removeClass('checked');
    },

    clearAllReviews: () => {
        $('.removable').remove();
    },

    clearComment: () => {
        $("#comment-content").val('');
    },

    processRatingRes: (templateObj, ratingList, avgRating) => {
        let reviews = [];

        if(avgRating >= 0){
            document.getElementById("avg-rating").innerHTML = avgRating + " stars";
        }
        else {
            document.getElementById("avg-rating").innerText = 0;
        }

        ratingList.forEach((rating, index) => {
            let cloneObj = templateObj.clone();
            cloneObj.removeClass('hidden');
            cloneObj.addClass('removable');
            if (rating.userId == CURRENT_USER_ID) {
                starRatingNS.myRatingId = rating.ratingId;
                starRatingNS.handleCommentedBefore(cloneObj);
            }
            cloneObj.find('.rated-by').text(rating.username);
            cloneObj.find('.rated-date').text(rating.rateTime);
            cloneObj.find('.comment').text(rating.comments);
            starRatingNS.changeStaticStars(cloneObj.find('.static-stars'), rating.stars);
            reviews.push(cloneObj);
        });

        return reviews;
    },

    changeStaticStars: (starObj, num) => {
        for (let i = 0; i < num; ++i) {
            starObj.children('.fa-star').eq(i).addClass('checked');
        }
    },

    handleError: err => console.error(err),

    handleCommentedBefore: (cloneObj) => {
        cloneObj.addClass('user-rating');
        // I'm gonna just hide the rating stars and comment box to prohibit user from
        // rating multiple times, there is NO endpoint protection for this. If a user
        // remove the hidden class from them, they can still submit a comment.
        $('#rating-form').addClass('hidden');
        let deleteRatingLink = cloneObj.find('.delete-rating');
        // Let user delete this comment, NO ENDPOINT PROTECTION!
        deleteRatingLink.removeClass('hidden');
        // Bind the link with ajax delete request
        deleteRatingLink.click((e) => {
            e.preventDefault();
            $.ajax({
                type: "POST",
                url: DELETE_RATING_ENDPOINT + "?rating_id=" + starRatingNS.myRatingId,
            })
                .done((res) => {
                    if (res.code == 200) {
                        $('.user-rating').remove();
                        $('#rating-form').removeClass('hidden');
                    } else {
                        alert("delete failed");
                    }
                })
                .fail(starRatingNS.handleError)
        })
    }
};

((s) => {
    s.bindStarClick();
    $('#view-rating').click((e) => {
        document.getElementById("item-id").innerText = sessionStorage.getItem("itemID");
        document.getElementById("item-name").innerText = sessionStorage.getItem("itemName")
        e.preventDefault();
        // do the ajax
        $.ajax({
            type: "GET",
            url: LIST_RATING_ENDPOINT,
            data: {
                item_id: ITEM_ID
            },
        })
            .done(s.handleRatingRes)
            .fail(s.handleError);
    });

    $('#rating-form').submit((e) => {
        e.preventDefault();

        $.ajax({
            type: "POST",
            url: POST_RATING_ENDPOINT,
            data: {
                item_id: ITEM_ID,
                user_id: CURRENT_USER_ID,
                comments: $("#comment-content").val(),
                stars: starRatingNS.numOfStars
            },
        })
            .done(s.handleRatingRes)
            .fail(s.handleError);
    });

})(starRatingNS);