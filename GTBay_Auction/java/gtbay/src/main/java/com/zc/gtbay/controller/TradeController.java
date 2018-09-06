package com.zc.gtbay.controller;

import com.zc.gtbay.service.TradeService;
import com.zc.gtbay.vo.*;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;

import java.math.BigDecimal;
import java.sql.Timestamp;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.*;

@CrossOrigin("http://localhost:8888")
@RestController
@RequestMapping(value = "/trade")
public class TradeController {
    @Autowired
    TradeService tradeService;
    @RequestMapping(value = "/create",method = RequestMethod.POST)
    public Result create(@RequestParam(value = "item_name") String itemName,
                         @RequestParam(value = "description") String description,
                         @RequestParam(value = "category_id") int categoryId,
                         @RequestParam(value = "condition_id") int conditionId,
                         @RequestParam(value = "start_bid_price")BigDecimal startBidPrice,
                         @RequestParam(value = "minimum_sale_price") BigDecimal minimumSalePrice,
                         @RequestParam(value = "auction_length") int auction_length,
                         @RequestParam(value = "get_now_price") BigDecimal getNowPrice,
                         @RequestParam(value = "return_accepted") int returnAccepted,
                         @RequestParam(value = "user_id") int userId){

        Result result = new Result();
        Item item = new Item();
        item.setItemName(itemName);
        item.setDescription(description);
        item.setCategoryId(categoryId);
        item.setConditionId(conditionId);
        item.setStartBidPrice(startBidPrice);
        item.setMinimumSalePrice(minimumSalePrice);
        item.setAuction_length(auction_length);
        item.setGetNowPrice(getNowPrice);
        item.setReturnAccepted(returnAccepted);
        item.setUserId(userId);
        item.setAcutionStartTime(new Timestamp(System.currentTimeMillis()));
        int row = tradeService.insertItem(item);
        if(row > 0){
            result.setCode(200);
        }else{
            result.setCode(400);
            result.setMsg("internal database error");
        }




        return result;
    }





    @RequestMapping(value="/option/list", method = RequestMethod.GET)
    public Result queryOptionList(@RequestParam(value = "type")OptionEnum optionEnum){
        Result result = new Result();
        List<Option> optionList = tradeService.queryByType(optionEnum);
        Map<String, Object> data = new HashMap<>();
        data.put("options", optionList);
        result.setData(data);
        result.setCode(200);
        return result;
    }

    @RequestMapping(value = "detail", method = RequestMethod.GET)
    public Result queryDetail(@RequestParam(value = "item_id") int itemId){
        Result result = new Result();
        Item item = tradeService.queryById(itemId);
        Map<String,Object>  data = new HashMap<>();
        data.put("item", item);
        List<Bid> bidList = tradeService.queryBidByItemId(itemId);
        data.put("bidList", bidList);
        result.setData(data);
        return result;

    }



    @RequestMapping(value = "bid/add", method = RequestMethod.POST)
    public Result addBid(@RequestParam(value = "item_id") int itemId,
                         @RequestParam(value = "user_id") int userId,
                         @RequestParam(value = "amount") BigDecimal amount){
        Result result = new Result();
        Bid bid = new Bid();
        bid.setItemId(itemId);
        bid.setAmount(amount);
        bid.setUserId(userId);
        BidState bidState = tradeService.getState(bid);
        if(bidState == BidState.OK){
            tradeService.insertBid(bid);
            result.setCode(200);
        }else{
            result.setCode(400);
            result.setMsg(bidState.getMsg());
        }
        return result;
    }


    @RequestMapping(value="/search", method = RequestMethod.GET)
    public Result search(@RequestParam(value = "keyword") String keyword,
                         @RequestParam(value = "category_id") int categoryId,
                         @RequestParam(value = "min_price") BigDecimal minPrice,
                         @RequestParam(value = "max_price") BigDecimal maxPirce,
                         @RequestParam(value = "condition_id") int conditionId){
        Result result = new Result();
        List<Item> itemList = tradeService.search(keyword,categoryId, minPrice,maxPirce,conditionId);
        Map<String, Object> data = new HashMap<>();
        data.put("item_list", itemList);
        result.setCode(200);
        result.setData(data);
        return result;

    }

    @RequestMapping(value="/update/description", method = RequestMethod.POST)
    public Result updateDescription(@RequestParam(value = "item_id") int itemId,
                                    @RequestParam(value = "") String description){
        Result result = new Result();
        int row = tradeService.updateDescription(itemId, description);
        if(row > 0){
            result.setCode(200);
        }else{
            result.setCode(400);
        }
        return result;
    }


    @RequestMapping(value="/rating", method = RequestMethod.POST)
    public Result addRating(@RequestParam(value = "item_id") int itemId,
                            @RequestParam(value = "user_id") int userId,
                            @RequestParam(value = "comments") String comments,
                            @RequestParam(value = "stars") int stars){
        Result result = new Result();
        Rating rating = new Rating();
        // TODO: need to generate a rating id here and insert it into db
        rating.setComments(comments);
        rating.setItemId(itemId);
        rating.setUserId(userId);
        rating.setStars(stars);
        tradeService.insertRating(rating);
        return listRatings(itemId);

    }

    @RequestMapping(value="/delete_rating", method = RequestMethod.POST)
    public Result deleteRating(@RequestParam(value = "rating_id") int ratingId){
        Result result = new Result();
        Rating rating = new Rating();
        int row = tradeService.deleteRatingByRatingId(ratingId);
        if(row > 0){
            result.setCode(200);
        }else{
            result.setCode(400);
        }
        return result;

    }

    @RequestMapping(value = "/list_ratings", method = RequestMethod.GET)
    public Result listRatings(@RequestParam(value = "item_id") int itemId){
        Result result = new Result();
        List<Rating> ratingList = tradeService.queryRatingListByItemId(itemId);
        int sum = 0;
        double averageRating = -1;
        if(ratingList != null && ratingList.size() != 0){
            for(Rating rating: ratingList){
                sum += rating.getStars();
            }
            averageRating = sum * 1.0 / ratingList.size();
        }

        Map<String, Object> data = new HashMap<>();
        ratingList.sort(new Comparator<Rating>() {
            @Override
            public int compare(Rating o1, Rating o2) {
                return o1.getRateTime().compareTo(o2.getRateTime());
            }
        });
        data.put("ratingList", ratingList);
        data.put("averageRating", averageRating);
        result.setCode(200);
        result.setData(data);
        return result;


    }

    @RequestMapping(value = "/auction_report", method = RequestMethod.GET)
    public Result auctionReport(){
        Result result = new Result();
        List<Item> items = tradeService.auctionReport();
        Map<String, Object> data = new HashMap<>();
        data.put("items", items);
        result.setCode(200);
        result.setData(data);
        return result;

    }

    @RequestMapping(value = "/category_report", method = RequestMethod.GET)
    public Result categoryReport(){
        Result result = new Result();
        List<CategoryReport> categoryReports = tradeService.categoryReport();
        Map<String, Object> data = new HashMap<>();
        data.put("categoryReports", categoryReports);
        result.setCode(200);
        result.setData(data);
        return result;

    }

















}
