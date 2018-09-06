package com.zc.gtbay.service;

import com.zc.gtbay.vo.*;

import java.math.BigDecimal;
import java.util.List;

public interface TradeService {
    int insertItem(Item item);
    List<Option> queryByType(OptionEnum optionEnum);
    int insertBid(Bid bid);
    BidState getState(Bid bid);
    List<Item> search(String keyword, int categoryId, BigDecimal minPrice, BigDecimal maxPrice, int conditionId);
    Item queryById(int itemId);
    List<Bid> queryBidByItemId(int itemId);
    int updateDescription(int itemId, String description);
    int insertRating(Rating rating);
    int deleteRatingByRatingId(int ratingId);
    List<Rating> queryRatingListByItemId(int itemId);
    List<Item> auctionReport();
    List<CategoryReport> categoryReport();
}
