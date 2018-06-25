package com.zc.gtbay.vo;

import java.math.BigDecimal;
import java.sql.Timestamp;


public class Item {
    private int itemId;
    private String itemName;
    private String description;
    private int categoryId;
    private int conditionId;
    private BigDecimal startBidPrice;
    private BigDecimal minimumSalePrice;
    private int auction_length;
    private BigDecimal getNowPrice;
    private int returnAccepted;
    private Timestamp acutionStartTime;
    private int userId;
    private int auctionEnd;
    private String userName;
    private BigDecimal amount;
    private Timestamp endTime;
    private BigDecimal winnerPrice;

    public BigDecimal getWinnerPrice() {
        return winnerPrice;
    }

    public void setWinnerPrice(BigDecimal winnerPrice) {
        this.winnerPrice = winnerPrice;
    }

    public Timestamp getEndTime() {
        return endTime;
    }

    public void setEndTime(Timestamp endTime) {
        this.endTime = endTime;
    }

    public String getUserName() {
        return userName;
    }

    public void setUserName(String userName) {
        this.userName = userName;
    }

    public BigDecimal getAmount() {
        return amount;
    }

    public void setAmount(BigDecimal amount) {
        this.amount = amount;
    }

    public int getAuctionEnd() {
        return auctionEnd;
    }

    public void setAuctionEnd(int auctionEnd) {
        this.auctionEnd = auctionEnd;
    }

    public int getItemId() {
        return itemId;
    }

    public void setItemId(int itemId) {
        this.itemId = itemId;
    }

    public String getItemName() {
        return itemName;
    }

    public void setItemName(String itemName) {
        this.itemName = itemName;
    }

    public String getDescription() {
        return description;
    }

    public void setDescription(String description) {
        this.description = description;
    }

    public int getCategoryId() {
        return categoryId;
    }

    public void setCategoryId(int categoryId) {
        this.categoryId = categoryId;
    }

    public int getConditionId() {
        return conditionId;
    }

    public void setConditionId(int conditionId) {
        this.conditionId = conditionId;
    }

    public BigDecimal getStartBidPrice() {
        return startBidPrice;
    }

    public void setStartBidPrice(BigDecimal startBidPrice) {
        this.startBidPrice = startBidPrice;
    }

    public BigDecimal getMinimumSalePrice() {
        return minimumSalePrice;
    }

    public void setMinimumSalePrice(BigDecimal minimumSalePrice) {
        this.minimumSalePrice = minimumSalePrice;
    }

    public int getAuction_length() {
        return auction_length;
    }

    public void setAuction_length(int auction_length) {
        this.auction_length = auction_length;
    }

    public BigDecimal getGetNowPrice() {
        return getNowPrice;
    }

    public void setGetNowPrice(BigDecimal getNowPrice) {
        this.getNowPrice = getNowPrice;
    }

    public int getReturnAccepted() {
        return returnAccepted;
    }

    public void setReturnAccepted(int returnAccepted) {
        this.returnAccepted = returnAccepted;
    }

    public Timestamp getAcutionStartTime() {
        return acutionStartTime;
    }

    public void setAcutionStartTime(Timestamp acutionStartTime) {
        this.acutionStartTime = acutionStartTime;
    }

    public int getUserId() {
        return userId;
    }

    public void setUserId(int userId) {
        this.userId = userId;
    }
}
