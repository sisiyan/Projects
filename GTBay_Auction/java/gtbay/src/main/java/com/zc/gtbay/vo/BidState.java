package com.zc.gtbay.vo;

public enum BidState {
    TIME_ENDS(10,"TIME_ENDS"),
    MAX_PRICE(11,"MAX_PRICE"),
    LESS_THAN_START_PRICE(12,"LESS_THAN_START_PRICE"),
    LESS_THAN_CURRENT_PRICE(13,"LESS_THAN_CURRENT_PRICE"),
    AUCTION_ENDS(14,"AUCTION_ENDS"),
    OK(15,"OK");

    private int code;
    private String msg;
    private BidState(int code, String msg){
        this.code = code;
        this.msg = msg;
    }

    public int getCode() {
        return code;
    }

    public void setCode(int code) {
        this.code = code;
    }

    public String getMsg() {
        return msg;
    }

    public void setMsg(String msg) {
        this.msg = msg;
    }
}
