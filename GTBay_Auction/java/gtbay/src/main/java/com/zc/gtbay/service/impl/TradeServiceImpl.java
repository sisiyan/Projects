package com.zc.gtbay.service.impl;

import com.zc.gtbay.service.TradeService;
import com.zc.gtbay.vo.*;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.dao.DataAccessException;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.jdbc.core.ResultSetExtractor;
import org.springframework.jdbc.core.RowMapper;
import org.springframework.stereotype.Service;

import java.math.BigDecimal;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.Timestamp;
import java.util.Calendar;
import java.util.List;


@Service(value = "tradeService")
public class TradeServiceImpl implements TradeService {
    @Autowired
    JdbcTemplate jdbcTemplate;
    @Override
    public int insertItem(Item item) {

        String sql = "INSERT INTO Item (item_name, description, categoryID, conditionID, start_bid_price, minimum_sale_price, auction_start_time, auction_length, get_now_price, return_accepted, userID, end_time)" +
                " values(?,?,?,?,?,?,?,?,?,?,?,?)";

        Timestamp startTime = item.getAcutionStartTime();
        int day = item.getAuction_length();
        Calendar calendar = Calendar.getInstance();
        calendar.setTimeInMillis(startTime.getTime());
        calendar.add(Calendar.DAY_OF_MONTH, day);
        System.out.print(new Timestamp(calendar.getTimeInMillis()));

        int row = jdbcTemplate.update(sql, item.getItemName(), item.getDescription(), item.getCategoryId(), item.getConditionId(), item.getStartBidPrice(), item.getMinimumSalePrice(),
                item.getAcutionStartTime(),item.getAuction_length(),item.getGetNowPrice(), item.getReturnAccepted(),item.getUserId(), new Timestamp(calendar.getTimeInMillis()));
        return row;
    }

    @Override
    public List<Option> queryByType(OptionEnum optionEnum) {
        String sql = "";
        if(optionEnum == OptionEnum.CATEGORY){
            sql = "select categoryid value, category_name name from category order by categoryid asc";
        }else{
            sql = "select conditionid value, condition_name name from conditions order by conditionid asc";
        }
        List<Option> list = jdbcTemplate.query(sql, new Object[0], new RowMapper<Option>() {
            @Override
            public Option mapRow(ResultSet resultSet, int i) throws SQLException {
                Option option = new Option();
                option.setName(resultSet.getString("name"));
                option.setValue(resultSet.getInt("value"));
                return option;
            }
        });
        return list;
    }

    @Override
    public int insertBid(Bid bid) {

        String sql = "INSERT INTO Bid (itemID, amount, userID) values (?,?,?)";
        int row = jdbcTemplate.update(sql, new Object[]{bid.getItemId(), bid.getAmount(), bid.getUserId()});
        if(row > 0){
            String updateSql = "update item set auction_end=1, end_time = ? where itemId = ? and get_now_price = ?";
            jdbcTemplate.update(updateSql, new Object[]{new Timestamp(System.currentTimeMillis()),bid.getItemId(), bid.getAmount()});
        }
        return row;


    }

    @Override
    public BidState getState(Bid bid) {
        String sql = "SELECT start_bid_price, auction_start_time, auction_length, get_now_price, auction_end FROM Item WHERE itemId = ?";
        BidState bidState = null;
        Item item = jdbcTemplate.query(sql, new Object[]{bid.getItemId()}, new ResultSetExtractor<Item>() {
            @Override
            public Item extractData(ResultSet resultSet) throws SQLException, DataAccessException {
                Item item = new Item();
                if(resultSet.next()){
                    item.setStartBidPrice(resultSet.getBigDecimal("start_bid_price"));
                    item.setAuction_length(resultSet.getInt("auction_length"));
                    item.setAcutionStartTime(resultSet.getTimestamp("auction_start_time"));
                    item.setAuctionEnd(resultSet.getInt("auction_end"));
                    item.setGetNowPrice(resultSet.getBigDecimal("get_now_price"));
                }
                return item;
            }
        });
        if(item != null){
            BigDecimal startBidPrice = item.getStartBidPrice();
            if(bid.getAmount().compareTo(startBidPrice) == -1){
                bidState = BidState.LESS_THAN_START_PRICE;
                return bidState;
            }
            int auctionEnd = item.getAuctionEnd();
            if(auctionEnd == 1){
                bidState = bidState.AUCTION_ENDS;
                return bidState;
            }
            Timestamp startTime = item.getAcutionStartTime();
            int day = item.getAuction_length();
            long currentTime = System.currentTimeMillis();
            Calendar calendar = Calendar.getInstance();
            calendar.setTimeInMillis(startTime.getTime());
            calendar.add(Calendar.DAY_OF_MONTH, day);
            if(currentTime >= calendar.getTimeInMillis()){
                bidState = BidState.TIME_ENDS;
                return bidState;
            }
            BigDecimal nowPrice = item.getGetNowPrice();
            if(nowPrice != null) {
                if(bid.getAmount().compareTo(nowPrice) == 1){
                    bidState = BidState.MAX_PRICE;
                    //bidState = bidState.AUCTION_ENDS;
                    return bidState;
                }
            }
//            if(bid.getAmount().compareTo(nowPrice) == 1){
//                bidState = BidState.MAX_PRICE;
//            }
        }

        String maxAmountSql = "SELECT max(amount) amount FROM Bid " +
                "WHERE itemID  = ?";

        BigDecimal amount = jdbcTemplate.query(maxAmountSql, new Object[]{bid.getItemId()}, new ResultSetExtractor<BigDecimal>() {
            @Override
            public BigDecimal extractData(ResultSet resultSet) throws SQLException, DataAccessException {
                if(resultSet.next()){
                    BigDecimal amount = resultSet.getBigDecimal("amount");
                    return amount == null ? new BigDecimal(0) : amount;
                }
                return new BigDecimal(0);
            }
        });

        if(bid.getAmount().compareTo(amount)<=0){
            bidState = BidState.LESS_THAN_CURRENT_PRICE;
            return bidState;
        }


        return BidState.OK;
    }

    @Override
    public List<Item> search(String keyword, int categoryId, BigDecimal minPrice, BigDecimal maxPrice, int conditionId) {

//        String sql = "SELECT a.itemID, a.item_name, a.get_now_price, a.end_time, b.winner_price, b.userID,u.username " +
//                "FROM Item as a " +
//                "LEFT JOIN high_bid as b " +
//                "ON a.itemID = b.itemID " +
//                "left join user as u " +
//                "on u.userid = b.userid " +
//                "WHERE  a.categoryID=? and (a.item_name like ?  or a.description like ?)  " +
//                "and b.winner_price>=? and b.winner_price<=? and a.conditionID<=? ";

        String sql1 = "SELECT a.itemID, a.item_name, a.get_now_price, a.auction_end, a.end_time, b.winner_price, b.userID,u.username,a.start_bid_price " +
                "FROM Item as a " +
                "LEFT JOIN high_bid as b " +
                "ON a.itemID = b.itemID " +
                "left join user as u " +
                "on u.userid = b.userid " +
                "WHERE a.categoryID=? and (a.item_name like ?  or a.description like ?) " +
                "and a.conditionID<=? and ((b.winner_price>=? and b.winner_price<=?) || (b.winner_price is NULL and a.start_bid_price >=? and a.start_bid_price<=?)) " +
                "and a.end_time > NOW() and a.auction_end = 0 " +
                "ORDER BY a.end_time";

        String sql2 = "SELECT a.itemID, a.item_name, a.get_now_price, a.auction_end, a.end_time, b.winner_price, b.userID,u.username,a.start_bid_price " +
                "FROM Item as a " +
                "LEFT JOIN high_bid as b " +
                "ON a.itemID = b.itemID " +
                "left join user as u " +
                "on u.userid = b.userid " +
                "WHERE (a.item_name like ?  or a.description like ?) " +
                " and a.conditionID<=? and ((b.winner_price>=? and b.winner_price<=?) || (b.winner_price is NULL and a.start_bid_price >=? and a.start_bid_price<=?)) " +
                "and a.end_time > NOW() and a.auction_end = 0 " +
                "ORDER BY a.end_time";
        //System.out.println("categoryId" + categoryId);
        if (categoryId != 1000) {
            List<Item> itemList = jdbcTemplate.query(sql1, new Object[]{categoryId, "%" + keyword + "%", "%" + keyword + "%",  conditionId, minPrice, maxPrice,minPrice, maxPrice,}, new RowMapper<Item>() {
                @Override
                public Item mapRow(ResultSet resultSet, int i) throws SQLException {
                    Item item = new Item();
                    item.setItemId(resultSet.getInt("itemid"));
                    item.setItemName(resultSet.getString("item_name"));
                    item.setGetNowPrice(resultSet.getBigDecimal("get_now_price"));
                    item.setEndTime(resultSet.getTimestamp("end_time"));
                    item.setAmount(resultSet.getBigDecimal("winner_price"));
                    item.setUserId(resultSet.getInt("userid"));
                    item.setUserName(resultSet.getString("username"));
                    return item;
                }
            });
            return itemList;
        }
        else {
            List<Item> itemList = jdbcTemplate.query(sql2, new Object[]{"%" + keyword + "%", "%" + keyword + "%",  conditionId, minPrice, maxPrice,minPrice, maxPrice,}, new RowMapper<Item>() {
                @Override
                public Item mapRow(ResultSet resultSet, int i) throws SQLException {
                    Item item = new Item();
                    item.setItemId(resultSet.getInt("itemid"));
                    item.setItemName(resultSet.getString("item_name"));
                    item.setGetNowPrice(resultSet.getBigDecimal("get_now_price"));
                    item.setEndTime(resultSet.getTimestamp("end_time"));
                    item.setAmount(resultSet.getBigDecimal("winner_price"));
                    item.setUserId(resultSet.getInt("userid"));
                    item.setUserName(resultSet.getString("username"));
                    return item;
                }
            });
            return itemList;
        }

    }

    @Override
    public Item queryById(int itemId) {
        String sql = "select item_name, description, categoryID, conditionID, start_bid_price, minimum_sale_price, auction_start_time, auction_length, get_now_price, return_accepted, userID, end_time" +
                " from item where itemid = ?";
        Item item = jdbcTemplate.query(sql, new Object[]{itemId}, new ResultSetExtractor<Item>() {
            @Override
            public Item extractData(ResultSet resultSet) throws SQLException, DataAccessException {
                Item item = new Item();
                if(resultSet.next()){
                    item.setItemId(itemId);
                    item.setItemName(resultSet.getString("item_name"));
                    item.setDescription(resultSet.getString("description"));
                    item.setCategoryId(resultSet.getInt("categoryid"));
                    item.setConditionId(resultSet.getInt("conditionid"));
                    item.setStartBidPrice(resultSet.getBigDecimal("start_bid_price"));
                    item.setMinimumSalePrice(resultSet.getBigDecimal("minimum_sale_price"));
                    item.setAcutionStartTime(resultSet.getTimestamp("auction_start_time"));
                    item.setAuction_length(resultSet.getInt("auction_length"));
                    item.setGetNowPrice(resultSet.getBigDecimal("get_now_price"));
                    item.setReturnAccepted(resultSet.getInt("return_accepted"));
                    item.setUserId(resultSet.getInt("userID"));
                    item.setEndTime(resultSet.getTimestamp("end_time"));
                }
                return item;
            }
        });
        return item;
    }

    @Override
    public List<Bid> queryBidByItemId(int itemId) {
        String sql = "select b.bidid, b.itemid, b.amount, b.userid, b.bid_time, u.username from bid b left join user u on u.userid = b.userid where b.itemid = ? order by b.bid_time desc";
        List<Bid> bidList = jdbcTemplate.query(sql, new Object[]{itemId}, new RowMapper<Bid>() {
            @Override
            public Bid mapRow(ResultSet resultSet, int i) throws SQLException {
                Bid bid = new Bid();
                bid.setBidId(resultSet.getInt("bidid"));
                bid.setItemId(resultSet.getInt("itemid"));
                bid.setAmount(resultSet.getBigDecimal("amount"));
                bid.setUserId(resultSet.getInt("userId"));
                bid.setBidTime(resultSet.getTimestamp("bid_time"));
                bid.setUsername(resultSet.getString("username"));
                return bid;
            }

        });
        return bidList;
    }

    @Override
    public int updateDescription(int itemId, String description) {
        String sql = "update item set description=? where itemid=?";
        int row = jdbcTemplate.update(sql, new Object[]{description, itemId});
        return row;
    }

    @Override
    public int insertRating(Rating rating) {
        String sql = "insert into rating(itemid, comments, stars, userid) values(?,?,?,?)";
        int row = jdbcTemplate.update(sql, new Object[]{rating.getItemId(), rating.getComments(), rating.getStars(), rating.getUserId()});
        return row;
    }

    @Override
    public int deleteRatingByRatingId(int ratingId) {
        String sql = "delete from rating where ratingid = ?";
        int row = jdbcTemplate.update(sql, new Object[]{ratingId});
        return row;
    }

    @Override
    public List<Rating> queryRatingListByItemId(int itemId) {
        String sql = "select r.ratingid, r.itemid, r.rate_time, r.comments, r.stars, r.userid, u.username from rating r left join user u on u.userid=r.userid where itemid=?";
        List<Rating> ratingList = jdbcTemplate.query(sql, new Object[]{itemId}, new RowMapper<Rating>() {
            @Override
            public Rating mapRow(ResultSet resultSet, int i) throws SQLException {
                Rating rating = new Rating();
                rating.setRatingId(resultSet.getInt("ratingid"));
                rating.setItemId(itemId);
                rating.setRateTime(resultSet.getTimestamp("rate_time"));
                rating.setComments(resultSet.getString("comments"));
                rating.setStars(resultSet.getInt("stars"));
                rating.setUserId(resultSet.getInt("userid"));
                rating.setUsername(resultSet.getString("username"));
                return rating;
            }
        });
        return ratingList;
    }

    @Override
    public List<Item> auctionReport() {
        String sql = "SELECT a.itemID, a.item_name, b.winner_price, b.userID, a.end_time, a.userid, u.username FROM Item as a " +
                "LEFT JOIN winner as b " +
                "ON a.itemID = b.itemID left join user as u on u.userid = b.userid " +
                "WHERE now() > a.end_time ORDER BY a.end_time desc";

        List<Item> itemList = jdbcTemplate.query(sql, new Object[]{}, new RowMapper<Item>() {

            @Override
            public Item mapRow(ResultSet resultSet, int i) throws SQLException {
                Item item = new Item();
                item.setItemId(resultSet.getInt("itemid"));
                item.setItemName(resultSet.getString("item_name"));
                item.setWinnerPrice(resultSet.getBigDecimal("winner_price"));
                item.setUserName(resultSet.getString("username"));
                item.setEndTime(resultSet.getTimestamp("end_time"));
                return item;
            }
        });
        return itemList;
    }

    @Override
    public List<CategoryReport> categoryReport() {
        String sql = "SELECT categoryid, (select category_name from Category c where c.categoryID = i.categoryID) category_name, min(get_now_price) as min_price, max(get_now_price) as max_price, avg(get_now_price) as average_price, count(categoryID) as total_items " +
                "FROM Item i " +
                "GROUP BY categoryID";

        //String sql2 = "COUN";where get_now_price is not null
        List<CategoryReport> categoryReports = jdbcTemplate.query(sql, new Object[]{}, new RowMapper<CategoryReport>() {
            @Override
            public CategoryReport mapRow(ResultSet resultSet, int i) throws SQLException {
                CategoryReport categoryReport = new CategoryReport();
                categoryReport.setCategoryId(resultSet.getInt("categoryid"));
                categoryReport.setCategoryName(resultSet.getString("category_name"));
                categoryReport.setMinPrice(resultSet.getBigDecimal("min_price"));
                categoryReport.setMaxPrice(resultSet.getBigDecimal("max_price"));
                categoryReport.setAveragePrice(resultSet.getBigDecimal("average_price"));
                categoryReport.setCount(resultSet.getInt("total_items"));
                return categoryReport;
            }
        });
        return categoryReports;
    }
}
