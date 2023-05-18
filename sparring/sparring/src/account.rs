use merovingian::hlcv::Hlcv;

#[derive(Debug, Clone, Default)]
pub struct Account {
    pub balance: f32,
    pub entry_price: f32,
    pub max_balance: f32,
    pub max_drawdown: f32,
    pub trade_bar: u32,
    pub n_trades: u32,
    pub taker_fee: f32,
    pub maker_fee: f32,
    pub slippage: f32,
}

impl Account {
    pub fn new() -> Self {
        Self {
            balance: 1.,
            entry_price: 0.,
            max_drawdown: 0.,
            max_balance: 1.,
            trade_bar: 0,
            n_trades: 0,
            taker_fee: 0.00075,
            maker_fee: 0.,
            slippage: 0.00075,
        }
    }

    pub fn open_short(&mut self, should_open_short: bool, price: f32, fee: f32) {
        if should_open_short {
            // println!("open short: {}, {}, {}", should_open_short, price, fee);
            self.balance -= self.balance * fee;
            self.entry_price = -price;
        }
    }

    pub fn close_short(&mut self, should_close_short: bool, price: f32, fee: f32) {
        if should_close_short {
            // println!("close short: {}, {}, {}", should_close_short, price, fee);
            let entry_price_abs: f32 = self.entry_price.abs();
            let position: f32 = self.balance / self.entry_price;
            self.balance += (price - entry_price_abs) * position + price * position * fee;
            self.entry_price = 0.;
        }
    }

    pub fn open_long(&mut self, should_open_long: bool, price: f32, fee: f32) {
        if should_open_long {
            // println!("open long: {}, {}, {}", should_open_long, price, fee);
            self.balance -= self.balance * fee;
            self.entry_price = price;
        }
    }
    pub fn close_long(&mut self, should_close_long: bool, price: f32, fee: f32) {
        if should_close_long {
            // println!("close long: {}, {}, {}", should_close_long, price, fee);
            // even tough we know that price is abs we leave it like that so that compiler can easily
            // optimize code
            let entry_price_abs: f32 = self.entry_price.abs();
            let position: f32 = self.balance / self.entry_price;
            self.balance += (price - entry_price_abs) * position - price * position * fee;
            self.entry_price = 0.;
        }
    }

    pub fn closed_position(&mut self, should_close_position: bool, input_i: usize) {
        if should_close_position {
            self.max_balance = self.max_balance.max(self.balance);
            let max_drawdown = (1. - self.balance / self.max_balance);
            self.max_drawdown = self.max_drawdown.max(max_drawdown);
            self.trade_bar = input_i as u32;
            self.n_trades += 1;
        }
    }

    pub fn market_open_short(&mut self, should_open_short: bool, price: f32) {
        self.open_short(should_open_short, price, self.taker_fee + self.slippage);
    }

    pub fn market_close_short(&mut self, should_close_short: bool, price: f32) {
        self.close_short(should_close_short, price, self.taker_fee + self.slippage);
    }

    pub fn market_open_long(&mut self, should_open_long: bool, price: f32) {
        self.open_long(should_open_long, price, self.taker_fee + self.slippage);
    }

    pub fn market_close_long(&mut self, should_close_long: bool, price: f32) {
        self.close_long(should_close_long, price, self.taker_fee + self.slippage);
    }

    pub fn limit_open_short(&mut self, should_open_short: bool, price: f32) {
        self.open_short(should_open_short, price, 0.);
    }

    pub fn limit_close_short(&mut self, should_close_short: bool, price: f32) {
        self.close_short(should_close_short, price, 0.);
    }

    pub fn limit_open_long(&mut self, should_open_long: bool, price: f32) {
        self.open_long(should_open_long, price, 0.);
    }

    pub fn limit_close_long(&mut self, should_close_long: bool, price: f32) {
        self.close_long(should_close_long, price, 0.);
    }

    pub fn short_stopped(&mut self, stop_price: f32, high: f32) -> bool {
        let should_stop_short = self.entry_price < 0. && high > stop_price;
        if should_stop_short {
            // println!("stop");
        }
        self.market_close_short(should_stop_short, stop_price);
        should_stop_short
    }

    pub fn long_stopped(&mut self, stop_price: f32, low: f32) -> bool {
        let should_stop_long = self.entry_price > 0. && low < stop_price;
        if should_stop_long {
            // println!("stop");
        }
        self.market_close_long(should_stop_long, stop_price);
        should_stop_long
    }

    pub fn update(
        &mut self,
        hlcvs: &[Hlcv],
        i: usize,
        prev_close: f32,
        prev_rsi: f32,
        rsi: f32,
        max_risk: f32,
        hline: f32,
        lline: f32,
    ) {
        let hlcv = unsafe { hlcvs.get_unchecked(i) };
        // println!("{} {}, {}, {}", prev_rsi, rsi, prev_close, hlcv.close);
        if hlcv.close > prev_close {
            let hline_condition = prev_rsi < hline && rsi >= hline;
            let stop_price = self.entry_price * -(1. + max_risk);
            let stop_short = self.short_stopped(stop_price, hlcv.high);
            let open_short = self.entry_price == 0. && hline_condition;
            let close_long = self.entry_price > 0. && hline_condition;
            let close_position = close_long | stop_short;
            self.market_open_short(open_short, hlcv.close);
            self.market_close_long(close_long, hlcv.close);
            self.closed_position(close_position, i);
            if open_short || close_long || close_position {
                // println!("{} {}", i, stop_short);
                // dbg!(&self);
            }
        } else if hlcv.close < prev_close {
            let lline_condition = prev_rsi >= lline && rsi < lline;
            let stop_price = self.entry_price * (1. - max_risk);
            let stop_long = self.long_stopped(stop_price, hlcv.low);
            let open_long = self.entry_price == 0. && lline_condition;
            let close_short = self.entry_price < 0. && lline_condition;
            let close_position = close_short | stop_long;
            self.market_open_long(open_long, hlcv.close);
            self.market_close_short(close_short, hlcv.close);
            self.closed_position(close_position, i);
            if close_short || open_long || close_position {
                // println!("{} {}", i, stop_long);
                // dbg!(&self);
            }
        } else {
            let stop_short_price = self.entry_price * -(1. + max_risk);
            let stop_long_price = self.entry_price * (1. - max_risk);
            let stop_short = self.short_stopped(stop_short_price, hlcv.high);
            let stop_long = self.long_stopped(stop_long_price, hlcv.low);
            let close_position = stop_long | stop_short;
            self.closed_position(close_position, i);
            if close_position {
                // println!("{} stopped", i);
                // dbg!(&self);
            }
        }
    }
}
