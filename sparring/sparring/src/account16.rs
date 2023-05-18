use half::f16;
use num_traits::Float;
use num_traits::FromPrimitive;

#[repr(C)]
#[derive(PartialEq, Debug, Clone, Default)]
pub struct Hlcv {
    pub high: f16,
    pub low: f16,
    pub close: f16,
    pub volume: f16,
}

#[derive(Debug, Clone)]
pub struct Account16 {
    pub balance: f16,
    pub entry_price: f16,
    pub max_balance: f16,
    pub max_drawdown: f16,
    pub trade_bar: u32,
    pub n_trades: u32,
    pub taker_fee: f16,
    pub maker_fee: f16,
    pub slippage: f16,
}

impl Account16 {
    pub fn new() -> Self {
        Self {
            balance: f16::from_f32(1.),
            entry_price: f16::from_f32(0.),
            max_drawdown: f16::from_f32(0.),
            max_balance: f16::from_f32(1.),
            trade_bar: 0,
            n_trades: 0,
            taker_fee: f16::from_f32(0.00075),
            maker_fee: f16::from_f32(0.),
            slippage: f16::from_f32(0.00075),
        }
    }

    pub fn open_short(&mut self, should_open_short: bool, price: f16, fee: f16) {
        if should_open_short {
            self.balance -= self.balance * fee;
            self.entry_price = -price;
        }
    }

    pub fn close_short(&mut self, should_close_short: bool, price: f16, fee: f16) {
        if should_close_short {
            let entry_price_abs: f16 = self.entry_price.abs();
            let position: f16 = self.balance / self.entry_price;
            self.balance += (price - entry_price_abs) * position + price * position * fee;
            self.entry_price = f16::from_f32(0.);
        }
    }

    pub fn open_long(&mut self, should_open_long: bool, price: f16, fee: f16) {
        if should_open_long {
            self.balance -= self.balance * fee;
            self.entry_price = price;
        }
    }
    pub fn close_long(&mut self, should_close_long: bool, price: f16, fee: f16) {
        if should_close_long {
            // even tough we know that price is abs we leave it like that so that compiler can easily
            // optimize code
            let entry_price_abs: f16 = self.entry_price.abs();
            let position: f16 = self.balance / self.entry_price;
            self.balance += (price - entry_price_abs) * position - price * position * fee;
            self.entry_price = f16::from_f32(0.);
        }
    }

    pub fn closed_position(&mut self, should_close_position: bool, input_i: usize) {
        if should_close_position {
            self.max_balance = self.max_balance.max(self.balance);
            let max_drawdown = (f16::from_f32(1.) - self.balance / self.max_balance);
            self.max_drawdown = self.max_drawdown.max(max_drawdown);
            self.trade_bar = input_i as u32;
            self.n_trades += 1;
        }
    }

    pub fn market_open_short(&mut self, should_open_short: bool, price: f16) {
        self.open_short(should_open_short, price, self.taker_fee + self.slippage);
    }

    pub fn market_close_short(&mut self, should_close_short: bool, price: f16) {
        self.close_short(should_close_short, price, self.taker_fee + self.slippage);
    }

    pub fn market_open_long(&mut self, should_open_long: bool, price: f16) {
        self.open_long(should_open_long, price, self.taker_fee + self.slippage);
    }

    pub fn market_close_long(&mut self, should_close_long: bool, price: f16) {
        self.close_long(should_close_long, price, self.taker_fee + self.slippage);
    }

    pub fn limit_open_short(&mut self, should_open_short: bool, price: f16) {
        self.open_short(should_open_short, price, f16::from_f32(0.));
    }

    pub fn limit_close_short(&mut self, should_close_short: bool, price: f16) {
        self.close_short(should_close_short, price, f16::from_f32(0.));
    }

    pub fn limit_open_long(&mut self, should_open_long: bool, price: f16) {
        self.open_long(should_open_long, price, f16::from_f32(0.));
    }

    pub fn limit_close_long(&mut self, should_close_long: bool, price: f16) {
        self.close_long(should_close_long, price, f16::from_f32(0.));
    }

    pub fn short_stopped(&mut self, stop_price: f16, high: f16) -> bool {
        let should_stop_short = self.entry_price < f16::from_f32(0.) && high > stop_price;
        self.market_close_short(should_stop_short, stop_price);
        should_stop_short
    }

    pub fn long_stopped(&mut self, stop_price: f16, low: f16) -> bool {
        let should_stop_long = self.entry_price > f16::from_f32(0.) && low < stop_price;
        self.market_close_long(should_stop_long, stop_price);
        should_stop_long
    }

    pub fn update(
        &mut self,
        hlcvs: &[Hlcv],
        i: usize,
        prev_close: f16,
        prev_rsi: f16,
        rsi: f16,
        max_risk: f16,
        hline: f16,
        lline: f16,
    ) {
        let hlcv = unsafe { hlcvs.get_unchecked(i) };
        if hlcv.close > prev_close {
            let hline_condition = prev_rsi < hline && rsi >= hline;
            let stop_price = self.entry_price * -(f16::from_f32(1.) + max_risk);
            let stop_short = self.short_stopped(stop_price, hlcv.high);
            let open_short = self.entry_price == f16::from_f32(0.) && hline_condition;
            let close_long = self.entry_price > f16::from_f32(0.) && hline_condition;
            let close_position = close_long | stop_short;
            self.market_open_short(open_short, hlcv.close);
            self.market_close_long(close_long, hlcv.close);
            self.closed_position(close_position, i);
        } else if hlcv.close < prev_close {
            let lline_condition = prev_rsi >= lline && rsi < lline;
            let stop_price = self.entry_price * (f16::from_f32(1.) - max_risk);
            let stop_long = self.long_stopped(stop_price, hlcv.low);
            let open_long = self.entry_price == f16::from_f32(0.) && lline_condition;
            let close_short = self.entry_price < f16::from_f32(0.) && lline_condition;
            let close_position = close_short | stop_long;
            self.market_open_long(open_long, hlcv.close);
            self.market_close_short(close_short, hlcv.close);
            self.closed_position(close_position, i);
        } else {
            let stop_short_price = self.entry_price * -(f16::from_f32(1.) + max_risk);
            let stop_long_price = self.entry_price * (f16::from_f32(1.) - max_risk);
            let stop_short = self.short_stopped(stop_short_price, hlcv.high);
            let stop_long = self.long_stopped(stop_long_price, hlcv.low);
            let close_position = stop_long | stop_short;
            self.closed_position(close_position, i);
        }
    }
}

pub struct RsiState {
    period: usize,
    avg_gain: f16,
    avg_loss: f16,
}

impl RsiState {
    pub fn new(period: usize, hlcvs: &[Hlcv]) -> Self {
        let mut avg_gain = f16::from_f32(0.);
        let mut avg_loss = f16::from_f32(0.);
        for i in 1..period + 1 {
            let price = hlcvs[i].close;
            let prev_price = hlcvs[i - 1].close;
            let diff = price - prev_price;
            avg_gain += f16::from_u16((diff > f16::from_f32(0.)) as u16).unwrap() * diff / f16::from_usize(period).unwrap();
            avg_loss -= f16::from_u16((diff < f16::from_f32(0.)) as u16).unwrap() * diff / f16::from_usize(period).unwrap();
        }

        Self {
            period,
            avg_gain,
            avg_loss,
        }
    }

    pub fn update_at(&self) -> usize {
        self.period + 1
    }

    pub fn update(&mut self, hlcvs: &[Hlcv], offset: usize) -> f16 {
        let price = hlcvs[offset].close;
        let prev_price = hlcvs[offset - 1].close;
        let diff = price - prev_price;
        let last_price = hlcvs[offset - self.period].close;
        let last_prev_price = hlcvs[offset - self.period - 1].close;
        let last_diff = last_price - last_prev_price;
        // Using rolling average because it is faster, but it is prone to prcision errors
        // First remove from average to minimize floating point precision errors
        self.avg_gain -=
            f16::from_u16((last_diff > f16::from_f32(0.)) as u16).unwrap() * last_diff / f16::from_usize(self.period).unwrap();
        self.avg_loss +=
            f16::from_u16((last_diff < f16::from_f32(0.)) as u16).unwrap() * last_diff / f16::from_usize(self.period).unwrap();

        self.avg_gain += f16::from_u16((diff > f16::from_f32(0.)) as u16).unwrap() * diff / f16::from_usize(self.period).unwrap();
        self.avg_loss -= f16::from_u16((diff < f16::from_f32(0.)) as u16).unwrap() * diff / f16::from_usize(self.period).unwrap();

        let mut rs = self.avg_gain / self.avg_loss;
        rs = if rs.is_nan() { f16::from_f32(1.) } else { rs };
        let rsi = f16::from_f32(100.) - (f16::from_f32(100.) / (f16::from_f32(1.) + rs));
        // we could clamp the value between 0 and 100, no need to bother, happens rarely
        //        assert!(rsi >= 0.);
        //        assert!(rsi <= 100.);

        return rsi;
    }
}
