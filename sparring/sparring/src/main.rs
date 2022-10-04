#![feature(bench_black_box)]

mod load_hlcv;
mod plot;

use cmaes::objective_function::Scale;
use cmaes::{CMAESOptions, PlotOptions};
use cmaes::{DVector, Mode, ObjectiveFunction};
use load_hlcv::load_hlcv;
use merovingian::hlcv::{Hlcv, MappedHlcvs};
use mouse::prelude::*;
use ordered_float::OrderedFloat;
use plotters::style::RGBAColor;
use rand::prelude::*;
use rayon::prelude::*;
use simplers_optimization::Optimizer;
use std::collections::{HashMap, HashSet};
use std::hint::black_box;
use std::ops::Range;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use crate::account::Account;
use crate::context::{convert, Context, Pair};

mod account;
mod context;
pub mod map;

#[tokio::main]
async fn main() -> Result<()> {
    unsafe {
        config::load("config.yaml");
    }
    //    let mut context = Context::new().await?;
    //    for i in 0..100 {
    //        dbg!(context.backtest(3540, 19, 21., 40.));
    //    }
    //    context.summary();
    //    drop(context);
    //    return Ok(());
    //        let mut mapper = map::Mapper::new()?;
    //    //    mapper.map().await?;
    //        let top10 = mapper.read_top_10();
    //        dbg!(top10
    //            .iter()
    //            .sorted_by_key(|x| OrderedFloat(x.balance))
    //            .collect_vec());
    //        println!("{}", mapper.state().elapsed_ns as f32 / 1e9);
    //        //    //    mapper.map().await?;
    //        return Ok(());

    let mut context = Context::new().await?;
    context.cache_timeframe = true;
    context.sharded_genetic_shearch(10);
    context.summary();
    //    context.hybrid_search();

    //    dbg!(context.backtest(2640, 28, 22., 42.));
    //    context.bayes_search();
    //    context.random_search();
    //    context.genetic_search();
    //    context.hybrid_search();
    //    context.summary();
    return Ok(());
    //    let mut changed_hlcvs = vec![];
    //    let hlcvs;
    //    let timeframe = 1;
    //
    //    if timeframe != 1 {
    //        let now = Instant::now();
    //        changed_hlcvs = vec![
    //            Default::default();
    //            merovingian::hlcv::change_timeframe_dest_len(
    //                context.mapped_hlcvs.as_ref().len(),
    //                context.mapped_hlcvs.start_ts,
    //                1,
    //                timeframe,
    //            )
    //        ];
    //        merovingian::hlcv::change_timeframe(
    //            context.mapped_hlcvs.as_ref(),
    //            context.mapped_hlcvs.start_ts,
    //            1,
    //            timeframe,
    //            &mut changed_hlcvs,
    //        );
    //        hlcvs = &changed_hlcvs[..];
    //    } else {
    //        hlcvs = context.mapped_hlcvs.as_ref();
    //    }
    //    let mut h = Vec::new();
    //    h.extend_from_slice(hlcvs);
    //    drop(context);
    //    let hlcvs = black_box(h);
    //    Context::bla(&hlcvs, 14, 30., 70.);
    //    Context::bla(&hlcvs, 14, 30., 70.);
    //    return Ok(());
    //    context.random_search();
    //    let f = move |x: &[f64]| {
    //        let account =
    //            context.backtest_non_cached(x[0] as u32, x[1] as usize, x[2] as f32, x[3] as f32);
    //        dbg!(x);
    //        dbg!(&account);
    //        account.balance
    //    };
    //    let input_interval = vec![(1., 60. * 60. * 24.), (2., 200.), (0., 101.), (0., 101.)];
    //    let nb_iterations = 1000;
    //
    //    let (max_value, coordinates) = Optimizer::new(&f, &input_interval, false)
    //        .set_exploration_depth(10)
    //        .skip(nb_iterations)
    //        .next()
    //        .unwrap();
    //    println!(
    //        "max value: {} found in [{}, {}, {}, {}]",
    //        max_value, coordinates[0], coordinates[1], coordinates[2], coordinates[3]
    //    );
    //    return Ok(());
    //    context.grid_search();
    //    let acc = context.backtest(60 * 20, 14, 30., 70.);
    //    dbg!(acc);
    //    return Ok(());

    //    Total: 190006 ms
    //    Timef: 184335 ms
    //    Cache: 1306 ms
    //    Compu: 1149 ms
    //    GenAl: 3216 ms

    //    Total: 1616548 ms
    //    Timef: 1555611 ms
    //    Cache: 14371 ms
    //    Compu: 12677 ms
    //    GA: 33889 ms

    // i3-3320m
    // Total: 431132 ms
    // Timef: 427746 ms
    // Cache: 741 ms
    // Compu: 561 ms
    // GA: 2084 ms

    // cached
    // Total: 280166 ms
    // Timef: 276554 ms
    // Cache: 741 ms
    // Compu: 603 ms
    // GA: 18446744073709277284 ms

    //    [sparring/src/main.rs:121] cmaes_state.function_evals() = 2824
    //    [sparring/src/main.rs:130] account = Account {
    //        balance: 144.76009,
    //        entry_price: 0.0,
    //        max_balance: 149.8666,
    //        max_drawdown: 0.32544893,
    //        trade_bar: 7111,
    //        n_trades: 58,
    //        taker_fee: 0.00075,
    //        maker_fee: 0.0,
    //        slippage: 0.0,
    //    }

    Ok(())
}

fn plot(
    data: impl Iterator<Item = (f32, f32, f32)>,
    name: &str,
    range_x: Range<f32>,
    range_y: Range<f32>,
    range_z: Range<f32>,
) -> Result<()> {
    use plotters::prelude::*;
    let file = format!("{}.png", name);
    let root = BitMapBackend::new(&file, (1280, 720)).into_drawing_area();
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .margin(5)
        .x_label_area_size(30)
        .y_label_area_size(40)
        .build_cartesian_2d(range_x.clone(), range_y.clone())?;

    chart.configure_mesh().draw()?;

    chart
        .draw_series(data.into_iter().map(|x| {
            Circle::new(
                (x.0, x.1),
                1,
                ShapeStyle::from(&RGBColor(
                    255,
                    //                    255 - (x.2 / range_z.end * 255.) as u8,
                    //                    255 - (x.2 / range_z.end * 255.) as u8,
                    0, 0,
                ))
                .filled(),
            )
        }))?
        .label(name)
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));

    //    chart
    //        .draw_series(PointSeries::of_element(
    //            data,
    //            1,
    //            ShapeStyle::from(&RED).filled(),
    //            &|coord, size, style| EmptyElement::at(coord) + Circle::new((0, 0), size, style),
    //        ))?
    //        .label(name)
    //        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));

    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()?;

    root.present()?;

    Ok(())
}

struct RsiState {
    period: usize,
    avg_gain: f32,
    avg_loss: f32,
}

impl RsiState {
    pub fn new(period: usize, hlcvs: &[Hlcv]) -> Self {
        let mut avg_gain = 0.;
        let mut avg_loss = 0.;
        for i in 1..period + 1 {
            let price = hlcvs[i].close;
            let prev_price = hlcvs[i - 1].close;
            let diff = price - prev_price;
            avg_gain += (diff > 0.) as u32 as f32 * diff / period as f32;
            avg_loss -= (diff < 0.) as u32 as f32 * diff / period as f32;
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

    pub fn update(&mut self, hlcvs: &[Hlcv], offset: usize) -> f32 {
        let price = hlcvs[offset].close;
        let prev_price = hlcvs[offset - 1].close;
        let diff = price - prev_price;
        let last_price = hlcvs[offset - self.period].close;
        let last_prev_price = hlcvs[offset - self.period - 1].close;
        let last_diff = last_price - last_prev_price;
        // Using rolling average because it is faster, but it is prone to prcision errors
        // First remove from average to minimize floating point precision errors
        self.avg_gain -= (last_diff > 0.) as u32 as f32 * last_diff / self.period as f32;
        self.avg_loss += (last_diff < 0.) as u32 as f32 * last_diff / self.period as f32;

        self.avg_gain += (diff > 0.) as u32 as f32 * diff / self.period as f32;
        self.avg_loss -= (diff < 0.) as u32 as f32 * diff / self.period as f32;

        let mut rs = self.avg_gain / self.avg_loss;
        rs = if rs.is_nan() { 1. } else { rs };
        let rsi = 100. - (100. / (1. + rs));
        // we could clamp the value between 0 and 100, no need to bother, happens rarely
        //        assert!(rsi >= 0.);
        //        assert!(rsi <= 100.);

        return rsi;
    }
}

/*
grid search
[sparring/src/map.rs:86] &pairs[i] = Pair {
    params: [
        3600,
        11,
        6,
        19,
    ],
    balance: 475.63037,
}
[sparring/src/map.rs:86] &pairs[i] = Pair {
    params: [
        3600,
        11,
        7,
        19,
    ],
    balance: 437.71353,
}
[sparring/src/map.rs:86] &pairs[i] = Pair {
    params: [
        3600,
        14,
        14,
        67,
    ],
    balance: 404.26514,
}
[sparring/src/map.rs:86] &pairs[i] = Pair {
    params: [
        3600,
        10,
        6,
        21,
    ],
    balance: 378.5838,
}
[sparring/src/map.rs:86] &pairs[i] = Pair {
    params: [
        3600,
        14,
        14,
        37,
    ],
    balance: 372.4461,
}
[sparring/src/map.rs:86] &pairs[i] = Pair {
    params: [
        3600,
        14,
        13,
        67,
    ],
    balance: 362.8052,
}
[sparring/src/map.rs:86] &pairs[i] = Pair {
    params: [
        3600,
        11,
        5,
        19,
    ],
    balance: 344.35306,
}
[sparring/src/map.rs:86] &pairs[i] = Pair {
    params: [
        3600,
        11,
        3,
        17,
    ],
    balance: 341.1855,
}
[sparring/src/map.rs:86] &pairs[i] = Pair {
    params: [
        3600,
        11,
        3,
        19,
    ],
    balance: 336.65463,
}
[sparring/src/map.rs:86] &pairs[i] = Pair {
    params: [
        3600,
        11,
        7,
        21,
    ],
    balance: 333.66168,
}

random 100k combs
compute_cost: 24717567281,
cache_cost: 24717567281,
timeframe_cost: 12394519800,
elapsed_compute_s: 1469.1398,
elapsed_cache_s: 1674.4476,
elapsed_timeframe_s: 39.012115,
elapsed_total_s: 3182.5996,
Top n: IntoIter(
    [
        Pair {
            params: [
                2880,
                23,
                18,
                42,
            ],
            balance: 185.19232,
        },
        Pair {
            params: [
                1620,
                40,
                23,
                40,
            ],
            balance: 197.04764,
        },
        Pair {
            params: [
                2280,
                26,
                20,
                35,
            ],
            balance: 197.18811,
        },
        Pair {
            params: [
                3360,
                23,
                27,
                41,
            ],
            balance: 198.95486,
        },
        Pair {
            params: [
                3600,
                17,
                13,
                39,
            ],
            balance: 207.43898,
        },
        Pair {
            params: [
                3600,
                21,
                21,
                45,
            ],
            balance: 214.06186,
        },
        Pair {
            params: [
                3600,
                9,
                5,
                84,
            ],
            balance: 223.02179,
        },
        Pair {
            params: [
                1680,
                30,
                17,
                80,
            ],
            balance: 224.97255,
        },
        Pair {
            params: [
                1620,
                46,
                25,
                44,
            ],
            balance: 227.08669,
        },
        Pair {
            params: [
                3600,
                10,
                12,
                82,
            ],
            balance: 229.8173,
        },
    ],
)

random 10k
compute_cost: 2504612533,
cache_cost: 2504612533,
timeframe_cost: 12775889640,
elapsed_compute_s: 134.98618,
elapsed_cache_s: 158.84773,
elapsed_timeframe_s: 38.544376,
elapsed_total_s: 332.3783,
Top n: IntoIter(
    [
        Pair {
            params: [
                3420,
                28,
                18,
                36,
            ],
            balance: 110.15507,
        },
        Pair {
            params: [
                2280,
                35,
                23,
                43,
            ],
            balance: 110.43089,
        },
        Pair {
            params: [
                2340,
                18,
                17,
                56,
            ],
            balance: 110.59013,
        },
        Pair {
            params: [
                2820,
                24,
                21,
                73,
            ],
            balance: 113.11365,
        },
        Pair {
            params: [
                3000,
                24,
                19,
                77,
            ],
            balance: 123.600525,
        },
        Pair {
            params: [
                2460,
                32,
                22,
                74,
            ],
            balance: 133.8299,
        },
        Pair {
            params: [
                2640,
                13,
                8,
                86,
            ],
            balance: 134.98138,
        },
        Pair {
            params: [
                2940,
                39,
                27,
                37,
            ],
            balance: 154.9129,
        },
        Pair {
            params: [
                3600,
                30,
                25,
                67,
            ],
            balance: 165.05014,
        },
        Pair {
            params: [
                3240,
                22,
                22,
                41,
            ],
            balance: 377.18643,
        },
    ],
)

random 1k
compute_cost: 248544824,
cache_cost: 248544824,
timeframe_cost: 12013149960,
elapsed_compute_s: 13.594854,
elapsed_cache_s: 15.771128,
elapsed_timeframe_s: 35.46154,
elapsed_total_s: 64.82752,
Top n: IntoIter(
    [
        Pair {
            params: [
                2700,
                34,
                22,
                65,
            ],
            balance: 58.57169,
        },
        Pair {
            params: [
                3300,
                24,
                87,
                77,
            ],
            balance: 61.342533,
        },
        Pair {
            params: [
                1500,
                17,
                5,
                30,
            ],
            balance: 65.38169,
        },
        Pair {
            params: [
                3360,
                15,
                12,
                64,
            ],
            balance: 68.76233,
        },
        Pair {
            params: [
                3600,
                49,
                86,
                42,
            ],
            balance: 76.3867,
        },
        Pair {
            params: [
                2640,
                30,
                21,
                47,
            ],
            balance: 81.01342,
        },
        Pair {
            params: [
                1980,
                22,
                12,
                82,
            ],
            balance: 91.746704,
        },
        Pair {
            params: [
                3480,
                10,
                8,
                31,
            ],
            balance: 99.56228,
        },
        Pair {
            params: [
                900,
                46,
                19,
                71,
            ],
            balance: 111.49764,
        },
        Pair {
            params: [
                2460,
                64,
                31,
                42,
            ],
            balance: 123.600525,
        },
    ],
)

random 100
compute_cost: 26555189,
cache_cost: 26555189,
timeframe_cost: 9152876160,
elapsed_compute_s: 1.414942,
elapsed_cache_s: 1.636153,
elapsed_timeframe_s: 28.866749,
elapsed_total_s: 31.917843,
Top n: IntoIter(
    [
        Pair {
            params: [
                120,
                80,
                3,
                88,
            ],
            balance: 6.5123515,
        },
        Pair {
            params: [
                360,
                62,
                29,
                78,
            ],
            balance: 6.780918,
        },
        Pair {
            params: [
                720,
                85,
                35,
                70,
            ],
            balance: 7.5611854,
        },
        Pair {
            params: [
                720,
                71,
                83,
                59,
            ],
            balance: 12.914693,
        },
        Pair {
            params: [
                2040,
                73,
                82,
                57,
            ],
            balance: 15.30367,
        },
        Pair {
            params: [
                3180,
                19,
                19,
                84,
            ],
            balance: 15.992647,
        },
        Pair {
            params: [
                2460,
                77,
                86,
                63,
            ],
            balance: 29.055056,
        },
        Pair {
            params: [
                660,
                56,
                16,
                22,
            ],
            balance: 37.903862,
        },
        Pair {
            params: [
                3600,
                67,
                82,
                62,
            ],
            balance: 38.36777,
        },
        Pair {
            params: [
                2640,
                60,
                28,
                46,
            ],
            balance: 53.97545,
        },
    ],
)

random 10
compute_cost: 1040536,
cache_cost: 1040536,
timeframe_cost: 1906849200,
elapsed_compute_s: 0.049976,
elapsed_cache_s: 0.060806,
elapsed_timeframe_s: 5.497741,
elapsed_total_s: 5.6085234,
Top n: IntoIter(
    [
        Pair {
            params: [
                840,
                25,
                62,
                63,
            ],
            balance: 1.6324617e-7,
        },
        Pair {
            params: [
                2400,
                15,
                27,
                18,
            ],
            balance: 0.05147737,
        },
        Pair {
            params: [
                3600,
                38,
                51,
                27,
            ],
            balance: 0.060962245,
        },
        Pair {
            params: [
                2580,
                72,
                54,
                78,
            ],
            balance: 0.11448978,
        },
        Pair {
            params: [
                2760,
                73,
                55,
                99,
            ],
            balance: 0.12636212,
        },
        Pair {
            params: [
                2340,
                64,
                8,
                85,
            ],
            balance: 0.9742327,
        },
        Pair {
            params: [
                3300,
                33,
                5,
                51,
            ],
            balance: 0.9985,
        },
        Pair {
            params: [
                3420,
                59,
                97,
                5,
            ],
            balance: 1.0,
        },
        Pair {
            params: [
                960,
                77,
                21,
                25,
            ],
            balance: 10.695549,
        },
        Pair {
            params: [
                1320,
                97,
                21,
                38,
            ],
            balance: 17.19032,
        },
    ],
)

bayes 10
compute_cost: 17328779,
cache_cost: 17328779,
timeframe_cost: 1525479360,
elapsed_compute_s: 0.783124,
elapsed_cache_s: 0.924763,
elapsed_timeframe_s: 1.566858,
elapsed_total_s: 3.274745,
Top n: IntoIter(
    [
        Pair {
            params: [
                3660,
                2,
                0,
                0,
            ],
            balance: 1.0,
        },
        Pair {
            params: [
                60,
                2,
                101,
                0,
            ],
            balance: 1.0,
        },
        Pair {
            params: [
                60,
                2,
                0,
                101,
            ],
            balance: 1.0,
        },
        Pair {
            params: [
                60,
                2,
                0,
                0,
            ],
            balance: 1.0,
        },
        Pair {
            params: [
                3480,
                96,
                96,
                96,
            ],
            balance: 1.0,
        },
        Pair {
            params: [
                480,
                76,
                76,
                76,
            ],
            balance: 1.0982119,
        },
        Pair {
            params: [
                120,
                68,
                6,
                68,
            ],
            balance: 4.0051103,
        },
        Pair {
            params: [
                480,
                71,
                21,
                71,
            ],
            balance: 20.768347,
        },
        Pair {
            params: [
                480,
                70,
                13,
                70,
            ],
            balance: 21.395294,
        },
        Pair {
            params: [
                180,
                70,
                15,
                70,
            ],
            balance: 29.564167,
        },
    ],
)

bayes 100
compute_cost: 40116367,
cache_cost: 40116367,
timeframe_cost: 7436711880,
elapsed_compute_s: 1.378206,
elapsed_cache_s: 1.823525,
elapsed_timeframe_s: 7.438887,
elapsed_total_s: 10.640618,
Top n: IntoIter(
    [
        Pair {
            params: [
                420,
                66,
                20,
                27,
            ],
            balance: 61.428204,
        },
        Pair {
            params: [
                2040,
                28,
                19,
                42,
            ],
            balance: 61.661804,
        },
        Pair {
            params: [
                2700,
                32,
                22,
                48,
            ],
            balance: 62.837284,
        },
        Pair {
            params: [
                2340,
                21,
                10,
                21,
            ],
            balance: 70.686195,
        },
        Pair {
            params: [
                2880,
                25,
                13,
                33,
            ],
            balance: 73.731346,
        },
        Pair {
            params: [
                3000,
                26,
                13,
                27,
            ],
            balance: 84.23508,
        },
        Pair {
            params: [
                2640,
                35,
                24,
                53,
            ],
            balance: 86.37532,
        },
        Pair {
            params: [
                480,
                78,
                24,
                33,
            ],
            balance: 93.67146,
        },
        Pair {
            params: [
                2700,
                33,
                22,
                49,
            ],
            balance: 93.68103,
        },
        Pair {
            params: [
                2880,
                25,
                13,
                27,
            ],
            balance: 97.71133,
            },
    ],
)

bayes 1k
compute_cost: 203508979,
cache_cost: 203508979,
timeframe_cost: 11441095200,
elapsed_compute_s: 8.72996,
elapsed_cache_s: 10.648926,
elapsed_timeframe_s: 13.348413,
elapsed_total_s: 32.727303,
Top n: IntoIter(
    [
        Pair {
            params: [
                2820,
                16,
                12,
                34,
            ],
            balance: 120.122444,
        },
        Pair {
            params: [
                1980,
                29,
                18,
                26,
            ],
            balance: 126.50907,
        },
        Pair {
            params: [
                2460,
                33,
                25,
                44,
            ],
            balance: 130.35327,
        },
        Pair {
            params: [
                2820,
                19,
                13,
                20,
            ],
            balance: 132.7809,
        },
        Pair {
            params: [
                2220,
                31,
                21,
                43,
            ],
            balance: 151.01424,
        },
        Pair {
            params: [
                2460,
                33,
                26,
                43,
            ],
            balance: 152.85992,
        },
        Pair {
            params: [
                2280,
                29,
                19,
                41,
            ],
            balance: 174.50792,
        },
        Pair {
            params: [
                2280,
                31,
                21,
                44,
            ],
            balance: 191.86713,
        },
        Pair {
            params: [
                2220,
                30,
                20,
                44,
            ],
            balance: 204.45781,
        },
        Pair {
            params: [
                2280,
                30,
                19,
                42,
            ],
            balance: 213.71512,
        },
    ],
)

bayes 10k
compute_cost: 2054102215,
cache_cost: 2054102215,
timeframe_cost: 11631780120,
elapsed_compute_s: 70.55814,
elapsed_cache_s: 93.09398,
elapsed_timeframe_s: 10.383355,
elapsed_total_s: 174.03549,
Top n: IntoIter(
    [
        Pair {
            params: [
                3060,
                7,
                1,
                84,
            ],
            balance: 195.42802,
        },
        Pair {
            params: [
                2220,
                30,
                20,
                44,
            ],
            balance: 204.45781,
        },
        Pair {
            params: [
                2280,
                31,
                19,
                55,
            ],
            balance: 206.85373,
        },
        Pair {
            params: [
                2280,
                30,
                19,
                42,
            ],
            balance: 213.71512,
        },
        Pair {
            params: [
                2640,
                28,
                21,
                44,
            ],
            balance: 216.89798,
        },
        Pair {
            params: [
                2640,
                30,
                25,
                40,
            ],
            balance: 219.43802,
        },
        Pair {
            params: [
                2700,
                30,
                23,
                47,
            ],
            balance: 229.21971,
        },
        Pair {
            params: [
                2640,
                27,
                21,
                42,
            ],
            balance: 235.03368,
        },
        Pair {
            params: [
                1800,
                39,
                28,
                43,
            ],
            balance: 239.98772,
        },
        Pair {
            params: [
                2280,
                38,
                26,
                44,
            ],
            balance: 271.193,
        },
    ],
)

bayes 100k
compute_cost: 21263870520,
cache_cost: 21263870520,
timeframe_cost: 11631780120,
elapsed_compute_s: 731.9564,
elapsed_cache_s: 966.7298,
elapsed_timeframe_s: 14.737782,
elapsed_total_s: 1713.4241,
Top n: IntoIter(
    [
        Pair {
            params: [
                2640,
                28,
                22,
                43,
            ],
            balance: 517.40356,
        },
        Pair {
            params: [
                2640,
                28,
                22,
                43,
            ],
            balance: 517.40356,
        },
        Pair {
            params: [
                2640,
                28,
                22,
                43,
            ],
            balance: 517.40356,
        },
        Pair {
            params: [
                2640,
                28,
                22,
                43,
            ],
            balance: 517.40356,
        },
        Pair {
            params: [
                2640,
                28,
                22,
                42,
            ],
            balance: 557.80023,
        },
        Pair {
            params: [
                2640,
                28,
                22,
                42,
            ],
            balance: 557.80023,
        },
        Pair {
            params: [
                2640,
                28,
                22,
                42,
            ],
            balance: 557.80023,
        },
        Pair {
            params: [
                2640,
                28,
                22,
                42,
            ],
            balance: 557.80023,
        },
        Pair {
            params: [
                2640,
                28,
                22,
                42,
            ],
            balance: 557.80023,
        },
        Pair {
            params: [
                2640,
                28,
                22,
                42,
            ],
            balance: 557.80023,
        },
    ],
)

ga 10pop 10gen
ompute_cost: 6396717,
cache_cost: 6396717,
timeframe_cost: 5911232520,
elapsed_compute_s: 0.286605,
elapsed_cache_s: 0.357739,
elapsed_timeframe_s: 15.974192,
elapsed_total_s: 16.618536,
Top n: IntoIter(
    [
        Pair {
            params: [
                3540,
                74,
                85,
                28,
            ],
            balance: 9.763032,
        },
        Pair {
            params: [
                2760,
                52,
                83,
                26,
            ],
            balance: 12.256326,
        },
        Pair {
            params: [
                3300,
                47,
                32,
                64,
            ],
            balance: 12.597843,
        },
        Pair {
            params: [
                2700,
                74,
                88,
                36,
            ],
            balance: 13.95096,
        },
        Pair {
            params: [
                3480,
                72,
                88,
                49,
            ],
            balance: 15.553964,
        },
        Pair {
            params: [
                2520,
                84,
                85,
                60,
            ],
            balance: 16.004335,
        },
        Pair {
            params: [
                2760,
                87,
                40,
                55,
            ],
            balance: 17.452631,
        },
        Pair {
            params: [
                2220,
                92,
                86,
                50,
            ],
            balance: 17.64258,
        },
        Pair {
            params: [
                1560,
                50,
                85,
                37,
            ],
            balance: 35.29802,
        },
        Pair {
            params: [
                3420,
                71,
                30,
                57,
            ],
            balance: 40.669014,
        },
    ],
)

ga 10pop 100gen
compute_cost: 54936706,
cache_cost: 54936706,
timeframe_cost: 7246026960,
elapsed_compute_s: 2.794731,
elapsed_cache_s: 3.300838,
elapsed_timeframe_s: 17.102297,
elapsed_total_s: 23.197865,
Top n: IntoIter(
    [
        Pair {
            params: [
                3240,
                56,
                82,
                64,
            ],
            balance: 64.461296,
        },
        Pair {
            params: [
                3240,
                56,
                82,
                64,
            ],
            balance: 64.461296,
        },
        Pair {
            params: [
                3240,
                56,
                82,
                64,
            ],
            balance: 64.461296,
        },
        Pair {
            params: [
                3240,
                56,
                82,
                64,
            ],
            balance: 64.461296,
        },
        Pair {
            params: [
                3240,
                56,
                82,
                64,
            ],
            balance: 64.461296,
        },
        Pair {
            params: [
                3240,
                56,
                82,
                64,
            ],
            balance: 64.461296,
        },
        Pair {
            params: [
                3240,
                56,
                82,
                64,
            ],
            balance: 64.461296,
        },
        Pair {
            params: [
                3240,
                56,
                82,
                64,
            ],
            balance: 64.461296,
        },
        Pair {
            params: [
                3180,
                58,
                81,
                64,
            ],
            balance: 68.029305,
        },
        Pair {
            params: [
                2340,
                31,
                21,
                75,
            ],
            balance: 76.11356,
        },
    ],
)

ga 10pop 1000gen
compute_cost: 130206680,
cache_cost: 130206680,
timeframe_cost: 10106300760,
elapsed_compute_s: 4.756285,
elapsed_cache_s: 5.797197,
elapsed_timeframe_s: 8.960727,
elapsed_total_s: 19.514209,
Top n: IntoIter(
    [
        Pair {
            params: [
                2280,
                33,
                22,
                43,
            ],
            balance: 309.1273,
        },
        Pair {
            params: [
                2280,
                33,
                22,
                43,
            ],
            balance: 309.48526,
        },
        Pair {
            params: [
                2280,
                33,
                22,
                43,
            ],
            balance: 309.48526,
        },
        Pair {
            params: [
                2280,
                33,
                22,
                43,
            ],
            balance: 309.48526,
        },
        Pair {
            params: [
                2280,
                33,
                22,
                43,
            ],
            balance: 309.48526,
        },
        Pair {
            params: [
                2640,
                29,
                23,
                43,
            ],
            balance: 313.032,
        },
        Pair {
            params: [
                2280,
                33,
                22,
                43,
            ],
            balance: 313.18814,
        },
        Pair {
            params: [
                2280,
                33,
                22,
                43,
            ],
            balance: 313.18814,
        },
        Pair {
            params: [
                2280,
                33,
                22,
                43,
            ],
            balance: 313.18814,
        },
        Pair {
            params: [
                2280,
                36,
                23,
                45,
            ],
            balance: 338.00708,
        },
    ],
)

ga 100pop 10gen
compute_cost: 83915085,
cache_cost: 83915085,
timeframe_cost: 10869040440,
elapsed_compute_s: 2.748988,
elapsed_cache_s: 3.752641,
elapsed_timeframe_s: 9.622249,
elapsed_total_s: 16.123878,
Top n: IntoIter(
    [
        Pair {
            params: [
                2340,
                43,
                22,
                54,
            ],
            balance: 100.107445,
        },
        Pair {
            params: [
                2220,
                31,
                18,
                49,
            ],
            balance: 105.93791,
        },
        Pair {
            params: [
                2280,
                31,
                21,
                56,
            ],
            balance: 114.67247,
        },
        Pair {
            params: [
                2280,
                20,
                17,
                57,
            ],
            balance: 128.19205,
        },
        Pair {
            params: [
                2280,
                32,
                22,
                49,
            ],
            balance: 132.57103,
        },
        Pair {
            params: [
                2040,
                54,
                31,
                48,
            ],
            balance: 145.49425,
        },
        Pair {
            params: [
                3000,
                41,
                25,
                44,
            ],
            balance: 153.91768,
        },
        Pair {
            params: [
                1800,
                46,
                24,
                46,
            ],
            balance: 166.4525,
        },
        Pair {
            params: [
                2340,
                31,
                20,
                52,
            ],
            balance: 180.96411,
        },
        Pair {
            params: [
                2100,
                42,
                23,
                37,
            ],
            balance: 289.5834,
        },
    ],
)
ga 100pop 100gen
compute_cost: 617432167,
cache_cost: 617432167,
timeframe_cost: 10678355520,
elapsed_compute_s: 20.646317,
elapsed_cache_s: 27.554161,
elapsed_timeframe_s: 9.937522,
elapsed_total_s: 58.138,
Top n: IntoIter(
    [
        Pair {
            params: [
                2640,
                28,
                21,
                42,
            ],
            balance: 675.41675,
        },
        Pair {
            params: [
                2640,
                28,
                21,
                42,
            ],
            balance: 675.41675,
        },
        Pair {
            params: [
                2640,
                28,
                21,
                42,
            ],
            balance: 675.41675,
        },
        Pair {
            params: [
                2640,
                28,
                21,
                42,
            ],
            balance: 675.41675,
        },
        Pair {
            params: [
                2640,
                28,
                21,
                42,
            ],
            balance: 675.41675,
        },
        Pair {
            params: [
                2640,
                28,
                21,
                42,
            ],
            balance: 675.41675,
        },
        Pair {
            params: [
                2640,
                28,
                21,
                42,
            ],
            balance: 675.41675,
        },
        Pair {
            params: [
                2640,
                28,
                21,
                42,
            ],
            balance: 675.41675,
        },
        Pair {
            params: [
                2640,
                28,
                21,
                42,
            ],
            balance: 675.41675,
        },
        Pair {
            params: [
                2640,
                28,
                21,
                42,
            ],
            balance: 675.41675,
        },
    ],
)
ga 100pop 1000gen
compute_cost: 433662202,
cache_cost: 433662202,
timeframe_cost: 11250410280,
elapsed_compute_s: 14.099861,
elapsed_cache_s: 19.379557,
elapsed_timeframe_s: 9.997656,
elapsed_total_s: 43.477074,
Top n: IntoIter(
    [
        Pair {
            params: [
                2820,
                25,
                19,
                38,
            ],
            balance: 745.4738,
        },
        Pair {
            params: [
                2820,
                25,
                19,
                38,
            ],
            balance: 745.4738,
        },
        Pair {
            params: [
                2820,
                25,
                19,
                38,
            ],
            balance: 745.4738,
        },
        Pair {
            params: [
                2820,
                25,
                19,
                38,
            ],
            balance: 745.4738,
        },
        Pair {
            params: [
                2820,
                25,
                19,
                38,
            ],
            balance: 745.4738,
        },
        Pair {
            params: [
                2820,
                25,
                19,
                38,
            ],
            balance: 745.4738,
        },
        Pair {
            params: [
                2820,
                25,
                19,
                38,
            ],
            balance: 745.4738,
        },
        Pair {
            params: [
                2820,
                25,
                19,
                38,
            ],
            balance: 745.4738,
        },
        Pair {
            params: [
                2820,
                25,
                19,
                38,
            ],
            balance: 745.4738,
        },
        Pair {
            params: [
                2820,
                25,
                19,
                38,
            ],
            balance: 745.4738,
        },
    ],
)
ga 1000pop 10gen
compute_cost: 1039984285,
cache_cost: 1039984285,
timeframe_cost: 11250410280,
elapsed_compute_s: 34.044113,
elapsed_cache_s: 46.498325,
elapsed_timeframe_s: 10.209258,
elapsed_total_s: 90.75169,
Top n: IntoIter(
    [
        Pair {
            params: [
                2340,
                29,
                20,
                53,
            ],
            balance: 189.4048,
        },
        Pair {
            params: [
                2280,
                36,
                22,
                46,
            ],
            balance: 190.39441,
        },
        Pair {
            params: [
                3300,
                18,
                17,
                39,
            ],
            balance: 215.53116,
        },
        Pair {
            params: [
                2220,
                42,
                25,
                40,
            ],
            balance: 216.17363,
        },
        Pair {
            params: [
                2640,
                25,
                21,
                42,
            ],
            balance: 223.93332,
        },
        Pair {
            params: [
                2940,
                14,
                10,
                53,
            ],
            balance: 237.64221,
        },
        Pair {
            params: [
                3420,
                14,
                14,
                52,
            ],
            balance: 243.4387,
        },
        Pair {
            params: [
                2280,
                48,
                26,
                38,
            ],
            balance: 247.57135,
        },
        Pair {
            params: [
                3120,
                18,
                14,
                39,
            ],
            balance: 287.20755,
        },
        Pair {
            params: [
                2640,
                25,
                21,
                41,
            ],
            balance: 352.9534,
        },
    ],
)

ga 1000pop 100gen
compute_cost: 4902319850,
cache_cost: 4902319850,
timeframe_cost: 11250410280,
elapsed_compute_s: 159.03088,
elapsed_cache_s: 218.68188,
elapsed_timeframe_s: 10.269542,
elapsed_total_s: 387.9823,
Top n: IntoIter(
    [
        Pair {
            params: [
                3240,
                22,
                21,
                41,
            ],
            balance: 759.38184,
        },
        Pair {
            params: [
                3240,
                22,
                21,
                41,
            ],
            balance: 759.38184,
        },
        Pair {
            params: [
                3240,
                22,
                21,
                41,
            ],
            balance: 759.38184,
        },
        Pair {
            params: [
                3240,
                22,
                21,
                41,
            ],
            balance: 759.38184,
        },
        Pair {
            params: [
                3240,
                22,
                21,
                41,
            ],
            balance: 759.38184,
        },
        Pair {
            params: [
                3240,
                22,
                21,
                41,
            ],
            balance: 759.38184,
        },
        Pair {
            params: [
                3240,
                22,
                21,
                41,
            ],
            balance: 759.38184,
        },
        Pair {
            params: [
                3240,
                22,
                21,
                41,
            ],
            balance: 759.38184,
        },
        Pair {
            params: [
                3240,
                22,
                21,
                41,
            ],
            balance: 759.38184,
        },
        Pair {
            params: [
                3240,
                22,
                21,
                41,
            ],
            balance: 759.38184,
        },
    ],
)
ga 1000pop 1000gen
compute_cost: 6042925591,
cache_cost: 6042925591,
timeframe_cost: 11250410280,
elapsed_compute_s: 194.34259,
elapsed_cache_s: 267.2354,
elapsed_timeframe_s: 10.186386,
elapsed_total_s: 471.7644,
Top n: IntoIter(
    [
        Pair {
            params: [
                2640,
                27,
                22,
                40,
            ],
            balance: 707.50305,
        },
        Pair {
            params: [
                2640,
                27,
                22,
                40,
            ],
            balance: 707.50305,
        },
        Pair {
            params: [
                2640,
                27,
                22,
                40,
            ],
            balance: 707.50305,
        },
        Pair {
            params: [
                2640,
                27,
                22,
                40,
            ],
            balance: 709.78656,
        },
        Pair {
            params: [
                2640,
                27,
                22,
                40,
            ],
            balance: 710.6311,
        },
        Pair {
            params: [
                2640,
                27,
                22,
                40,
            ],
            balance: 711.8946,
        },
        Pair {
            params: [
                2640,
                27,
                22,
                40,
            ],
            balance: 712.89844,
        },
        Pair {
            params: [
                2640,
                27,
                22,
                40,
            ],
            balance: 716.8214,
        },
        Pair {
            params: [
                2640,
                27,
                22,
                40,
            ],
            balance: 724.34314,
        },
        Pair {
            params: [
                2640,
                27,
                22,
                40,
            ],
            balance: 724.34314,
        },
    ],
)

hybrid 10 10pop 10gen
compute_cost: 306522912,
cache_cost: 306522912,
timeframe_cost: 2478903960,
elapsed_compute_s: 10.153179,
elapsed_cache_s: 13.507719,
elapsed_timeframe_s: 2.244725,
elapsed_total_s: 25.905622,
Top n: IntoIter(
    [
        Pair {
            params: [
                2760,
                63,
                28,
                41,
            ],
            balance: 45.237144,
        },
        Pair {
            params: [
                2520,
                78,
                32,
                44,
            ],
            balance: 57.155975,
        },
        Pair {
            params: [
                720,
                56,
                16,
                66,
            ],
            balance: 60.98636,
        },
        Pair {
            params: [
                480,
                85,
                24,
                67,
            ],
            balance: 64.92354,
        },
        Pair {
            params: [
                3420,
                26,
                17,
                46,
            ],
            balance: 71.21261,
        },
        Pair {
            params: [
                1860,
                56,
                27,
                62,
            ],
            balance: 84.24218,
        },
        Pair {
            params: [
                3180,
                43,
                26,
                54,
            ],
            balance: 92.08949,
        },
        Pair {
            params: [
                3660,
                62,
                26,
                42,
            ],
            balance: 94.283325,
        },
        Pair {
            params: [
                960,
                26,
                15,
                80,
            ],
            balance: 106.816605,
        },
        Pair {
            params: [
                2280,
                16,
                12,
                55,
            ],
            balance: 143.58241,
        },
    ],
)

hybrid 10 10pop 100gen
compute_cost: 3404078089,
cache_cost: 3404078089,
timeframe_cost: 2478903960,
elapsed_compute_s: 107.54804,
elapsed_cache_s: 149.61111,
elapsed_timeframe_s: 2.232527,
elapsed_total_s: 259.39166,
Top n: IntoIter(
    [
        Pair {
            params: [
                2520,
                63,
                28,
                42,
            ],
            balance: 197.16766,
        },
        Pair {
            params: [
                2940,
                32,
                23,
                54,
            ],
            balance: 200.99287,
        },
        Pair {
            params: [
                2040,
                84,
                32,
                45,
            ],
            balance: 214.39824,
        },
        Pair {
            params: [
                1860,
                42,
                25,
                44,
            ],
            balance: 259.8285,
        },
        Pair {
            params: [
                2760,
                60,
                30,
                43,
            ],
            balance: 275.77414,
        },
        Pair {
            params: [
                2160,
                38,
                24,
                45,
            ],
            balance: 307.46112,
        },
        Pair {
            params: [
                3180,
                25,
                21,
                43,
            ],
            balance: 338.61655,
        },
        Pair {
            params: [
                3660,
                22,
                18,
                41,
            ],
            balance: 352.37793,
        },
        Pair {
            params: [
                1380,
                64,
                28,
                39,
            ],
            balance: 380.4573,
        },
        Pair {
            params: [
                2280,
                37,
                22,
                38,
            ],
            balance: 404.63754,
        },
    ],
)
hybrid 10 10pop 1000gen
compute_cost: 3858658369,
cache_cost: 3858658369,
timeframe_cost: 2478903960,
elapsed_compute_s: 121.82603,
elapsed_cache_s: 169.52773,
elapsed_timeframe_s: 2.214145,
elapsed_total_s: 293.5679,
Top n: IntoIter(
    [
        Pair {
            params: [
                2820,
                65,
                30,
                48,
            ],
            balance: 149.82085,
        },
        Pair {
            params: [
                3660,
                71,
                29,
                54,
            ],
            balance: 175.72424,
        },
        Pair {
            params: [
                2760,
                65,
                29,
                43,
            ],
            balance: 294.63766,
        },
        Pair {
            params: [
                2400,
                18,
                16,
                46,
            ],
            balance: 296.42712,
        },
        Pair {
            params: [
                3180,
                22,
                17,
                43,
            ],
            balance: 324.22882,
        },
        Pair {
            params: [
                2040,
                42,
                26,
                47,
            ],
            balance: 330.1075,
        },
        Pair {
            params: [
                2940,
                33,
                22,
                37,
            ],
            balance: 336.47348,
        },
        Pair {
            params: [
                2520,
                24,
                20,
                73,
            ],
            balance: 344.1203,
        },
        Pair {
            params: [
                2160,
                38,
                21,
                45,
            ],
            balance: 374.80103,
        },
        Pair {
            params: [
                2280,
                31,
                23,
                43,
            ],
            balance: 411.746,
        },
    ],
)
hybrid 10 100pop 10gen
compute_cost: 3035680540,
cache_cost: 3035680540,
timeframe_cost: 2478903960,
elapsed_compute_s: 97.159004,
elapsed_cache_s: 154.51614,
elapsed_timeframe_s: 2.326808,
elapsed_total_s: 254.00195,
Top n: IntoIter(
    [
        Pair {
            params: [
                960,
                86,
                31,
                44,
            ],
            balance: 147.11638,
        },
        Pair {
            params: [
                2760,
                66,
                29,
                45,
            ],
            balance: 163.43506,
        },
        Pair {
            params: [
                2040,
                86,
                32,
                49,
            ],
            balance: 173.18988,
        },
        Pair {
            params: [
                2280,
                31,
                21,
                52,
            ],
            balance: 175.626,
        },
        Pair {
            params: [
                1620,
                42,
                31,
                43,
            ],
            balance: 184.27649,
        },
        Pair {
            params: [
                3660,
                12,
                10,
                49,
            ],
            balance: 195.10973,
        },
        Pair {
            params: [
                1140,
                72,
                29,
                47,
            ],
            balance: 210.95575,
        },
        Pair {
            params: [
                1380,
                81,
                29,
                39,
            ],
            balance: 212.51007,
        },
        Pair {
            params: [
                3180,
                27,
                22,
                44,
            ],
            balance: 230.9076,
        },
        Pair {
            params: [
                1860,
                41,
                24,
                40,
            ],
            balance: 232.88274,
        },
    ],
)
hybrid 10 100pop 100gen
compute_cost: 17384068172,
cache_cost: 17384068172,
timeframe_cost: 2478903960,
elapsed_compute_s: 551.59705,
elapsed_cache_s: 767.24725,
elapsed_timeframe_s: 2.268069,
elapsed_total_s: 1321.1123,
Top n: IntoIter(
    [
        Pair {
            params: [
                3660,
                21,
                18,
                43,
            ],
            balance: 218.91711,
        },
        Pair {
            params: [
                2580,
                69,
                30,
                49,
            ],
            balance: 259.94702,
        },
        Pair {
            params: [
                2040,
                48,
                26,
                40,
            ],
            balance: 275.27216,
        },
        Pair {
            params: [
                1860,
                43,
                25,
                44,
            ],
            balance: 305.20923,
        },
        Pair {
            params: [
                2400,
                31,
                20,
                43,
            ],
            balance: 360.44604,
        },
        Pair {
            params: [
                1380,
                64,
                28,
                39,
            ],
            balance: 381.8689,
        },
        Pair {
            params: [
                2760,
                29,
                22,
                44,
            ],
            balance: 393.33847,
        },
        Pair {
            params: [
                2520,
                35,
                23,
                38,
            ],
            balance: 524.40814,
        },
        Pair {
            params: [
                2280,
                39,
                24,
                39,
            ],
            balance: 533.39014,
        },
        Pair {
            params: [
                2640,
                28,
                21,
                43,
            ],
            balance: 645.25665,
        },
    ],
)

hybrid 10 100pop 1000gen
compute_cost: 19965894448,
cache_cost: 19965894448,
timeframe_cost: 2478903960,
elapsed_compute_s: 633.0583,
elapsed_cache_s: 1021.1015,
elapsed_timeframe_s: 2.250441,
elapsed_total_s: 1656.4103,
Top n: IntoIter(
    [
        Pair {
            params: [
                1860,
                43,
                24,
                65,
            ],
            balance: 263.72772,
        },
        Pair {
            params: [
                2760,
                36,
                21,
                33,
            ],
            balance: 287.06046,
        },
        Pair {
            params: [
                2520,
                72,
                30,
                48,
            ],
            balance: 319.66055,
        },
        Pair {
            params: [
                2040,
                42,
                26,
                47,
            ],
            balance: 330.8478,
        },
        Pair {
            params: [
                2400,
                14,
                13,
                53,
            ],
            balance: 335.18515,
        },
        Pair {
            params: [
                3180,
                25,
                21,
                42,
            ],
            balance: 351.20306,
        },
        Pair {
            params: [
                2160,
                37,
                24,
                39,
            ],
            balance: 362.15906,
        },
        Pair {
            params: [
                3300,
                23,
                21,
                44,
            ],
            balance: 363.18658,
        },
        Pair {
            params: [
                2280,
                39,
                24,
                39,
            ],
            balance: 431.63718,
        },
        Pair {
            params: [
                3420,
                21,
                19,
                38,
            ],
            balance: 577.86584,
        },
    ],
)

hybrid 10 1000pop 10gen
compute_cost: 29910088251,
cache_cost: 29910088251,
timeframe_cost: 2478903960,
elapsed_compute_s: 1034.9928,
elapsed_cache_s: 1377.8097,
elapsed_timeframe_s: 15.976849,
elapsed_total_s: 2428.7793,
Top n: IntoIter(
    [
        Pair {
            params: [
                2760,
                31,
                19,
                35,
            ],
            balance: 200.33066,
        },
        Pair {
            params: [
                1380,
                56,
                27,
                39,
            ],
            balance: 232.57237,
        },
        Pair {
            params: [
                3180,
                15,
                15,
                49,
            ],
            balance: 316.79764,
        },
        Pair {
            params: [
                1860,
                37,
                25,
                43,
            ],
            balance: 334.80307,
        },
        Pair {
            params: [
                3420,
                14,
                15,
                48,
            ],
            balance: 366.3102,
        },
        Pair {
            params: [
                2280,
                35,
                22,
                38,
            ],
            balance: 386.22348,
        },
        Pair {
            params: [
                2520,
                33,
                22,
                36,
            ],
            balance: 447.21106,
        },
        Pair {
            params: [
                3660,
                21,
                17,
                39,
            ],
            balance: 484.07056,
        },
        Pair {
            params: [
                3540,
                19,
                20,
                39,
            ],
            balance: 504.0864,
        },
        Pair {
            params: [
                3600,
                21,
                20,
                37,
            ],
            balance: 570.05,
        },
    ],
)
hybrid 10 1000pop 100gen
compute_cost: 197161604420,
cache_cost: 197161604420,
timeframe_cost: 2478903960,
elapsed_compute_s: 7053.041,
elapsed_cache_s: 9344.401,
elapsed_timeframe_s: 136.14291,
elapsed_total_s: 16533.584,
Top n: IntoIter(
    [
        Pair {
            params: [
                3180,
                15,
                15,
                49,
            ],
            balance: 355.66992,
        },
        Pair {
            params: [
                2400,
                30,
                21,
                39,
            ],
            balance: 376.4355,
        },
        Pair {
            params: [
                1860,
                37,
                25,
                43,
            ],
            balance: 378.8169,
        },
        Pair {
            params: [
                2760,
                29,
                22,
                44,
            ],
            balance: 393.33847,
        },
        Pair {
            params: [
                3480,
                21,
                21,
                41,
            ],
            balance: 484.43918,
        },
        Pair {
            params: [
                3660,
                21,
                17,
                39,
            ],
            balance: 496.8898,
        },
        Pair {
            params: [
                2520,
                33,
                21,
                36,
            ],
            balance: 518.8472,
        },
        Pair {
            params: [
                2280,
                39,
                24,
                39,
            ],
            balance: 533.39014,
        },
        Pair {
            params: [
                3420,
                21,
                19,
                38,
            ],
            balance: 589.10114,
        },
        Pair {
            params: [
                3540,
                19,
                22,
                40,
            ],
            balance: 1009.7302,
        },
    ],
)
hybrid 10 1000pop 1000gen
compute_cost: 202022867262,
cache_cost: 202022867262,
timeframe_cost: 2478903960,
elapsed_compute_s: 7060.048,
elapsed_cache_s: 9447.537,
elapsed_timeframe_s: 57.805393,
elapsed_total_s: 16565.39,
Top n: IntoIter(
    [
        Pair {
            params: [
                3180,
                25,
                20,
                43,
            ],
            balance: 356.37988,
        },
        Pair {
            params: [
                2400,
                30,
                21,
                39,
            ],
            balance: 376.4355,
        },
        Pair {
            params: [
                1380,
                64,
                28,
                39,
            ],
            balance: 386.22513,
        },
        Pair {
            params: [
                2760,
                29,
                22,
                44,
            ],
            balance: 393.33847,
        },
        Pair {
            params: [
                3660,
                21,
                17,
                39,
            ],
            balance: 496.8898,
        },
        Pair {
            params: [
                2520,
                35,
                24,
                38,
            ],
            balance: 531.3792,
        },
        Pair {
            params: [
                2280,
                43,
                25,
                38,
            ],
            balance: 542.72986,
        },
        Pair {
            params: [
                3480,
                21,
                21,
                41,
            ],
            balance: 552.3362,
        },
        Pair {
            params: [
                3420,
                17,
                19,
                33,
            ],
            balance: 661.34106,
        },
        Pair {
            params: [
                3540,
                19,
                22,
                40,
            ],
            balance: 1009.7302,
        },
    ],
)
hybrid 15 10po 10gen
compute_cost: 315305762,
cache_cost: 315305762,
timeframe_cost: 3432328560,
elapsed_compute_s: 13.557411,
elapsed_cache_s: 16.585636,
elapsed_timeframe_s: 8.17436,
elapsed_total_s: 38.317406,
Top n: IntoIter(
    [
        Pair {
            params: [
                3600,
                31,
                20,
                66,
            ],
            balance: 65.86082,
        },
        Pair {
            params: [
                1860,
                74,
                28,
                41,
            ],
            balance: 67.443855,
        },
        Pair {
            params: [
                1140,
                74,
                31,
                47,
            ],
            balance: 67.82524,
        },
        Pair {
            params: [
                1620,
                49,
                25,
                70,
            ],
            balance: 75.903595,
        },
        Pair {
            params: [
                2760,
                41,
                23,
                41,
            ],
            balance: 87.44673,
        },
        Pair {
            params: [
                3660,
                30,
                23,
                38,
            ],
            balance: 96.48518,
        },
        Pair {
            params: [
                1500,
                59,
                24,
                38,
            ],
            balance: 100.11322,
        },
        Pair {
            params: [
                3420,
                20,
                14,
                42,
            ],
            balance: 114.28564,
        },
        Pair {
            params: [
                3540,
                30,
                25,
                38,
            ],
            balance: 134.0334,
        },
        Pair {
            params: [
                1380,
                61,
                28,
                42,
            ],
            balance: 189.83562,
        },
    ],
)
hybrid 15 10pop 100gen
compute_cost: 3995088993,
cache_cost: 3995088993,
timeframe_cost: 3241643640,
elapsed_compute_s: 145.00249,
elapsed_cache_s: 193.102,
elapsed_timeframe_s: 25.629286,
elapsed_total_s: 363.73376,
Top n: IntoIter(
    [
        Pair {
            params: [
                2520,
                91,
                76,
                50,
            ],
            balance: 104.440956,
        },
        Pair {
            params: [
                2040,
                32,
                83,
                35,
            ],
            balance: 106.91997,
        },
        Pair {
            params: [
                1860,
                56,
                27,
                59,
            ],
            balance: 123.185814,
        },
        Pair {
            params: [
                2940,
                90,
                74,
                52,
            ],
            balance: 124.8075,
        },
        Pair {
            params: [
                960,
                52,
                25,
                55,
            ],
            balance: 147.34352,
        },
        Pair {
            params: [
                2640,
                66,
                32,
                55,
            ],
            balance: 191.93227,
        },
        Pair {
            params: [
                2280,
                23,
                16,
                64,
            ],
            balance: 219.34068,
        },
        Pair {
            params: [
                2760,
                60,
                30,
                43,
            ],
            balance: 292.232,
        },
        Pair {
            params: [
                3180,
                22,
                17,
                43,
            ],
            balance: 324.4479,
        },
        Pair {
            params: [
                2700,
                11,
                8,
                83,
            ],
            balance: 414.3159,
        },
    ],
)
hybrid 15 10pop 1000gen
compute_cost: 4306684439,
cache_cost: 4306684439,
timeframe_cost: 3432328560,
elapsed_compute_s: 171.03067,
elapsed_cache_s: 218.43852,
elapsed_timeframe_s: 4.105565,
elapsed_total_s: 393.57474,
Top n: IntoIter(
    [
        Pair {
            params: [
                2040,
                32,
                21,
                62,
            ],
            balance: 192.75117,
        },
        Pair {
            params: [
                3420,
                11,
                10,
                81,
            ],
            balance: 212.95255,
        },
        Pair {
            params: [
                1500,
                58,
                25,
                40,
            ],
            balance: 216.48526,
        },
        Pair {
            params: [
                3660,
                67,
                28,
                54,
            ],
            balance: 222.60011,
        },
        Pair {
            params: [
                960,
                26,
                15,
                81,
            ],
            balance: 229.42938,
        },
        Pair {
            params: [
                1620,
                50,
                25,
                63,
            ],
            balance: 240.37624,
        },
        Pair {
            params: [
                2280,
                68,
                29,
                46,
            ],
            balance: 260.42642,
        },
        Pair {
            params: [
                1140,
                72,
                29,
                46,
            ],
            balance: 266.28473,
        },
        Pair {
            params: [
                3540,
                95,
                72,
                57,
            ],
            balance: 320.50006,
        },
        Pair {
            params: [
                1380,
                55,
                28,
                65,
            ],
            balance: 341.49133,
        },
    ],
)
hybrid 15 100pop 10gen
compute_cost: 3425676868,
cache_cost: 3425676868,
timeframe_cost: 3432328560,
elapsed_compute_s: 109.29615,
elapsed_cache_s: 151.30467,
elapsed_timeframe_s: 3.017818,
elapsed_total_s: 263.61865,
Top n: IntoIter(
    [
        Pair {
            params: [
                3660,
                57,
                27,
                41,
            ],
            balance: 193.93752,
        },
        Pair {
            params: [
                2520,
                31,
                22,
                44,
            ],
            balance: 215.37404,
        },
        Pair {
            params: [
                2760,
                21,
                20,
                35,
            ],
            balance: 216.10732,
        },
        Pair {
            params: [
                2940,
                22,
                19,
                42,
            ],
            balance: 224.4349,
        },
        Pair {
            params: [
                2820,
                26,
                21,
                41,
            ],
            balance: 230.94089,
        },
        Pair {
            params: [
                3300,
                26,
                22,
                46,
            ],
            balance: 236.65898,
        },
        Pair {
            params: [
                3420,
                24,
                18,
                42,
            ],
            balance: 246.44807,
        },
        Pair {
            params: [
                3180,
                15,
                15,
                51,
            ],
            balance: 283.4989,
        },
        Pair {
            params: [
                3480,
                52,
                28,
                45,
            ],
            balance: 296.03226,
        },
        Pair {
            params: [
                3540,
                22,
                20,
                42,
            ],
            balance: 343.07587,
        },
    ],
)
hybrid 15 100pop 100gen
compute_cost: 20718023659,
cache_cost: 20718023659,
timeframe_cost: 3241643640,
elapsed_compute_s: 653.98987,
elapsed_cache_s: 911.7451,
elapsed_timeframe_s: 2.868559,
elapsed_total_s: 1568.6035,
Top n: IntoIter(
    [
        Pair {
            params: [
                3180,
                25,
                20,
                43,
            ],
            balance: 344.53873,
        },
        Pair {
            params: [
                2760,
                29,
                22,
                44,
            ],
            balance: 365.29166,
        },
        Pair {
            params: [
                1620,
                46,
                24,
                44,
            ],
            balance: 416.77725,
        },
        Pair {
            params: [
                3660,
                21,
                17,
                39,
            ],
            balance: 496.8898,
        },
        Pair {
            params: [
                2520,
                35,
                23,
                38,
            ],
            balance: 523.52783,
        },
        Pair {
            params: [
                2280,
                39,
                24,
                39,
            ],
            balance: 533.39014,
        },
        Pair {
            params: [
                3480,
                21,
                21,
                41,
            ],
            balance: 552.20856,
        },
        Pair {
            params: [
                3420,
                21,
                18,
                38,
            ],
            balance: 568.26196,
        },
        Pair {
            params: [
                3600,
                19,
                16,
                39,
            ],
            balance: 820.8183,
        },
        Pair {
            params: [
                3540,
                19,
                22,
                40,
            ],
            balance: 1009.7302,
        },
    ],
)
hybrid 15 100pop 1000gen
compute_cost: 23783810246,
cache_cost: 23783810246,
timeframe_cost: 3241643640,
elapsed_compute_s: 750.8473,
elapsed_cache_s: 1046.9257,
elapsed_timeframe_s: 2.841762,
elapsed_total_s: 1800.6147,
Top n: IntoIter(
    [
        Pair {
            params: [
                3180,
                25,
                20,
                43,
            ],
            balance: 348.93884,
        },
        Pair {
            params: [
                1380,
                64,
                28,
                39,
            ],
            balance: 370.74768,
        },
        Pair {
            params: [
                1140,
                65,
                28,
                44,
            ],
            balance: 372.36722,
        },
        Pair {
            params: [
                2940,
                33,
                22,
                37,
            ],
            balance: 410.01343,
        },
        Pair {
            params: [
                2280,
                31,
                23,
                43,
            ],
            balance: 411.746,
        },
        Pair {
            params: [
                1620,
                46,
                24,
                44,
            ],
            balance: 416.77725,
        },
        Pair {
            params: [
                3660,
                21,
                17,
                39,
            ],
            balance: 496.8898,
        },
        Pair {
            params: [
                3540,
                23,
                20,
                44,
            ],
            balance: 571.70197,
        },
        Pair {
            params: [
                3600,
                21,
                20,
                37,
            ],
            balance: 755.1101,
        },
        Pair {
            params: [
                3540,
                19,
                21,
                40,
            ],
            balance: 1009.748,
        },
    ],
)
hybrid 15 1000pop 10gen
compute_cost: 33644741787,
cache_cost: 33644741787,
timeframe_cost: 3432328560,
elapsed_compute_s: 1156.3745,
elapsed_cache_s: 1554.8888,
elapsed_timeframe_s: 3.018126,
elapsed_total_s: 2714.2813,
Top n: IntoIter(
    [
        Pair {
            params: [
                1620,
                24,
                18,
                51,
            ],
            balance: 265.40408,
        },
        Pair {
            params: [
                1380,
                64,
                28,
                40,
            ],
            balance: 297.0899,
        },
        Pair {
            params: [
                2580,
                16,
                15,
                44,
            ],
            balance: 300.43988,
        },
        Pair {
            params: [
                3180,
                25,
                20,
                42,
            ],
            balance: 305.13803,
        },
        Pair {
            params: [
                2400,
                31,
                19,
                37,
            ],
            balance: 310.26038,
        },
        Pair {
            params: [
                2280,
                37,
                22,
                40,
            ],
            balance: 330.29306,
        },
        Pair {
            params: [
                3420,
                17,
                16,
                34,
            ],
            balance: 389.68152,
        },
        Pair {
            params: [
                2520,
                35,
                24,
                38,
            ],
            balance: 398.92914,
        },
        Pair {
            params: [
                3300,
                21,
                26,
                44,
            ],
            balance: 440.32306,
        },
        Pair {
            params: [
                2640,
                28,
                21,
                42,
            ],
            balance: 551.2344,
        },
    ],
)
hybrid 15 1000pop 100gen
compute_cost: 213130371082,
cache_cost: 213130371082,
timeframe_cost: 3050958720,
elapsed_compute_s: 8140.574,
elapsed_cache_s: 10183.317,
elapsed_timeframe_s: 27.726837,
elapsed_total_s: 18351.617,
Top n: IntoIter(
    [
        Pair {
            params: [
                1620,
                46,
                24,
                44,
            ],
            balance: 416.77725,
        },
        Pair {
            params: [
                2280,
                43,
                24,
                38,
            ],
            balance: 421.35916,
        },
        Pair {
            params: [
                3660,
                21,
                17,
                39,
            ],
            balance: 496.8898,
        },
        Pair {
            params: [
                2520,
                33,
                21,
                36,
            ],
            balance: 518.8472,
        },
        Pair {
            params: [
                3480,
                21,
                21,
                41,
            ],
            balance: 552.3362,
        },
        Pair {
            params: [
                3420,
                21,
                19,
                38,
            ],
            balance: 593.1501,
        },
        Pair {
            params: [
                3600,
                19,
                16,
                39,
            ],
            balance: 836.11926,
        },
        Pair {
            params: [
                3540,
                19,
                22,
                40,
            ],
            balance: 1009.7302,
        },
        Pair {
            params: [
                3540,
                19,
                22,
                40,
            ],
            balance: 1009.7302,
        },
        Pair {
            params: [
                3540,
                19,
                22,
                40,
            ],
            balance: 1009.7302,
        },
    ],
)
hybrid 15 1000pop 1000gen
compute_cost: 206900144187,
cache_cost: 206900144187,
timeframe_cost: 3050958720,
elapsed_compute_s: 7772.7944,
elapsed_cache_s: 9787.715,
elapsed_timeframe_s: 154.06601,
elapsed_total_s: 17714.576,
Top n: IntoIter(
    [
        Pair {
            params: [
                2940,
                33,
                22,
                37,
            ],
            balance: 410.01343,
        },
        Pair {
            params: [
                3660,
                21,
                17,
                39,
            ],
            balance: 496.8898,
        },
        Pair {
            params: [
                3480,
                21,
                21,
                41,
            ],
            balance: 499.9938,
        },
        Pair {
            params: [
                2520,
                33,
                21,
                36,
            ],
            balance: 502.11694,
        },
        Pair {
            params: [
                2280,
                39,
                24,
                39,
            ],
            balance: 533.39014,
        },
        Pair {
            params: [
                3420,
                21,
                19,
                38,
            ],
            balance: 589.10114,
        },
        Pair {
            params: [
                3600,
                19,
                16,
                39,
            ],
            balance: 831.02435,
        },
        Pair {
            params: [
                3540,
                19,
                22,
                40,
            ],
            balance: 1009.7302,
        },
        Pair {
            params: [
                3540,
                19,
                22,
                40,
            ],
            balance: 1009.7302,
        },
        Pair {
            params: [
                3540,
                19,
                22,
                40,
            ],
            balance: 1009.7302,
        },
    ],
)
hybrid 30 10pop 10gen
compute_cost: 415717829,
cache_cost: 415717829,
timeframe_cost: 5720547600,
elapsed_compute_s: 15.069839,
elapsed_cache_s: 19.436417,
elapsed_timeframe_s: 6.306756,
elapsed_total_s: 40.81301,
Top n: IntoIter(
    [
        Pair {
            params: [
                2820,
                26,
                24,
                40,
            ],
            balance: 72.87514,
        },
        Pair {
            params: [
                2940,
                15,
                13,
                74,
            ],
            balance: 76.5857,
        },
        Pair {
            params: [
                3120,
                21,
                8,
                34,
            ],
            balance: 77.00227,
        },
        Pair {
            params: [
                1500,
                59,
                25,
                45,
            ],
            balance: 79.98053,
        },
        Pair {
            params: [
                3540,
                18,
                21,
                55,
            ],
            balance: 101.116356,
        },
        Pair {
            params: [
                1620,
                43,
                30,
                43,
            ],
            balance: 116.48545,
        },
        Pair {
            params: [
                2280,
                34,
                19,
                29,
            ],
            balance: 123.31756,
        },
        Pair {
            params: [
                3000,
                26,
                21,
                39,
            ],
            balance: 124.11926,
        },
        Pair {
            params: [
                3600,
                15,
                14,
                34,
            ],
            balance: 170.37953,
        },
        Pair {
            params: [
                3060,
                26,
                21,
                51,
            ],
            balance: 203.14163,
        },
    ],
)
hybrid 30 10pop 100gen
compute_cost: 6281341131,
cache_cost: 6281341131,
timeframe_cost: 5529862680,
elapsed_compute_s: 241.42343,
elapsed_cache_s: 309.67993,
elapsed_timeframe_s: 18.325085,
elapsed_total_s: 569.42847,
Top n: IntoIter(
    [
        Pair {
            params: [
                3180,
                71,
                29,
                40,
            ],
            balance: 312.28333,
        },
        Pair {
            params: [
                3180,
                25,
                20,
                43,
            ],
            balance: 353.7601,
        },
        Pair {
            params: [
                3300,
                23,
                21,
                44,
            ],
            balance: 363.18658,
        },
        Pair {
            params: [
                1920,
                38,
                24,
                40,
            ],
            balance: 364.68216,
        },
        Pair {
            params: [
                2160,
                39,
                25,
                37,
            ],
            balance: 368.75708,
        },
        Pair {
            params: [
                2040,
                15,
                13,
                82,
            ],
            balance: 379.41278,
        },
        Pair {
            params: [
                2220,
                45,
                25,
                40,
            ],
            balance: 467.27954,
        },
        Pair {
            params: [
                3660,
                21,
                17,
                39,
            ],
            balance: 496.88943,
        },
        Pair {
            params: [
                3240,
                23,
                19,
                38,
            ],
            balance: 949.66864,
        },
        Pair {
            params: [
                3240,
                23,
                18,
                38,
            ],
            balance: 992.5113,
        },
    ],
)
hybrid 30 10pop 1000gen
compute_cost: 7166401254,
cache_cost: 7166401254,
timeframe_cost: 5339177760,
elapsed_compute_s: 334.0619,
elapsed_cache_s: 393.72803,
elapsed_timeframe_s: 58.42503,
elapsed_total_s: 786.21497,
Top n: IntoIter(
    [
        Pair {
            params: [
                3060,
                23,
                20,
                44,
            ],
            balance: 304.79868,
        },
        Pair {
            params: [
                1920,
                38,
                23,
                40,
            ],
            balance: 307.57596,
        },
        Pair {
            params: [
                1860,
                42,
                25,
                44,
            ],
            balance: 307.7051,
        },
        Pair {
            params: [
                3180,
                25,
                20,
                43,
            ],
            balance: 356.37988,
        },
        Pair {
            params: [
                3300,
                23,
                20,
                44,
            ],
            balance: 367.58963,
        },
        Pair {
            params: [
                2280,
                36,
                23,
                46,
            ],
            balance: 389.57092,
        },
        Pair {
            params: [
                3420,
                23,
                17,
                38,
            ],
            balance: 453.86127,
        },
        Pair {
            params: [
                3480,
                21,
                21,
                41,
            ],
            balance: 533.09735,
        },
        Pair {
            params: [
                3420,
                21,
                17,
                38,
            ],
            balance: 539.05505,
        },
        Pair {
            params: [
                3420,
                21,
                17,
                38,
            ],
            balance: 562.76306,
        },
    ],
)
hybrid 30 100pop 10gen
compute_cost: 4503589410,
cache_cost: 4503589410,
timeframe_cost: 5720547600,
elapsed_compute_s: 181.354,
elapsed_cache_s: 224.69789,
elapsed_timeframe_s: 16.01682,
elapsed_total_s: 422.0687,
Top n: IntoIter(
    [
        Pair {
            params: [
                2820,
                25,
                29,
                40,
            ],
            balance: 245.65938,
        },
        Pair {
            params: [
                2640,
                25,
                22,
                40,
            ],
            balance: 246.75967,
        },
        Pair {
            params: [
                3660,
                21,
                18,
                38,
            ],
            balance: 253.45097,
        },
        Pair {
            params: [
                2520,
                34,
                22,
                35,
            ],
            balance: 258.83133,
        },
        Pair {
            params: [
                3240,
                22,
                21,
                43,
            ],
            balance: 260.8039,
        },
        Pair {
            params: [
                3300,
                23,
                20,
                42,
            ],
            balance: 278.4027,
        },
        Pair {
            params: [
                2280,
                17,
                12,
                53,
            ],
            balance: 283.33344,
        },
        Pair {
            params: [
                2700,
                25,
                18,
                43,
            ],
            balance: 285.5663,
        },
        Pair {
            params: [
                2400,
                14,
                13,
                55,
            ],
            balance: 289.7754,
        },
        Pair {
            params: [
                2640,
                31,
                23,
                42,
            ],
            balance: 312.27255,
        },
    ],
)
hybrid 30 100pop 100gen
compute_cost: 34110482328,
cache_cost: 34110482328,
timeframe_cost: 5720547600,
elapsed_compute_s: 1258.4264,
elapsed_cache_s: 1623.7084,
elapsed_timeframe_s: 162.73622,
elapsed_total_s: 3044.871,
Top n: IntoIter(
    [
        Pair {
            params: [
                1620,
                46,
                24,
                44,
            ],
            balance: 416.77725,
        },
        Pair {
            params: [
                2700,
                25,
                22,
                41,
            ],
            balance: 440.0808,
        },
        Pair {
            params: [
                2520,
                35,
                23,
                38,
            ],
            balance: 496.45782,
        },
        Pair {
            params: [
                3540,
                23,
                20,
                44,
            ],
            balance: 574.8794,
        },
        Pair {
            params: [
                3420,
                21,
                19,
                38,
            ],
            balance: 589.10114,
        },
        Pair {
            params: [
                2580,
                43,
                27,
                38,
            ],
            balance: 607.9208,
        },
        Pair {
            params: [
                2640,
                28,
                21,
                42,
            ],
            balance: 680.7284,
        },
        Pair {
            params: [
                1740,
                42,
                25,
                40,
            ],
            balance: 685.064,
        },
        Pair {
            params: [
                3600,
                19,
                16,
                39,
            ],
            balance: 820.8183,
        },
        Pair {
            params: [
                3540,
                19,
                21,
                40,
            ],
            balance: 1009.748,
        },
    ],
)
hybrid 30 100pop 1000gen
compute_cost: 32109748092,
cache_cost: 32109748092,
timeframe_cost: 5911232520,
elapsed_compute_s: 1076.4839,
elapsed_cache_s: 1449.2573,
elapsed_timeframe_s: 48.324783,
elapsed_total_s: 2574.066,
Top n: IntoIter(
    [
        Pair {
            params: [
                2820,
                12,
                11,
                83,
            ],
            balance: 425.11636,
        },
        Pair {
            params: [
                2160,
                39,
                25,
                37,
            ],
            balance: 462.46677,
        },
        Pair {
            params: [
                3000,
                23,
                17,
                45,
            ],
            balance: 476.36185,
        },
        Pair {
            params: [
                1800,
                37,
                23,
                41,
            ],
            balance: 496.52432,
        },
        Pair {
            params: [
                3660,
                21,
                17,
                39,
            ],
            balance: 496.8898,
        },
        Pair {
            params: [
                3060,
                23,
                19,
                43,
            ],
            balance: 508.10544,
        },
        Pair {
            params: [
                2280,
                39,
                24,
                39,
            ],
            balance: 533.39014,
        },
        Pair {
            params: [
                2640,
                28,
                21,
                42,
            ],
            balance: 680.7284,
        },
        Pair {
            params: [
                1740,
                42,
                26,
                40,
            ],
            balance: 681.368,
        },
        Pair {
            params: [
                1740,
                42,
                26,
                40,
            ],
            balance: 681.368,
        },
    ],
)
hybrid 30 1000pop 10gen
compute_cost: 43969342463,
cache_cost: 43969342463,
timeframe_cost: 5720547600,
elapsed_compute_s: 1822.041,
elapsed_cache_s: 2231.4255,
elapsed_timeframe_s: 265.90826,
elapsed_total_s: 4319.375,
Top n: IntoIter(
    [
        Pair {
            params: [
                2220,
                30,
                20,
                43,
            ],
            balance: 382.73065,
        },
        Pair {
            params: [
                3540,
                22,
                22,
                42,
            ],
            balance: 384.40213,
        },
        Pair {
            params: [
                3420,
                21,
                16,
                38,
            ],
            balance: 387.29083,
        },
        Pair {
            params: [
                3600,
                17,
                15,
                40,
            ],
            balance: 394.6116,
        },
        Pair {
            params: [
                2280,
                39,
                23,
                39,
            ],
            balance: 395.6467,
        },
        Pair {
            params: [
                2880,
                31,
                22,
                40,
            ],
            balance: 448.665,
        },
        Pair {
            params: [
                2880,
                34,
                23,
                39,
            ],
            balance: 457.19052,
        },
        Pair {
            params: [
                3540,
                23,
                20,
                43,
            ],
            balance: 461.39008,
        },
        Pair {
            params: [
                2640,
                28,
                21,
                41,
            ],
            balance: 470.3536,
        },
        Pair {
            params: [
                2820,
                26,
                18,
                40,
            ],
            balance: 471.551,
        },
    ],
)
hybrid 30 1000pop 100gen
compute_cost: 312149062073,
cache_cost: 312149062073,
timeframe_cost: 5148492840,
elapsed_compute_s: 10628.289,
elapsed_cache_s: 14059.175,
elapsed_timeframe_s: 134.58731,
elapsed_total_s: 24822.053,
Top n: IntoIter(
    [
        Pair {
            params: [
                2580,
                43,
                27,
                38,
            ],
            balance: 608.455,
        },
        Pair {
            params: [
                2640,
                28,
                21,
                42,
            ],
            balance: 680.7284,
        },
        Pair {
            params: [
                2820,
                25,
                19,
                38,
            ],
            balance: 745.4738,
        },
        Pair {
            params: [
                2880,
                31,
                22,
                39,
            ],
            balance: 780.0614,
        },
        Pair {
            params: [
                2880,
                31,
                22,
                39,
            ],
            balance: 780.0614,
        },
        Pair {
            params: [
                3600,
                19,
                16,
                39,
            ],
            balance: 836.11926,
        },
        Pair {
            params: [
                3540,
                19,
                22,
                40,
            ],
            balance: 1009.7302,
        },
        Pair {
            params: [
                3540,
                19,
                22,
                40,
            ],
            balance: 1009.7302,
        },
        Pair {
            params: [
                3540,
                19,
                22,
                40,
            ],
            balance: 1009.7302,
        },
        Pair {
            params: [
                3540,
                19,
                22,
                40,
            ],
            balance: 1009.7302,
        },
    ],
)

hybrid 30 1000pop 1000pop
compute_cost: 350004795023,
cache_cost: 350004795023,
timeframe_cost: 5148492840,
elapsed_compute_s: 12431.987,
elapsed_cache_s: 16429.65,
elapsed_timeframe_s: 176.32323,
elapsed_total_s: 29037.959,
Top n: IntoIter(
    [
        Pair {
            params: [
                3420,
                21,
                19,
                38,
            ],
            balance: 593.1501,
        },
        Pair {
            params: [
                2640,
                28,
                21,
                42,
            ],
            balance: 680.7284,
        },
        Pair {
            params: [
                2820,
                25,
                19,
                38,
            ],
            balance: 745.4738,
        },
        Pair {
            params: [
                2880,
                31,
                22,
                39,
            ],
            balance: 780.0614,
        },
        Pair {
            params: [
                2880,
                31,
                22,
                39,
            ],
            balance: 780.0614,
        },
        Pair {
            params: [
                3600,
                19,
                16,
                39,
            ],
            balance: 836.11926,
        },
        Pair {
            params: [
                3540,
                19,
                22,
                40,
            ],
            balance: 1009.7302,
        },
        Pair {
            params: [
                3540,
                19,
                22,
                40,
            ],
            balance: 1009.7302,
        },
        Pair {
            params: [
                3540,
                19,
                22,
                40,
            ],
            balance: 1009.7302,
        },
        Pair {
            params: [
                3540,
                19,
                22,
                40,
            ],
            balance: 1009.7302,
        },
    ],
)
corrected
ga 10pop 10gen
compute_cost: 16632060,
cache_cost: 16632060,
timeframe_cost: 6673972200,
elapsed_compute_s: 1.148414,
elapsed_cache_s: 1.331776,
elapsed_timeframe_s: 11.102385,
elapsed_total_s: 13.582575,
Top n: IntoIter(
    [
        Pair {
            params: [
                840,
                41,
                94,
                20,
            ],
            balance: 15.869984,
        },
        Pair {
            params: [
                720,
                97,
                17,
                49,
            ],
            balance: 16.769033,
        },
        Pair {
            params: [
                1200,
                98,
                26,
                45,
            ],
            balance: 16.993992,
        },
        Pair {
            params: [
                2100,
                89,
                40,
                50,
            ],
            balance: 21.437586,
        },
        Pair {
            params: [
                1140,
                99,
                81,
                41,
            ],
            balance: 22.486605,
        },
        Pair {
            params: [
                240,
                63,
                23,
                35,
            ],
            balance: 26.795078,
        },
        Pair {
            params: [
                420,
                96,
                21,
                49,
            ],
            balance: 28.037704,
        },
        Pair {
            params: [
                480,
                80,
                26,
                52,
            ],
            balance: 31.959637,
        },
        Pair {
            params: [
                1800,
                92,
                34,
                60,
            ],
            balance: 32.31645,
        },
        Pair {
            params: [
                300,
                99,
                22,
                39,
            ],
            balance: 34.907894,
        },
    ],
)
ga 10pop 100gen
compute_cost: 74971502,
cache_cost: 74971502,
timeframe_cost: 8580821400,
elapsed_compute_s: 2.426157,
elapsed_cache_s: 3.864645,
elapsed_timeframe_s: 7.665974,
elapsed_total_s: 13.956776,
Top n: IntoIter(
    [
        Pair {
            params: [
                2100,
                39,
                24,
                47,
            ],
            balance: 312.605,
        },
        Pair {
            params: [
                2100,
                39,
                24,
                47,
            ],
            balance: 312.605,
        },
        Pair {
            params: [
                2100,
                39,
                24,
                47,
            ],
            balance: 312.605,
        },
        Pair {
            params: [
                2100,
                39,
                24,
                47,
            ],
            balance: 312.605,
        },
        Pair {
            params: [
                2100,
                39,
                24,
                47,
            ],
            balance: 312.605,
        },
        Pair {
            params: [
                2100,
                39,
                24,
                47,
            ],
            balance: 312.605,
        },
        Pair {
            params: [
                2100,
                39,
                24,
                47,
            ],
            balance: 312.605,
        },
        Pair {
            params: [
                2100,
                39,
                24,
                47,
            ],
            balance: 312.605,
        },
        Pair {
            params: [
                2100,
                38,
                24,
                48,
            ],
            balance: 314.87256,
        },
        Pair {
            params: [
                2160,
                38,
                22,
                46,
            ],
            balance: 332.0676,
        },
    ],
)
ga 10pop 1000gen
compute_cost: 171661284,
cache_cost: 171661284,
timeframe_cost: 10678355520,
elapsed_compute_s: 7.002634,
elapsed_cache_s: 8.875617,
elapsed_timeframe_s: 13.260614,
elapsed_total_s: 29.138866,
Top n: IntoIter(
    [
        Pair {
            params: [
                1380,
                63,
                28,
                64,
            ],
            balance: 270.21634,
        },
        Pair {
            params: [
                1380,
                63,
                28,
                64,
            ],
            balance: 270.21634,
        },
        Pair {
            params: [
                1380,
                63,
                28,
                64,
            ],
            balance: 270.21634,
        },
        Pair {
            params: [
                1380,
                63,
                28,
                64,
            ],
            balance: 270.21634,
        },
        Pair {
            params: [
                1380,
                63,
                28,
                64,
            ],
            balance: 270.21634,
        },
        Pair {
            params: [
                1380,
                63,
                28,
                64,
            ],
            balance: 270.21634,
        },
        Pair {
            params: [
                1380,
                63,
                28,
                64,
            ],
            balance: 270.21634,
        },
        Pair {
            params: [
                1380,
                63,
                28,
                64,
            ],
            balance: 270.21634,
        },
        Pair {
            params: [
                1380,
                63,
                28,
                64,
            ],
            balance: 270.21634,
        },
        Pair {
            params: [
                1380,
                63,
                28,
                64,
            ],
            balance: 270.21634,
        },
    ],
)
ga 100pop 10gen
compute_cost: 85361261,
cache_cost: 85361261,
timeframe_cost: 10106300760,
elapsed_compute_s: 2.929642,
elapsed_cache_s: 3.955469,
elapsed_timeframe_s: 9.605208,
elapsed_total_s: 16.490318,
Top n: IntoIter(
    [
        Pair {
            params: [
                1800,
                46,
                24,
                47,
            ],
            balance: 138.0048,
        },
        Pair {
            params: [
                2280,
                33,
                21,
                41,
            ],
            balance: 140.94705,
        },
        Pair {
            params: [
                1800,
                37,
                22,
                45,
            ],
            balance: 141.25911,
        },
        Pair {
            params: [
                2340,
                46,
                23,
                37,
            ],
            balance: 144.37146,
        },
        Pair {
            params: [
                2820,
                74,
                31,
                42,
            ],
            balance: 144.44649,
        },
        Pair {
            params: [
                1980,
                40,
                23,
                44,
            ],
            balance: 150.3913,
        },
        Pair {
            params: [
                2100,
                37,
                23,
                48,
            ],
            balance: 153.0248,
        },
        Pair {
            params: [
                1800,
                50,
                26,
                49,
            ],
            balance: 159.34251,
        },
        Pair {
            params: [
                2220,
                50,
                26,
                45,
            ],
            balance: 164.34584,
        },
        Pair {
            params: [
                2280,
                37,
                23,
                39,
            ],
            balance: 274.9236,
        },
    ],
)
ga 100pop 100gen
compute_cost: 439712317,
cache_cost: 439712317,
timeframe_cost: 10869040440,
elapsed_compute_s: 14.969129,
elapsed_cache_s: 20.72782,
elapsed_timeframe_s: 10.284252,
elapsed_total_s: 45.9812,
Top n: IntoIter(
    [
        Pair {
            params: [
                2280,
                36,
                24,
                46,
            ],
            balance: 312.52213,
        },
        Pair {
            params: [
                2640,
                28,
                20,
                43,
            ],
            balance: 318.4468,
        },
        Pair {
            params: [
                3240,
                22,
                19,
                40,
            ],
            balance: 341.5089,
        },
        Pair {
            params: [
                2280,
                37,
                22,
                39,
            ],
            balance: 347.72113,
        },
        Pair {
            params: [
                2220,
                45,
                26,
                41,
            ],
            balance: 375.89032,
        },
        Pair {
            params: [
                2340,
                30,
                22,
                45,
            ],
            balance: 381.81024,
        },
        Pair {
            params: [
                2340,
                30,
                22,
                45,
            ],
            balance: 381.81024,
        },
        Pair {
            params: [
                2340,
                30,
                22,
                45,
            ],
            balance: 381.81024,
        },
        Pair {
            params: [
                2340,
                30,
                22,
                45,
            ],
            balance: 381.81024,
        },
        Pair {
            params: [
                2340,
                30,
                22,
                45,
            ],
            balance: 381.81024,
        },
    ],
)
ga 100pop 1000gen
compute_cost: 344251377,
cache_cost: 344251377,
timeframe_cost: 10869040440,
elapsed_compute_s: 11.186355,
elapsed_cache_s: 15.383453,
elapsed_timeframe_s: 9.803795,
elapsed_total_s: 36.373604,
Top n: IntoIter(
    [
        Pair {
            params: [
                2640,
                28,
                22,
                42,
            ],
            balance: 557.80023,
        },
        Pair {
            params: [
                2640,
                28,
                22,
                42,
            ],
            balance: 557.80023,
        },
        Pair {
            params: [
                2640,
                28,
                22,
                42,
            ],
            balance: 557.80023,
        },
        Pair {
            params: [
                2640,
                28,
                22,
                42,
            ],
            balance: 557.80023,
        },
        Pair {
            params: [
                2640,
                28,
                22,
                42,
            ],
            balance: 557.80023,
        },
        Pair {
            params: [
                2640,
                28,
                22,
                42,
            ],
            balance: 557.80023,
        },
        Pair {
            params: [
                2640,
                28,
                22,
                42,
            ],
            balance: 557.80023,
        },
        Pair {
            params: [
                2640,
                28,
                22,
                42,
            ],
            balance: 557.80023,
        },
        Pair {
            params: [
                2640,
                28,
                22,
                42,
            ],
            balance: 557.80023,
        },
        Pair {
            params: [
                2640,
                28,
                22,
                42,
            ],
            balance: 557.80023,
        },
    ],
)
ga 1000pop 10gen
compute_cost: 1007497923,
cache_cost: 1007497923,
timeframe_cost: 11250410280,
elapsed_compute_s: 39.042168,
elapsed_cache_s: 50.353447,
elapsed_timeframe_s: 10.562438,
elapsed_total_s: 99.95805,
Top n: IntoIter(
    [
        Pair {
            params: [
                3000,
                27,
                21,
                47,
            ],
            balance: 201.6508,
        },
        Pair {
            params: [
                3120,
                21,
                18,
                43,
            ],
            balance: 204.57007,
        },
        Pair {
            params: [
                2280,
                44,
                25,
                41,
            ],
            balance: 205.61948,
        },
        Pair {
            params: [
                3300,
                25,
                21,
                46,
            ],
            balance: 216.46088,
        },
        Pair {
            params: [
                2880,
                26,
                21,
                38,
            ],
            balance: 217.83844,
        },
        Pair {
            params: [
                1800,
                42,
                23,
                45,
            ],
            balance: 219.73592,
        },
        Pair {
            params: [
                2220,
                45,
                25,
                40,
            ],
            balance: 221.80893,
        },
        Pair {
            params: [
                2880,
                39,
                23,
                42,
            ],
            balance: 237.96632,
        },
        Pair {
            params: [
                2520,
                35,
                23,
                39,
            ],
            balance: 251.25558,
        },
        Pair {
            params: [
                2280,
                39,
                24,
                36,
            ],
            balance: 291.52325,
        },
    ],
)
ga 1000pop 100gen
compute_cost: 3781475634,
cache_cost: 3781475634,
timeframe_cost: 11250410280,
elapsed_compute_s: 136.47737,
elapsed_cache_s: 180.08362,
elapsed_timeframe_s: 12.744197,
elapsed_total_s: 329.30518,
Top n: IntoIter(
    [
        Pair {
            params: [
                2820,
                25,
                20,
                39,
            ],
            balance: 567.4214,
        },
        Pair {
            params: [
                2820,
                25,
                20,
                39,
            ],
            balance: 567.4214,
        },
        Pair {
            params: [
                2820,
                25,
                20,
                39,
            ],
            balance: 567.4214,
        },
        Pair {
            params: [
                2820,
                25,
                20,
                39,
            ],
            balance: 567.4214,
        },
        Pair {
            params: [
                2820,
                25,
                20,
                39,
            ],
            balance: 567.4214,
        },
        Pair {
            params: [
                2820,
                25,
                20,
                39,
            ],
            balance: 567.4214,
        },
        Pair {
            params: [
                2820,
                25,
                20,
                39,
            ],
            balance: 567.4214,
        },
        Pair {
            params: [
                2820,
                25,
                20,
                39,
            ],
            balance: 567.4214,
        },
        Pair {
            params: [
                2820,
                25,
                20,
                39,
            ],
            balance: 567.4214,
        },
        Pair {
            params: [
                3240,
                23,
                19,
                38,
            ],
            balance: 743.23956,
        },
    ],
)
ga 1000pop 1000gen
compute_cost: 4074073002,
cache_cost: 4074073002,
timeframe_cost: 11250410280,
elapsed_compute_s: 148.09857,
elapsed_cache_s: 221.59726,
elapsed_timeframe_s: 12.003036,
elapsed_total_s: 381.69885,
Top n: IntoIter(
    [
        Pair {
            params: [
                2820,
                25,
                20,
                39,
            ],
            balance: 567.4214,
        },
        Pair {
            params: [
                2820,
                25,
                20,
                39,
            ],
            balance: 567.4214,
        },
        Pair {
            params: [
                2820,
                25,
                20,
                39,
            ],
            balance: 567.4214,
        },
        Pair {
            params: [
                2820,
                25,
                20,
                39,
            ],
            balance: 567.4214,
        },
        Pair {
            params: [
                2820,
                25,
                20,
                39,
            ],
            balance: 567.4214,
        },
        Pair {
            params: [
                2820,
                25,
                20,
                39,
            ],
            balance: 567.4214,
        },
        Pair {
            params: [
                2820,
                25,
                20,
                39,
            ],
            balance: 567.4214,
        },
        Pair {
            params: [
                2820,
                25,
                20,
                39,
            ],
            balance: 567.4214,
        },
        Pair {
            params: [
                2820,
                25,
                20,
                39,
            ],
            balance: 567.4214,
        },
        Pair {
            params: [
                2820,
                25,
                20,
                39,
            ],
            balance: 567.4214,
        },
    ],
)
compute_cost: 291570152,
cache_cost: 291570152,
timeframe_cost: 2478903960,
elapsed_compute_s: 9.70386,
elapsed_cache_s: 13.116971,
elapsed_timeframe_s: 2.346523,
elapsed_total_s: 25.167355,
Top n: IntoIter(
    [
        Pair {
            params: [
                2760,
                26,
                86,
                75,
            ],
            balance: 47.007153,
        },
        Pair {
            params: [
                1380,
                20,
                15,
                81,
            ],
            balance: 52.333862,
        },
        Pair {
            params: [
                2040,
                94,
                34,
                52,
            ],
            balance: 55.3627,
        },
        Pair {
            params: [
                2520,
                70,
                34,
                50,
            ],
            balance: 65.87674,
        },
        Pair {
            params: [
                3660,
                81,
                78,
                54,
            ],
            balance: 71.5484,
        },
        Pair {
            params: [
                3420,
                29,
                18,
                37,
            ],
            balance: 74.44096,
        },
        Pair {
            params: [
                2400,
                32,
                19,
                55,
            ],
            balance: 82.10174,
        },
        Pair {
            params: [
                1860,
                48,
                26,
                41,
            ],
            balance: 85.60014,
        },
        Pair {
            params: [
                3180,
                74,
                40,
                58,
            ],
            balance: 86.47081,
        },
        Pair {
            params: [
                2280,
                64,
                30,
                50,
            ],
            balance: 104.18372,
        },
    ],
)
hybrid 10 10pop 100gen
compute_cost: 2308396251,
cache_cost: 2308396251,
timeframe_cost: 2478903960,
elapsed_compute_s: 90.132286,
elapsed_cache_s: 114.94262,
elapsed_timeframe_s: 3.291782,
elapsed_total_s: 208.36668,
Top n: IntoIter(
    [
        Pair {
            params: [
                3660,
                72,
                81,
                52,
            ],
            balance: 98.63147,
        },
        Pair {
            params: [
                1860,
                48,
                25,
                66,
            ],
            balance: 126.32534,
        },
        Pair {
            params: [
                2940,
                28,
                20,
                69,
            ],
            balance: 126.50504,
        },
        Pair {
            params: [
                3300,
                44,
                26,
                58,
            ],
            balance: 148.39236,
        },
        Pair {
            params: [
                960,
                89,
                30,
                43,
            ],
            balance: 155.7078,
        },
        Pair {
            params: [
                3540,
                81,
                31,
                55,
            ],
            balance: 189.61049,
        },
        Pair {
            params: [
                1380,
                60,
                28,
                45,
            ],
            balance: 230.124,
        },
        Pair {
            params: [
                2760,
                29,
                22,
                45,
            ],
            balance: 263.72806,
        },
        Pair {
            params: [
                3180,
                25,
                21,
                44,
            ],
            balance: 280.67798,
        },
        Pair {
            params: [
                3420,
                18,
                20,
                35,
            ],
            balance: 392.57397,
        },
    ],
)
hybrid 10 10pop 1000gen
compute_cost: 2388267234,
cache_cost: 2388267234,
timeframe_cost: 2478903960,
elapsed_compute_s: 77.0326,
elapsed_cache_s: 106.21758,
elapsed_timeframe_s: 2.411448,
elapsed_total_s: 185.66164,
Top n: IntoIter(
    [
        Pair {
            params: [
                3540,
                52,
                84,
                70,
            ],
            balance: 58.357975,
        },
        Pair {
            params: [
                2940,
                56,
                82,
                43,
            ],
            balance: 64.0797,
        },
        Pair {
            params: [
                3240,
                24,
                13,
                45,
            ],
            balance: 64.510635,
        },
        Pair {
            params: [
                3660,
                60,
                27,
                50,
            ],
            balance: 134.71269,
        },
        Pair {
            params: [
                1860,
                43,
                25,
                48,
            ],
            balance: 164.54079,
        },
        Pair {
            params: [
                3360,
                78,
                39,
                56,
            ],
            balance: 196.98268,
        },
        Pair {
            params: [
                2760,
                17,
                17,
                31,
            ],
            balance: 218.12138,
        },
        Pair {
            params: [
                3300,
                27,
                20,
                70,
            ],
            balance: 248.97241,
        },
        Pair {
            params: [
                3420,
                53,
                28,
                43,
            ],
            balance: 274.54846,
        },
        Pair {
            params: [
                3180,
                25,
                21,
                44,
            ],
            balance: 280.67798,
        },
    ],
)
hybrid 10 100pop 10gen
compute_cost: 3026812510,
cache_cost: 3026812510,
timeframe_cost: 2478903960,
elapsed_compute_s: 128.51848,
elapsed_cache_s: 159.41586,
elapsed_timeframe_s: 15.362621,
elapsed_total_s: 303.29694,
Top n: IntoIter(
    [
        Pair {
            params: [
                2940,
                61,
                32,
                49,
            ],
            balance: 135.9335,
        },
        Pair {
            params: [
                2760,
                60,
                28,
                44,
            ],
            balance: 167.80664,
        },
        Pair {
            params: [
                1860,
                38,
                24,
                45,
            ],
            balance: 170.16534,
        },
        Pair {
            params: [
                2040,
                15,
                13,
                55,
            ],
            balance: 177.34227,
        },
        Pair {
            params: [
                3180,
                24,
                20,
                54,
            ],
            balance: 184.9607,
        },
        Pair {
            params: [
                2160,
                40,
                24,
                47,
            ],
            balance: 216.05402,
        },
        Pair {
            params: [
                2520,
                31,
                21,
                45,
            ],
            balance: 223.97289,
        },
        Pair {
            params: [
                1380,
                66,
                30,
                49,
            ],
            balance: 231.24287,
        },
        Pair {
            params: [
                2400,
                14,
                13,
                56,
            ],
            balance: 235.17955,
        },
        Pair {
            params: [
                2280,
                35,
                22,
                42,
            ],
            balance: 240.89867,
        },
    ],
)
hybrid 10 100pop 100gen
compute_cost: 16037149113,
cache_cost: 16037149113,
timeframe_cost: 2478903960,
elapsed_compute_s: 612.3284,
elapsed_cache_s: 794.70874,
elapsed_timeframe_s: 24.770767,
elapsed_total_s: 1431.8079,
Top n: IntoIter(
    [
        Pair {
            params: [
                2760,
                30,
                23,
                68,
            ],
            balance: 159.17754,
        },
        Pair {
            params: [
                1860,
                43,
                26,
                44,
            ],
            balance: 206.56311,
        },
        Pair {
            params: [
                3540,
                19,
                23,
                35,
            ],
            balance: 214.31036,
        },
        Pair {
            params: [
                2040,
                42,
                26,
                47,
            ],
            balance: 241.61688,
        },
        Pair {
            params: [
                1380,
                58,
                28,
                42,
            ],
            balance: 262.82138,
        },
        Pair {
            params: [
                2520,
                72,
                31,
                49,
            ],
            balance: 281.77634,
        },
        Pair {
            params: [
                3300,
                27,
                20,
                44,
            ],
            balance: 324.39722,
        },
        Pair {
            params: [
                3660,
                22,
                17,
                43,
            ],
            balance: 343.60617,
        },
        Pair {
            params: [
                2280,
                37,
                22,
                38,
            ],
            balance: 349.2373,
        },
        Pair {
            params: [
                3420,
                21,
                19,
                39,
            ],
            balance: 487.2926,
        },
    ],
)
hybrid 10 100pop 1000gen
compute_cost: 15671972324,
cache_cost: 15671972324,
timeframe_cost: 2478903960,
elapsed_compute_s: 653.32007,
elapsed_cache_s: 824.259,
elapsed_timeframe_s: 184.52162,
elapsed_total_s: 1662.1007,
Top n: IntoIter(
    [
        Pair {
            params: [
                1860,
                43,
                26,
                44,
            ],
            balance: 206.56311,
        },
        Pair {
            params: [
                2520,
                70,
                31,
                48,
            ],
            balance: 227.02823,
        },
        Pair {
            params: [
                2040,
                42,
                26,
                47,
            ],
            balance: 241.61688,
        },
        Pair {
            params: [
                2760,
                29,
                22,
                45,
            ],
            balance: 263.72806,
        },
        Pair {
            params: [
                3180,
                25,
                21,
                44,
            ],
            balance: 280.67798,
        },
        Pair {
            params: [
                2940,
                33,
                22,
                37,
            ],
            balance: 297.60178,
        },
        Pair {
            params: [
                2160,
                38,
                22,
                46,
            ],
            balance: 332.0676,
        },
        Pair {
            params: [
                2400,
                30,
                21,
                40,
            ],
            balance: 345.01608,
        },
        Pair {
            params: [
                2280,
                37,
                22,
                39,
            ],
            balance: 347.72113,
        },
        Pair {
            params: [
                2340,
                30,
                22,
                45,
            ],
            balance: 381.81024,
        },
    ],
)
hybrid 10 1000pop 10gen
compute_cost: 29224965512,
cache_cost: 29224965512,
timeframe_cost: 2478903960,
elapsed_compute_s: 1091.2856,
elapsed_cache_s: 1420.3417,
elapsed_timeframe_s: 38.5468,
elapsed_total_s: 2550.1743,
Top n: IntoIter(
    [
        Pair {
            params: [
                3180,
                28,
                21,
                42,
            ],
            balance: 213.58124,
        },
        Pair {
            params: [
                2040,
                42,
                26,
                47,
            ],
            balance: 241.61688,
        },
        Pair {
            params: [
                3480,
                21,
                18,
                41,
            ],
            balance: 253.84991,
        },
        Pair {
            params: [
                2520,
                35,
                23,
                38,
            ],
            balance: 304.46722,
        },
        Pair {
            params: [
                2760,
                21,
                19,
                40,
            ],
            balance: 312.63156,
        },
        Pair {
            params: [
                3660,
                23,
                23,
                44,
            ],
            balance: 343.78613,
        },
        Pair {
            params: [
                2280,
                37,
                22,
                38,
            ],
            balance: 349.2373,
        },
        Pair {
            params: [
                3420,
                21,
                17,
                39,
            ],
            balance: 389.34967,
        },
        Pair {
            params: [
                3600,
                19,
                18,
                38,
            ],
            balance: 393.2344,
        },
        Pair {
            params: [
                3540,
                19,
                22,
                41,
            ],
            balance: 573.63153,
        },
    ],
)
hybrid 10 1000pop 100gen
compute_cost: 120057043606,
cache_cost: 120057043606,
timeframe_cost: 2478903960,
elapsed_compute_s: 4275.8955,
elapsed_cache_s: 5784.6997,
elapsed_timeframe_s: 38.18716,
elapsed_total_s: 10098.783,
Top n: IntoIter(
    [
        Pair {
            params: [
                2940,
                33,
                21,
                37,
            ],
            balance: 300.0186,
        },
        Pair {
            params: [
                2520,
                33,
                22,
                35,
            ],
            balance: 309.46555,
        },
        Pair {
            params: [
                2760,
                21,
                19,
                40,
            ],
            balance: 312.63156,
        },
        Pair {
            params: [
                3480,
                21,
                22,
                42,
            ],
            balance: 313.00763,
        },
        Pair {
            params: [
                3300,
                26,
                22,
                46,
            ],
            balance: 324.57056,
        },
        Pair {
            params: [
                3180,
                37,
                26,
                38,
            ],
            balance: 335.95743,
        },
        Pair {
            params: [
                3660,
                22,
                17,
                43,
            ],
            balance: 343.60617,
        },
        Pair {
            params: [
                2280,
                43,
                25,
                39,
            ],
            balance: 475.26324,
        },
        Pair {
            params: [
                3420,
                21,
                19,
                39,
            ],
            balance: 487.2926,
        },
        Pair {
            params: [
                3540,
                19,
                22,
                39,
            ],
            balance: 599.8812,
        },
    ],
)
hybrid 10 1000pop 1000gen
compute_cost: 157911720265,
cache_cost: 157911720265,
timeframe_cost: 2478903960,
elapsed_compute_s: 5287.6187,
elapsed_cache_s: 7225.856,
elapsed_timeframe_s: 22.718733,
elapsed_total_s: 12536.193,
Top n: IntoIter(
    [
        Pair {
            params: [
                2940,
                33,
                21,
                37,
            ],
            balance: 300.0186,
        },
        Pair {
            params: [
                2520,
                33,
                22,
                35,
            ],
            balance: 309.46555,
        },
        Pair {
            params: [
                2760,
                21,
                19,
                40,
            ],
            balance: 312.63156,
        },
        Pair {
            params: [
                3480,
                22,
                20,
                36,
            ],
            balance: 323.31586,
        },
        Pair {
            params: [
                3300,
                26,
                22,
                46,
            ],
            balance: 324.57056,
        },
        Pair {
            params: [
                3180,
                37,
                26,
                38,
            ],
            balance: 335.95743,
        },
        Pair {
            params: [
                3660,
                22,
                17,
                43,
            ],
            balance: 343.60617,
        },
        Pair {
            params: [
                2280,
                43,
                25,
                39,
            ],
            balance: 475.26324,
        },
        Pair {
            params: [
                3420,
                17,
                20,
                33,
            ],
            balance: 596.20874,
        },
        Pair {
            params: [
                3540,
                19,
                22,
                39,
            ],
            balance: 599.8812,
        },
    ],
)
hybrid 15 10pop 10gen
compute_cost: 272716235,
cache_cost: 272716235,
timeframe_cost: 3241643640,
elapsed_compute_s: 8.869875,
elapsed_cache_s: 12.155466,
elapsed_timeframe_s: 2.98287,
elapsed_total_s: 24.008211,
Top n: IntoIter(
    [
        Pair {
            params: [
                2520,
                45,
                28,
                41,
            ],
            balance: 58.27234,
        },
        Pair {
            params: [
                1140,
                56,
                32,
                69,
            ],
            balance: 65.343864,
        },
        Pair {
            params: [
                2760,
                64,
                31,
                45,
            ],
            balance: 69.7458,
        },
        Pair {
            params: [
                1380,
                28,
                24,
                71,
            ],
            balance: 78.92081,
        },
        Pair {
            params: [
                2160,
                39,
                24,
                48,
            ],
            balance: 84.76145,
        },
        Pair {
            params: [
                960,
                47,
                24,
                47,
            ],
            balance: 89.94429,
        },
        Pair {
            params: [
                2340,
                21,
                12,
                31,
            ],
            balance: 90.18166,
        },
        Pair {
            params: [
                2280,
                57,
                28,
                45,
            ],
            balance: 104.141815,
        },
        Pair {
            params: [
                2220,
                28,
                22,
                39,
            ],
            balance: 116.71121,
        },
        Pair {
            params: [
                2280,
                37,
                22,
                38,
            ],
            balance: 349.2373,
        },
    ],
)
hybrid 15 10pop 100gen
compute_cost: 2541638169,
cache_cost: 2541638169,
timeframe_cost: 3432328560,
elapsed_compute_s: 81.66827,
elapsed_cache_s: 112.8626,
elapsed_timeframe_s: 3.091106,
elapsed_total_s: 197.62198,
Top n: IntoIter(
    [
        Pair {
            params: [
                1860,
                43,
                25,
                64,
            ],
            balance: 222.7781,
        },
        Pair {
            params: [
                2760,
                29,
                23,
                45,
            ],
            balance: 233.43857,
        },
        Pair {
            params: [
                2940,
                22,
                20,
                38,
            ],
            balance: 239.8337,
        },
        Pair {
            params: [
                3300,
                23,
                21,
                43,
            ],
            balance: 283.01868,
        },
        Pair {
            params: [
                2520,
                35,
                23,
                38,
            ],
            balance: 304.46722,
        },
        Pair {
            params: [
                2280,
                17,
                13,
                53,
            ],
            balance: 311.84467,
        },
        Pair {
            params: [
                2400,
                30,
                21,
                40,
            ],
            balance: 345.01608,
        },
        Pair {
            params: [
                2340,
                30,
                22,
                45,
            ],
            balance: 381.81024,
        },
        Pair {
            params: [
                3420,
                21,
                18,
                39,
            ],
            balance: 409.54993,
        },
        Pair {
            params: [
                2640,
                28,
                22,
                42,
            ],
            balance: 557.80023,
        },
    ],
)
hybrid 15 10pop 1000gen
compute_cost: 3807116405,
cache_cost: 3807116405,
timeframe_cost: 3432328560,
elapsed_compute_s: 121.9636,
elapsed_cache_s: 168.46448,
elapsed_timeframe_s: 3.047476,
elapsed_total_s: 293.47556,
Top n: IntoIter(
    [
        Pair {
            params: [
                1140,
                68,
                28,
                51,
            ],
            balance: 160.64975,
        },
        Pair {
            params: [
                720,
                61,
                25,
                68,
            ],
            balance: 171.71454,
        },
        Pair {
            params: [
                480,
                87,
                33,
                63,
            ],
            balance: 179.25446,
        },
        Pair {
            params: [
                1380,
                60,
                28,
                45,
            ],
            balance: 230.124,
        },
        Pair {
            params: [
                3180,
                29,
                21,
                46,
            ],
            balance: 231.67537,
        },
        Pair {
            params: [
                2760,
                21,
                19,
                36,
            ],
            balance: 242.16997,
        },
        Pair {
            params: [
                2520,
                35,
                23,
                38,
            ],
            balance: 304.46722,
        },
        Pair {
            params: [
                3300,
                27,
                20,
                44,
            ],
            balance: 324.39722,
        },
        Pair {
            params: [
                3540,
                32,
                25,
                46,
            ],
            balance: 327.98148,
        },
        Pair {
            params: [
                3420,
                21,
                18,
                39,
            ],
            balance: 409.54993,
        },
    ],
)
hybrid 15 100pop 10gen
compute_cost: 3551765292,
cache_cost: 3551765292,
timeframe_cost: 3432328560,
elapsed_compute_s: 117.585266,
elapsed_cache_s: 157.77869,
elapsed_timeframe_s: 3.050927,
elapsed_total_s: 278.4149,
Top n: IntoIter(
    [
        Pair {
            params: [
                3540,
                32,
                25,
                47,
            ],
            balance: 189.27522,
        },
        Pair {
            params: [
                2160,
                42,
                27,
                46,
            ],
            balance: 189.59872,
        },
        Pair {
            params: [
                2040,
                49,
                26,
                45,
            ],
            balance: 190.87498,
        },
        Pair {
            params: [
                1140,
                63,
                28,
                45,
            ],
            balance: 191.4861,
        },
        Pair {
            params: [
                3180,
                15,
                16,
                48,
            ],
            balance: 195.77118,
        },
        Pair {
            params: [
                3660,
                21,
                16,
                41,
            ],
            balance: 226.02141,
        },
        Pair {
            params: [
                2520,
                27,
                22,
                45,
            ],
            balance: 239.02908,
        },
        Pair {
            params: [
                1380,
                63,
                28,
                63,
            ],
            balance: 244.80258,
        },
        Pair {
            params: [
                2280,
                20,
                18,
                55,
            ],
            balance: 258.63873,
        },
        Pair {
            params: [
                3420,
                20,
                16,
                41,
            ],
            balance: 288.43024,
        },
    ],
)
hybrid 15 100pop 100gen
compute_cost: 18112757741,
cache_cost: 18112757741,
timeframe_cost: 3432328560,
elapsed_compute_s: 582.78204,
elapsed_cache_s: 804.858,
elapsed_timeframe_s: 3.206861,
elapsed_total_s: 1390.8469,
Top n: IntoIter(
    [
        Pair {
            params: [
                2760,
                29,
                22,
                45,
            ],
            balance: 263.72806,
        },
        Pair {
            params: [
                2340,
                30,
                21,
                45,
            ],
            balance: 278.2208,
        },
        Pair {
            params: [
                1380,
                63,
                28,
                40,
            ],
            balance: 298.21088,
        },
        Pair {
            params: [
                2520,
                33,
                22,
                35,
            ],
            balance: 309.46555,
        },
        Pair {
            params: [
                2160,
                38,
                22,
                46,
            ],
            balance: 332.0676,
        },
        Pair {
            params: [
                2400,
                30,
                21,
                40,
            ],
            balance: 345.01608,
        },
        Pair {
            params: [
                2280,
                38,
                24,
                44,
            ],
            balance: 348.3299,
        },
        Pair {
            params: [
                2700,
                26,
                22,
                38,
            ],
            balance: 353.52902,
        },
        Pair {
            params: [
                2220,
                45,
                26,
                41,
            ],
            balance: 375.89032,
        },
        Pair {
            params: [
                2640,
                28,
                22,
                42,
            ],
            balance: 557.80023,
        },
    ],
)
hybrid 15 100pop 1000gen
compute_cost: 13285364956,
cache_cost: 13285364956,
timeframe_cost: 3241643640,
elapsed_compute_s: 438.22467,
elapsed_cache_s: 599.02094,
elapsed_timeframe_s: 16.407808,
elapsed_total_s: 1053.6534,
Top n: IntoIter(
    [
        Pair {
            params: [
                1380,
                63,
                28,
                40,
            ],
            balance: 298.21088,
        },
        Pair {
            params: [
                2520,
                35,
                23,
                38,
            ],
            balance: 304.46722,
        },
        Pair {
            params: [
                3480,
                22,
                20,
                36,
            ],
            balance: 323.31586,
        },
        Pair {
            params: [
                3660,
                47,
                26,
                43,
            ],
            balance: 328.93573,
        },
        Pair {
            params: [
                2400,
                30,
                21,
                40,
            ],
            balance: 345.01608,
        },
        Pair {
            params: [
                2280,
                43,
                25,
                39,
            ],
            balance: 475.26324,
        },
        Pair {
            params: [
                3420,
                21,
                19,
                39,
            ],
            balance: 487.2926,
        },
        Pair {
            params: [
                3540,
                19,
                21,
                39,
            ],
            balance: 516.53375,
        },
        Pair {
            params: [
                3540,
                19,
                22,
                41,
            ],
            balance: 573.63153,
        },
        Pair {
            params: [
                3600,
                19,
                17,
                39,
            ],
            balance: 613.6229,
        },
    ],
)
hybrid 15 1000pop 10gen
compute_cost: 32888648400,
cache_cost: 32888648400,
timeframe_cost: 3241643640,
elapsed_compute_s: 1089.5535,
elapsed_cache_s: 1495.0144,
elapsed_timeframe_s: 42.31548,
elapsed_total_s: 2626.8833,
Top n: IntoIter(
    [
        Pair {
            params: [
                3480,
                26,
                21,
                42,
            ],
            balance: 267.4995,
        },
        Pair {
            params: [
                2040,
                55,
                31,
                64,
            ],
            balance: 276.50677,
        },
        Pair {
            params: [
                1380,
                63,
                28,
                40,
            ],
            balance: 298.21088,
        },
        Pair {
            params: [
                3180,
                15,
                15,
                51,
            ],
            balance: 319.025,
        },
        Pair {
            params: [
                3660,
                47,
                26,
                43,
            ],
            balance: 328.93573,
        },
        Pair {
            params: [
                3540,
                26,
                20,
                40,
            ],
            balance: 342.42963,
        },
        Pair {
            params: [
                2280,
                37,
                22,
                38,
            ],
            balance: 349.2373,
        },
        Pair {
            params: [
                3420,
                24,
                18,
                34,
            ],
            balance: 416.6413,
        },
        Pair {
            params: [
                3600,
                23,
                20,
                44,
            ],
            balance: 537.59247,
        },
        Pair {
            params: [
                3540,
                19,
                22,
                39,
            ],
            balance: 599.8812,
        },
    ],
)
hybrid 15 1000pop 100gen
compute_cost: 138201766598,
cache_cost: 138201766598,
timeframe_cost: 3241643640,
elapsed_compute_s: 5193.3887,
elapsed_cache_s: 6752.035,
elapsed_timeframe_s: 193.29321,
elapsed_total_s: 12138.717,
Top n: IntoIter(
    [
        Pair {
            params: [
                2760,
                21,
                19,
                40,
            ],
            balance: 312.63156,
        },
        Pair {
            params: [
                3480,
                22,
                20,
                36,
            ],
            balance: 323.31586,
        },
        Pair {
            params: [
                3300,
                30,
                20,
                37,
            ],
            balance: 338.41608,
        },
        Pair {
            params: [
                3660,
                22,
                17,
                43,
            ],
            balance: 343.60617,
        },
        Pair {
            params: [
                2400,
                30,
                21,
                40,
            ],
            balance: 345.01608,
        },
        Pair {
            params: [
                2280,
                43,
                25,
                39,
            ],
            balance: 475.26324,
        },
        Pair {
            params: [
                3420,
                17,
                20,
                33,
            ],
            balance: 596.20874,
        },
        Pair {
            params: [
                3540,
                19,
                22,
                39,
            ],
            balance: 599.8812,
        },
        Pair {
            params: [
                3540,
                19,
                22,
                39,
            ],
            balance: 599.8812,
        },
        Pair {
            params: [
                3600,
                21,
                21,
                37,
            ],
            balance: 679.58966,
        },
    ],
)
hybrid 15 1000pop 1000gen
compute_cost: 145172656903,
cache_cost: 145172656903,
timeframe_cost: 3241643640,
elapsed_compute_s: 5290.0435,
elapsed_cache_s: 6976.6113,
elapsed_timeframe_s: 99.23288,
elapsed_total_s: 12365.887,
Top n: IntoIter(
    [
        Pair {
            params: [
                3480,
                22,
                20,
                36,
            ],
            balance: 323.31586,
        },
        Pair {
            params: [
                3180,
                37,
                26,
                38,
            ],
            balance: 335.95743,
        },
        Pair {
            params: [
                3300,
                30,
                20,
                37,
            ],
            balance: 338.41608,
        },
        Pair {
            params: [
                3660,
                22,
                17,
                43,
            ],
            balance: 343.60617,
        },
        Pair {
            params: [
                2400,
                30,
                21,
                40,
            ],
            balance: 345.01608,
        },
        Pair {
            params: [
                2280,
                43,
                25,
                39,
            ],
            balance: 475.26324,
        },
        Pair {
            params: [
                3420,
                21,
                19,
                39,
            ],
            balance: 487.2926,
        },
        Pair {
            params: [
                3540,
                19,
                22,
                39,
            ],
            balance: 599.8812,
        },
        Pair {
            params: [
                3540,
                19,
                22,
                39,
            ],
            balance: 599.8812,
        },
        Pair {
            params: [
                3600,
                19,
                17,
                39,
            ],
            balance: 613.6229,
        },
    ],
)
hybrid 30 10pop 10gen
compute_cost: 499836854,
cache_cost: 499836854,
timeframe_cost: 5529862680,
elapsed_compute_s: 16.443737,
elapsed_cache_s: 22.352774,
elapsed_timeframe_s: 5.062363,
elapsed_total_s: 43.85887,
Top n: IntoIter(
    [
        Pair {
            params: [
                3180,
                67,
                31,
                42,
            ],
            balance: 92.11942,
        },
        Pair {
            params: [
                1500,
                11,
                1,
                53,
            ],
            balance: 103.55827,
        },
        Pair {
            params: [
                1740,
                41,
                27,
                61,
            ],
            balance: 103.81322,
        },
        Pair {
            params: [
                2220,
                31,
                15,
                27,
            ],
            balance: 104.02185,
        },
        Pair {
            params: [
                3660,
                54,
                27,
                33,
            ],
            balance: 104.96378,
        },
        Pair {
            params: [
                960,
                27,
                15,
                79,
            ],
            balance: 108.757484,
        },
        Pair {
            params: [
                1740,
                74,
                32,
                50,
            ],
            balance: 112.01303,
        },
        Pair {
            params: [
                2160,
                34,
                21,
                51,
            ],
            balance: 118.814896,
        },
        Pair {
            params: [
                2220,
                50,
                25,
                36,
            ],
            balance: 121.583664,
        },
        Pair {
            params: [
                1800,
                50,
                27,
                58,
            ],
            balance: 147.24904,
        },
    ],
)
hybrid 30 10pop 100gen
compute_cost: 4278371772,
cache_cost: 4278371772,
timeframe_cost: 5148492840,
elapsed_compute_s: 168.17381,
elapsed_cache_s: 215.95775,
elapsed_timeframe_s: 58.856785,
elapsed_total_s: 442.98834,
Top n: IntoIter(
    [
        Pair {
            params: [
                2280,
                32,
                23,
                44,
            ],
            balance: 255.36589,
        },
        Pair {
            params: [
                2820,
                25,
                19,
                41,
            ],
            balance: 257.56262,
        },
        Pair {
            params: [
                2280,
                38,
                26,
                44,
            ],
            balance: 271.193,
        },
        Pair {
            params: [
                3300,
                27,
                20,
                35,
            ],
            balance: 304.99487,
        },
        Pair {
            params: [
                2220,
                45,
                26,
                41,
            ],
            balance: 375.89032,
        },
        Pair {
            params: [
                1740,
                42,
                25,
                41,
            ],
            balance: 421.93964,
        },
        Pair {
            params: [
                1740,
                42,
                25,
                41,
            ],
            balance: 421.93964,
        },
        Pair {
            params: [
                1740,
                42,
                25,
                41,
            ],
            balance: 421.93964,
        },
        Pair {
            params: [
                3540,
                21,
                19,
                43,
            ],
            balance: 458.91266,
        },
        Pair {
            params: [
                3420,
                21,
                19,
                39,
            ],
            balance: 487.2926,
        },
    ],
)
hybrid 30 10pop 100gen
compute_cost: 4078564164,
cache_cost: 4078564164,
timeframe_cost: 5911232520,
elapsed_compute_s: 159.05833,
elapsed_cache_s: 204.82547,
elapsed_timeframe_s: 33.53911,
elapsed_total_s: 397.4229,
Top n: IntoIter(
    [
        Pair {
            params: [
                2040,
                15,
                12,
                55,
            ],
            balance: 187.84428,
        },
        Pair {
            params: [
                1140,
                63,
                28,
                45,
            ],
            balance: 191.4861,
        },
        Pair {
            params: [
                1320,
                39,
                21,
                74,
            ],
            balance: 194.7783,
        },
        Pair {
            params: [
                2280,
                35,
                22,
                42,
            ],
            balance: 240.89867,
        },
        Pair {
            params: [
                1260,
                66,
                29,
                47,
            ],
            balance: 269.43814,
        },
        Pair {
            params: [
                3180,
                22,
                17,
                43,
            ],
            balance: 270.38684,
        },
        Pair {
            params: [
                1380,
                63,
                28,
                40,
            ],
            balance: 298.21088,
        },
        Pair {
            params: [
                3360,
                23,
                21,
                43,
            ],
            balance: 315.08017,
        },
        Pair {
            params: [
                3660,
                22,
                17,
                43,
            ],
            balance: 343.60617,
        },
        Pair {
            params: [
                3420,
                21,
                19,
                39,
            ],
            balance: 487.2926,
        },
    ],
)
hybrid 30 10pop 1000gen
compute_cost: 3623408187,
cache_cost: 3623408187,
timeframe_cost: 5529862680,
elapsed_compute_s: 130.68448,
elapsed_cache_s: 174.07826,
elapsed_timeframe_s: 41.36272,
elapsed_total_s: 346.1255,
Top n: IntoIter(
    [
        Pair {
            params: [
                2340,
                31,
                21,
                53,
            ],
            balance: 236.82265,
        },
        Pair {
            params: [
                2280,
                32,
                23,
                44,
            ],
            balance: 255.36589,
        },
        Pair {
            params: [
                3300,
                27,
                20,
                38,
            ],
            balance: 278.96448,
        },
        Pair {
            params: [
                2160,
                38,
                24,
                37,
            ],
            balance: 281.6913,
        },
        Pair {
            params: [
                1380,
                63,
                28,
                40,
            ],
            balance: 298.21088,
        },
        Pair {
            params: [
                2520,
                33,
                22,
                35,
            ],
            balance: 309.46555,
        },
        Pair {
            params: [
                2220,
                30,
                21,
                44,
            ],
            balance: 354.8458,
        },
        Pair {
            params: [
                1740,
                42,
                25,
                41,
            ],
            balance: 421.93964,
        },
        Pair {
            params: [
                2280,
                43,
                25,
                39,
            ],
            balance: 475.26324,
        },
        Pair {
            params: [
                2280,
                43,
                25,
                39,
            ],
            balance: 475.26324,
        },
    ],
)
hybrid 30 100pop 10gen
compute_cost: 5159848289,
cache_cost: 5159848289,
timeframe_cost: 5911232520,
elapsed_compute_s: 192.95425,
elapsed_cache_s: 251.24037,
elapsed_timeframe_s: 108.30361,
elapsed_total_s: 552.4982,
Top n: IntoIter(
    [
        Pair {
            params: [
                2940,
                12,
                11,
                33,
            ],
            balance: 225.58148,
        },
        Pair {
            params: [
                2340,
                25,
                19,
                34,
            ],
            balance: 244.80681,
        },
        Pair {
            params: [
                3180,
                37,
                24,
                38,
            ],
            balance: 249.30508,
        },
        Pair {
            params: [
                1260,
                58,
                29,
                42,
            ],
            balance: 249.97662,
        },
        Pair {
            params: [
                2400,
                32,
                23,
                44,
            ],
            balance: 252.78268,
        },
        Pair {
            params: [
                2280,
                11,
                8,
                78,
            ],
            balance: 258.6101,
        },
        Pair {
            params: [
                3420,
                23,
                18,
                39,
            ],
            balance: 269.63992,
        },
        Pair {
            params: [
                1740,
                41,
                26,
                41,
            ],
            balance: 275.23135,
        },
        Pair {
            params: [
                2280,
                38,
                24,
                44,
            ],
            balance: 348.3299,
        },
        Pair {
            params: [
                2220,
                30,
                21,
                44,
            ],
            balance: 354.8458,
        },
    ],
)
hybrid 30 100pop 100gen
compute_cost: 19279817841,
cache_cost: 19279817841,
timeframe_cost: 5720547600,
elapsed_compute_s: 692.63873,
elapsed_cache_s: 915.7718,
elapsed_timeframe_s: 211.78557,
elapsed_total_s: 1820.196,
Top n: IntoIter(
    [
        Pair {
            params: [
                2280,
                37,
                22,
                38,
            ],
            balance: 349.2373,
        },
        Pair {
            params: [
                3060,
                23,
                20,
                44,
            ],
            balance: 363.80692,
        },
        Pair {
            params: [
                2340,
                30,
                22,
                45,
            ],
            balance: 381.81024,
        },
        Pair {
            params: [
                2700,
                25,
                18,
                43,
            ],
            balance: 390.99402,
        },
        Pair {
            params: [
                2220,
                48,
                28,
                42,
            ],
            balance: 405.0882,
        },
        Pair {
            params: [
                3420,
                21,
                19,
                39,
            ],
            balance: 487.2926,
        },
        Pair {
            params: [
                2640,
                28,
                22,
                42,
            ],
            balance: 557.80023,
        },
        Pair {
            params: [
                2640,
                28,
                22,
                42,
            ],
            balance: 557.80023,
        },
        Pair {
            params: [
                2640,
                28,
                22,
                42,
            ],
            balance: 557.80023,
        },
        Pair {
            params: [
                2820,
                25,
                20,
                39,
            ],
            balance: 567.4214,
        },
    ],
)
hybrid 30 100pop 100gen
hybrid 30 compute_cost: 16984818479,
cache_cost: 16984818479,
timeframe_cost: 5148492840,
elapsed_compute_s: 572.5969,
elapsed_cache_s: 775.3382,
elapsed_timeframe_s: 27.662815,
elapsed_total_s: 1375.5979,
Top n: IntoIter(
    [
        Pair {
            params: [
                2700,
                25,
                18,
                43,
            ],
            balance: 390.99402,
        },
        Pair {
            params: [
                3540,
                21,
                19,
                43,
            ],
            balance: 458.91266,
        },
        Pair {
            params: [
                3420,
                21,
                19,
                39,
            ],
            balance: 487.2926,
        },
        Pair {
            params: [
                2640,
                28,
                22,
                42,
            ],
            balance: 557.80023,
        },
        Pair {
            params: [
                2640,
                28,
                22,
                42,
            ],
            balance: 557.80023,
        },
        Pair {
            params: [
                2640,
                28,
                22,
                42,
            ],
            balance: 557.80023,
        },
        Pair {
            params: [
                3540,
                19,
                22,
                39,
            ],
            balance: 599.8812,
        },
        Pair {
            params: [
                3600,
                19,
                17,
                39,
            ],
            balance: 613.6229,
        },
        Pair {
            params: [
                3600,
                19,
                17,
                39,
            ],
            balance: 613.6229,
        },
        Pair {
            params: [
                3600,
                19,
                17,
                39,
            ],
            balance: 613.6229,
        },
    ],
)
hybrid 30 100pop 1000gen
compute_cost: 16836051409,
cache_cost: 16836051409,
timeframe_cost: 5529862680,
elapsed_compute_s: 560.4923,
elapsed_cache_s: 759.6484,
elapsed_timeframe_s: 7.473574,
elapsed_total_s: 1327.6143,
Top n: IntoIter(
    [
        Pair {
            params: [
                2340,
                30,
                22,
                45,
            ],
            balance: 381.81024,
        },
        Pair {
            params: [
                2400,
                14,
                13,
                75,
            ],
            balance: 408.1628,
        },
        Pair {
            params: [
                3420,
                24,
                18,
                34,
            ],
            balance: 416.6413,
        },
        Pair {
            params: [
                3540,
                21,
                19,
                43,
            ],
            balance: 458.91266,
        },
        Pair {
            params: [
                3540,
                21,
                19,
                43,
            ],
            balance: 458.91266,
        },
        Pair {
            params: [
                2640,
                28,
                22,
                42,
            ],
            balance: 557.80023,
        },
        Pair {
            params: [
                3540,
                19,
                22,
                39,
            ],
            balance: 599.8812,
        },
        Pair {
            params: [
                3600,
                19,
                17,
                39,
            ],
            balance: 613.6229,
        },
        Pair {
            params: [
                3600,
                19,
                17,
                39,
            ],
            balance: 613.6229,
        },
        Pair {
            params: [
                3600,
                19,
                17,
                39,
            ],
            balance: 613.6229,
        },
    ],
)
hybrid 30 1000pop 10gen
compute_cost: 45895600168,
cache_cost: 45895600168,
timeframe_cost: 5529862680,
elapsed_compute_s: 1513.1284,
elapsed_cache_s: 2055.9387,
elapsed_timeframe_s: 37.86825,
elapsed_total_s: 3606.9353,
Top n: IntoIter(
    [
        Pair {
            params: [
                3420,
                21,
                16,
                39,
            ],
            balance: 350.68286,
        },
        Pair {
            params: [
                2580,
                45,
                28,
                39,
            ],
            balance: 373.3086,
        },
        Pair {
            params: [
                3240,
                22,
                23,
                39,
            ],
            balance: 381.19946,
        },
        Pair {
            params: [
                2220,
                48,
                28,
                42,
            ],
            balance: 405.0882,
        },
        Pair {
            params: [
                3600,
                23,
                18,
                44,
            ],
            balance: 411.6594,
        },
        Pair {
            params: [
                3540,
                21,
                18,
                43,
            ],
            balance: 420.11035,
        },
        Pair {
            params: [
                2640,
                28,
                26,
                42,
            ],
            balance: 452.46817,
        },
        Pair {
            params: [
                3540,
                21,
                19,
                43,
            ],
            balance: 458.91266,
        },
        Pair {
            params: [
                2280,
                43,
                25,
                39,
            ],
            balance: 475.26324,
        },
        Pair {
            params: [
                2280,
                43,
                25,
                39,
            ],
            balance: 475.26324,
        },
    ],
)
hybrid 30 1000pop 100gen
compute_cost: 185424521511,
cache_cost: 185424521511,
timeframe_cost: 5529862680,
elapsed_compute_s: 6242.427,
elapsed_cache_s: 8453.742,
elapsed_timeframe_s: 28.981466,
elapsed_total_s: 14725.15,
Top n: IntoIter(
    [
        Pair {
            params: [
                2280,
                43,
                25,
                39,
            ],
            balance: 475.26324,
        },
        Pair {
            params: [
                3420,
                21,
                19,
                39,
            ],
            balance: 487.2926,
        },
        Pair {
            params: [
                2880,
                34,
                22,
                39,
            ],
            balance: 540.58887,
        },
        Pair {
            params: [
                2880,
                34,
                22,
                39,
            ],
            balance: 540.58887,
        },
        Pair {
            params: [
                2640,
                28,
                22,
                42,
            ],
            balance: 557.80023,
        },
        Pair {
            params: [
                2820,
                25,
                20,
                39,
            ],
            balance: 567.4214,
        },
        Pair {
            params: [
                3540,
                19,
                22,
                39,
            ],
            balance: 599.8812,
        },
        Pair {
            params: [
                3540,
                19,
                22,
                39,
            ],
            balance: 599.8812,
        },
        Pair {
            params: [
                3540,
                19,
                22,
                39,
            ],
            balance: 599.8812,
        },
        Pair {
            params: [
                3600,
                21,
                21,
                37,
            ],
            balance: 679.58966,
        },
    ],
)
hybrid 30 1000pop 1000gen
compute_cost: 190321355344,
cache_cost: 190321355344,
timeframe_cost: 5529862680,
elapsed_compute_s: 6502.0664,
elapsed_cache_s: 8681.886,
elapsed_timeframe_s: 112.61642,
elapsed_total_s: 15296.568,
Top n: IntoIter(
    [
        Pair {
            params: [
                2280,
                43,
                25,
                39,
            ],
            balance: 475.26324,
        },
        Pair {
            params: [
                3420,
                21,
                19,
                39,
            ],
            balance: 487.2926,
        },
        Pair {
            params: [
                2880,
                34,
                22,
                39,
            ],
            balance: 540.58887,
        },
        Pair {
            params: [
                2880,
                34,
                22,
                39,
            ],
            balance: 540.58887,
        },
        Pair {
            params: [
                2640,
                28,
                22,
                42,
            ],
            balance: 557.80023,
        },
        Pair {
            params: [
                2820,
                25,
                20,
                39,
            ],
            balance: 567.4214,
        },
        Pair {
            params: [
                3540,
                19,
                22,
                39,
            ],
            balance: 599.8812,
        },
        Pair {
            params: [
                3540,
                19,
                22,
                39,
            ],
            balance: 599.8812,
        },
        Pair {
            params: [
                3540,
                19,
                22,
                39,
            ],
            balance: 599.8812,
        },
        Pair {
            params: [
                3600,
                21,
                21,
                37,
            ],
            balance: 679.58966,
        },
    ],
)
sharded 10pop 10gen
compute_cost: 116797103,
cache_cost: 116797103,
timeframe_cost: 10869040440,
elapsed_compute_s: 3.660687,
elapsed_cache_s: 5.127423,
elapsed_timeframe_s: 9.787669,
elapsed_total_s: 18.575779,
Top n: IntoIter(
    [
        Pair {
            params: [
                240,
                82,
                84,
                52,
            ],
            balance: 33.54827,
        },
        Pair {
            params: [
                3000,
                38,
                86,
                44,
            ],
            balance: 42.31916,
        },
        Pair {
            params: [
                1740,
                89,
                78,
                46,
            ],
            balance: 46.48601,
        },
        Pair {
            params: [
                600,
                64,
                24,
                64,
            ],
            balance: 50.47154,
        },
        Pair {
            params: [
                1380,
                67,
                31,
                46,
            ],
            balance: 57.47093,
        },
        Pair {
            params: [
                2700,
                63,
                84,
                43,
            ],
            balance: 63.27787,
        },
        Pair {
            params: [
                1980,
                52,
                28,
                63,
            ],
            balance: 67.521225,
        },
        Pair {
            params: [
                2520,
                28,
                23,
                44,
            ],
            balance: 75.505775,
        },
        Pair {
            params: [
                1020,
                59,
                24,
                38,
            ],
            balance: 90.89385,
        },
        Pair {
            params: [
                3600,
                50,
                30,
                65,
            ],
            balance: 105.697014,
        },
    ],
)
sharded 10pop 100gen
compute_cost: 1342532121,
cache_cost: 1342532121,
timeframe_cost: 11250410280,
elapsed_compute_s: 41.58593,
elapsed_cache_s: 58.62253,
elapsed_timeframe_s: 10.099989,
elapsed_total_s: 110.30846,
Top n: IntoIter(
    [
        Pair {
            params: [
                2700,
                36,
                85,
                38,
            ],
            balance: 67.53487,
        },
        Pair {
            params: [
                3120,
                79,
                78,
                52,
            ],
            balance: 75.99725,
        },
        Pair {
            params: [
                300,
                86,
                19,
                69,
            ],
            balance: 101.64587,
        },
        Pair {
            params: [
                840,
                53,
                23,
                67,
            ],
            balance: 113.06457,
        },
        Pair {
            params: [
                1200,
                10,
                100,
                91,
            ],
            balance: 139.88574,
        },
        Pair {
            params: [
                3540,
                60,
                31,
                51,
            ],
            balance: 157.15842,
        },
        Pair {
            params: [
                720,
                61,
                25,
                68,
            ],
            balance: 171.71454,
        },
        Pair {
            params: [
                1980,
                42,
                24,
                53,
            ],
            balance: 186.42853,
        },
        Pair {
            params: [
                1740,
                50,
                26,
                40,
            ],
            balance: 287.855,
        },
        Pair {
            params: [
                2400,
                30,
                21,
                40,
            ],
            balance: 345.01608,
        },
    ],
)
sharded 10pop 1000gen
compute_cost: 1142598141,
cache_cost: 1142598141,
timeframe_cost: 11250410280,
elapsed_compute_s: 35.48996,
elapsed_cache_s: 49.84927,
elapsed_timeframe_s: 10.123007,
elapsed_total_s: 95.46224,
Top n: IntoIter(
    [
        Pair {
            params: [
                1380,
                27,
                89,
                76,
            ],
            balance: 49.635822,
        },
        Pair {
            params: [
                1740,
                87,
                23,
                61,
            ],
            balance: 54.45603,
        },
        Pair {
            params: [
                960,
                71,
                81,
                56,
            ],
            balance: 55.90445,
        },
        Pair {
            params: [
                480,
                58,
                21,
                65,
            ],
            balance: 74.31776,
        },
        Pair {
            params: [
                2400,
                85,
                76,
                45,
            ],
            balance: 87.02823,
        },
        Pair {
            params: [
                360,
                64,
                20,
                70,
            ],
            balance: 99.95986,
        },
        Pair {
            params: [
                3060,
                99,
                74,
                53,
            ],
            balance: 110.3755,
        },
        Pair {
            params: [
                2100,
                90,
                34,
                50,
            ],
            balance: 175.92363,
        },
        Pair {
            params: [
                2640,
                28,
                22,
                42,
            ],
            balance: 557.80023,
        },
        Pair {
            params: [
                3600,
                19,
                17,
                39,
            ],
            balance: 613.6229,
        },
    ],
)
sharded 100pop 10gen
compute_cost: 1521767522,
cache_cost: 1521767522,
timeframe_cost: 11441095200,
elapsed_compute_s: 47.426727,
elapsed_cache_s: 66.44739,
elapsed_timeframe_s: 10.260082,
elapsed_total_s: 124.1342,
Top n: IntoIter(
    [
        Pair {
            params: [
                300,
                62,
                27,
                68,
            ],
            balance: 64.986336,
        },
        Pair {
            params: [
                540,
                81,
                28,
                66,
            ],
            balance: 104.491165,
        },
        Pair {
            params: [
                840,
                80,
                26,
                43,
            ],
            balance: 105.29919,
        },
        Pair {
            params: [
                1680,
                49,
                28,
                39,
            ],
            balance: 131.31525,
        },
        Pair {
            params: [
                2760,
                10,
                5,
                35,
            ],
            balance: 133.17545,
        },
        Pair {
            params: [
                1260,
                61,
                30,
                46,
            ],
            balance: 169.1841,
        },
        Pair {
            params: [
                3240,
                23,
                21,
                41,
            ],
            balance: 176.39182,
        },
        Pair {
            params: [
                2280,
                31,
                23,
                40,
            ],
            balance: 197.27788,
        },
        Pair {
            params: [
                2160,
                39,
                24,
                39,
            ],
            balance: 219.82231,
        },
        Pair {
            params: [
                3600,
                21,
                18,
                42,
            ],
            balance: 286.00897,
        },
    ],
)
sharded 100pop 100gen
compute_cost: 6854506571,
cache_cost: 6854506571,
timeframe_cost: 11441095200,
elapsed_compute_s: 226.16148,
elapsed_cache_s: 308.55115,
elapsed_timeframe_s: 10.819622,
elapsed_total_s: 545.5323,
Top n: IntoIter(
    [
        Pair {
            params: [
                360,
                64,
                20,
                70,
            ],
            balance: 99.95986,
        },
        Pair {
            params: [
                480,
                54,
                21,
                72,
            ],
            balance: 208.01686,
        },
        Pair {
            params: [
                1020,
                87,
                31,
                43,
            ],
            balance: 216.69336,
        },
        Pair {
            params: [
                1620,
                46,
                25,
                44,
            ],
            balance: 227.08669,
        },
        Pair {
            params: [
                1260,
                59,
                27,
                46,
            ],
            balance: 238.16562,
        },
        Pair {
            params: [
                3240,
                22,
                18,
                44,
            ],
            balance: 308.31873,
        },
        Pair {
            params: [
                2100,
                38,
                24,
                48,
            ],
            balance: 314.87256,
        },
        Pair {
            params: [
                2340,
                30,
                22,
                45,
            ],
            balance: 381.81024,
        },
        Pair {
            params: [
                2640,
                28,
                22,
                42,
            ],
            balance: 557.80023,
        },
        Pair {
            params: [
                3540,
                19,
                22,
                39,
            ],
            balance: 599.8812,
        },
    ],
)
sharded 1000pop 1000gen
compute_cost: 55347225370,
cache_cost: 55347225370,
timeframe_cost: 11441095200,
elapsed_compute_s: 1739.082,
elapsed_cache_s: 2433.272,
elapsed_timeframe_s: 10.278027,
elapsed_total_s: 4182.632,
Top n: IntoIter(
    [
        Pair {
            params: [
                300,
                91,
                20,
                69,
            ],
            balance: 110.86595,
        },
        Pair {
            params: [
                480,
                55,
                21,
                72,
            ],
            balance: 210.04955,
        },
        Pair {
            params: [
                1380,
                63,
                28,
                40,
            ],
            balance: 298.21088,
        },
        Pair {
            params: [
                900,
                50,
                24,
                71,
            ],
            balance: 336.55966,
        },
        Pair {
            params: [
                2220,
                40,
                25,
                36,
            ],
            balance: 412.15137,
        },
        Pair {
            params: [
                1740,
                42,
                25,
                41,
            ],
            balance: 421.93964,
        },
        Pair {
            params: [
                3240,
                23,
                19,
                37,
            ],
            balance: 525.1039,
        },
        Pair {
            params: [
                2820,
                25,
                20,
                39,
            ],
            balance: 567.4214,
        },
        Pair {
            params: [
                2100,
                42,
                25,
                37,
            ],
            balance: 602.05066,
        },
        Pair {
            params: [
                3600,
                21,
                21,
                37,
            ],
            balance: 679.58966,
        },
    ],
)
 */
/*
sharded 10pop 10gen
compute_cost: 120300268,
cache_cost: 120300268,
timeframe_cost: 11441095200,
elapsed_compute_s: 3.754076,
elapsed_cache_s: 5.276735,
elapsed_timeframe_s: 10.053675,
elapsed_total_s: 19.084486,
Top n: IntoIter(
    [
        Pair {
            params: [
                2640,
                68,
                80,
                67,
            ],
            balance: 28.150444,
        },
        Pair {
            params: [
                540,
                45,
                87,
                49,
            ],
            balance: 35.83357,
        },
        Pair {
            params: [
                2340,
                79,
                82,
                57,
            ],
            balance: 39.463417,
        },
        Pair {
            params: [
                300,
                73,
                22,
                42,
            ],
            balance: 40.272903,
        },
        Pair {
            params: [
                1920,
                76,
                38,
                56,
            ],
            balance: 40.886845,
        },
        Pair {
            params: [
                960,
                45,
                22,
                34,
            ],
            balance: 68.0488,
        },
        Pair {
            params: [
                3000,
                33,
                25,
                66,
            ],
            balance: 81.8685,
        },
        Pair {
            params: [
                1320,
                56,
                27,
                45,
            ],
            balance: 94.89741,
        },
        Pair {
            params: [
                3600,
                81,
                39,
                55,
            ],
            balance: 99.357124,
        },
        Pair {
            params: [
                1740,
                44,
                27,
                68,
            ],
            balance: 148.26797,
        },
    ],
)
sharded 10pop 100gen
compute_cost: 1336705015,
cache_cost: 1336705015,
timeframe_cost: 10678355520,
elapsed_compute_s: 42.61764,
elapsed_cache_s: 58.883823,
elapsed_timeframe_s: 9.716582,
elapsed_total_s: 111.21805,
Top n: IntoIter(
    [
        Pair {
            params: [
                1260,
                61,
                27,
                65,
            ],
            balance: 198.55684,
        },
        Pair {
            params: [
                1260,
                61,
                27,
                65,
            ],
            balance: 198.55684,
        },
        Pair {
            params: [
                1260,
                61,
                27,
                65,
            ],
            balance: 198.55684,
        },
        Pair {
            params: [
                1260,
                61,
                27,
                65,
            ],
            balance: 198.55684,
        },
        Pair {
            params: [
                1260,
                61,
                27,
                65,
            ],
            balance: 198.55684,
        },
        Pair {
            params: [
                1260,
                61,
                27,
                65,
            ],
            balance: 198.55684,
        },
        Pair {
            params: [
                1260,
                61,
                27,
                65,
            ],
            balance: 198.55684,
        },
        Pair {
            params: [
                1260,
                61,
                27,
                65,
            ],
            balance: 198.55684,
        },
        Pair {
            params: [
                1260,
                61,
                27,
                65,
            ],
            balance: 198.55684,
        },
        Pair {
            params: [
                1260,
                61,
                27,
                65,
            ],
            balance: 198.55684,
        },
    ],
)
sharded 10pop 1000gen
compute_cost: 1336034285,
cache_cost: 1336034285,
timeframe_cost: 11059725360,
elapsed_compute_s: 41.453728,
elapsed_cache_s: 58.431545,
elapsed_timeframe_s: 9.718003,
elapsed_total_s: 109.60327,
Top n: IntoIter(
    [
        Pair {
            params: [
                840,
                39,
                22,
                73,
            ],
            balance: 255.76738,
        },
        Pair {
            params: [
                840,
                39,
                22,
                73,
            ],
            balance: 255.76738,
        },
        Pair {
            params: [
                840,
                39,
                22,
                73,
            ],
            balance: 255.76738,
        },
        Pair {
            params: [
                840,
                39,
                22,
                73,
            ],
            balance: 255.76738,
        },
        Pair {
            params: [
                840,
                39,
                22,
                73,
            ],
            balance: 255.76738,
        },
        Pair {
            params: [
                840,
                39,
                22,
                73,
            ],
            balance: 255.76738,
        },
        Pair {
            params: [
                840,
                39,
                22,
                73,
            ],
            balance: 255.76738,
        },
        Pair {
            params: [
                840,
                39,
                22,
                73,
            ],
            balance: 255.76738,
        },
        Pair {
            params: [
                840,
                39,
                22,
                73,
            ],
            balance: 255.76738,
        },
        Pair {
            params: [
                2640,
                28,
                21,
                43,
            ],
            balance: 355.48788,
        },
    ],
)
sharded 100pop 10gen
compute_cost: 1487060447,
cache_cost: 1487060447,
timeframe_cost: 11441095200,
elapsed_compute_s: 46.224724,
elapsed_cache_s: 64.94512,
elapsed_timeframe_s: 10.07904,
elapsed_total_s: 121.248886,
Top n: IntoIter(
    [
        Pair {
            params: [
                240,
                64,
                22,
                58,
            ],
            balance: 45.84282,
        },
        Pair {
            params: [
                540,
                78,
                23,
                33,
            ],
            balance: 80.32329,
        },
        Pair {
            params: [
                780,
                76,
                26,
                40,
            ],
            balance: 102.73707,
        },
        Pair {
            params: [
                2400,
                27,
                22,
                70,
            ],
            balance: 125.14108,
        },
        Pair {
            params: [
                3180,
                25,
                24,
                37,
            ],
            balance: 132.83992,
        },
        Pair {
            params: [
                2880,
                36,
                25,
                39,
            ],
            balance: 147.03839,
        },
        Pair {
            params: [
                1260,
                65,
                28,
                47,
            ],
            balance: 161.03497,
        },
        Pair {
            params: [
                1740,
                42,
                28,
                44,
            ],
            balance: 170.1655,
        },
        Pair {
            params: [
                2100,
                44,
                25,
                47,
            ],
            balance: 217.40285,
        },
        Pair {
            params: [
                3420,
                21,
                19,
                38,
            ],
            balance: 432.0451,
        },
    ],
)
sharded 100pop 100gen
compute_cost: 6921355874,
cache_cost: 6921355874,
timeframe_cost: 11441095200,
elapsed_compute_s: 256.11804,
elapsed_cache_s: 336.35092,
elapsed_timeframe_s: 13.820202,
elapsed_total_s: 606.2892,
Top n: IntoIter(
    [
        Pair {
            params: [
                1740,
                42,
                25,
                41,
            ],
            balance: 421.93964,
        },
        Pair {
            params: [
                1740,
                42,
                25,
                41,
            ],
            balance: 421.93964,
        },
        Pair {
            params: [
                1740,
                42,
                25,
                41,
            ],
            balance: 421.93964,
        },
        Pair {
            params: [
                1740,
                42,
                25,
                41,
            ],
            balance: 421.93964,
        },
        Pair {
            params: [
                1740,
                42,
                25,
                41,
            ],
            balance: 421.93964,
        },
        Pair {
            params: [
                1740,
                42,
                25,
                41,
            ],
            balance: 421.93964,
        },
        Pair {
            params: [
                1740,
                42,
                25,
                41,
            ],
            balance: 421.93964,
        },
        Pair {
            params: [
                1740,
                42,
                25,
                41,
            ],
            balance: 421.93964,
        },
        Pair {
            params: [
                1740,
                42,
                25,
                41,
            ],
            balance: 421.93964,
        },
        Pair {
            params: [
                1740,
                42,
                25,
                41,
            ],
            balance: 421.93964,
        },
    ],
)
sharded 100pop 1000gen
compute_cost: 6897966849,
cache_cost: 6897966849,
timeframe_cost: 11441095200,
elapsed_compute_s: 214.3238,
elapsed_cache_s: 301.70914,
elapsed_timeframe_s: 10.128082,
elapsed_total_s: 526.161,
Top n: IntoIter(
    [
        Pair {
            params: [
                2640,
                25,
                21,
                41,
            ],
            balance: 490.45026,
        },
        Pair {
            params: [
                2640,
                25,
                21,
                41,
            ],
            balance: 490.45026,
        },
        Pair {
            params: [
                2640,
                25,
                21,
                41,
            ],
            balance: 490.45026,
        },
        Pair {
            params: [
                2640,
                25,
                21,
                41,
            ],
            balance: 490.45026,
        },
        Pair {
            params: [
                2640,
                25,
                21,
                41,
            ],
            balance: 490.45026,
        },
        Pair {
            params: [
                2640,
                25,
                21,
                41,
            ],
            balance: 490.45026,
        },
        Pair {
            params: [
                2640,
                25,
                21,
                41,
            ],
            balance: 490.45026,
        },
        Pair {
            params: [
                2640,
                25,
                21,
                41,
            ],
            balance: 490.45026,
        },
        Pair {
            params: [
                2640,
                25,
                21,
                41,
            ],
            balance: 490.45026,
        },
        Pair {
            params: [
                2640,
                25,
                21,
                41,
            ],
            balance: 490.45026,
        },
    ],
)
sharded 1000pop 10gen
compute_cost: 15226214748,
cache_cost: 15226214748,
timeframe_cost: 11441095200,
elapsed_compute_s: 505.2313,
elapsed_cache_s: 695.28235,
elapsed_timeframe_s: 11.53886,
elapsed_total_s: 1212.0525,
Top n: IntoIter(
    [
        Pair {
            params: [
                660,
                59,
                18,
                28,
            ],
            balance: 152.14384,
        },
        Pair {
            params: [
                1680,
                43,
                27,
                41,
            ],
            balance: 176.09875,
        },
        Pair {
            params: [
                1380,
                63,
                28,
                63,
            ],
            balance: 244.80258,
        },
        Pair {
            params: [
                2160,
                40,
                28,
                46,
            ],
            balance: 262.99493,
        },
        Pair {
            params: [
                900,
                48,
                24,
                71,
            ],
            balance: 308.48178,
        },
        Pair {
            params: [
                2400,
                30,
                21,
                40,
            ],
            balance: 345.01608,
        },
        Pair {
            params: [
                3420,
                17,
                17,
                34,
            ],
            balance: 354.5796,
        },
        Pair {
            params: [
                1740,
                42,
                25,
                41,
            ],
            balance: 421.93964,
        },
        Pair {
            params: [
                2640,
                28,
                22,
                41,
            ],
            balance: 438.71878,
        },
        Pair {
            params: [
                3000,
                12,
                12,
                36,
            ],
            balance: 649.1493,
        },
    ],
)
sharded 1000pop 100gen
compute_cost: 54344686730,
cache_cost: 54344686730,
timeframe_cost: 11441095200,
elapsed_compute_s: 1794.4093,
elapsed_cache_s: 2485.604,
elapsed_timeframe_s: 11.414354,
elapsed_total_s: 4291.4277,
Top n: IntoIter(
    [
        Pair {
            params: [
                3600,
                19,
                17,
                39,
            ],
            balance: 613.6229,
        },
        Pair {
            params: [
                3600,
                19,
                17,
                39,
            ],
            balance: 613.6229,
        },
        Pair {
            params: [
                3600,
                19,
                17,
                39,
            ],
            balance: 613.6229,
        },
        Pair {
            params: [
                3600,
                19,
                17,
                39,
            ],
            balance: 613.6229,
        },
        Pair {
            params: [
                3600,
                19,
                17,
                39,
            ],
            balance: 613.6229,
        },
        Pair {
            params: [
                3600,
                19,
                17,
                39,
            ],
            balance: 613.6229,
        },
        Pair {
            params: [
                3600,
                19,
                17,
                39,
            ],
            balance: 613.6229,
        },
        Pair {
            params: [
                3600,
                19,
                17,
                39,
            ],
            balance: 613.6229,
        },
        Pair {
            params: [
                3600,
                19,
                17,
                39,
            ],
            balance: 613.6229,
        },
        Pair {
            params: [
                3240,
                23,
                19,
                38,
            ],
            balance: 743.23956,
        },
    ],
)
sharded 1000pop 1000gen
compute_cost: 53341183599,
cache_cost: 53341183599,
timeframe_cost: 11441095200,
elapsed_compute_s: 1741.7816,
elapsed_cache_s: 2414.4243,
elapsed_timeframe_s: 10.780214,
elapsed_total_s: 4166.9863,
Top n: IntoIter(
    [
        Pair {
            params: [
                3600,
                19,
                17,
                39,
            ],
            balance: 613.6229,
        },
        Pair {
            params: [
                3600,
                19,
                17,
                39,
            ],
            balance: 613.6229,
        },
        Pair {
            params: [
                3600,
                19,
                17,
                39,
            ],
            balance: 613.6229,
        },
        Pair {
            params: [
                3600,
                19,
                17,
                39,
            ],
            balance: 613.6229,
        },
        Pair {
            params: [
                3600,
                19,
                17,
                39,
            ],
            balance: 613.6229,
        },
        Pair {
            params: [
                3600,
                19,
                17,
                39,
            ],
            balance: 613.6229,
        },
        Pair {
            params: [
                3600,
                19,
                17,
                39,
            ],
            balance: 613.6229,
        },
        Pair {
            params: [
                3600,
                19,
                17,
                39,
            ],
            balance: 613.6229,
        },
        Pair {
            params: [
                3600,
                19,
                17,
                39,
            ],
            balance: 613.6229,
        },
        Pair {
            params: [
                3600,
                19,
                17,
                39,
            ],
            balance: 613.6229,
        },
    ],
)
*/
